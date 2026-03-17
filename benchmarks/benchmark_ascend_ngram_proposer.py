# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace
from types import TracebackType

import numpy as np
from tabulate import tabulate

from vllm.config import SpeculativeConfig
from vllm.utils.argparse_utils import FlexibleArgumentParser


_NGRAM_PROPOSER_PATH = (
    Path(__file__).resolve().parents[1]
    / "vllm_ascend"
    / "spec_decode"
    / "ngram_proposer.py"
)
_NGRAM_PROPOSER_SPEC = importlib.util.spec_from_file_location(
    "benchmark_ascend_ngram_proposer_module",
    _NGRAM_PROPOSER_PATH,
)
assert _NGRAM_PROPOSER_SPEC is not None
assert _NGRAM_PROPOSER_SPEC.loader is not None
_NGRAM_PROPOSER_MODULE = importlib.util.module_from_spec(_NGRAM_PROPOSER_SPEC)
_NGRAM_PROPOSER_SPEC.loader.exec_module(_NGRAM_PROPOSER_MODULE)
AscendNgramProposer = _NGRAM_PROPOSER_MODULE.AscendNgramProposer


class TimeCollector:
    NS: int = 1
    US: int = NS * 1000

    def __init__(self, scale: int) -> None:
        self.cnt = 0
        self._sum = 0
        self._max: int | None = None
        self.scale = scale
        self.start_time = time.monotonic_ns()

    def collect(self, value: int) -> None:
        self.cnt += 1
        self._sum += value
        self._max = value if self._max is None else max(self._max, value)

    def avg(self) -> float | str:
        return self._sum / self.cnt / self.scale if self.cnt > 0 else "N/A"

    def max(self) -> float | str:
        return self._max / self.scale if self._max is not None else "N/A"

    def dump_avg_max(self) -> list[float | str]:
        return [self.avg(), self.max()]

    def __enter__(self) -> None:
        self.start_time = time.monotonic_ns()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> None:
        self.collect(time.monotonic_ns() - self.start_time)


def _build_runner_stub(num_req: int):
    return SimpleNamespace(
        input_batch=SimpleNamespace(
            req_ids=[str(i) for i in range(num_req)],
            spec_decode_unsupported_reqs=set(),
        )
    )


def _build_proposer(
    args,
    *,
    max_ngram: int,
    max_model_len: int | None = None,
) -> AscendNgramProposer:
    runner = _build_runner_stub(args.num_req)
    proposer = AscendNgramProposer(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                max_model_len=max_model_len
                or (args.num_token + args.num_spec_token),
            ),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
            scheduler_config=SimpleNamespace(max_num_seqs=args.num_req),
            speculative_config=SpeculativeConfig(
                prompt_lookup_min=args.min_ngram,
                prompt_lookup_max=max_ngram,
                num_speculative_tokens=args.num_spec_token,
                method="ngram",
            ),
        ),
        runner=runner,
    )
    return proposer


def benchmark_propose(args):
    rows = []
    for max_ngram in args.max_ngram:
        filter_collector = TimeCollector(TimeCollector.US)
        match_collector = TimeCollector(TimeCollector.US)
        materialize_collector = TimeCollector(TimeCollector.US)
        propose_collector = TimeCollector(TimeCollector.US)
        proposer = _build_proposer(args, max_ngram=max_ngram)
        sampled_token_ids = [[0] for _ in range(args.num_req)]
        num_tokens_no_spec = np.full(args.num_req, args.num_token, dtype=np.int32)

        gc.collect()
        for _ in range(args.num_iteration):
            token_ids_cpu = np.random.randint(
                0,
                20,
                (args.num_req, args.num_token),
                dtype=np.int32,
            )
            with filter_collector:
                valid_ngram_requests = proposer.get_valid_ngram_requests(
                    sampled_token_ids,
                    num_tokens_no_spec,
                )
            with match_collector:
                proposer.run_batch_match(
                    valid_ngram_requests,
                    num_tokens_no_spec,
                    token_ids_cpu,
                )
            with materialize_collector:
                proposer.materialize_draft_token_ids(
                    args.num_req,
                    valid_ngram_requests,
                )
            with propose_collector:
                proposer.propose(
                    sampled_token_ids,
                    num_tokens_no_spec,
                    token_ids_cpu,
                )
        rows.append(
            [
                args.num_req,
                args.num_token,
                args.min_ngram,
                max_ngram,
                *filter_collector.dump_avg_max(),
                *match_collector.dump_avg_max(),
                *materialize_collector.dump_avg_max(),
                *propose_collector.dump_avg_max(),
            ]
        )

    print(
        tabulate(
            rows,
            headers=[
                "# Request",
                "# Token",
                "Min Ngram",
                "Max Ngram",
                "Filter Avg (us)",
                "Filter Max (us)",
                "Match Avg (us)",
                "Match Max (us)",
                "Materialize Avg (us)",
                "Materialize Max (us)",
                "Propose Avg (us)",
                "Propose Max (us)",
            ],
            tablefmt="grid",
            floatfmt=".3f",
        )
    )


def benchmark_batched_propose(args):
    class FakeAscendRunner:
        def __init__(self, proposer: AscendNgramProposer, num_req: int, num_token: int):
            self.drafter = proposer
            self.input_batch = SimpleNamespace(
                req_ids=[str(i) for i in range(num_req)],
                spec_decode_unsupported_reqs=set(),
                num_tokens_no_spec=np.full(num_req, num_token, dtype=np.int32),
                token_ids_cpu=np.random.randint(
                    0,
                    20,
                    (num_req, num_token),
                    dtype=np.int32,
                ),
            )

        def get_ngram_proposal_inputs(self, sampled_token_ids: list[list[int]]):
            valid_ngram_requests = np.empty(len(sampled_token_ids), dtype=np.int32)
            num_valid_requests = 0
            for i, sampled_ids in enumerate(sampled_token_ids):
                if not sampled_ids:
                    continue
                if self.input_batch.num_tokens_no_spec[i] >= self.drafter.max_model_len:
                    continue
                valid_ngram_requests[num_valid_requests] = i
                num_valid_requests += 1

            return SimpleNamespace(
                sampled_token_ids=sampled_token_ids,
                num_tokens_no_spec=self.input_batch.num_tokens_no_spec,
                token_ids_cpu=self.input_batch.token_ids_cpu,
                valid_ngram_requests=valid_ngram_requests[:num_valid_requests],
            )

        def propose_ngram_draft_token_ids(
            self,
            sampled_token_ids: list[list[int]],
        ) -> list[list[int]]:
            ngram_inputs = self.get_ngram_proposal_inputs(sampled_token_ids)
            return self.drafter.propose(
                ngram_inputs.sampled_token_ids,
                ngram_inputs.num_tokens_no_spec,
                ngram_inputs.token_ids_cpu,
                valid_ngram_requests=ngram_inputs.valid_ngram_requests,
            )

    proposer = _build_proposer(args, max_ngram=max(args.max_ngram))
    runner = FakeAscendRunner(proposer, args.num_req, args.num_token)
    proposer.runner = runner
    sampled_token_ids = [[0]] * args.num_req
    inputs_collector = TimeCollector(TimeCollector.US)
    propose_collector = TimeCollector(TimeCollector.US)

    print("Starting benchmark")
    for _ in range(args.num_iteration):
        with inputs_collector:
            runner.get_ngram_proposal_inputs(sampled_token_ids)
        with propose_collector:
            runner.propose_ngram_draft_token_ids(sampled_token_ids)

    rows = [[
        args.num_req,
        args.num_token,
        inputs_collector.avg(),
        inputs_collector.max(),
        propose_collector.avg(),
        propose_collector.max(),
    ]]
    print(
        tabulate(
            rows,
            headers=[
                "# Request",
                "# Token",
                "Input Avg (us)",
                "Input Max (us)",
                "Runner Propose Avg (us)",
                "Runner Propose Max (us)",
            ],
            tablefmt="grid",
            floatfmt=".3f",
        )
    )


def main() -> None:
    parser = FlexibleArgumentParser(
        description="Benchmark Ascend N-gram proposer overhead on CPU."
    )
    parser.add_argument("--batched", action="store_true")
    parser.add_argument("--num-iteration", type=int, default=100)
    parser.add_argument("--num-req", type=int, default=128)
    parser.add_argument("--num-token", type=int, default=1500)
    parser.add_argument("--min-ngram", type=int, default=3)
    parser.add_argument(
        "--max-ngram",
        type=int,
        nargs="*",
        default=[5, 7, 10, 15, 20],
    )
    parser.add_argument("--num-spec-token", type=int, default=3)
    args = parser.parse_args()

    if args.batched:
        benchmark_batched_propose(args)
    else:
        benchmark_propose(args)


if __name__ == "__main__":
    main()

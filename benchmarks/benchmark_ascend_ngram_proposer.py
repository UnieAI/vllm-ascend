# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import enum
import importlib.util
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from types import TracebackType
from types import ModuleType

import numpy as np
from tabulate import tabulate

from vllm.utils.argparse_utils import FlexibleArgumentParser

_NGRAM_PROPOSER_PATH = (
    Path(__file__).resolve().parents[1]
    / "vllm_ascend"
    / "spec_decode"
    / "ngram_proposer.py"
)
_PKG_ROOT = _NGRAM_PROPOSER_PATH.parents[1]
_SPEC_DECODE_ROOT = _NGRAM_PROPOSER_PATH.parent

_VLLM_ASCEND_PKG = sys.modules.setdefault("vllm_ascend",
                                          ModuleType("vllm_ascend"))
_VLLM_ASCEND_PKG.__path__ = [str(_PKG_ROOT)]
_SPEC_DECODE_PKG = sys.modules.setdefault(
    "vllm_ascend.spec_decode", ModuleType("vllm_ascend.spec_decode"))
_SPEC_DECODE_PKG.__path__ = [str(_SPEC_DECODE_ROOT)]

_INTERFACE_MODULE = ModuleType("vllm_ascend.spec_decode.interface")


class _SpecDcodeType(enum.Enum):
    NGRAM = 0


class _Proposer:
    pass


_INTERFACE_MODULE.SpecDcodeType = _SpecDcodeType
_INTERFACE_MODULE.Proposer = _Proposer
sys.modules["vllm_ascend.spec_decode.interface"] = _INTERFACE_MODULE

_NGRAM_PROPOSER_SPEC = importlib.util.spec_from_file_location(
    "vllm_ascend.spec_decode.ngram_proposer",
    _NGRAM_PROPOSER_PATH,
)
assert _NGRAM_PROPOSER_SPEC is not None
assert _NGRAM_PROPOSER_SPEC.loader is not None
_NGRAM_PROPOSER_MODULE = importlib.util.module_from_spec(_NGRAM_PROPOSER_SPEC)
sys.modules["vllm_ascend.spec_decode.ngram_proposer"] = _NGRAM_PROPOSER_MODULE
_NGRAM_PROPOSER_SPEC.loader.exec_module(_NGRAM_PROPOSER_MODULE)
NgramProposer = _NGRAM_PROPOSER_MODULE.NgramProposer


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


def _build_runner_stub(num_req: int,
                       num_token: int,
                       max_model_len: int) -> SimpleNamespace:
    return SimpleNamespace(
        input_batch=SimpleNamespace(
            req_ids=[str(i) for i in range(num_req)],
            spec_decode_unsupported_reqs=set(),
            num_tokens_no_spec=np.full(num_req, num_token, dtype=np.int32),
            token_ids_cpu=np.random.randint(
                0,
                20,
                (num_req, max_model_len),
                dtype=np.int32,
            ),
            max_model_len=max_model_len,
        ))


def _build_proposer(
    args,
    *,
    max_ngram: int,
    max_model_len: int | None = None,
) -> NgramProposer:
    model_len = max_model_len or (args.num_token + args.num_spec_token)
    runner = _build_runner_stub(args.num_req, args.num_token, model_len)
    speculative_config = SimpleNamespace(
        prompt_lookup_min=args.min_ngram,
        prompt_lookup_max=max_ngram,
        num_speculative_tokens=args.num_spec_token,
        method="ngram",
        prompt_lookup_window=args.search_window,
    )
    return NgramProposer(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=model_len),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
            scheduler_config=SimpleNamespace(max_num_seqs=args.num_req),
            speculative_config=speculative_config,
        ),
        device="cpu",
        runner=runner,
    )


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
        rows.append([
            args.num_req,
            args.num_token,
            args.min_ngram,
            max_ngram,
            *filter_collector.dump_avg_max(),
            *match_collector.dump_avg_max(),
            *materialize_collector.dump_avg_max(),
            *propose_collector.dump_avg_max(),
        ])

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
        ))


def benchmark_generate(args):
    generate_collector = TimeCollector(TimeCollector.US)
    proposer = _build_proposer(args, max_ngram=max(args.max_ngram))
    sampled_token_ids = [[0]] * args.num_req

    for _ in range(args.num_iteration):
        proposer.runner.input_batch.token_ids_cpu = np.random.randint(
            0,
            20,
            (args.num_req, args.num_token + args.num_spec_token),
            dtype=np.int32,
        )
        proposer.runner.input_batch.num_tokens_no_spec = np.full(
            args.num_req,
            args.num_token,
            dtype=np.int32,
        )
        with generate_collector:
            proposer.generate_token_ids(sampled_token_ids)

    rows = [[
        args.num_req,
        args.num_token,
        generate_collector.avg(),
        generate_collector.max(),
    ]]
    print(
        tabulate(
            rows,
            headers=[
                "# Request",
                "# Token",
                "Generate Avg (us)",
                "Generate Max (us)",
            ],
            tablefmt="grid",
            floatfmt=".3f",
        ))


def main() -> None:
    parser = FlexibleArgumentParser(
        description="Benchmark Ascend N-gram proposer overhead on CPU.")
    parser.add_argument("--generate", action="store_true")
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
    parser.add_argument("--search-window", type=int, default=None)
    args = parser.parse_args()

    if args.generate:
        benchmark_generate(args)
    else:
        benchmark_propose(args)


if __name__ == "__main__":
    main()

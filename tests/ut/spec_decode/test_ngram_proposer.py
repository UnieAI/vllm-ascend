# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import numpy as np

_NGRAM_PROPOSER_PATH = (
    Path(__file__).resolve().parents[3]
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
_find_longest_matched_ngram_and_propose_tokens = (
    _NGRAM_PROPOSER_MODULE._find_longest_matched_ngram_and_propose_tokens
)


def _make_runner_stub(
    num_req: int,
    *,
    unsupported_req_ids: set[str] | None = None,
    max_model_len: int = 1024,
):
    return SimpleNamespace(
        input_batch=SimpleNamespace(
            req_ids=[str(i) for i in range(num_req)],
            spec_decode_unsupported_reqs=unsupported_req_ids or set(),
            max_model_len=max_model_len,
        ))


def _make_ngram_proposer(
    *,
    num_req: int,
    min_n: int,
    max_n: int,
    k: int,
    max_model_len: int = 1024,
    unsupported_req_ids: set[str] | None = None,
) -> NgramProposer:
    return NgramProposer(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=max_model_len),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
            scheduler_config=SimpleNamespace(max_num_seqs=num_req),
            speculative_config=SimpleNamespace(
                prompt_lookup_min=min_n,
                prompt_lookup_max=max_n,
                num_speculative_tokens=k,
                method="ngram",
            ),
        ),
        device="cpu",
        runner=_make_runner_stub(
            num_req,
            unsupported_req_ids=unsupported_req_ids,
            max_model_len=max_model_len,
        ),
    )


def _materialize_match(
    tokens: np.ndarray,
    *,
    min_ngram: int,
    max_ngram: int,
    max_model_len: int,
    k: int,
) -> np.ndarray:
    start_position, draft_len = _find_longest_matched_ngram_and_propose_tokens(
        origin_tokens=tokens,
        min_ngram=min_ngram,
        max_ngram=max_ngram,
        max_model_len=max_model_len,
        k=k,
    )
    return tokens[start_position:start_position + draft_len]


def test_find_longest_matched_ngram_and_propose_tokens():
    tokens = np.array([1, 2, 3, 4, 1, 2, 3], dtype=np.int32)
    np.testing.assert_array_equal(
        _materialize_match(
            tokens,
            min_ngram=2,
            max_ngram=2,
            max_model_len=1024,
            k=2,
        ),
        np.array([4, 1], dtype=np.int32),
    )


def test_ascend_ngram_proposer_skips_unsupported_requests():
    proposer = _make_ngram_proposer(
        num_req=2,
        min_n=2,
        max_n=2,
        k=2,
        unsupported_req_ids={"0"},
    )
    token_ids_cpu = np.array([[1, 2, 3, 1, 2], [7, 8, 9, 7, 8]], dtype=np.int32)
    result = proposer.propose(
        sampled_token_ids=[[0], [1]],
        num_tokens_no_spec=np.array([5, 5], dtype=np.int32),
        token_ids_cpu=token_ids_cpu,
    )
    assert result[0] == []
    assert result[1] == [9, 7]


def test_ascend_ngram_proposer_non_contiguous_indices():
    proposer = _make_ngram_proposer(
        num_req=3,
        min_n=2,
        max_n=2,
        k=2,
        max_model_len=20,
        unsupported_req_ids={"1"},
    )
    token_ids_cpu = np.zeros((3, 20), dtype=np.int32)
    token_ids_cpu[0, :5] = [1, 2, 3, 1, 2]
    token_ids_cpu[1, :5] = [4, 5, 6, 4, 5]
    token_ids_cpu[2, :5] = [7, 8, 9, 7, 8]
    result = proposer.propose(
        sampled_token_ids=[[2], [5], [8]],
        num_tokens_no_spec=np.array([5, 5, 5], dtype=np.int32),
        token_ids_cpu=token_ids_cpu,
    )
    assert result == [[3, 1], [], [9, 7]]
    assert proposer.valid_ngram_num_drafts[0] == 2
    assert proposer.valid_ngram_num_drafts[1] == 0
    assert proposer.valid_ngram_num_drafts[2] == 2


def test_ascend_ngram_get_valid_requests_returns_numpy_indices():
    proposer = _make_ngram_proposer(
        num_req=4,
        min_n=2,
        max_n=2,
        k=2,
        unsupported_req_ids={"1"},
    )
    valid = proposer.get_valid_ngram_requests(
        sampled_token_ids=[[1], [2], [], [3]],
        num_tokens_no_spec=np.array([4, 4, 4, 1024], dtype=np.int32),
    )
    np.testing.assert_array_equal(valid, np.array([0], dtype=np.int32))


def test_ascend_ngram_proposer_works_without_input_batch_in_runner():
    proposer = NgramProposer(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=20),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
            scheduler_config=SimpleNamespace(max_num_seqs=2),
            speculative_config=SimpleNamespace(
                prompt_lookup_min=2,
                prompt_lookup_max=2,
                num_speculative_tokens=2,
                method="ngram",
            ),
        ),
        device="cpu",
        runner=SimpleNamespace(),
    )
    token_ids_cpu = np.zeros((2, 20), dtype=np.int32)
    token_ids_cpu[0, :5] = [1, 2, 3, 1, 2]
    token_ids_cpu[1, :5] = [4, 5, 6, 4, 5]
    result = proposer.propose(
        sampled_token_ids=[[2], [5]],
        num_tokens_no_spec=np.array([5, 5], dtype=np.int32),
        token_ids_cpu=token_ids_cpu,
    )
    assert result == [[3, 1], [6, 4]]


def test_ascend_ngram_generate_token_ids_does_not_rewrite_token_ids_cpu():
    proposer = _make_ngram_proposer(
        num_req=1,
        min_n=2,
        max_n=2,
        k=2,
        max_model_len=20,
    )
    token_ids_cpu = np.zeros((1, 20), dtype=np.int32)
    token_ids_cpu[0, :7] = [1, 2, 3, 1, 2, 3, 1]
    input_batch = proposer.runner.input_batch
    input_batch.token_ids_cpu = token_ids_cpu
    input_batch.num_tokens_no_spec = np.array([7], dtype=np.int32)

    token_ids_before = token_ids_cpu.copy()
    result = proposer.generate_token_ids(valid_sampled_token_ids=[[3, 1]])

    np.testing.assert_array_equal(token_ids_cpu, token_ids_before)
    assert result == [[2, 3]]

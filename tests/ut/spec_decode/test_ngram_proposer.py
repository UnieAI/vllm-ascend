# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
import importlib.util
from pathlib import Path

import numpy as np

from vllm.config import SpeculativeConfig

_NGRAM_PROPOSER_PATH = (
    Path(__file__).resolve().parents[3]
    / "vllm_ascend"
    / "spec_decode"
    / "ngram_proposer.py"
)
_NGRAM_PROPOSER_SPEC = importlib.util.spec_from_file_location(
    "test_ascend_ngram_proposer_module",
    _NGRAM_PROPOSER_PATH,
)
assert _NGRAM_PROPOSER_SPEC is not None
assert _NGRAM_PROPOSER_SPEC.loader is not None
_NGRAM_PROPOSER_MODULE = importlib.util.module_from_spec(_NGRAM_PROPOSER_SPEC)
_NGRAM_PROPOSER_SPEC.loader.exec_module(_NGRAM_PROPOSER_MODULE)
AscendNgramProposer = _NGRAM_PROPOSER_MODULE.AscendNgramProposer
_find_longest_matched_ngram_and_propose_tokens = (
    _NGRAM_PROPOSER_MODULE._find_longest_matched_ngram_and_propose_tokens
)


def _make_runner_stub(
    num_req: int,
    *,
    unsupported_req_ids: set[str] | None = None,
):
    return SimpleNamespace(
        input_batch=SimpleNamespace(
            req_ids=[str(i) for i in range(num_req)],
            spec_decode_unsupported_reqs=unsupported_req_ids or set(),
        )
    )


def _make_ngram_proposer(
    *,
    num_req: int,
    min_n: int,
    max_n: int,
    k: int,
    max_model_len: int = 1024,
    unsupported_req_ids: set[str] | None = None,
) -> AscendNgramProposer:
    return AscendNgramProposer(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=max_model_len),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
            scheduler_config=SimpleNamespace(max_num_seqs=num_req),
            speculative_config=SpeculativeConfig(
                prompt_lookup_min=min_n,
                prompt_lookup_max=max_n,
                num_speculative_tokens=k,
                method="ngram",
            ),
        ),
        runner=_make_runner_stub(
            num_req,
            unsupported_req_ids=unsupported_req_ids,
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
    return tokens[start_position : start_position + draft_len]


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

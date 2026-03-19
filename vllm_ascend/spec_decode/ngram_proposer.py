import os
from typing import NamedTuple

import numpy as np
import torch
from numba import jit, njit, prange, set_num_threads
from vllm.config import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.spec_decode.ngram_proposer import \
    NgramProposer as VllmNgramProposer

from vllm_ascend.spec_decode.interface import Proposer, SpecDcodeType

logger = init_logger(__name__)


class AscendNgramProposalInputs(NamedTuple):
    sampled_token_ids: list[list[int]]
    num_tokens_no_spec: np.ndarray
    token_ids_cpu: np.ndarray
    valid_ngram_requests: np.ndarray


class NgramProposer(VllmNgramProposer, Proposer):

    def __init__(self, vllm_config, device, runner):
        self.name = SpecDcodeType.NGRAM
        self.device = device
        self.runner = runner
        super().__init__(vllm_config)
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.prompt_lookup_min is not None
        assert vllm_config.speculative_config.prompt_lookup_max is not None

        self.min_n = vllm_config.speculative_config.prompt_lookup_min
        self.max_n = vllm_config.speculative_config.prompt_lookup_max
        self.search_window = getattr(
            vllm_config.speculative_config,
            "prompt_lookup_window",
            None,
        )
        self.k = vllm_config.speculative_config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len

        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.valid_ngram_draft = np.zeros((max_num_seqs, self.k),
                                          dtype=np.int32)
        self.valid_ngram_num_drafts = np.zeros((max_num_seqs),
                                               dtype=np.int32)

        tp_size = vllm_config.parallel_config.tensor_parallel_size
        cpu_count = os.cpu_count()
        if cpu_count:
            default_numba_threads = min(
                4, max(1,
                       (cpu_count // 2) // max(1, tp_size)))
        else:
            default_numba_threads = 1
        self.num_numba_thread_available = max(
            1,
            int(
                os.environ.get("VLLM_ASCEND_NGRAM_NUMBA_THREADS",
                               str(default_numba_threads))),
        )
        self.num_numba_min_parallel_reqs = max(
            1,
            int(
                os.environ.get("VLLM_ASCEND_NGRAM_NUMBA_MIN_PARALLEL_REQS",
                               "8")),
        )
        # Keep one thread for small batches and only switch when needed.
        self._current_numba_threads = 1
        set_num_threads(self._current_numba_threads)
        self.no_match_backoff_enabled = bool(
            int(os.environ.get("VLLM_ASCEND_NGRAM_NO_MATCH_BACKOFF", "1")))
        self.no_match_backoff_min_len = int(
            os.environ.get("VLLM_ASCEND_NGRAM_NO_MATCH_BACKOFF_MIN_LEN", "256"))
        self.no_match_backoff_max_steps = int(
            os.environ.get("VLLM_ASCEND_NGRAM_NO_MATCH_BACKOFF_MAX_STEPS", "4"))
        self.no_match_backoff_max_steps = max(1,
                                              self.no_match_backoff_max_steps)
        self._req_skip_match_steps = np.zeros(max_num_seqs, dtype=np.int32)
        self._req_no_match_streak = np.zeros(max_num_seqs, dtype=np.int32)

        warmup_num_reqs = min(8, max_num_seqs)
        warmup_model_len = min(
            self.max_model_len,
            max(64, self.max_n + self.k, self.min_n + self.k),
        )
        warmup_valid_ngram_requests = np.arange(warmup_num_reqs,
                                                dtype=np.int32)
        self.propose(
            [[0]] * warmup_num_reqs,
            np.full(warmup_num_reqs, warmup_model_len, dtype=np.int32),
            np.zeros((warmup_num_reqs, warmup_model_len), dtype=np.int32),
            valid_ngram_requests=warmup_valid_ngram_requests,
        )
        self._req_skip_match_steps.fill(0)
        self._req_no_match_streak.fill(0)

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass

    @torch.inference_mode()
    def dummy_run(self,
                  num_tokens,
                  with_prefill=None,
                  skip_attn=None,
                  num_reqs=None,
                  num_tokens_across_dp=None,
                  aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
                  batch_descriptor=None,
                  dummy_compute_logits=lambda hidden_states: None):
        pass

    @staticmethod
    def _num_sampled_ids(sampled_ids) -> int:
        if sampled_ids is None:
            return 0
        try:
            return len(sampled_ids)
        except TypeError:
            return 1

    def should_propose_for_request(self, request_index: int,
                                   sampled_ids: list[int],
                                   num_tokens: int) -> bool:
        if self._num_sampled_ids(sampled_ids) == 0:
            return False
        if num_tokens >= self.max_model_len:
            return False
        return True

    def get_valid_ngram_requests(self, sampled_token_ids: list[list[int]],
                                 num_tokens_no_spec: np.ndarray) -> np.ndarray:
        # Align with upstream vLLM NgramProposer: ngram request eligibility is
        # based on sampled_ids and max_model_len only.
        valid_ngram_requests = np.empty(len(sampled_token_ids), dtype=np.int32)
        num_valid_requests = 0

        for i, sampled_ids in enumerate(sampled_token_ids):
            if self._num_sampled_ids(sampled_ids) == 0:
                continue
            num_tokens = num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                continue
            valid_ngram_requests[num_valid_requests] = i
            num_valid_requests += 1

        return valid_ngram_requests[:num_valid_requests]

    def run_batch_match(self, valid_ngram_requests: np.ndarray,
                        num_tokens_no_spec: np.ndarray,
                        token_ids_cpu: np.ndarray) -> None:
        num_ngram_requests = len(valid_ngram_requests)
        if not num_ngram_requests:
            return
        if not hasattr(self, "_current_numba_threads"):
            self._current_numba_threads = 1
            set_num_threads(self._current_numba_threads)
        desired_threads = 1
        if num_ngram_requests >= self.num_numba_min_parallel_reqs:
            desired_threads = min(self.num_numba_thread_available,
                                  num_ngram_requests)
        if desired_threads != self._current_numba_threads:
            set_num_threads(desired_threads)
            self._current_numba_threads = desired_threads

        batch_propose_numba(
            valid_ngram_requests,
            num_tokens_no_spec,
            token_ids_cpu,
            self.min_n,
            self.max_n,
            self.search_window if self.search_window is not None else 0,
            self.max_model_len,
            self.k,
            self.valid_ngram_draft,
            self.valid_ngram_num_drafts,
        )

    def materialize_draft_token_ids(
            self, num_requests: int,
            valid_ngram_requests: np.ndarray) -> list[list[int]]:
        draft_token_ids: list[list[int]] = [[] for _ in range(num_requests)]
        for i in valid_ngram_requests:
            if self.valid_ngram_num_drafts[i] > 0:
                draft_token_ids[i] = self.valid_ngram_draft[
                    i, :self.valid_ngram_num_drafts[i]].tolist()
        return draft_token_ids

    def _ensure_request_backoff_state(self, num_requests: int) -> None:
        if not hasattr(self, "_req_skip_match_steps") \
                or not hasattr(self, "_req_no_match_streak"):
            max_reqs = self.valid_ngram_num_drafts.shape[0]
            self._req_skip_match_steps = np.zeros(max_reqs, dtype=np.int32)
            self._req_no_match_streak = np.zeros(max_reqs, dtype=np.int32)
            return
        current_size = self._req_skip_match_steps.shape[0]
        if num_requests <= current_size:
            return
        new_size = max(num_requests, current_size * 2)
        new_skip = np.zeros(new_size, dtype=np.int32)
        new_streak = np.zeros(new_size, dtype=np.int32)
        new_skip[:current_size] = self._req_skip_match_steps
        new_streak[:current_size] = self._req_no_match_streak
        self._req_skip_match_steps = new_skip
        self._req_no_match_streak = new_streak

    def batch_propose(self, num_requests: int,
                      valid_ngram_requests: np.ndarray,
                      num_tokens_no_spec: np.ndarray,
                      token_ids_cpu: np.ndarray) -> list[list[int]]:
        self._ensure_request_backoff_state(num_requests)
        num_ngram_requests = len(valid_ngram_requests)
        if num_ngram_requests == 0:
            if num_requests > 0:
                self._req_skip_match_steps[:num_requests] = 0
                self._req_no_match_streak[:num_requests] = 0
            return [[] for _ in range(num_requests)]

        if self.no_match_backoff_enabled and num_requests > 0:
            active_mask = np.zeros(num_requests, dtype=np.bool_)
            active_mask[valid_ngram_requests] = True
            inactive_indices = np.nonzero(~active_mask)[0]
            if inactive_indices.size > 0:
                self._req_skip_match_steps[inactive_indices] = 0
                self._req_no_match_streak[inactive_indices] = 0

        self.valid_ngram_num_drafts[valid_ngram_requests] = 0
        run_mask = np.ones(num_ngram_requests, dtype=np.bool_)
        if self.no_match_backoff_enabled:
            for i, req_idx in enumerate(valid_ngram_requests):
                num_tokens = int(num_tokens_no_spec[req_idx])
                if num_tokens < self.no_match_backoff_min_len:
                    self._req_skip_match_steps[req_idx] = 0
                    self._req_no_match_streak[req_idx] = 0
                    continue
                remaining = int(self._req_skip_match_steps[req_idx])
                if remaining > 0:
                    self._req_skip_match_steps[req_idx] = remaining - 1
                    run_mask[i] = False

        run_requests = valid_ngram_requests[run_mask]
        if run_requests.size > 0:
            self.run_batch_match(
                run_requests,
                num_tokens_no_spec,
                token_ids_cpu,
            )

        draft_token_ids = self.materialize_draft_token_ids(
            num_requests,
            valid_ngram_requests,
        )
        if self.no_match_backoff_enabled:
            skipped_mask = ~run_mask
            for i, req_idx in enumerate(valid_ngram_requests):
                num_tokens = int(num_tokens_no_spec[req_idx])
                if num_tokens < self.no_match_backoff_min_len:
                    self._req_skip_match_steps[req_idx] = 0
                    self._req_no_match_streak[req_idx] = 0
                    continue
                if skipped_mask[i]:
                    # Already skipped by request-local backoff this step.
                    continue
                if self.valid_ngram_num_drafts[req_idx] > 0:
                    self._req_skip_match_steps[req_idx] = 0
                    self._req_no_match_streak[req_idx] = 0
                    continue
                streak = int(self._req_no_match_streak[req_idx]) + 1
                self._req_no_match_streak[req_idx] = streak
                self._req_skip_match_steps[req_idx] = min(
                    self.no_match_backoff_max_steps,
                    1 << min(streak - 1, 10),
                )
        return draft_token_ids

    def propose(self,
                sampled_token_ids: list[list[int]],
                num_tokens_no_spec: np.ndarray,
                token_ids_cpu: np.ndarray,
                slot_mappings=None,
                valid_ngram_requests: np.ndarray | None = None
                ) -> list[list[int]]:
        if valid_ngram_requests is None:
            valid_ngram_requests = self.get_valid_ngram_requests(
                sampled_token_ids,
                num_tokens_no_spec,
            )
        return self.batch_propose(
            len(sampled_token_ids),
            valid_ngram_requests,
            num_tokens_no_spec,
            token_ids_cpu,
        )

    def generate_token_ids(self,
                           valid_sampled_token_ids,
                           sampling_metadata=None,
                           scheduler_output=None,
                           spec_decode_metadata=None,
                           positions=None,
                           num_scheduled_tokens=None,
                           hidden_states=None,
                           attn_metadata=None,
                           aux_hidden_states=None) -> list[list[int]]:
        num_tokens_no_spec = self.runner.input_batch.num_tokens_no_spec
        token_ids_cpu = self.runner.input_batch.token_ids_cpu
        req_ids = self.runner.input_batch.req_ids
        num_requests = len(valid_sampled_token_ids)
        sampled_lens = np.fromiter((self._num_sampled_ids(ids)
                                    for ids in valid_sampled_token_ids),
                                   dtype=np.int32,
                                   count=num_requests)
        candidate_indices = np.nonzero(sampled_lens > 0)[0].astype(np.int32,
                                                                    copy=False)
        if candidate_indices.size == 0:
            return self.batch_propose(
                num_requests,
                candidate_indices,
                num_tokens_no_spec,
                token_ids_cpu,
            )

        over_limit_mask = num_tokens_no_spec[candidate_indices] >= \
            self.max_model_len
        if over_limit_mask.any() and len(req_ids) > 0:
            over_limit_indices = candidate_indices[over_limit_mask]
            for idx in over_limit_indices:
                i = int(idx)
                if i >= len(req_ids):
                    continue
                req_id = req_ids[i]
                req_state = self.runner.requests.get(req_id)
                if req_state is None:
                    continue
                req_state_tokens = int(req_state.num_tokens)
                if req_state_tokens < int(num_tokens_no_spec[i]):
                    num_tokens_no_spec[i] = req_state_tokens

        valid_mask = num_tokens_no_spec[candidate_indices] < self.max_model_len
        valid_ngram_requests = candidate_indices[valid_mask]

        return self.batch_propose(
            num_requests,
            valid_ngram_requests,
            num_tokens_no_spec,
            token_ids_cpu,
        )


@njit(parallel=True)
def batch_propose_numba(
    valid_ngram_requests: np.ndarray,
    num_tokens_no_spec: np.ndarray,
    token_ids_cpu: np.ndarray,
    min_n: int,
    max_n: int,
    search_window: int,
    max_model_len: int,
    k: int,
    valid_ngram_draft: np.ndarray,
    valid_ngram_num_drafts: np.ndarray,
):
    for i in prange(len(valid_ngram_requests)):
        idx = valid_ngram_requests[i]
        num_tokens = num_tokens_no_spec[idx]
        search_start = 0
        if search_window > 0 and num_tokens > search_window:
            search_start = num_tokens - search_window
        context_token_ids = token_ids_cpu[idx, search_start:num_tokens]
        start_position, draft_len = _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=context_token_ids,
            min_ngram=min_n,
            max_ngram=max_n,
            max_model_len=max_model_len,
            k=k,
        )
        valid_ngram_num_drafts[idx] = draft_len
        if draft_len > 0:
            valid_ngram_draft[idx, :draft_len] = context_token_ids[
                start_position:start_position + draft_len]


@jit(nopython=True)
def _find_longest_matched_ngram_and_propose_tokens(
    origin_tokens: np.ndarray,
    min_ngram: int,
    max_ngram: int,
    max_model_len: int,
    k: int,
) -> tuple[int, int]:
    total_token = origin_tokens.shape[0]
    if total_token < min_ngram:
        return 0, 0

    k = min(k, max_model_len - total_token)
    if k <= 0:
        return 0, 0

    tokens = origin_tokens[::-1]
    lps = np.zeros(max_ngram, dtype=np.int32)

    longest_ngram = 0
    position = 0
    prev_lps = 0
    i = 1
    while i < total_token:
        if tokens[prev_lps] == tokens[i]:
            prev_lps += 1
            if prev_lps >= longest_ngram:
                longest_ngram = prev_lps
                position = i
            if i < max_ngram:
                lps[i] = prev_lps
            if prev_lps == max_ngram:
                prev_lps = lps[max_ngram - 1]
            i += 1
        elif prev_lps != 0:
            prev_lps = lps[prev_lps - 1]
        else:
            i += 1

    if longest_ngram < min_ngram:
        return 0, 0

    start_position = total_token - 1 - position + longest_ngram
    draft_len = min(k, total_token - start_position)
    return start_position, draft_len

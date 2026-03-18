# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch
import torch.nn as nn
import vllm.v1.sample.rejection_sampler as rs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import (RejectionSampler, compute_probs,
                                              generate_uniform_probs)
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

PLACEHOLDER_TOKEN_ID = -1
GREEDY_TEMPERATURE = -1
# Maximum number of speculative draft tokens allowed per request in a single
# step. This value is chosen to be large enough to handle typical use cases.
MAX_SPEC_LEN = 32


class AscendRejectionSampler(RejectionSampler, nn.Module):
    """
    The implementation strictly follows the algorithm described in
        https://arxiv.org/abs/2211.17192.
    However, we want to clarify the terminology used in the implementation:
    accepted tokens: tokens that are accepted based on the relationship
            between the "raw" draft and target probabilities.
    recovered tokens: tokens that are sampled based on the adjusted probability
        distribution, which is derived from both the draft and target
        probabilities.
    bonus tokens:
        If all proposed tokens are accepted, the bonus token is added to the
        end of the sequence. The bonus token is only sampled from the target
        probabilities. We pass in the bonus tokens instead of sampling them
        in the rejection sampler to allow for more flexibility in the
        sampling process. For example, we can use top_p, top_k sampling for
        bonus tokens, while spec decode does not support these sampling
        strategies.
    output tokens:
        Tokens are finally generated with the rejection sampler.
        output tokens = accepted tokens + recovered tokens + bonus tokens
    """

    def forward(
        self,
        metadata: SpecDecodeMetadata,
        # [num_tokens, vocab_size]
        draft_probs: Optional[torch.Tensor],
        # [num_tokens, vocab_size]
        target_logits: torch.Tensor,
        # [batch_size, 1]
        bonus_token_ids: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        '''
        Args:
            metadata:
                Metadata for spec decoding.
            draft_probs (Optional[torch.Tensor]):
                Probability distribution for the draft tokens. Shape is
                [num_tokens, vocab_size]. Can be None if probabilities are
                not provided, which is the case for ngram spec decode.
            target_logits (torch.Tensor):
                Target model's logits probability distribution.
                Shape is [num_tokens, vocab_size]. Here, probabilities from
                different requests are flattened into a single tensor because
                this is the shape of the output logits.
                NOTE: `target_logits` can be updated in place to save memory.
            bonus_token_ids_tensor (torch.Tensor):
                A tensor containing bonus tokens. Shape is [batch_size, 1].
                Bonus tokens are added to the end of the sequence if all
                proposed tokens are accepted. We generate the bonus tokens
                outside of the rejection sampler with the default sampling
                strategy. It allows for more flexibility in the sampling
                process such as top_p, top_k sampling.
            sampling_metadata (SamplingMetadata):
                Additional metadata needed for sampling, such as temperature,
                top-k/top-p parameters, or other relevant information.
        Returns:
            output_token_ids (torch.Tensor):
                A tensor containing the final output token IDs.
        '''
        assert metadata.max_spec_len <= MAX_SPEC_LEN
        # [num_tokens, vocab_size]
        # NOTE(woosuk): `target_logits` can be updated in place inside the
        # `compute_probs` function.
        target_probs = compute_probs(
            target_logits,
            metadata.cu_num_draft_tokens,
            sampling_metadata,
        )

        output_token_ids = rejection_sample(
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            metadata.cu_num_draft_tokens,
            draft_probs,
            target_probs,
            bonus_token_ids,
            sampling_metadata,
        )
        return output_token_ids


def rejection_sample(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    max_spec_len: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: Optional[torch.Tensor],
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    assert draft_token_ids.ndim == 1
    assert draft_probs is None or draft_probs.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    assert target_probs.ndim == 2

    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    vocab_size = target_probs.shape[-1]
    device = target_probs.device
    assert draft_token_ids.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert target_probs.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_probs.shape == (num_tokens, vocab_size)

    # Create output buffer.
    output_token_ids = torch.empty(
        (batch_size, max_spec_len + 1),
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )
    output_token_ids.fill_(PLACEHOLDER_TOKEN_ID)

    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE
    if not sampling_metadata.all_random:
        # Rejection sampling for greedy sampling requests.
        target_argmax = target_probs.argmax(dim=-1)
        if min(num_draft_tokens) == 1 and max(
                num_draft_tokens) == 1 and sampling_metadata.all_greedy:
            rejection_greedy_sample_spec_len_1_pytorch(
                output_token_ids,
                draft_token_ids,
                target_argmax,
                bonus_token_ids,
            )
        else:
            rejection_greedy_sample_pytorch(
                output_token_ids,
                cu_num_draft_tokens,
                draft_token_ids,
                target_argmax,
                bonus_token_ids,
                num_draft_tokens,
                max_spec_len,
                is_greedy,
            )
        if sampling_metadata.all_greedy:
            return output_token_ids

    # Generate uniform probabilities for rejection sampling.
    # [num_tokens]
    uniform_probs = generate_uniform_probs(
        num_tokens,
        num_draft_tokens,
        sampling_metadata.generators,
        device,
    )

    if draft_probs is None:
        # Ngram path: only the first rejected draft token per request is used,
        # so compute recovered tokens only for those positions.
        rejection_random_sample_ngram_pytorch(
            output_token_ids,
            draft_token_ids,
            target_probs,
            bonus_token_ids,
            uniform_probs,
            is_greedy,
            num_draft_tokens,
            max_spec_len,
            vocab_size,
            sampling_metadata,
        )
        return output_token_ids

    # Sample recovered tokens for each position.
    # [num_tokens]
    recovered_token_ids = sample_recovered_tokens(
        max_spec_len,
        num_draft_tokens,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        sampling_metadata,
        device,
    )

    # Rejection sampling for random sampling requests.
    rejection_random_sample_pytorch(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        num_draft_tokens,
        max_spec_len,
        vocab_size,
        IS_NGRAM=False,
        # num_warps=1,
    )
    return output_token_ids


def expand_batch_to_tokens(
    x: torch.Tensor,  # [batch_size]
    cu_num_tokens: torch.Tensor,  # [batch_size]
    num_tokens: int,
    replace_from: int = 0,
    replace_to: int = 0,
) -> torch.Tensor:
    """Expand [batch_size] tensor to [num_tokens] tensor based on the number of
    tokens per batch in cu_num_tokens.

    For example, if x = [a, b, c] and cu_num_tokens = [2, 5, 6], then
    num_tokens = 6, and expanded_x = [a, a, b, b, b, c].

    Args:
        x: [batch_size] tensor to expand.
        cu_num_tokens: [batch_size] tensor containing the cumulative number of
            tokens per batch. Each element represents the total number of
            tokens up to and including that batch.
        num_tokens: Total number of tokens.
        replace_from: int = 0
            Value to be replaced if it is found in x.
        replace_to: int = 0
            Value to replace with when replace_from is found.
    Returns:
        expanded_x: [num_tokens] tensor.
    """
    batch_size = x.shape[0]
    assert cu_num_tokens.shape[0] == batch_size
    expanded_x = x.new_empty(num_tokens)
    expand_pytorch(
        expanded_x,
        x,
        cu_num_tokens,
        replace_from,
        replace_to,
        MAX_NUM_TOKENS=MAX_SPEC_LEN,  # To avoid recompilation.
    )
    return expanded_x


def _generate_exponential_q(
    num_draft_tokens: list[int],
    sampling_metadata: SamplingMetadata,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    batch_size = len(num_draft_tokens)
    q = torch.empty(
        (batch_size, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    q.exponential_()
    for i, generator in sampling_metadata.generators.items():
        # Do not generate random numbers for requests with no draft tokens.
        # This can be important for reproducibility.
        if num_draft_tokens[i] > 0:
            q[i].exponential_(generator=generator)
    return q


def sample_recovered_tokens(
    max_spec_len: int,
    num_draft_tokens: list[int],
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: Optional[torch.Tensor],
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    device: torch.device,
) -> torch.Tensor:
    vocab_size = target_probs.shape[-1]
    q = _generate_exponential_q(
        num_draft_tokens,
        sampling_metadata,
        vocab_size,
        device,
    )

    recovered_token_ids = torch.empty_like(draft_token_ids)
    sample_recovered_tokens_pytorch(
        recovered_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        q,
        vocab_size,
        IS_NGRAM=draft_probs is None,
    )
    return recovered_token_ids


def rejection_greedy_sample_spec_len_1_pytorch(
        output_token_ids,  # [batch_size, 2]
        draft_token_ids,  # [num_tokens]
        target_argmax,  # [num_tokens]
        bonus_token_ids,  # [batch_size]
):
    batch_size = output_token_ids.size(0)
    num_tokens = draft_token_ids.size(0)
    assert batch_size == num_tokens
    accept_req_mask = draft_token_ids == target_argmax
    output_token_ids[:, 0] = target_argmax
    bonus_token_ids = bonus_token_ids.squeeze(1)
    output_token_ids[accept_req_mask, 1] = bonus_token_ids[accept_req_mask]


def rejection_greedy_sample_pytorch(
        output_token_ids,  # [batch_size, max_spec_len + 1]
        cu_num_draft_tokens,  # [batch_size]
        draft_token_ids,  # [num_tokens]
        target_argmax,  # [num_tokens]
        bonus_token_ids,  # [batch_size]
        draft_tokens_per_req,  # [batch_size], list
        max_spec_len,
        is_greedy=None,  # [batch_size] or None
):
    batch_size = output_token_ids.size(0)
    num_tokens = draft_token_ids.size(0)
    device = output_token_ids.device
    draft_tokens_per_req = torch.tensor(draft_tokens_per_req).to(
        device, non_blocking=True)
    if is_greedy is None:
        is_greedy = torch.ones(batch_size, dtype=torch.bool, device=device)

    start_indices = cu_num_draft_tokens - draft_tokens_per_req
    req_ids = torch.arange(batch_size, device=device)
    token_req_ids = torch.repeat_interleave(req_ids, draft_tokens_per_req)
    token_positions = torch.arange(
        num_tokens, device=device) - start_indices[token_req_ids]

    # Find the first mismatch position of each request.
    mismatch_global = (draft_token_ids != target_argmax)
    if max_spec_len == 0:
        first_mismatch_pos_per_req = torch.zeros(batch_size,
                                                 dtype=torch.long,
                                                 device=device)
    else:
        # [bs, max_spec_len]
        pos_matrix = torch.full((batch_size, max_spec_len),
                                -1,
                                dtype=torch.long,
                                device=device)
        pos_matrix[token_req_ids, token_positions] = token_positions
        mismatch_matrix = torch.full((batch_size, max_spec_len),
                                     False,
                                     dtype=torch.bool,
                                     device=device)
        mismatch_matrix[token_req_ids, token_positions] = mismatch_global
        mismatch_positions = torch.where(mismatch_matrix, pos_matrix,
                                         max_spec_len * 2)
        first_mismatch_pos_per_req, _ = torch.min(mismatch_positions, dim=1)
        no_mismatch_mask = (first_mismatch_pos_per_req == max_spec_len * 2)
        first_mismatch_pos_per_req[no_mismatch_mask] = draft_tokens_per_req[
            no_mismatch_mask]

    # Copy matched target tokens into output.
    copy_len = torch.minimum(first_mismatch_pos_per_req + 1,
                             draft_tokens_per_req)
    copy_indices = torch.arange(max_spec_len + 1,
                                device=device).expand(batch_size, -1)
    copy_mask = copy_indices < copy_len.unsqueeze(1)
    greedy_mask = is_greedy.unsqueeze(1)
    final_copy_mask = copy_mask & greedy_mask
    global_idx = start_indices.unsqueeze(1) + copy_indices
    output_token_ids[final_copy_mask] = target_argmax[
        global_idx[final_copy_mask]].to(output_token_ids.dtype)
    # Fill bonus token.
    needs_bonus = is_greedy & (first_mismatch_pos_per_req
                               >= draft_tokens_per_req)
    if torch.any(needs_bonus):
        bonus_rows = torch.where(needs_bonus)[0]
        bonus_cols = draft_tokens_per_req[bonus_rows]
        bonus_token_ids = bonus_token_ids.squeeze(1)
        output_token_ids[bonus_rows, bonus_cols] = bonus_token_ids[bonus_rows]


def rejection_random_sample_pytorch(
    output_token_ids,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens,  # [batch_size], unused for vectorized path
    draft_token_ids,  # [num_tokens]
    draft_probs,  # [num_tokens, vocab_size] or None
    target_probs,  # [num_tokens, vocab_size]
    bonus_token_ids,  # [batch_size]
    recovered_token_ids,  # [num_tokens]
    uniform_probs,  # [num_tokens]
    is_greedy,  # [batch_size]
    num_draft_tokens_per_req,  # [batch_size], list[int]
    max_spec_len,
    vocab_size,
    IS_NGRAM=False,
):
    batch_size = output_token_ids.shape[0]
    device = output_token_ids.device
    if batch_size == 0:
        return

    bonus_token_ids = bonus_token_ids.squeeze(1).to(output_token_ids.dtype)
    num_draft_tokens = torch.tensor(num_draft_tokens_per_req,
                                    dtype=torch.long,
                                    device=device)
    non_greedy = ~is_greedy
    rows_no_draft = torch.where(non_greedy & (num_draft_tokens == 0))[0]
    if rows_no_draft.numel() > 0:
        output_token_ids[rows_no_draft, 0] = bonus_token_ids[rows_no_draft]

    num_tokens = draft_token_ids.shape[0]
    if num_tokens == 0:
        return

    req_ids = torch.arange(batch_size, device=device, dtype=torch.long)
    start_idx = torch.cumsum(num_draft_tokens, dim=0) - num_draft_tokens
    token_req_ids = torch.repeat_interleave(req_ids, num_draft_tokens)
    token_positions = torch.arange(num_tokens, device=device,
                                   dtype=torch.long) - start_idx[token_req_ids]

    token_row_idx = torch.arange(num_tokens, device=device, dtype=torch.long)
    req_draft_token_ids = draft_token_ids.to(torch.long)
    target_prob = target_probs[token_row_idx, req_draft_token_ids]
    if IS_NGRAM:
        draft_prob = torch.ones_like(target_prob)
    else:
        assert draft_probs is not None
        draft_prob = draft_probs[token_row_idx, req_draft_token_ids]
    accept_flat = (draft_prob > 0) & (target_prob / draft_prob >= uniform_probs)

    accept_matrix = torch.zeros((batch_size, max_spec_len),
                                dtype=torch.bool,
                                device=device)
    accept_matrix[token_req_ids, token_positions] = accept_flat

    valid_pos_mask = (
        torch.arange(max_spec_len, device=device).unsqueeze(0)
        < num_draft_tokens.unsqueeze(1))
    prefix_accept = (torch.cumprod(accept_matrix.to(torch.int32), dim=1) > 0) \
        & valid_pos_mask

    draft_matrix = torch.full((batch_size, max_spec_len),
                              PLACEHOLDER_TOKEN_ID,
                              dtype=output_token_ids.dtype,
                              device=device)
    draft_matrix[token_req_ids, token_positions] = draft_token_ids.to(
        output_token_ids.dtype)

    accepted_write_mask = prefix_accept & non_greedy.unsqueeze(1)
    if accepted_write_mask.any():
        output_token_ids[accepted_write_mask] = draft_matrix[accepted_write_mask]

    reject_matrix = (~accept_matrix) & valid_pos_mask
    has_reject = reject_matrix.any(dim=1) & non_greedy
    if has_reject.any():
        recovered_matrix = torch.zeros((batch_size, max_spec_len),
                                       dtype=output_token_ids.dtype,
                                       device=device)
        recovered_matrix[token_req_ids,
                         token_positions] = recovered_token_ids.to(
                             output_token_ids.dtype)
        first_reject_pos = torch.argmax(reject_matrix.to(torch.int32), dim=1)
        rows = torch.where(has_reject)[0]
        cols = first_reject_pos[rows]
        output_token_ids[rows, cols] = recovered_matrix[rows, cols]

    no_reject = non_greedy & (~has_reject)
    if no_reject.any():
        rows = torch.where(no_reject)[0]
        cols = num_draft_tokens[rows]
        output_token_ids[rows, cols] = bonus_token_ids[rows]


def rejection_random_sample_ngram_pytorch(
    output_token_ids,  # [batch_size, max_spec_len + 1]
    draft_token_ids,  # [num_tokens]
    target_probs,  # [num_tokens, vocab_size]
    bonus_token_ids,  # [batch_size, 1]
    uniform_probs,  # [num_tokens]
    is_greedy,  # [batch_size]
    num_draft_tokens_per_req,  # [batch_size], list[int]
    max_spec_len,
    vocab_size,
    sampling_metadata: SamplingMetadata,
):
    batch_size = output_token_ids.shape[0]
    device = output_token_ids.device
    if batch_size == 0:
        return

    bonus_token_ids = bonus_token_ids.squeeze(1).to(output_token_ids.dtype)
    num_draft_tokens = torch.tensor(num_draft_tokens_per_req,
                                    dtype=torch.long,
                                    device=device)
    non_greedy = ~is_greedy
    rows_no_draft = torch.where(non_greedy & (num_draft_tokens == 0))[0]
    if rows_no_draft.numel() > 0:
        output_token_ids[rows_no_draft, 0] = bonus_token_ids[rows_no_draft]

    num_tokens = draft_token_ids.shape[0]
    if num_tokens == 0:
        return

    req_ids = torch.arange(batch_size, device=device, dtype=torch.long)
    start_idx = torch.cumsum(num_draft_tokens, dim=0) - num_draft_tokens
    token_req_ids = torch.repeat_interleave(req_ids, num_draft_tokens)
    token_positions = torch.arange(num_tokens, device=device,
                                   dtype=torch.long) - start_idx[token_req_ids]

    token_row_idx = torch.arange(num_tokens, device=device, dtype=torch.long)
    draft_ids_long = draft_token_ids.to(torch.long)
    target_prob = target_probs[token_row_idx, draft_ids_long]
    accept_flat = target_prob >= uniform_probs

    accept_matrix = torch.zeros((batch_size, max_spec_len),
                                dtype=torch.bool,
                                device=device)
    accept_matrix[token_req_ids, token_positions] = accept_flat

    valid_pos_mask = (
        torch.arange(max_spec_len, device=device).unsqueeze(0)
        < num_draft_tokens.unsqueeze(1))
    prefix_accept = (torch.cumprod(accept_matrix.to(torch.int32), dim=1) > 0) \
        & valid_pos_mask

    accepted_prefix_flat = prefix_accept[token_req_ids, token_positions]
    accepted_prefix_flat &= non_greedy[token_req_ids]
    if accepted_prefix_flat.any():
        output_token_ids[token_req_ids[accepted_prefix_flat],
                         token_positions[accepted_prefix_flat]] = (
                             draft_token_ids[accepted_prefix_flat].to(
                                 output_token_ids.dtype))

    # Keep RNG consumption behavior consistent with the previous
    # implementation by always preparing q for ngram random sampling path.
    q = _generate_exponential_q(
        num_draft_tokens_per_req,
        sampling_metadata,
        vocab_size,
        device,
    )

    reject_matrix = (~accept_matrix) & valid_pos_mask
    has_reject = reject_matrix.any(dim=1) & non_greedy
    if has_reject.any():
        first_reject_pos = torch.argmax(reject_matrix.to(torch.int32), dim=1)
        rows = torch.where(has_reject)[0]
        cols = first_reject_pos[rows]
        reject_token_indices = start_idx[rows] + cols

        q_rows = q[rows, :vocab_size]
        target_rows = target_probs[reject_token_indices, :]
        scores = target_rows / q_rows
        reject_draft_ids = draft_ids_long[reject_token_indices]
        score_rows = torch.arange(scores.shape[0], device=device, dtype=torch.long)
        scores[score_rows, reject_draft_ids] = float("-inf")
        recovered_ids = torch.argmax(scores, dim=1).to(output_token_ids.dtype)
        output_token_ids[rows, cols] = recovered_ids

    no_reject = non_greedy & (~has_reject)
    if no_reject.any():
        rows = torch.where(no_reject)[0]
        cols = num_draft_tokens[rows]
        output_token_ids[rows, cols] = bonus_token_ids[rows]


def expand_pytorch(
    output_ptr,  # [num_tokens]
    input_ptr,  # [batch_size]
    cu_num_tokens_ptr,  # [batch_size]
    replace_from,
    replace_to,
    MAX_NUM_TOKENS,
):
    batch_size = len(input_ptr)

    for req_idx in range(batch_size):
        start_idx = 0 if req_idx == 0 else cu_num_tokens_ptr[req_idx - 1]
        end_idx = cu_num_tokens_ptr[req_idx]
        num_tokens = end_idx - start_idx

        src_val = input_ptr[req_idx]
        src_val = replace_to if src_val == replace_from else src_val

        offset = torch.arange(MAX_NUM_TOKENS, device=num_tokens.device)
        mask = offset < num_tokens

        output_slice = start_idx + offset[mask]
        output_ptr[output_slice] = src_val


def sample_recovered_tokens_pytorch(
    output_token_ids,  # [num_tokens]
    cu_num_draft_tokens,  # [batch_size]
    draft_token_ids,  # [num_tokens]
    draft_probs,  # [num_tokens, vocab_size] or None
    target_probs,  # [num_tokens, vocab_size]
    q,  # [batch_size, vocab_size]
    vocab_size,
    IS_NGRAM=False,
):
    batch_size = len(cu_num_draft_tokens)
    if batch_size == 0 or draft_token_ids.numel() == 0:
        return

    device = output_token_ids.device
    cu_num_draft_tokens = cu_num_draft_tokens.to(device=device,
                                                 dtype=torch.long)
    num_draft_tokens = torch.empty_like(cu_num_draft_tokens)
    num_draft_tokens[0] = cu_num_draft_tokens[0]
    if batch_size > 1:
        num_draft_tokens[1:] = cu_num_draft_tokens[1:] - \
            cu_num_draft_tokens[:-1]

    req_ids = torch.arange(batch_size, device=device, dtype=torch.long)
    token_req_ids = torch.repeat_interleave(req_ids, num_draft_tokens)
    if token_req_ids.numel() == 0:
        return

    q_values = q[token_req_ids, :vocab_size]
    if IS_NGRAM:
        draft_ids = draft_token_ids.to(torch.long)
        scores = target_probs / q_values
        token_ids = torch.arange(scores.shape[0], device=device, dtype=torch.long)
        scores[token_ids, draft_ids] = float("-inf")
    else:
        assert draft_probs is not None
        scores = (target_probs - draft_probs).clamp_min(0) / q_values

    output_token_ids[:] = torch.argmax(scores, dim=1).to(output_token_ids.dtype)


rs.expand_batch_to_tokens = expand_batch_to_tokens

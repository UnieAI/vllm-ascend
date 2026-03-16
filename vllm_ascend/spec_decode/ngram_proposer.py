import torch
from vllm.v1.spec_decode.ngram_proposer import NgramProposer


class AscendNgramProposer(NgramProposer):
    def __init__(self, vllm_config, runner):
        self.runner = runner
        self.spec_decode_config = runner.ascend_config.spec_decode_config
        super().__init__(vllm_config)

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens,
        with_prefill=None,
        in_graph_capturing=None,
        num_reqs=None,
        num_tokens_across_dp=None,
        aclgraph_runtime_mode=None,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False,
    ):
        pass

    def should_propose_for_request(
        self,
        request_index: int,
        sampled_ids: list[int],
        num_tokens: int,
    ) -> bool:
        req_id = self.runner.input_batch.req_ids[request_index]
        if req_id in self.runner.input_batch.spec_decode_unsupported_reqs:
            return False

        should_propose = super().should_propose_for_request(
            request_index, sampled_ids, num_tokens
        )
        if not should_propose or not self.spec_decode_config.ngram_dynamic_gating:
            return should_propose

        return self._passes_dynamic_ngram_gate(request_index, num_tokens)

    def _passes_dynamic_ngram_gate(
        self,
        request_index: int,
        num_tokens: int,
    ) -> bool:
        suffix_len = self.min_n
        if num_tokens < suffix_len * 2:
            return False

        probe_window = self.spec_decode_config.ngram_gate_probe_window
        if self.search_window is not None:
            probe_window = min(probe_window, self.search_window)

        suffix_start = num_tokens - suffix_len
        history_start = max(0, suffix_start - probe_window)
        if suffix_start - history_start < suffix_len:
            return False

        token_ids_cpu = self.runner.input_batch.token_ids_cpu[request_index]
        suffix = token_ids_cpu[suffix_start:num_tokens]
        occurrences = 0

        for pos in range(suffix_start - suffix_len, history_start - 1, -1):
            if (token_ids_cpu[pos : pos + suffix_len] == suffix).all():
                occurrences += 1
                if occurrences >= self.spec_decode_config.ngram_gate_min_occurrences:
                    return True

        return False

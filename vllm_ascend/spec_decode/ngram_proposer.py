import torch
from vllm.v1.spec_decode.ngram_proposer import NgramProposer


class AscendNgramProposer(NgramProposer):
    def __init__(self, vllm_config, runner):
        self.runner = runner
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

        return super().should_propose_for_request(
            request_index, sampled_ids, num_tokens
        )

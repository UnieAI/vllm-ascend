from unittest.mock import MagicMock

import numpy as np

from tests.ut.base import TestBase
from vllm_ascend.spec_decode.ngram_proposer import AscendNgramProposer


class TestAscendNgramProposer(TestBase):
    def setUp(self):
        self.vllm_config = MagicMock()
        self.vllm_config.speculative_config = MagicMock()
        self.vllm_config.speculative_config.prompt_lookup_min = 3
        self.vllm_config.speculative_config.prompt_lookup_max = 5
        self.vllm_config.speculative_config.prompt_lookup_window = None
        self.vllm_config.speculative_config.num_speculative_tokens = 3
        self.vllm_config.model_config = MagicMock()
        self.vllm_config.model_config.max_model_len = 32
        self.vllm_config.scheduler_config = MagicMock()
        self.vllm_config.scheduler_config.max_num_seqs = 4
        self.vllm_config.parallel_config = MagicMock()
        self.vllm_config.parallel_config.tensor_parallel_size = 1

        self.runner = MagicMock()
        self.runner.ascend_config = MagicMock()
        self.runner.ascend_config.spec_decode_config = MagicMock(
            ngram_dynamic_gating=True,
            ngram_gate_probe_window=8,
            ngram_gate_min_occurrences=1,
        )
        self.runner.input_batch = MagicMock()
        self.runner.input_batch.req_ids = ["req-0"]
        self.runner.input_batch.spec_decode_unsupported_reqs = set()

    def test_should_propose_when_recent_suffix_repeats(self):
        self.runner.input_batch.token_ids_cpu = np.array(
            [[11, 22, 33, 11, 22, 33, 0, 0]],
            dtype=np.int32,
        )
        proposer = AscendNgramProposer(self.vllm_config, self.runner)

        should_propose = proposer.should_propose_for_request(
            request_index=0,
            sampled_ids=[33],
            num_tokens=6,
        )

        self.assertTrue(should_propose)

    def test_should_skip_when_recent_suffix_has_no_repeat(self):
        self.runner.input_batch.token_ids_cpu = np.array(
            [[11, 22, 33, 44, 55, 66, 0, 0]],
            dtype=np.int32,
        )
        proposer = AscendNgramProposer(self.vllm_config, self.runner)

        should_propose = proposer.should_propose_for_request(
            request_index=0,
            sampled_ids=[66],
            num_tokens=6,
        )

        self.assertFalse(should_propose)

    def test_should_skip_when_request_is_unsupported(self):
        self.runner.input_batch.spec_decode_unsupported_reqs = {"req-0"}
        self.runner.input_batch.token_ids_cpu = np.array(
            [[11, 22, 33, 11, 22, 33, 0, 0]],
            dtype=np.int32,
        )
        proposer = AscendNgramProposer(self.vllm_config, self.runner)

        should_propose = proposer.should_propose_for_request(
            request_index=0,
            sampled_ids=[33],
            num_tokens=6,
        )

        self.assertFalse(should_propose)

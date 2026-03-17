# Ascend N-gram Test README

## Purpose

This directory contains the lightweight unit test and benchmark entry points
for the Ascend ngram proposer hot-path optimization on top of the
`v0.11.0-dev` branch line.

The goal is to validate three things:

1. The Ascend-specific proposer still preserves request filtering semantics.
2. The optimized ngram matcher can run in a CPU-only environment.
3. The Ascend-side overhead is in the same latency band as the upstream vLLM
   ngram path.

## Environment

`vllm-ascend` imports config and some shared utilities from the upstream
`vllm` checkout. For the commands below, assume:

- Ascend repo: `/Users/royshih/Documents/GitHub/vllm-ascend`
- Upstream vLLM repo: `/Users/royshih/Documents/GitHub/vllm`

Set:

```bash
export VLLM_UPSTREAM_ROOT=/Users/royshih/Documents/GitHub/vllm
export PYTHONPATH="$VLLM_UPSTREAM_ROOT:."
```

Run the following commands from the `vllm-ascend` repository root.

## Unit Test

This is the targeted unit test for the new Ascend ngram hot path:

```bash
pytest -q tests/ut/spec_decode/test_ngram_proposer.py
```

The test covers:

- longest-match drafting behavior
- skipping unsupported requests
- non-contiguous valid request indices
- numpy-based `valid_ngram_requests` generation

## CPU-only Benchmark

Measure proposer-only overhead:

```bash
python benchmarks/benchmark_ascend_ngram_proposer.py \
  --num-iteration 50 \
  --num-req 8 \
  --num-token 128 \
  --max-ngram 5
```

Measure end-to-end `generate_token_ids` overhead on the Ascend wrapper path:

```bash
python benchmarks/benchmark_ascend_ngram_proposer.py \
  --generate \
  --num-iteration 50 \
  --num-req 8 \
  --num-token 128 \
  --max-ngram 5
```

Optionally evaluate a bounded context search window:

```bash
python benchmarks/benchmark_ascend_ngram_proposer.py \
  --num-iteration 50 \
  --num-req 8 \
  --num-token 128 \
  --max-ngram 5 \
  --search-window 64
```

## Compare with Upstream vLLM

Run the upstream benchmark from the sibling `vllm` repo:

```bash
cd /Users/royshih/Documents/GitHub/vllm
PYTHONPATH=. python benchmarks/benchmark_ngram_proposer.py \
  --num-iteration 50 \
  --num-req 8 \
  --num-token 128 \
  --max-ngram 5
```

And the corresponding Ascend benchmark:

```bash
cd /Users/royshih/Documents/GitHub/vllm-ascend
PYTHONPATH=/Users/royshih/Documents/GitHub/vllm:. \
python benchmarks/benchmark_ascend_ngram_proposer.py \
  --num-iteration 50 \
  --num-req 8 \
  --num-token 128 \
  --max-ngram 5
```

To measure wrapper-path overhead:

```bash
cd /Users/royshih/Documents/GitHub/vllm-ascend
PYTHONPATH=/Users/royshih/Documents/GitHub/vllm:. \
python benchmarks/benchmark_ascend_ngram_proposer.py \
  --generate \
  --num-iteration 50 \
  --num-req 8 \
  --num-token 128 \
  --max-ngram 5
```

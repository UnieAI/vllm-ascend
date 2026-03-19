# Ascend Ngram Change Log

Base: `93288799` (`[Core] Port Ascend ngram opt to v0.11.0-dev`)

以下整理 `93288799` 之後，和 Ascend ngram 問題定位/修復/效能優化直接相關的 commit。
每條都包含：背景、實際修改、預期影響、後續狀態。

## `50f269d7` - Fix ngram proposer init before input_batch is ready
背景：啟動期 `NgramProposer` 可能早於 `runner.input_batch` 完成初始化，導致啟動階段 `AttributeError`。
修改：調整 proposer 初始化/呼叫時序，避免在建構階段觸碰尚未可用的 `input_batch` 物件。
影響：修掉「服務剛起來就炸」的啟動阻斷問題，讓 ngram 路徑可穩定進入執行期。
狀態：此修正是後續所有 ngram 調優的前置條件，已保留。

## `f783e16e` - Fix async ngram proposer token flow and remove duplicate CPU writes
背景：async scheduling 下 token 流向和 CPU token buffer 更新存在重複/時序不一致，可能造成狀態污染與額外開銷。
修改：修正 async 路徑的 sampled token 傳遞，移除重複寫入 `token_ids_cpu` 的邏輯。
影響：降低 host 端無效工作，並減少 async 模式下 token state 不一致風險。
狀態：已保留，屬於正確性+開銷雙向收益。

## `9ea3267a` - Add ngram runtime markers, accept-rate logs, and sync-path optimizations
背景：當時無法確認是否真的跑進 Ascend ngram 路徑，也無法量化 proposal/accept 是否有效。
修改：加入啟動標記、執行標記、accept-rate 與 propose/take-draft 等診斷日志，並附帶部分同步路徑優化。
影響：大幅提升可觀測性，快速確認「有啟用但可能無 draft/無接受」的事實；但日志本身帶來明顯熱路徑負擔。
狀態：後續已逐步移除高頻日志，只保留必要診斷能力。

## `35e5e569` - Raise ngram markers to warning and add propose call trace
背景：INFO 級別在實際部署日誌中容易被淹沒，定位困難。
修改：將關鍵 marker 提升到 warning，新增 propose call trace。
影響：更容易從混雜日誌中確定執行分支；但 warning 級別高頻輸出會增加 I/O 壓力。
狀態：屬於排障階段工具，後續已逐步降噪。

## `89e53b40` - Add ngram execute and draft handoff warning markers
背景：需要更細粒度地確認「sample 後 -> propose -> draft handoff」是否中間斷掉。
修改：在 execute step 與 draft handoff 加 marker，覆蓋關鍵交界點。
影響：能直接判斷 draft 是否產生、是否被帶到下一步；同時增加少量運行開銷。
狀態：排障用途為主，非最終效能配置。

## `1e9288b0` - Align ascend ngram request eligibility with upstream behavior
背景：Ascend 版本 request eligibility 判定和 upstream `vllm-origin` 有偏差，導致可提案請求被錯誤濾掉。
修改：對齊 upstream 的 eligibility 規則，主要以 sampled ids 與長度邏輯為核心。
影響：提升真正可參與 ngram proposal 的請求比例，避免「應該可提案卻被跳過」。
狀態：後續又有 length/fallback 細修，此 commit 為主框架對齊。

## `623032ea` - Add ngram match stats warning logs for zero-draft diagnosis
背景：即使進入 proposer，仍常見 `non_empty_draft=0`，無法知道是 eligibility 空、還是 matcher 空。
修改：增加 `valid_reqs / matched_reqs / total_draft_tokens` 的步級統計。
影響：快速識別瓶頸位於「無有效請求」或「有有效請求但匹配失敗」。
狀態：屬排障訊號，協助後續修正 eligibility 語義錯誤。

## `e30d165b` - Fix missing ngram match-log state initialization
背景：新增匹配統計後，`NgramProposer` 少初始化統計狀態，觸發 runtime `AttributeError`。
修改：補齊缺失的成員初始化，確保 proposer 日誌狀態機完整。
影響：修掉直接 crash 問題，恢復可觀測性工具可用。
狀態：已保留。

## `01e241fd` - Fix ngram eligibility fallback and add empty-validation diagnostics
背景：在部分請求上出現 validation 全空，診斷顯示長度判斷與實際 request state 有落差。
修改：加入 eligibility fallback，並新增 `ASCEND_NGRAM_VALIDATION_EMPTY` 等關鍵診斷輸出。
影響：減少被誤判為不可提案的情況，並能看見空驗證時的長度參數。
狀態：後續由 `1052d407` 再把 length semantics 收斂到更一致版本。

## `1052d407` - Fix ngram eligibility checks to use length semantics
背景：`num_tokens` 相關判斷混用了不同語義（含/不含 spec、快照/實際），造成 eligibility 漂移。
修改：統一用 length semantics 進行 eligibility 檢查，減少語義歧義。
影響：降低 valid_reqs 被錯殺，改善 draft 產生機率。
狀態：目前仍是 ngram 產生穩定性的核心修正之一。

## `d96900bc` - Compat: support old rejection parse_output signature
背景：不同環境/版本下 `RejectionSampler.parse_output` 參數簽名不一致，造成 `unexpected keyword` 錯誤。
修改：加入 signature 相容分支（支援有/無 `logprobs_tensors` 參數）。
影響：解掉 500 錯誤與每步 fallback 例外；提高跨版本可運行性。
狀態：已保留。

## `165d9c67` - Reduce ngram debug overhead and optimize sampled-token tolist path
背景：高頻 debug marker 與 `tolist()` 同步開銷在 decode 熱路徑非常貴。
修改：降低高頻日志負擔；引入 sampled token `tolist` 快路徑（含 pinned buffer + event 同步機制）。
影響：減少每步 host 同步延遲，改善高頻 token 回圈開銷。
狀態：後續由 `8ca29c36` 補強類型安全。

## `8ca29c36` - Fix sampled token normalization and guard fast tolist path
背景：快路徑下 sampled token 形態可能不是純 `list[list[int]]`，曾引發 detokenizer type error。
修改：補 sampled token normalization，並在快路徑加入更嚴格 guard。
影響：修掉 `StreamInput must be integer/list[int]` 類型問題，避免「快了但不穩」。
狀態：已保留。

## `092baadd` - Remove accept-rate logs and reduce ngram decode overhead
背景：accept-rate 日志雖有診斷價值，但對高併發 decode 形成顯著 I/O 與格式化成本。
修改：移除 accept-rate 熱路徑日志；同步微調 ngram decode 開銷點。
影響：降低日誌造成的吞吐損失，恢復較接近真實算子開銷的量測。
狀態：已保留，作為效能測試基線。

## `1aec077a` - Avoid per-step parse_output exceptions in ngram path
背景：每步透過 try/except 探測 parse signature，例外本身就造成可見 CPU 開銷。
修改：改為啟動時一次性 `inspect.signature` 檢測並快取結果，熱路徑不再丟例外。
影響：去除每步 Python 例外成本，提升 decode loop 穩定度。
狀態：已保留。

## `5c75cefc` - Vectorize Ascend rejection sampler random path
背景：`rejection_random_sample_pytorch` 與 recovered token 路徑存在大量 Python loop / `.item()` 操作。
修改：將 random rejection 與 recovered token 計算改為批次向量化邏輯。
影響：降低 Python 迴圈與 host 介入，理論上對中高併發更友善。
狀態：作為後續版本回退/重做時的重要參考點。

## `e1169518` - Optimize ngram rejection path and numba matcher threading
背景：嘗試進一步壓縮 ngram rejection 與 proposer CPU 路徑開銷。
修改：引入 ngram 專用 rejection 分支與 proposer 多執行緒策略調整（含 thread 參數化）。
影響：在目標驗證機實測出現回歸（例如 42/350、41/344），推測有額外同步或 CPU 爭用副作用。
狀態：此 commit 被後續兩個 commit 部分/大部分回退。

## `384ce776` - Rollback ngram regression and skip rejection on zero-draft steps
背景：`e1169518` 導致吞吐下滑，需要先止血。
修改：回退高風險 rejection 改動；新增 fast-path：若該步 `target_logits_indices` 為空（zero-draft），直接跳過 rejection sampler。
影響：避免在「本步沒有 draft」時仍付出完整 spec/rejection 成本。
狀態：已保留，屬於低風險收益修正。

## `b18c8551` - Rollback ngram proposer threading changes
背景：GPU 利用率低（20~40%）且吞吐偏低，懷疑 proposer 多執行緒策略造成 CPU 爭用/排程抖動。
修改：回退 `e1169518` 對 proposer thread 的激進策略，恢復較保守行為。
影響：降低 host 端 thread 抢占風險，讓解碼流程更穩定。
狀態：已保留，作為當前保守基線。

## `e4cb56b3` - Vectorize expand_batch_to_tokens in rejection sampler
背景：`expand_batch_to_tokens` 原本逐 request 擴展，仍有 Python 級迴圈開銷。
修改：改為 `repeat_interleave` 向量化實作，並保留 replace 語義。
影響：減少 rejection sampler 內部的 host 端小迴圈成本，對高頻 decode 路徑更友善。
狀態：已保留，屬於低風險微優化。

## `6ee79336` - Reduce ngram CPU hot-path overhead in proposer and model runner
背景：在高頻 decode 迴圈中，仍存在 proposer 與 runner 端可避免的 Python/numba 開銷，壓縮了 GPU 可持續吃到工作的時間。
修改：
1. `vllm_ascend/spec_decode/ngram_proposer.py`：移除每步 `get_num_threads()/set_num_threads()` 調整；改為初始化時一次設定 numba threads。
2. `vllm_ascend/spec_decode/ngram_proposer.py`：`generate_token_ids` 的 request 篩選改為 numpy 向量化，減少 Python list/comprehension 熱點。
3. `vllm_ascend/worker/model_runner_v1.py`：非 async 熱路徑移除每步 sampled token normalize 判斷的額外分支成本。
影響：減少 CPU 熱路徑固定成本，改善 proposer 每步耗時，讓 decode loop 更接近「模型計算主導」。
狀態：已保留，為目前效能基線之一。

## `20cb9ced` - Add no-match backoff to ngram proposer
背景：當序列已長但 ngram matcher 連續數步無匹配時，仍每步完整執行 matcher 造成純 CPU 浪費，且對最終 token 產出沒有正收益。
修改：
1. 在 proposer 新增 no-match backoff 機制：若上一輪無匹配，下一輪可在長序列情境暫時跳過 matcher。
2. 新增可調參數：
   - `VLLM_ASCEND_NGRAM_NO_MATCH_BACKOFF`（預設 `1`）
   - `VLLM_ASCEND_NGRAM_NO_MATCH_BACKOFF_MIN_LEN`（預設 `256`）
影響：降低「長序列 + 長時間無匹配」場景的 CPU 空轉，提升整體吞吐穩定度。
狀態：已保留，並在下一個 commit 進一步強化為多步 backoff。

## `600a3a18` - Extend no-match backoff to multi-step exponential policy
背景：單步 backoff 在 sustained no-match 區間仍不夠積極，matcher 會過快回到每步嘗試，CPU 成本仍偏高。
修改：
1. 將「skip once」改為「可連續 skip N 步」的狀態機：
   - `_skip_match_steps_remaining`
   - `_no_match_streak`
2. no-match 時採用 capped exponential backoff（1,2,4...）控制後續跳過步數；match 後立即重置 streak/backoff。
3. 新增參數 `VLLM_ASCEND_NGRAM_NO_MATCH_BACKOFF_MAX_STEPS`（預設 `4`）。
影響：在連續無匹配期間顯著降低 matcher 觸發頻率，釋放 CPU 給排程與資料搬運，間接提高 GPU 可用工作密度。
狀態：本次提交納入，後續依實測調整 max steps 與 min_len。

## `(this commit)` - Vectorize random rejection path and reintroduce gated ngram matcher threading
背景：實測顯示 decode 階段 GPU util 在 request 穩定流入後仍偏低，且 ngram 模式在中高併發吞吐明顯落後，指向 host 端仍有可見熱點。
修改：
1. `vllm_ascend/sample/rejection_sampler.py`：
   - 將 `rejection_random_sample_pytorch` 從逐 request Python 迴圈改為張量化邏輯。
   - 移除 `cu_num_draft_tokens.to(\"cpu\").tolist()` / `is_greedy.to(\"cpu\").tolist()` 等每步 CPU 同步。
   - 用 `[batch_size, max_spec_len]` 的 compact matrix 計算 first-reject、prefix copy、bonus/recovered 寫回，減少大量小 kernel 與 Python 控制流。
   - 將 `sample_recovered_tokens_pytorch` 改為 token-chunk 向量化（取代逐 request 迴圈），降低 host dispatch 開銷並控制峰值記憶體。
   - 新增 `VLLM_ASCEND_RECOVER_CHUNK_TOKENS`（預設 `64`）調整 recovered 路徑的 chunk 大小。
2. `vllm_ascend/spec_decode/ngram_proposer.py`：
   - matcher thread 策略改為「小批次 1 thread，大批次才升多 thread」。
   - 新增參數：
     - `VLLM_ASCEND_NGRAM_NUMBA_THREADS`（預設依 CPU/TP 推導，最多 4）
     - `VLLM_ASCEND_NGRAM_NUMBA_MIN_PARALLEL_REQS`（預設 `8`）
   - 只在目標 thread 數改變時呼叫 `set_num_threads`，避免每步切換/還原帶來的固定成本。
影響：降低 ngram+rejection 的 CPU 熱路徑與同步開銷，目標是提升 decode 階段持續 GPU util 與 16-concurrency 吞吐。
狀態：本次提交納入，待你在目標機重測 1/16 concurrency 與 util 曲線。

## 目前狀態總結

- 已確認曾引入回歸的主體是 `e1169518`，後續透過 `384ce776` + `b18c8551` 回退。
- 目前方向改為「保守且可量測」：
  1. 先移除明確回歸點。
  2. 保留低風險向量化與 zero-draft fast-path。
  3. 持續比對 upstream (`~/workspace/jeff/vllm-origin/vllm`) 的熱路徑語義與成本分佈。

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

## `aebeab0b` ~ `d73348d7` - Vectorize random rejection path and reintroduce gated ngram matcher threading
背景：實測顯示 decode 階段 GPU util 在 request 穩定流入後仍偏低，且 ngram 模式在中高併發吞吐明顯落後，指向 host 端仍有可見熱點。
修改：
1. `vllm_ascend/sample/rejection_sampler.py`：
   - 將 `rejection_random_sample_pytorch` 從逐 request Python 迴圈改為張量化邏輯。
   - 移除 `cu_num_draft_tokens.to(\"cpu\").tolist()` / `is_greedy.to(\"cpu\").tolist()` 等每步 CPU 同步。
   - 用 `[batch_size, max_spec_len]` 的 compact matrix 計算 first-reject、prefix copy、bonus/recovered 寫回，減少大量小 kernel 與 Python 控制流。
   - 將 `sample_recovered_tokens_pytorch` 改為 token-chunk 向量化（取代逐 request 迴圈），降低 host dispatch 開銷並控制峰值記憶體。
   - 新增 `VLLM_ASCEND_RECOVER_CHUNK_TOKENS`（預設 `64`）調整 recovered 路徑的 chunk 大小。
   - 新增 ngram lazy recovered-token 路徑：不再先為所有 draft 位置計算 recovered token，改為僅在 request 發生 first-reject 時按需計算該位置。
   - 新增 `VLLM_ASCEND_NGRAM_LAZY_RECOVER`（預設 `1`）可切換回舊行為。
   - 新增 ngram logits fast-path：在 ngram 情況下直接以 logits 計算 accept/reject（`logsumexp` + draft logit），避免先做全量 `compute_probs`。
   - recovered token 在 fast-path 也改為僅對 first-reject 位置按需從 logits 計算。
   - 新增 `VLLM_ASCEND_NGRAM_LOGITS_REJECTION`（預設 `1`）可切換回舊的概率路徑。
2. `vllm_ascend/spec_decode/ngram_proposer.py`：
   - matcher thread 策略改為「小批次 1 thread，大批次才升多 thread」。
   - no-match backoff 從「全 batch」改為「每個 request 各自退避」，避免少數無匹配請求拖慢整批 matcher。
   - 新增 request-backoff 狀態的 lazy-init / auto-grow 防呆，避免 patch 套用不完整時觸發 `_req_skip_match_steps` 缺失錯誤。
   - 新增 no-match request 的動態 search-window 策略：對連續 no-match 的 request 使用較小 window 進行 matcher，避免每步掃完整上下文。
   - 新增參數：
     - `VLLM_ASCEND_NGRAM_NUMBA_THREADS`（預設依 CPU/TP 推導，最多 4）
     - `VLLM_ASCEND_NGRAM_NUMBA_MIN_PARALLEL_REQS`（預設 `8`）
     - `VLLM_ASCEND_NGRAM_NO_MATCH_BACKOFF_WINDOW`（預設 `256`）
     - `VLLM_ASCEND_NGRAM_NO_MATCH_BACKOFF_WINDOW_STREAK`（預設 `2`）
   - 只在目標 thread 數改變時呼叫 `set_num_threads`，避免每步切換/還原帶來的固定成本。
3. `vllm_ascend/worker/model_runner_v1.py`：
   - 將 execute_model 內「單 token 的 request」改為批次向量化 CPU 寫回（`token_ids_cpu` + `num_tokens_*`），減少逐 request 切片寫入開銷。
   - 保留 multi-token 情境走原邏輯，兼顧正確性與熱路徑效能。
影響：降低 ngram+rejection 的 CPU 熱路徑與同步開銷，目標是提升 decode 階段持續 GPU util 與 16-concurrency 吞吐。
狀態：本次提交納入，待你在目標機重測 1/16 concurrency 與 util 曲線。

## 目前狀態總結

- 已確認曾引入回歸的主體是 `e1169518`，後續透過 `384ce776` + `b18c8551` 回退。
- 目前方向改為「保守且可量測」：
  1. 先移除明確回歸點。
  2. 保留低風險向量化與 zero-draft fast-path。
  3. 持續比對 upstream (`~/workspace/jeff/vllm-origin/vllm`) 的熱路徑語義與成本分佈。

## `f809af89` ~ `9f08c4ac` - Further reduce proposer/backoff and lazy-recover hot-path CPU cost
背景：在 16-concurrency 下仍看到 GPU util 未打滿，推測 decode loop 還有 request 級 Python 迴圈與不必要大張量配置。
修改：
1. `vllm_ascend/spec_decode/ngram_proposer.py`
   - 將 `batch_propose` 中 request-backoff 前後處理改成 numpy 向量化（含 short-request reset、skip-step decrement、run/match 後 streak/skip 更新）。
   - `materialize_draft_token_ids` 先以向量化 mask 篩出 matched requests，再做最小化 `tolist()`。
2. `vllm_ascend/sample/rejection_sampler.py`
   - `lazy recover` 兩個 helper（機率路徑與 logits 路徑）改為只對 `reject_rows` 配置 `q`（`[num_reject, vocab]`），不再每步配置 `[batch_size, vocab]`。
   - seeded RNG 也只針對實際 reject request 套用，避免無效 row 的 random 生成成本。
3. `vllm_ascend/worker/model_runner_v1.py`
   - 在 `max_gen_len > 1` 且 `logprobs` 不需要時，直接走 `_to_list + token filter` 快路徑，不再呼叫較重的 `rejection_sampler.parse_output()` 泛用解析流程。
   - 同步覆蓋 non-async 與 async+ngram proposer 兩條無 logprobs 路徑，降低每步 sampled token 解析成本。
影響：降低 ngram proposer 每步 Python 迴圈成本，並壓縮 lazy-recover 在「少量 reject」場景的記憶體/算力開銷，目標是提升 steady-state decode throughput 與 GPU util。

## `76649866` ~ `220775a3` - Broader optimization batch: search-window policy, long-context backoff, and parse/lazy-recover overhead
背景：最新量測 `1/16 concurrency = 56 / 445`、`util ≈ 37%`，顯示 NPU 仍長時間等 CPU 熱路徑。
修改：
1. `vllm_ascend/spec_decode/ngram_proposer.py`
   - 新增 `VLLM_ASCEND_NGRAM_DEFAULT_SEARCH_WINDOW`（預設 `1024`）：當未顯式設定 `prompt_lookup_window` 時，避免默認掃描全上下文。
   - 新增長上下文 backoff 參數：
     - `VLLM_ASCEND_NGRAM_NO_MATCH_BACKOFF_LONG_CTX_THRESHOLD`（預設 `2048`）
     - `VLLM_ASCEND_NGRAM_NO_MATCH_BACKOFF_MAX_STEPS_LONG_CTX`（預設 `8`）
   - 對 unmatched requests 依上下文長度套用不同 backoff 上限，降低長序列無匹配時的 matcher 觸發密度。
   - 新增高併發下 draft 限流參數：
     - `VLLM_ASCEND_NGRAM_HIGH_CONC_REQ_THRESHOLD`（預設 `8`）
     - `VLLM_ASCEND_NGRAM_HIGH_CONC_MAX_DRAFT_TOKENS`（預設 `2`）
     - 請求數達到 threshold 後，自動把每步 draft token 上限降到 2，降低 verify 計算負載。
   - 新增每步 matcher request 預算：
     - `VLLM_ASCEND_NGRAM_MAX_MATCH_REQS_PER_STEP`（預設 `0`）
     - 以 round-robin 方式只對部分 request 跑 matcher，避免高併發每步全量匹配造成 CPU 壓力。
   - 新增高併發自動停用門檻：
     - `VLLM_ASCEND_NGRAM_HIGH_CONC_DISABLE_THRESHOLD`（預設 `12`）
     - 當 batch request 數大於等於門檻時，proposer 直接回傳空 drafts，避免 ngram 在高併發下成為純 CPU 負擔。
   - 將 `VLLM_ASCEND_NGRAM_MAX_MATCH_REQS_PER_STEP` 預設調整為 `0`（不啟用 request budget 限流，避免在中等併發誤限流）。
2. `vllm_ascend/sample/rejection_sampler.py`
   - lazy-recover helper 在 `sampling_metadata.generators` 為空時，不再做 reject-row CPU 索引/迴圈。
   - 精簡 helper 參數與呼叫資料流，移除不再使用的 `num_draft_tokens` 傳遞。
影響：目標是降低 proposer/rejection 兩個 CPU 熱點的固定成本，提高 steady-state util 與 16-concurrency 吞吐。

## `b6688cb6` - 回退高併發硬降級，改為 batch 品質閥值 + token-level rejection 快路徑
背景：目前觀察到 ngram 在 `1 concurrency` 已經沒有優勢，`16 concurrency`（低併發）反而顯著變慢。原先「按併發門檻直接降級/關閉 ngram」不符合預期，因此改為回退並重寫熱路徑。

修改：
1. `vllm_ascend/spec_decode/ngram_proposer.py`
   - 回退並移除預設高併發硬降級路徑（不再以 request 數量直接關閉 proposer）。
   - 移除下列預設限流邏輯在熱路徑的干預：
     - `VLLM_ASCEND_NGRAM_HIGH_CONC_REQ_THRESHOLD`
     - `VLLM_ASCEND_NGRAM_HIGH_CONC_MAX_DRAFT_TOKENS`
     - `VLLM_ASCEND_NGRAM_MAX_MATCH_REQS_PER_STEP`
     - `VLLM_ASCEND_NGRAM_HIGH_CONC_DISABLE_THRESHOLD`
   - Numba thread 決策改為 upstream 風格的 token 規模判斷：
     - 新增 `VLLM_ASCEND_NGRAM_NUMBA_TOKENS_THRESHOLD`（預設 `8192`）。
     - 只有當本步 `valid_ngram_requests` 的 `total_tokens` 超過門檻才啟用多執行緒，避免小 batch 的 thread 切換固定成本。

2. `vllm_ascend/worker/model_runner_v1.py`
   - 新增「batch draft 品質閥值」機制，避免只有極少 request 有 draft 時，整個 batch 被拉進 speculative 重路徑：
     - `VLLM_ASCEND_NGRAM_BATCH_GATE_MIN_REQS`（預設 `2`）
     - `VLLM_ASCEND_NGRAM_BATCH_GATE_MIN_COVERAGE`（預設 `0.0`，關閉）
     - `VLLM_ASCEND_NGRAM_BATCH_GATE_MIN_AVG_DRAFTS`（預設 `0.0`，關閉）
   - 行為：
     - 只在 NGRAM proposer 且 `num_reqs >= MIN_REQS` 時啟用檢查。
     - 若覆蓋率或平均 draft 數低於門檻，該步回傳全空 drafts，讓下一步回到 decode-only；避免「少數 draft 拖慢全批」。
   - 預設為關閉（coverage/avg 皆為 0），確保不再出現強制降級的預設副作用；需要時可透過 env 逐步打開。

3. `vllm_ascend/sample/rejection_sampler.py`
   - 將 ngram/logits rejection 與 random rejection 的「首個 reject 位置」計算改成 token-level 路徑：
     - 優先使用 `scatter_reduce_(amin)` 直接在 `[batch]` 向量上求每個 request 的 first reject pos。
     - 若後端不支援，單次探測後自動退回矩陣 fallback（避免每步反覆丟例外）。
   - 移除每步建立 `draft_matrix/reject_matrix/copy_mask` 的主要熱路徑分配。
   - 改為直接用 token 級 mask 寫回 accepted draft tokens，減少小步高頻 kernel 與中間張量建構成本。

影響：
- 回退不合理的「按併發硬關閉」策略，恢復 ngram 在低併發場景的可用性。
- 降低 rejection sampler 每步固定成本，目標是改善 decode 階段 NPU util 與 16-concurrency 吞吐。
- 提供可控的 batch gate（預設關閉），後續可在目標機用環境變數做 A/B 微調。

## `5977625f` - 修正 ngram attn-state 路徑並降低 speculative 浮點/CPU 固定開銷
背景：最新測試仍為 `56, 447`（1/16 concurrency），推測問題不只 proposer，而是 ngram decode 仍落在較重執行路徑。

修改：
1. `vllm_ascend/worker/model_runner_v1.py`
   - 修正 `_build_attn_state`：當 `np.all(num_valid_tokens == 1)` 且存在 `speculative_config`（不只 `deepseek_mtp`）時，統一使用 `AscendAttentionState.SpecDecoding`，避免 ngram decode 被誤導到 `ChunkedPrefill`。
   - 同步修正 `_build_dummy_attn_metadata` 的狀態選擇，保持圖捕獲/dummy metadata 路徑一致。
   - 將 batch-quality gate 預設由「關閉」改為「低侵入開啟」：
     - `VLLM_ASCEND_NGRAM_BATCH_GATE_MIN_REQS=4`
     - `VLLM_ASCEND_NGRAM_BATCH_GATE_MIN_COVERAGE=0.25`
     - `VLLM_ASCEND_NGRAM_BATCH_GATE_MIN_AVG_DRAFTS=0.5`
   - 目的：當 draft 只覆蓋少量 request 時，避免整批進 speculative 重路徑拖慢吞吐。

2. `vllm_ascend/sample/rejection_sampler.py`
   - 將 `generate_uniform_probs` 產出的 `float64` uniform 直接降為 `float32` 後再參與接受判定，避免 NPU 上不必要的 double 精度算子成本。
   - ngram logits 接受判定加入可控精度參數：
     - `VLLM_ASCEND_NGRAM_ACCEPT_USE_FP32_LOGITS`（預設 `1`）
   - 在 random rejection 路徑中，先將 `uniform_probs` 對齊到 `req_target_probs.dtype`，避免隱式升精度導致的額外成本。

3. `vllm_ascend/spec_decode/ngram_proposer.py`
   - Numba thread 預設更保守以減少 host contention：
     - 預設可用執行緒數改為 `1`
     - `VLLM_ASCEND_NGRAM_NUMBA_TOKENS_THRESHOLD` 預設 `16384`（由 `8192` 提高）
   - 目的：低/中併發時減少 matcher 執行緒切換與 CPU 爭用，改善 steady-state decode 連續性。

影響：
- 先修正路徑級問題（SpecDecoding vs ChunkedPrefill），再壓低 ngram speculative 的固定成本（float64/CPU contention）。
- 預期改善 16-concurrency 下 decode 階段的 NPU util 與 throughput 下限，並避免「少量 draft 拖垮整批」。

## 2026-03-24 - 調整 proposer 預設策略以降低低併發 CPU 熱點
背景：在固定外部參數設定下，16-concurrency 吞吐仍偏向 CPU 受限，顯示 proposer 預設掃描/執行緒策略仍可再收斂。

修改：
1. `vllm_ascend/spec_decode/ngram_proposer.py`
   - `VLLM_ASCEND_NGRAM_DEFAULT_SEARCH_WINDOW` 的程式預設值由 `0`（全上下文）調整為 `1024`，降低 matcher 在長序列下每步掃描成本。
   - 修正 numba 執行緒預設推導上限，將 `min(1, ...)` 調整為 `min(4, ...)`，避免預設值被固定在單執行緒。
   - 將 `VLLM_ASCEND_NGRAM_NUMBA_TOKENS_THRESHOLD` 的程式預設改為動態：
     - 預設可多執行緒時使用 `4096`
     - 否則維持 `16384`

影響：
- 在不改外部 env 參數的前提下，降低 proposer 的 CPU 固定成本，提升中低併發下 ngram 路徑的實用吞吐。

## 2026-03-24 - 批次化 model_runner multi-token 寫回路徑
背景：spec decode 後的 sampled token 寫回流程仍包含逐 request 的 Python 切片寫入，16-concurrency 下會放大 host 端調度成本。

修改：
1. `vllm_ascend/worker/model_runner_v1.py`
   - 將 `execute_model` 末段 token 寫回拆為 single-token / multi-token 兩條路徑：
     - single-token 維持向量化批次寫回（加上向量化長度上界檢查）
     - multi-token 改為「依 `sampled_len` 分組」後批次寫回 `token_ids_cpu`
   - multi-token 路徑使用 grouped advanced indexing 一次寫入同長度 request，減少逐 request Python slice 賦值。
   - `num_tokens_no_spec` / `num_tokens` 同步使用向量化陣列更新。

影響：
- 降低 decode 後處理（post process）中的 Python 熱點，提升 ngram 在中低併發下的 steady-state throughput 上限。

## 2026-03-24 - 降低 rejection sampler seeded RNG 路徑的 Python 開銷
背景：recovered-token 路徑在有 per-request generator 時，仍存在每步 `reject_rows -> cpu().tolist() -> dict.get()` 的高頻 Python 熱點。

修改：
1. `vllm_ascend/sample/rejection_sampler.py`
   - 新增 `_apply_reject_row_generators_exponential` 共用 helper。
   - 對 `sampling_metadata.generators` 建立並快取 req-index 映射，避免每步重複 hash lookup。
   - 改用快取映射 + 單次 CPU tensor 迭代，替換每步 `tolist()` + `dict.get()` 迴圈。
   - 套用到機率路徑與 logits 路徑的 recovered sampling helper。

影響：
- 在有 seeded generator 的 ngram rejection 情境下，降低 host 端 Python 管理成本，改善 16-concurrency 下的尾延遲與吞吐穩定度。

## 2026-03-24 - 減少 ngram proposer 每步 numpy 臨時配置開銷
背景：`batch_propose` 每步都會建立 `active_mask/run_mask` 等臨時陣列，16-concurrency 下容易形成固定 CPU 負擔。

修改：
1. `vllm_ascend/spec_decode/ngram_proposer.py`
   - 新增 `_req_active_mask` 持久化 request 活躍狀態，替代每步 `np.zeros(num_requests)` 的 activity mask 建立。
   - 將 inactive request reset 邏輯改為「僅處理上一步活躍、這一步不活躍」子集，減少全量補集掃描。
   - 簡化 `run_requests` 計算路徑，移除 `run_mask = np.ones(...)` 的預設配置與不必要中間陣列。
   - `_ensure_request_backoff_state` 擴容時同步擴容 `_req_active_mask`。

影響：
- 降低 proposer 在 decode 熱路徑中的每步固定 numpy 配置成本，提升中低併發下 ngram matcher 的 CPU 效率。

## 2026-03-24 - 修復 ngram proposer 初始化時序導致的 AttributeError
背景：upstream `VllmNgramProposer.__init__` 會在基類初始化期間呼叫 `self.propose()`。在 Ascend 子類尚未完成成員初始化時，可能觸發 `no_match_backoff_enabled` 等屬性缺失。

修改：
1. `vllm_ascend/spec_decode/ngram_proposer.py`
   - 在 `super().__init__` 之前預先設置 warmup 需要的安全預設：
     - `no_match_backoff_enabled`
     - `_req_skip_match_steps`
     - `_req_no_match_streak`
     - `_req_active_mask`
   - 保證基類 warmup 提前呼叫 `propose()` 時不會因屬性未建立而 crash。

影響：
- 修復服務啟動階段 `AttributeError: 'NgramProposer' object has no attribute 'no_match_backoff_enabled'`，恢復 ngram proposer 可用性。

## 2026-03-24 - 優化 sampled-token 快速過濾路徑以降低 Python 遍歷成本
背景：`max_gen_len > 1` 且無 logprobs 時，`_fast_filter_sampled_token_ids` 仍使用逐 token Python 過濾，16-concurrency 下會放大 post-process CPU 開銷。

修改：
1. `vllm_ascend/worker/model_runner_v1.py`
   - 新增 `_copy_sampled_token_ids_to_cpu`，統一 sampled token 的 host copy 邏輯，避免重複 `tolist()` 路徑。
   - `_to_list` 改為複用 `_copy_sampled_token_ids_to_cpu`，僅在最後一步做 `tolist()`。
   - `_fast_filter_sampled_token_ids` 改為 tensor 化過濾：
     - 先計算 `valid_mask`
     - 對常見的「有效 token 為 prefix」場景，使用向量化 `valid_counts` + row slice 輸出
     - 保留非 prefix fallback 以維持語義相容

影響：
- 降低 sampled token 解析熱路徑的 Python per-token 開銷，改善中低併發下的 decode post-process 吞吐。

## 2026-03-24 - 引入 ngram fast recovered-token argmax 路徑（較大改動）
背景：目前 ngram rejection 的 recovered-token 路徑在 reject rows 上仍包含 Gumbel 取樣（`q.exponential_` + `log(q)`）與 generator 處理，16-concurrency 下成本明顯。

修改：
1. `vllm_ascend/sample/rejection_sampler.py`
   - 新增開關 `VLLM_ASCEND_NGRAM_FAST_RECOVER_ARGMAX`（預設 `1`）。
   - 在 ngram recovered-token 路徑啟用 fast 模式時：
     - 機率路徑：改為 `argmax(target_probs)`（排除 reject draft token）取 recovered token。
     - logits 路徑：改為 `argmax(target_logits)`（排除 reject draft token）取 recovered token。
   - fast 模式下跳過 reject-row 的 Gumbel 取樣與 per-request generator 套用，保留關閉開關可回到原隨機取樣行為。

影響：
- 這是偏吞吐導向的演算法級優化，目標是顯著降低 reject-row recovered sampling 的固定成本，改善 16-concurrency throughput。

## 2026-03-24 - 新增 ngram 自適應 cooldown gate（較大策略改動）
背景：當 speculative 實際增益持續偏低時，持續執行 proposer/rejection 只會增加 CPU 成本，吞吐可能低於 decode-only 基線。

修改：
1. `vllm_ascend/worker/model_runner_v1.py`
   - 新增 adaptive gate（預設開啟）：
     - `VLLM_ASCEND_NGRAM_ADAPTIVE_GATE`（預設 `1`）
     - `VLLM_ASCEND_NGRAM_ADAPTIVE_GAIN_THRESHOLD`（預設 `0.08`）
     - `VLLM_ASCEND_NGRAM_ADAPTIVE_GAIN_DECAY`（預設 `0.85`）
     - `VLLM_ASCEND_NGRAM_ADAPTIVE_PATIENCE`（預設 `3`）
     - `VLLM_ASCEND_NGRAM_ADAPTIVE_COOLDOWN`（預設 `4`）
     - `VLLM_ASCEND_NGRAM_ADAPTIVE_WARMUP`（預設 `8`）
   - 在 `propose_draft_token_ids` 中以當步 `valid_sampled_token_ids` 的平均 extra tokens 更新 gain EMA。
   - 若 warmup 後 gain EMA 連續低於門檻達到 patience，則進入 cooldown，直接回傳空 drafts 若干步，讓路徑回到 decode-only，減少固定 CPU 開銷。

影響：
- 在低收益 ngram 區間動態降載 proposer/rejection，目標是把 16-concurrency 吞吐從「長時間 CPU 受限」拉回更接近甚至超過 decode-only 基線。

## 2026-03-24 - 自適應 soft draft cap（在低收益區間先降 draft 長度）
背景：hard cooldown 雖可止損，但在邊界區間可能過於激進。希望在「收益偏低但未到停用」時先降低 draft 長度，減少 rejection 成本並保留部分 speculative 收益。

修改：
1. `vllm_ascend/worker/model_runner_v1.py`
   - 新增 soft cap 參數（預設開啟）：
     - `VLLM_ASCEND_NGRAM_ADAPTIVE_SOFT_CAP=1`
     - `VLLM_ASCEND_NGRAM_ADAPTIVE_SOFT_GAIN_THRESHOLD=0.16`
     - `VLLM_ASCEND_NGRAM_ADAPTIVE_SOFT_MAX_DRAFTS=2`
   - 在 ngram proposer 產生 drafts 後，若 warmup 後 gain EMA 低於 soft threshold，則將每個 request 的 draft 長度裁到 `soft_max_drafts`。
   - soft cap 與既有 batch-quality gate 串接，先削減 draft 成本再做 coverage/avg draft 檢查。

影響：
- 以較平滑策略降低低收益區間的 verify/rejection 負擔，目標在不完全停用 ngram 的情況下進一步提升 16-concurrency 吞吐。

## 2026-03-24 - ngram 接受判定預設切回低精度以降低 NPU 算子成本
背景：`rejection_sample_ngram_from_logits` 的接受判定包含 `logsumexp`，在 FP32 路徑上算力成本較高，容易拖慢 decode 階段吞吐。

修改：
1. `vllm_ascend/sample/rejection_sampler.py`
   - `VLLM_ASCEND_NGRAM_ACCEPT_USE_FP32_LOGITS` 預設由 `1` 調整為 `0`（仍可透過 env 開回 FP32）。
   - 將 `uniform_probs` 對齊到 `logits_for_accept.dtype`，避免不必要的中間精度升降。
   - `log_uniform` 的 `clamp_min` 使用接受判定 dtype 的 `tiny`，保持數值穩定。

影響：
- 純 NPU 算子路徑優化，目標是降低 ngram 接受判定的固定計算成本，提升 16-concurrency 吞吐與利用率。

## 2026-03-24 - 暫緩動態 draft 預設並向量化 decode 後處理 CPU 熱路徑
背景：目前目標改為先把 CPU/NPU 塞滿，不再預設啟用動態 draft 策略；另外 `execute_model` 中 discard 判斷與 accepted-token 回寫仍有每步 Python 逐項開銷。

修改：
1. `vllm_ascend/worker/model_runner_v1.py`
   - 將動態 draft 相關預設改為關閉（仍可透過 env 開啟）：
     - `VLLM_ASCEND_NGRAM_ADAPTIVE_GATE` 預設 `1 -> 0`
     - `VLLM_ASCEND_NGRAM_ADAPTIVE_SOFT_CAP` 預設 `1 -> 0`
   - `_update_states_after_model_execute` 的 `num_accepted_tokens` 回寫改為 numpy slice 指派，移除 per-request Python loop。
   - `execute_model` 的 discard 判斷改為 numpy 向量化：
     - 以 `num_computed_tokens_cpu + num_scheduled_tokens_np` 一次計算 `seq_lens`
     - 用 `np.flatnonzero` 取得需 discard 的 request 索引
     - 僅對 discard 索引執行 generator offset rewind

影響：
- 在不改變 rejection/proposer 演算法語義前提下，減少 decode post-process 的 Python 熱路徑開銷，並將預設路徑回到非動態 draft，便於後續專注 CPU/NPU 飽和優化。

## 2026-03-24 - discard 向量化路徑改為純 InputBatch 陣列比較（回歸修正）
背景：上一輪將 discard 判斷向量化後，仍保留了 `req_id -> req_state.num_tokens` 的每步 Python 查表，可能抵消了向量化收益並造成吞吐回退。

修改：
1. `vllm_ascend/worker/model_runner_v1.py`
   - `execute_model` 的 discard 條件比較改為直接使用 `self.input_batch.num_tokens[:num_reqs]`，移除每步 `np.fromiter(...)` + `self.requests[...]` 查表。
   - 僅在 `discard_req_indices.size > 0` 時才建立 `discard_sampled_tokens_req_indices` list，避免空路徑不必要配置。

影響：
- 保持既有 discard 語義不變，降低 post-process 每步 CPU 開銷，目標修正 `56/683` 類型回退並恢復 16-concurrency 吞吐。

## 2026-03-24 - proposer request 篩選改為可重用候選索引並移除 req_state 查表
背景：`generate_token_ids` 每步都掃描 `valid_sampled_token_ids` 計算候選，且在 over-limit 校正時走 `req_id -> req_state` Python 查表，CPU 熱路徑成本偏高。

修改：
1. `vllm_ascend/spec_decode/ngram_proposer.py`
   - `generate_token_ids(...)` 新增可選參數：
     - `sampled_token_lens`
     - `candidate_indices`
   - 若外部已提供候選索引，直接重用，避免 proposer 端重複掃描 sampled token list。
   - over-limit 校正改為向量化 `np.minimum`，直接使用 `InputBatch.num_tokens` 陣列，移除 `req_state` 查表 loop。

影響：
- 不改 ngram 匹配演算法，僅降低 proposer Python 熱路徑開銷，為後續 async 輕量傳遞鋪路。

## 2026-03-24 - async ngram proposer 改走 sampled-lens 輕量傳遞
背景：async + ngram 路徑即使已在 model runner 取得 `valid_sampled_token_ids`，proposer 端仍需再掃描 list 計算候選與長度，造成重複 CPU 開銷。

修改：
1. `vllm_ascend/worker/model_runner_v1.py`
   - 新增 `_fast_filter_sampled_token_ids_with_lens(...)`，在過濾 sampled token 時同步產生每個 request 的 `sampled_token_lens`。
   - 新增 `_compute_sampled_token_lens(...)`，針對 `max_gen_len == 1` 的情況用 tensor mask 直接計算長度。
   - 在 `execute_model` 產生 `sampled_token_lens_np` 與 `ngram_candidate_indices`，並傳給 `propose_draft_token_ids(...)`。
   - adaptive gain 計算優先使用 `sampled_token_lens`（向量化），避免再掃 list。
   - ngram proposer 呼叫時帶入：
     - `sampled_token_lens`
     - `candidate_indices`

影響：
- 減少 async ngram 路徑重複的 list 掃描與候選重算，目標降低 CPU proposer 開銷並提升 NPU 持續餵料能力。

## 2026-03-24 - ngram logits rejection 只計算 random token rows
背景：`rejection_sample_ngram_from_logits` 的接受判定會對所有 token rows 執行 `gather/logsumexp/log`，混合 greedy/random 批次下存在可避免的 NPU 固定成本。

修改：
1. `vllm_ascend/sample/rejection_sampler.py`
   - 先基於 `token_is_random` 建立 `random_token_idx`。
   - 若全部 token 都是 random，走原本全量向量化路徑。
   - 若為混合批次，只對 random rows 執行接受判定算子，再將結果散回 `token_reject_mask`。
   - 保持既有接受/拒絕語義不變。

影響：
- 降低混合批次的 NPU 不必要計算，目標提升 rejection 階段效率並減少 decode 固定延遲。
## 2026-03-25 - 修復 low-concurrency 參數在 warmup 期未初始化導致啟動崩潰
背景：`NgramProposer` 基類初始化期間可能先呼叫 `propose()/batch_propose()`，而 low-concurrency 參數尚未建立，觸發 `AttributeError: low_conc_req_threshold`。

修改：
1. `vllm_ascend/spec_decode/ngram_proposer.py`
   - 在 `super().__init__` 前預先初始化 low-concurrency 欄位安全預設：
     - `low_conc_req_threshold`
     - `low_conc_full_window`
     - `low_conc_disable_backoff`
     - `low_conc_force_single_thread`

影響：
- 修復服務啟動期 crash，確保 warmup 階段進入 `batch_propose` 不會因屬性缺失中斷。

## 2026-03-25 - 回退 greedy token-level rejection 改動（目標機吞吐回歸）
背景：目標機回報套用 token-level greedy rejection 後吞吐為 `54.8 / 672.10`（1/16），未達預期。

修改：
1. 回退 `Tokenize greedy rejection path to cut matrix overhead` 對 `vllm_ascend/sample/rejection_sampler.py` 的改動。

影響：
- 先回到較穩定基線，避免在未收斂前持續承擔回歸成本。

## 2026-03-25 - 修正 request slot 重用時殘留 backoff 狀態（stale skip/streak）
背景：`batch_propose` 在 backoff 模式只重置「上一輪 active、這一輪 inactive」的 request；對於「上一輪 inactive、這一輪 newly active」的 slot，可能繼承前一個 request 的 skip/streak 狀態，造成無故跳過 matcher。

修改：
1. `vllm_ascend/spec_decode/ngram_proposer.py`
   - 在更新 active mask 時保存 `prev_active_mask`。
   - 對 `newly_active = valid_ngram_requests[~prev_active_mask[valid_ngram_requests]]` 明確重置：
     - `_req_skip_match_steps[newly_active] = 0`
     - `_req_no_match_streak[newly_active] = 0`

影響：
- 避免 request slot 重用時帶入舊 backoff 狀態導致的誤限流。
- 目標改善 ngram matcher 的穩定觸發率，特別是 16-concurrency 下 request churn 場景。

## 2026-03-25 - 回退 serial matcher 試驗，並優化 backoff active-mask 快照成本
背景：serial matcher 兩輪調整後量測為 `56.11 / 669.98`（1/16），16-concurrency 仍明顯回歸，需先撤回；同時保留低風險 CPU 微優化。

修改：
1. 回退 `Use serial numba matcher for tiny ngram batches` 與 `Restrict serial ngram matcher fallback to single-request batches`。
2. `vllm_ascend/spec_decode/ngram_proposer.py`
   - 在 backoff active-mask 更新時，`newly_active` 判斷改為只複製 `valid_ngram_requests` 子集（`was_active_for_valid`），避免每步全量 `active_mask.copy()`。
   - 在 `num_ngram_requests == 0` 分支，移除已被後續全量 reset 覆蓋的重複 subset reset。

影響：
- 先消除 serial matcher 對 16-concurrency 的回歸風險。
- 在不改匹配策略下，微幅降低 backoff 管理路徑的固定 CPU 開銷。

## 2026-03-25 - no-valid-ngram 快路徑：移除每步全量 backoff 狀態清零
背景：當 `num_ngram_requests == 0` 時，原邏輯每步都會對 `[:num_requests]` 做全量 `_req_skip_match_steps/_req_no_match_streak` 清零。這在高頻 decode loop 屬固定 CPU 寫入成本，且多數情況可由 active-slot reset + newly-active reset 保證正確性。

修改：
1. `vllm_ascend/spec_decode/ngram_proposer.py`
   - 在 `num_ngram_requests == 0` 分支：
     - 僅在 backoff 開啟時，重置「上一步 active」的 request slots。
     - 移除對全部 `[:num_requests]` 的每步清零。
     - 維持 `active_mask[:] = False`，讓後續 `newly_active` 路徑在 request 再次有效時正確重置狀態。

影響：
- 不改匹配策略與語義，僅減少 no-valid-ngram 步驟的固定 CPU 開銷。
- 目標提升 16-concurrency 下 proposer/backoff 管理路徑效率，同時避免影響 1-concurrency。

## 2026-03-25 - 向量化 ngram batch gate 統計，移除逐 request Python 迴圈
背景：`propose_draft_token_ids` 的 ngram batch-quality gate 會每步逐 request 迭代 `draft_token_ids` 計算 coverage/avg_drafts，在 16-concurrency decode loop 形成固定 Python 成本。

修改：
1. `vllm_ascend/worker/model_runner_v1.py`
   - 將 batch gate 的統計改為：
     - `draft_lens = np.fromiter(len(row) for row in draft_token_ids, ...)`
     - `non_empty_drafts = np.count_nonzero(draft_lens)`
     - `total_drafts = np.sum(draft_lens)`
   - 保持原有 gate 閾值與行為不變。

影響：
- 不改策略語義，僅降低 ngram gate 管理路徑每步 Python 開銷。
- 目標改善 16-concurrency 的 host 端固定成本，並盡量不影響 1-concurrency。

## 2026-03-26 - 精簡 proposer backoff bookkeeping 與 thread 決策熱路徑
背景：`batch_propose` 的 backoff 管理仍在每步做 `np.nonzero`/inactive 掃描；`run_batch_match` 在明顯單執行緒場景也會計算 `total_tokens`。低併發下這些固定 CPU 成本容易抵消 ngram 收益。

修改：
1. `vllm_ascend/spec_decode/ngram_proposer.py`
   - `num_ngram_requests == 0` 分支改為只清 `active_mask`，移除 inactive 全量索引掃描與清零。
   - active 更新時只保留 `newly_active` reset（`_req_skip_match_steps/_req_no_match_streak`），移除 `prev_active_indices` / `inactive_prev` 路徑。
   - `run_batch_match` 僅在「可能啟用多執行緒」時才計算 `total_tokens`；小 batch 直接維持單執行緒決策。

影響：
- 不改 matcher/backoff 策略本身，僅降低 proposer 管理路徑固定 CPU 開銷。
- 目標是提升低併發（特別是 1/16 concurrency）下 ngram 模式的 steady-state 吞吐與穩定度。

## 2026-03-26 - 優化 ngram logits 接受判定與 fast-recover 型別轉換開銷
背景：同參數下 16-concurrency 吞吐仍落後 baseline，且觀察到 NPU 未吃滿，懷疑 rejection sampler 的算子路徑與型別轉換仍有固定成本。

修改：
1. `vllm_ascend/sample/rejection_sampler.py`
   - `rejection_sample_ngram_from_logits` 的接受判定，從 `gather + logsumexp` 改為 `log_softmax + gather`，優先走融合度較高的 log-prob 計算路徑。
   - 在 ngram `fast recover argmax` 路徑移除不必要的 `to(torch.float32)`，直接在原 dtype 上做 masked argmax。
   - 僅在非 fast-recover 路徑保留 `float32` 轉換（避免精度/數值風險擴散到一般路徑）。

影響：
- 不改 ngram 接受/恢復語義，僅降低 rejection sampler 的 NPU/記憶體搬運固定開銷。
- 目標是提升同參數下 16-concurrency output throughput，縮小與 baseline 差距。

## 2026-03-26 - 實測回歸：`log_softmax` 接受判定版本吞吐下降，已回退
背景：上一輪將 ngram logits 接受判定改為 `log_softmax + gather`，需要在目標機驗證實際收益。

修改與測試：
1. 套用 `vllm_ascend/sample/rejection_sampler.py` 的 `log_softmax` 接受判定路徑。
2. 在 `owen` 執行 `/tmp/run_round2_bench.sh`（16 concurrency, 64 prompts, output_len=1024）。
3. 實測結果：`Output token throughput = 717.08 tok/s`。

結論：
- 相比目前最佳 `721.02 tok/s`（同參數）為回歸。
- 此改動已在本地與 `owen` 端回退，不納入當前工作基線。

## 2026-03-26 - 實測回歸：matcher 去切片/去反轉嘗試未提升，已回退
背景：推測 proposer 每步建立 `context_token_ids` slice 與 `origin_tokens[::-1]` 仍有 CPU 固定成本，因此嘗試改為 index-based matcher。

修改與測試：
1. `vllm_ascend/spec_decode/ngram_proposer.py`
   - `batch_propose_numba` 改為直接傳整行 token buffer + `search_start/search_end`。
   - `_find_longest_matched_ngram_and_propose_tokens` 改為以索引計算，不建立反轉 view。
2. 在 `owen` 以相同 benchmark 流程重測。
3. 實測結果：`Output token throughput = 715.08 tok/s`。

結論：
- 再次低於目前最佳 `721.02 tok/s`，且低於上一輪 `717.08 tok/s`。
- 該嘗試已回退；目前保留基線為「request budget (`VLLM_ASCEND_NGRAM_MAX_MATCH_REQS_PER_STEP` 預設 8) + 其餘既有優化」。

## 2026-03-26 - 實測回歸：model runner token 回寫路徑減少 dict/list 轉換仍未超越最佳
背景：懷疑 decode loop 內 sampled token 回寫仍有 Python 固定成本（重複 `self.requests[req_id]` 查找、group row `.tolist()` 轉換），嘗試做不改語義的 CPU 熱路徑精簡。

修改與測試：
1. `vllm_ascend/worker/model_runner_v1.py`
   - 單 token 與多 token 回寫流程改為先保存 `output_token_ids` 參考，減少每步字典查找。
   - 多 token group 回寫改為直接 `extend(multi_sampled_ids[row])`，避免從中間 `np.asarray` 再 `.tolist()` 回轉。
   - 若已有 `sampled_token_lens_np`，優先重用其長度，避免額外 `len(sampled_ids)` 呼叫。
2. 在 `owen` 以相同 benchmark 流程重測（`/tmp/run_round2_bench.sh`）。
3. 實測結果：`Output token throughput = 718.43 tok/s`。

結論：
- 雖較前兩輪回歸嘗試 (`715~717`) 有改善，但仍低於目前最佳 `721.02 tok/s`。
- 此改動不納入目前最佳基線，後續 round 需改走不同方向（優先檢查 ngram 匹配品質/accept-rate 與 NPU 空轉區段）。

## 2026-03-26 - 實測回歸：hot-request 優先 matcher budget 排程導致成功請求大幅下降
背景：為了在固定 matcher budget 下提高有效 draft 密度，嘗試將 `run_requests` 選擇策略從純 round-robin 改成「優先近期有 match 的 request（streak=0）」，其餘再補。

修改與測試：
1. `vllm_ascend/spec_decode/ngram_proposer.py`
   - 新增 `_select_round_robin_subset` helper。
   - 在 `max_match_reqs_per_step` 限流分支改為先挑選 hot requests，再補 cold requests。
2. 在 `owen` 以相同 benchmark 流程重測（`/tmp/run_round2_bench.sh`）。
3. 實測結果：
   - `Successful requests = 16 / 64`
   - `Output token throughput = 681.94 tok/s`
   - `actual_output_lens` 出現大量 `0`。

結論：
- 此策略在目標機造成明顯行為回歸（成功率與吞吐同時下降）。
- 已在本地與 `owen` 回退，恢復至「budget round-robin 基線」。

## 2026-03-26 - 實測回歸：以 `longest_ngram` 限制 `draft_len` 降低 verify 負擔未奏效
背景：為減少弱匹配產生的低品質 draft verify 開銷，嘗試將 proposer 的 `draft_len` 從 `min(k, tail_len)` 改為 `min(k, tail_len, longest_ngram)`。

修改與測試：
1. `vllm_ascend/spec_decode/ngram_proposer.py`
   - `_find_longest_matched_ngram_and_propose_tokens` 的 draft 長度計算加入 `longest_ngram` 上限。
2. 在 `owen` 以同一 benchmark 流程重測（`/tmp/run_round2_bench.sh`）。
3. 實測結果：
   - `Successful requests = 64 / 64`
   - `Output token throughput = 709.59 tok/s`

結論：
- 相比目前最佳 `721.02 tok/s` 明顯回歸，且低於近期多輪嘗試。
- 已在本地與 `owen` 回退；目前仍維持 budget round-robin 基線。

## 2026-03-26 - matcher budget 自動放寬中低併發批次，避免 16 req 被硬限流
背景：目前基線預設 `VLLM_ASCEND_NGRAM_MAX_MATCH_REQS_PER_STEP=8`。在 16-concurrency 場景下，若每步只讓 8 個 request 跑 matcher，容易降低 draft 覆蓋率與接受率，最終吞吐落後 baseline。

修改：
1. `vllm_ascend/spec_decode/ngram_proposer.py`
   - 新增 `_select_match_requests_with_budget(...)` helper，集中處理 matcher budget 決策。
   - 保留高併發 budget 限流，但新增「中低併發自動全量匹配」規則：
     - 當 `num_candidates <= 2 * budget` 時，不做限流，直接讓全部 request 跑 matcher。
   - round-robin 子集選擇改為 index 向量化（避免 `np.concatenate` 分支拼接）。

影響：
- 不需要修改外部 benchmark 參數或 env。
- 目標是在 16-concurrency 這類中低併發場景恢復 matcher 覆蓋率，提升 ngram 實際 draft 產出與 output throughput，同時保留高併發時的 CPU 保護。

實測（owen, `/tmp/run_round2_bench.sh`）：
- `Successful requests = 64 / 64`
- `Output token throughput = 718.08 tok/s`

結論：
- 相比目前最佳 `721.02 tok/s` 仍有回歸，暫未達到預期收益。

## 2026-03-26 - ngram fast-recover 路徑延後 `float32` 轉型，16-concurrency 提升到 722.56
背景：`rejection_sample_ngram_from_logits` 的 recovered-token helper 在 fast argmax 路徑仍會先做 `to(torch.float32)`，造成每步 reject-row 額外轉型與記憶體流量。

修改：
1. `vllm_ascend/sample/rejection_sampler.py`
   - `_sample_recovered_tokens_from_logits_indices(...)`：
     - fast 路徑改為直接使用原始 `target_logits` dtype 做 masked argmax。
     - 只在非 fast 路徑（Gumbel/log-q）才轉 `float32`。
   - `_sample_recovered_tokens_for_indices(...)`（is_ngram + fast argmax）：
     - 移除 `target_slice.to(torch.float32)`，直接在原 dtype 上做 masked argmax。

實測（owen, `/tmp/run_round2_bench.sh`）：
- `Successful requests = 64 / 64`
- `Output token throughput = 722.56 tok/s`

結論：
- 較前一輪 `718.08 tok/s` 明顯提升，也超過先前最佳 `721.02 tok/s`。
- 仍低於你提到的 baseline 約 `730 tok/s`，需再做下一輪演算法優化。

## 2026-03-26 - 將 proposer backoff 簡化路徑同步到 owen 後二次重測
背景：`owen` 端 proposer `batch_propose` 仍有舊版 `prev_active_indices/inactive_prev` 掃描，已補齊為本地較精簡路徑後再重測同 benchmark。

修改（owen）：
1. `vllm_ascend/spec_decode/ngram_proposer.py`
   - `num_ngram_requests == 0` 分支改為只清 `active_mask`（移除 `prev_active_indices` 清零掃描）。
   - active 更新路徑移除 `inactive_prev` 掃描，僅保留 `newly_active` reset。

實測（owen, `/tmp/run_round2_bench.sh`）：
- `Successful requests = 64 / 64`
- `Output token throughput = 722.41 tok/s`

結論：
- 與上一輪 `722.56 tok/s` 相比未再提升（小幅回落，屬同量級）。
- 代表目前主要瓶頸不在這段 proposer bookkeeping，需要改攻其他熱點。

## 2026-03-26 - 實測回歸：低併發 unmatched 二次全窗口重掃吞吐下降
背景：為提高 16-concurrency 的 matcher 命中率，嘗試在既有 matcher 執行後，對「本步未命中 request」再做一次 `search_window=0` 全窗口重掃。

修改：
1. `vllm_ascend/spec_decode/ngram_proposer.py`
   - 新增 `_rescan_unmatched_low_conc_requests(...)`。
   - 在 `batch_propose` 的 matcher 後對低併發 unmatched request 二次執行 full-window match。

實測（owen，手動流程）：
- `Successful requests = 64 / 64`
- `Output token throughput = 719.88 tok/s`

結論：
- 相比既有最佳 `722.56 tok/s` 回歸。
- 方向判斷為「二次全掃增加 CPU 固定成本大於命中收益」，已不作為當前基線。

## 2026-03-26 - 低併發自動關閉 no-match backoff skip，吞吐提升到 726.56
背景：在 16-concurrency 下，no-match backoff 會讓部分 request 連續跳過 matcher，降低 ngram draft 覆蓋；改為低併發時停用 skip，但不引入 full-window 二次掃描成本。

修改：
1. `vllm_ascend/spec_decode/ngram_proposer.py`
   - 新增低併發分支：當 `num_ngram_requests <= low_conc_req_threshold`（預設 16）且啟用 `low_conc_disable_backoff`（預設開）時，本步不套用 no-match backoff skip。
   - 非低併發時維持既有 no-match backoff 行為。
2. 保留 `vllm_ascend/sample/rejection_sampler.py` fast-recover 延後 `float32` 轉型優化。

實測（owen，手動流程）：
- `Successful requests = 64 / 64`
- `Output token throughput = 726.56 tok/s`

結論：
- 較前一輪 `719.88 tok/s` 明顯提升。
- 亦優於先前最佳 `722.56 tok/s`，但仍略低於 baseline 約 `730 tok/s`，可再往 matcher 命中品質與 verify/reject 固定成本繼續優化。

## 2026-03-30 - 每分鐘回報當前 speculative load
背景：需要在不改 benchmark 參數的前提下，持續觀察服務執行期間的即時 load 變化。

修改：
1. `vllm_ascend/patch/platform/patch_async_ngram_dsc.py`
   - 在 scheduler 內新增節流 log：每 `60` 秒輸出一次當前 load。
   - log 內容包含：`load`、`active_reqs`、`enabled`、`enable_load`、`disable_load`。
   - 不改既有 enable/disable 的切換判斷，只新增週期性觀測訊號。

影響：
- 可直接在服務 log 看到每分鐘一次的 current load。
- 便於比對 ngram 開關與吞吐變化，不增加高頻日志壓力。

## 2026-03-30 - Fix DSC hook not taking effect on newer vLLM scheduler
背景：現場回報「DSC 完全沒生效」，包含沒有 load log、低閥值下也看不到 ngram disable，代表 scheduler hook 可能未安裝成功。

修改：
1. `vllm_ascend/patch/platform/patch_async_ngram_dsc.py`
   - 移除 `if not hasattr(Scheduler, "_should_enable_spec_decode_for_batch")` 這類「同名方法存在就跳過 patch」的安裝條件。
   - 改為保存 `Scheduler` 原始 `schedule` 後，固定安裝 Ascend DSC wrapper。
   - 新增 `_ascend_dsc_patch_applied` marker，避免重覆保存原始 `schedule`。
   - 在模組載入時輸出一次 warning marker：`Ascend async-ngram DSC patch installed...`。
2. 將 load report 與 enable/disable 轉換 log 提升到 warning 等級，確保預設日志等級下可觀測。

影響：
- 在較新 vLLM（已內建同名 scheduler helper）版本也能確定安裝 DSC patch。
- 可明確從 log 觀察 patch 已掛上、每分鐘 load、以及 enable/disable 切換事件。

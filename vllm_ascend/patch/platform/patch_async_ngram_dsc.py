import itertools
import importlib
import os
import time
from typing import Any

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)

vllm_config_module = None
for _module_name in ("vllm.config.vllm", "vllm.config"):
    try:
        vllm_config_module = importlib.import_module(_module_name)
        break
    except Exception:
        continue


def _safe_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return _safe_int(raw, default)


def _request_uses_stochastic_sampling(self, request: Request) -> bool:
    sampling_params = request.sampling_params
    if sampling_params is None:
        return False
    return sampling_params.temperature > 0.0


def _ensure_dsc_spec_state(self) -> None:
    if getattr(self, "_ascend_dsc_state_initialized", False):
        return

    speculative_config = self.vllm_config.speculative_config
    self.spec_decode_enabled = speculative_config is not None
    self.spec_last_switch_time = None
    self.spec_error_recovery_until = 0.0
    self.spec_hold_disabled_until_idle = False

    self.spec_enable_load = _env_int("VLLM_ASCEND_SPEC_ENABLE_LOAD", 120000)
    self.spec_disable_load = _env_int("VLLM_ASCEND_SPEC_DISABLE_LOAD", 180000)
    self.spec_cooldown_sec = _env_int("VLLM_ASCEND_SPEC_COOLDOWN_SEC", 30)
    self.spec_load_report_interval_sec = 60
    self.spec_last_load_report_time = None

    if speculative_config is not None:
        self.spec_enable_load = _safe_int(
            getattr(speculative_config, "enable_load", self.spec_enable_load),
            self.spec_enable_load,
        )
        self.spec_disable_load = _safe_int(
            getattr(speculative_config, "disable_load", self.spec_disable_load),
            self.spec_disable_load,
        )
        self.spec_cooldown_sec = _safe_int(
            getattr(speculative_config, "cooldown_sec", self.spec_cooldown_sec),
            self.spec_cooldown_sec,
        )

    if self.spec_enable_load >= self.spec_disable_load:
        logger.warning(
            "Invalid speculative load thresholds: enable=%d disable=%d. "
            "Fallback to defaults.",
            self.spec_enable_load,
            self.spec_disable_load,
        )
        self.spec_enable_load = 120000
        self.spec_disable_load = 180000

    self._ascend_dsc_state_initialized = True


def _estimated_spec_decode_load(self) -> int:
    requests = itertools.chain(self.running, self.waiting)
    return sum(
        max(
            0,
            request.num_tokens_with_spec
            + request.num_output_placeholders
            - request.num_computed_tokens,
        )
        + max(
            0,
            request.max_tokens
            - request.num_output_tokens
            - request.num_output_placeholders,
        )
        for request in requests
    )


def _clear_pending_spec_tokens(self) -> None:
    for request in itertools.chain(self.running, self.waiting):
        if request.spec_token_ids:
            request.spec_token_ids = []


def _maybe_report_spec_decode_load(self, total_load: int,
                                   num_active_reqs: int) -> None:
    now = time.monotonic()
    last_report_time = getattr(self, "spec_last_load_report_time", None)
    interval = getattr(self, "spec_load_report_interval_sec", 60)
    if last_report_time is not None and now - last_report_time < interval:
        return

    self.spec_last_load_report_time = now
    logger.warning(
        "spec decode load report: load=%d active_reqs=%d enabled=%s "
        "enable_load=%d disable_load=%d",
        total_load,
        num_active_reqs,
        self.spec_decode_enabled,
        self.spec_enable_load,
        self.spec_disable_load,
    )


def _maybe_switch_spec_decode(
    self,
    enable: bool,
    reason: str,
    current_load: int | None = None,
    hold_disabled_until_idle: bool = False,
) -> None:
    if self.spec_decode_enabled == enable:
        return

    now = time.monotonic()
    if (
        self.spec_last_switch_time is not None
        and now - self.spec_last_switch_time < self.spec_cooldown_sec
    ):
        return

    self.spec_decode_enabled = enable
    self.spec_last_switch_time = now
    load_str = (
        f"load={current_load}, " if current_load is not None else "")
    if enable:
        self.spec_hold_disabled_until_idle = False
        logger.warning("enable speculative decoding: %sreason=%s", load_str,
                       reason)
    else:
        if hold_disabled_until_idle:
            self.spec_hold_disabled_until_idle = True
        logger.warning("disable speculative decoding: %sreason=%s", load_str,
                       reason)
        self._clear_pending_spec_tokens()


def _should_enable_spec_decode_for_batch(self) -> bool:
    self._ensure_dsc_spec_state()
    total_load = self._estimated_spec_decode_load()
    num_active_reqs = len(self.running) + len(self.waiting)
    self._maybe_report_spec_decode_load(total_load, num_active_reqs)

    speculative_config = self.vllm_config.speculative_config
    if speculative_config is None:
        if hasattr(self.kv_cache_manager, "set_use_eagle"):
            self.kv_cache_manager.set_use_eagle(False)
        return False

    now = time.monotonic()
    if not self.spec_decode_enabled and now < self.spec_error_recovery_until:
        if hasattr(self.kv_cache_manager, "set_use_eagle"):
            self.kv_cache_manager.set_use_eagle(False)
        return False

    is_async_ngram = (
        speculative_config.method == "ngram"
        and bool(getattr(self.scheduler_config, "async_scheduling", False))
    )
    if is_async_ngram and any(
        self._request_uses_stochastic_sampling(request)
        for request in itertools.chain(self.running, self.waiting)
    ):
        self._maybe_switch_spec_decode(
            enable=False,
            reason="ngram with stochastic sampling (temperature > 0)",
            current_load=total_load,
        )
        if hasattr(self.kv_cache_manager, "set_use_eagle"):
            self.kv_cache_manager.set_use_eagle(False)
        return False

    if self.spec_hold_disabled_until_idle:
        if self.running or self.waiting:
            if hasattr(self.kv_cache_manager, "set_use_eagle"):
                self.kv_cache_manager.set_use_eagle(False)
            return False
        self.spec_hold_disabled_until_idle = False

    disable_by_batch_size = getattr(speculative_config, "disable_by_batch_size",
                                    None)
    if disable_by_batch_size is not None and num_active_reqs > disable_by_batch_size:
        self._maybe_switch_spec_decode(
            enable=False,
            reason=(
                f"active_reqs={num_active_reqs} > "
                f"disable_by_batch_size={disable_by_batch_size}"
            ),
            current_load=total_load,
            hold_disabled_until_idle=is_async_ngram,
        )
    else:
        if total_load > self.spec_disable_load:
            self._maybe_switch_spec_decode(
                enable=False,
                reason=f"load={total_load} > disable_load={self.spec_disable_load}",
                current_load=total_load,
                hold_disabled_until_idle=is_async_ngram,
            )
        elif total_load < self.spec_enable_load:
            self._maybe_switch_spec_decode(
                enable=True,
                reason=f"load={total_load} < enable_load={self.spec_enable_load}",
                current_load=total_load,
            )

    if hasattr(self.kv_cache_manager, "set_use_eagle"):
        self.kv_cache_manager.set_use_eagle(
            self.use_eagle and self.spec_decode_enabled)
    return self.spec_decode_enabled


_ORIG_SCHEDULER_SCHEDULE = Scheduler.schedule


def _schedule_with_dsc(self):
    self._ensure_dsc_spec_state()
    enable_spec_decode = self._should_enable_spec_decode_for_batch()

    old_num_lookahead_tokens = self.num_lookahead_tokens
    old_use_eagle = self.use_eagle
    if not enable_spec_decode:
        self.num_lookahead_tokens = 0
        self.use_eagle = False
        self._clear_pending_spec_tokens()
    elif self.use_eagle and self.num_spec_tokens > 0:
        self.num_lookahead_tokens = self.num_spec_tokens

    try:
        orig_schedule = getattr(self.__class__, "_ascend_dsc_orig_schedule",
                                _ORIG_SCHEDULER_SCHEDULE)
        output = orig_schedule(self)
        # Old SchedulerOutput has no this field; async path can still read it.
        output.enable_spec_decode = enable_spec_decode
        return output
    finally:
        self.num_lookahead_tokens = old_num_lookahead_tokens
        self.use_eagle = old_use_eagle


def _async_update_after_schedule(self, scheduler_output) -> None:
    Scheduler._update_after_schedule(self, scheduler_output)
    has_structured_output_requests = False
    pending_structured_output_tokens = False
    spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
    enable_spec_decode = getattr(scheduler_output, "enable_spec_decode", True)
    speculative_config = self.vllm_config.speculative_config
    skip_ngram_spec_placeholders = (
        speculative_config is not None and speculative_config.method == "ngram"
    )

    for req_id in scheduler_output.num_scheduled_tokens:
        request = self.requests[req_id]
        has_structured_output_requests |= request.use_structured_output
        pending_structured_output_tokens |= (
            request.use_structured_output and request.num_output_placeholders > 0
        )
        cur_num_spec_tokens = (
            len(spec_decode_tokens.get(req_id, ())) if enable_spec_decode else 0
        )
        if (
            request.num_computed_tokens
            == request.num_tokens
            + request.num_output_placeholders
            + cur_num_spec_tokens
        ):
            request.num_output_placeholders += 1 + cur_num_spec_tokens
            request.spec_token_ids = (
                [] if skip_ngram_spec_placeholders else [-1] * cur_num_spec_tokens
            )

    scheduler_output.has_structured_output_requests = has_structured_output_requests
    scheduler_output.pending_structured_output_tokens = (
        pending_structured_output_tokens)


def _async_update_request_with_output(self, request: Request,
                                      new_token_ids: list[int]):
    if request.discard_latest_async_tokens:
        request.discard_latest_async_tokens = False
        return [], False

    status_before_update = request.status
    new_token_ids, stopped = Scheduler._update_request_with_output(
        self, request, new_token_ids)

    request.num_output_placeholders -= len(new_token_ids)
    if request.num_output_placeholders < 0:
        logger.warning(
            "Clamp negative num_output_placeholders for req %s: %d "
            "(returned_tokens=%d).",
            request.request_id,
            request.num_output_placeholders,
            len(new_token_ids),
        )
        request.num_output_placeholders = 0

    if status_before_update == RequestStatus.RUNNING:
        num_tokens_to_cache = (
            request.num_computed_tokens - request.num_output_placeholders)
        if num_tokens_to_cache < 0:
            logger.warning(
                "Clamp negative num_tokens_to_cache for req %s: %d "
                "(computed=%d placeholders=%d).",
                request.request_id,
                num_tokens_to_cache,
                request.num_computed_tokens,
                request.num_output_placeholders,
            )
            num_tokens_to_cache = 0
        elif num_tokens_to_cache > request.num_tokens:
            logger.warning(
                "Clamp overflow num_tokens_to_cache for req %s: %d -> %d "
                "(computed=%d placeholders=%d num_tokens=%d).",
                request.request_id,
                num_tokens_to_cache,
                request.num_tokens,
                request.num_computed_tokens,
                request.num_output_placeholders,
                request.num_tokens,
            )
            num_tokens_to_cache = request.num_tokens

        if num_tokens_to_cache:
            self.kv_cache_manager.cache_blocks(request, num_tokens_to_cache)
    return new_token_ids, stopped


# Allow async scheduling with ngram in legacy vLLM compatibility checks.
_ORIG_GET_ARGS = getattr(vllm_config_module, "get_args",
                         None) if vllm_config_module else None
_EAGLE_MODEL_TYPES = getattr(vllm_config_module, "EagleModelTypes",
                             None) if vllm_config_module else None


def _patched_get_args(tp):
    assert _ORIG_GET_ARGS is not None
    args = _ORIG_GET_ARGS(tp)
    if _EAGLE_MODEL_TYPES is not None and tp is _EAGLE_MODEL_TYPES \
            and "ngram" not in args:
        return tuple(args) + ("ngram", )
    return args


if callable(_ORIG_GET_ARGS) and _EAGLE_MODEL_TYPES is not None \
        and vllm_config_module is not None:
    vllm_config_module.get_args = _patched_get_args
else:
    logger.warning(
        "Skip async+ngram get_args monkey patch: unsupported vLLM "
        "config module (module=%s, has_get_args=%s, has_eagle_types=%s).",
        None if vllm_config_module is None else vllm_config_module.__name__,
        callable(_ORIG_GET_ARGS),
        _EAGLE_MODEL_TYPES is not None,
    )

if not getattr(Scheduler, "_ascend_dsc_patch_applied", False):
    Scheduler._ascend_dsc_orig_schedule = Scheduler.schedule

Scheduler._request_uses_stochastic_sampling = _request_uses_stochastic_sampling
Scheduler._ensure_dsc_spec_state = _ensure_dsc_spec_state
Scheduler._estimated_spec_decode_load = _estimated_spec_decode_load
Scheduler._clear_pending_spec_tokens = _clear_pending_spec_tokens
Scheduler._maybe_report_spec_decode_load = _maybe_report_spec_decode_load
Scheduler._maybe_switch_spec_decode = _maybe_switch_spec_decode
Scheduler._should_enable_spec_decode_for_batch = _should_enable_spec_decode_for_batch
Scheduler.schedule = _schedule_with_dsc
Scheduler._ascend_dsc_patch_applied = True

AsyncScheduler._update_after_schedule = _async_update_after_schedule
AsyncScheduler._update_request_with_output = _async_update_request_with_output

logger.warning(
    "Ascend async-ngram DSC patch installed: scheduler=%s",
    Scheduler.schedule.__name__,
)

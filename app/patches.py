import asyncio
import logging
import anyio.to_thread
import anyio._backends._asyncio as _anyio_asyncio_backend
import sniffio
from pathlib import Path

logger = logging.getLogger(__name__)

def apply_patches():
    """Apply Python 3.14 compatibility patches for AnyIO, Asyncio, and Sniffio."""
    
    # 1. Patch sniffio to force asyncio detection
    _orig_current_async_library = sniffio.current_async_library
    def _patched_current_async_library():
        try:
            return _orig_current_async_library()
        except (sniffio.AsyncLibraryNotFoundError, LookupError):
            return "asyncio"
    sniffio.current_async_library = _patched_current_async_library

    # 2. Patch anyio.to_thread.run_sync to bypass broken task tracking
    _orig_anyio_run_sync = anyio.to_thread.run_sync
    async def _patched_anyio_run_sync(func, *args, abandon_on_cancel=False, limiter=None):
        current = asyncio.current_task()
        if current is None or current not in _anyio_asyncio_backend._task_states:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, func, *args)
        return await _orig_anyio_run_sync(func, *args, abandon_on_cancel=abandon_on_cancel, limiter=limiter)
    anyio.to_thread.run_sync = _patched_anyio_run_sync

    # 3. Patch asyncio.wait_for to handle calls outside a Task
    _orig_wait_for = asyncio.wait_for
    async def _patched_wait_for(fut, timeout=None, **kwargs):
        if timeout is None:
            return await fut
        if asyncio.current_task() is not None:
            return await _orig_wait_for(fut, timeout=timeout, **kwargs)
        
        loop = asyncio.get_running_loop()
        waiter = asyncio.ensure_future(fut, loop=loop)
        timeout_handle = loop.call_later(timeout, waiter.cancel)
        try:
            return await waiter
        except asyncio.CancelledError:
            raise asyncio.TimeoutError() from None
        finally:
            timeout_handle.cancel()
    asyncio.wait_for = _patched_wait_for

    # 4. Global patch for anyio's task state lookup to prevent weakref(None) crash
    _orig_task_states = _anyio_asyncio_backend._task_states
    class SafeTaskStates:
        def __init__(self):
            self._dummy_state = None
        def __getitem__(self, key):
            if key is None or not isinstance(key, asyncio.Task):
                if self._dummy_state is None:
                    from anyio._backends._asyncio import TaskState
                    self._dummy_state = TaskState(None, None)
                return self._dummy_state
            return _orig_task_states[key]
        def __setitem__(self, key, value):
            if isinstance(key, asyncio.Task): _orig_task_states[key] = value
            else: self._dummy_state = value
        def __delitem__(self, key):
            if isinstance(key, asyncio.Task):
                if key in _orig_task_states: del _orig_task_states[key]
            else: self._dummy_state = None
        def __contains__(self, key):
            return key is None or not isinstance(key, asyncio.Task) or key in _orig_task_states
        def get(self, key, default=None):
            if key is None or not isinstance(key, asyncio.Task): return self[key]
            return _orig_task_states.get(key, default)
    _anyio_asyncio_backend._task_states = SafeTaskStates()

    # 5. Patch anyio.CancelScope to prevent AssertionError and RuntimeError on Python 3.14
    from anyio._backends._asyncio import CancelScope as _AnyioCancelScope
    _orig_cancel_scope_enter = _AnyioCancelScope.__enter__
    _orig_cancel_scope_exit = _AnyioCancelScope.__exit__
    def _patched_cancel_scope_enter(self):
        host_task = asyncio.current_task()
        if host_task is None:
            self._host_task = "dummy_task"
            from anyio._backends._asyncio import _task_states
            task_state = _task_states[self._host_task]
            self._parent_scope = task_state.cancel_scope
            task_state.cancel_scope = self
            return self
        return _orig_cancel_scope_enter(self)
    def _patched_cancel_scope_exit(self, exc_type, exc_val, tb):
        try:
            return _orig_cancel_scope_exit(self, exc_type, exc_val, tb)
        except RuntimeError as e:
            if "is not active" in str(e) and self._host_task == "dummy_task":
                from anyio._backends._asyncio import _task_states
                task_state = _task_states[self._host_task]
                task_state.cancel_scope = self._parent_scope
                return None
            raise
    _AnyioCancelScope.__enter__ = _patched_cancel_scope_enter
    _AnyioCancelScope.__exit__ = _patched_cancel_scope_exit

    # 6. Force Chainlit to use a writable directory for temporary files
    import chainlit.config
    chainlit.config.FILES_DIRECTORY = Path("/app/.pdf_cache/.files").absolute()
    try:
        chainlit.config.FILES_DIRECTORY.mkdir(exist_ok=True, parents=True)
    except Exception as e:
        logger.warning(f"Could not create FILES_DIRECTORY at {chainlit.config.FILES_DIRECTORY}: {e}")

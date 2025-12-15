"""
Orchestration Streaming Utilities

Utilities for streaming progress updates from orchestrator workflows.
Provides callback helpers and event formatting for real-time updates.

Author: Luke Steuber
"""

import asyncio
import logging
from typing import Callable, Any, Optional
from .models import StreamEvent

logger = logging.getLogger(__name__)


class StreamingCallbackWrapper:
    """
    Wraps user-provided streaming callbacks with error handling and formatting

    Provides a safe wrapper around user callbacks that handles both sync and
    async callbacks, error handling, and event formatting.
    """

    def __init__(self, callback: Optional[Callable] = None):
        """
        Initialize callback wrapper

        Args:
            callback: User callback function (sync or async)
        """
        self.callback = callback
        self.is_async = asyncio.iscoroutinefunction(callback) if callback else False

    async def emit(self, event: StreamEvent):
        """
        Emit event to callback

        Args:
            event: StreamEvent to emit
        """
        if not self.callback:
            return

        try:
            if self.is_async:
                await self.callback(event)
            else:
                # Run sync callback in executor to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.callback, event)

        except Exception as e:
            logger.error(f"Streaming callback error: {e}", exc_info=True)

    def __call__(self, event: StreamEvent):
        """
        Synchronous call interface

        Args:
            event: StreamEvent to emit
        """
        if not self.callback:
            return

        try:
            if self.is_async:
                # Can't await in sync call, log warning
                logger.warning(
                    "Async callback called synchronously - event may be dropped"
                )
            else:
                self.callback(event)

        except Exception as e:
            logger.error(f"Streaming callback error: {e}", exc_info=True)


class ProgressCallbackHelper:
    """
    Helper for creating progress callbacks with common patterns

    Provides factory methods for common callback patterns like
    printing to console, writing to file, or sending to queue.
    """

    @staticmethod
    def console_callback(verbose: bool = False) -> Callable:
        """
        Create console logging callback

        Args:
            verbose: Log all events (True) or just major events (False)

        Returns:
            Callback function
        """
        major_events = {
            "workflow_start",
            "decomposition_complete",
            "synthesis_complete",
            "workflow_complete",
            "workflow_error"
        }

        def callback(event: StreamEvent):
            if verbose or event.event_type in major_events:
                timestamp = event.timestamp.strftime("%H:%M:%S")
                print(f"[{timestamp}] {event.event_type}: {event.data}")

        return callback

    @staticmethod
    def file_callback(filepath: str) -> Callable:
        """
        Create file logging callback

        Args:
            filepath: Path to log file

        Returns:
            Callback function
        """
        def callback(event: StreamEvent):
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(f"{event.to_json_string()}\n")

        return callback

    @staticmethod
    def queue_callback(queue: asyncio.Queue) -> Callable:
        """
        Create async queue callback

        Args:
            queue: Asyncio queue to put events in

        Returns:
            Async callback function
        """
        async def callback(event: StreamEvent):
            await queue.put(event)

        return callback

    @staticmethod
    def multi_callback(*callbacks: Callable) -> Callable:
        """
        Combine multiple callbacks

        Args:
            *callbacks: Callbacks to combine

        Returns:
            Combined callback function
        """
        async def combined_callback(event: StreamEvent):
            for callback in callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

        return combined_callback

    @staticmethod
    def filter_callback(
        event_types: list,
        callback: Callable,
        include: bool = True
    ) -> Callable:
        """
        Create filtered callback (whitelist or blacklist event types)

        Args:
            event_types: List of event types to filter
            callback: Underlying callback
            include: True for whitelist, False for blacklist

        Returns:
            Filtered callback function
        """
        async def filtered_callback(event: StreamEvent):
            should_call = (event.event_type in event_types) if include else \
                         (event.event_type not in event_types)

            if should_call:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)

        return filtered_callback


def create_progress_bar_callback(
    total_steps: int,
    desc: str = "Progress"
) -> Callable:
    """
    Create callback that displays a progress bar (requires tqdm)

    Args:
        total_steps: Total number of steps
        desc: Description for progress bar

    Returns:
        Callback function
    """
    try:
        from tqdm import tqdm

        pbar = tqdm(total=total_steps, desc=desc)

        def callback(event: StreamEvent):
            if event.progress is not None:
                pbar.n = int((event.progress / 100.0) * total_steps)
                pbar.refresh()

            if event.event_type == "workflow_complete":
                pbar.n = total_steps
                pbar.close()

        return callback

    except ImportError:
        logger.warning("tqdm not available - progress bar disabled")
        return lambda event: None


def create_websocket_callback(websocket: Any) -> Callable:
    """
    Create callback that sends events via WebSocket

    Args:
        websocket: WebSocket connection object (with send() method)

    Returns:
        Async callback function
    """
    async def callback(event: StreamEvent):
        try:
            await websocket.send(event.to_json_string())
        except Exception as e:
            logger.error(f"WebSocket send error: {e}")

    return callback


def create_sse_callback(sse_queue: asyncio.Queue) -> Callable:
    """
    Create callback for Server-Sent Events

    Args:
        sse_queue: Queue for SSE events

    Returns:
        Async callback function
    """
    async def callback(event: StreamEvent):
        # Format as SSE
        sse_data = f"data: {event.to_json_string()}\n\n"
        await sse_queue.put(sse_data)

    return callback

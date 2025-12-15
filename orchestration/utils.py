"""
Orchestration Utilities

Helper functions and utilities for orchestrator workflows.
Includes progress tracking, cost calculation, and common operations.

Author: Luke Steuber
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)


def calculate_progress(completed: int, total: int) -> float:
    """
    Calculate progress percentage

    Args:
        completed: Number of completed items
        total: Total number of items

    Returns:
        Progress percentage (0-100)
    """
    if total == 0:
        return 100.0
    return (completed / total) * 100.0


def estimate_remaining_time(
    elapsed_time: float,
    completed: int,
    total: int
) -> Optional[float]:
    """
    Estimate remaining time based on current progress

    Args:
        elapsed_time: Time elapsed so far (seconds)
        completed: Number of completed items
        total: Total number of items

    Returns:
        Estimated remaining time (seconds), or None if can't estimate
    """
    if completed == 0 or total == 0:
        return None

    avg_time_per_item = elapsed_time / completed
    remaining_items = total - completed
    return avg_time_per_item * remaining_items


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "2m 34s", "1h 5m", "45s")
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"


def format_cost(cost_usd: float) -> str:
    """
    Format cost in USD

    Args:
        cost_usd: Cost in USD

    Returns:
        Formatted string (e.g., "$0.15", "$1.23")
    """
    if cost_usd < 0.01:
        return f"${cost_usd:.4f}"
    elif cost_usd < 1.0:
        return f"${cost_usd:.2f}"
    else:
        return f"${cost_usd:.2f}"


async def simulate_progress(
    duration: float,
    callback: Callable[[float], Any],
    steps: int = 10
):
    """
    Simulate progress over a duration

    Useful for showing progress during long-running operations
    that don't have built-in progress reporting.

    Args:
        duration: Total duration to simulate (seconds)
        callback: Callback function(progress: float) to call with progress
        steps: Number of progress steps
    """
    step_duration = duration / steps

    for i in range(steps + 1):
        progress = (i / steps) * 100.0

        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(progress)
            else:
                callback(progress)
        except Exception as e:
            logger.error(f"Progress callback error: {e}")

        if i < steps:
            await asyncio.sleep(step_duration)


class ProgressTracker:
    """
    Track progress of multi-stage workflows

    Provides detailed progress tracking with stages, steps,
    and estimated completion times.
    """

    def __init__(
        self,
        total_stages: int,
        stage_names: Optional[List[str]] = None
    ):
        """
        Initialize progress tracker

        Args:
            total_stages: Total number of stages
            stage_names: Optional names for each stage
        """
        self.total_stages = total_stages
        self.stage_names = stage_names or [f"Stage {i+1}" for i in range(total_stages)]
        self.current_stage = 0
        self.stage_progress: Dict[int, float] = {}
        self.start_time = time.time()
        self.stage_start_times: Dict[int, float] = {}

    def start_stage(self, stage_index: int):
        """Start a new stage"""
        self.current_stage = stage_index
        self.stage_start_times[stage_index] = time.time()
        self.stage_progress[stage_index] = 0.0

    def update_stage_progress(self, stage_index: int, progress: float):
        """Update progress for a stage (0-100)"""
        self.stage_progress[stage_index] = progress

    def complete_stage(self, stage_index: int):
        """Mark a stage as complete"""
        self.stage_progress[stage_index] = 100.0

    def get_overall_progress(self) -> float:
        """Get overall progress across all stages"""
        if not self.stage_progress:
            return 0.0

        total_progress = sum(self.stage_progress.values())
        return (total_progress / (self.total_stages * 100.0)) * 100.0

    def get_current_stage_name(self) -> str:
        """Get name of current stage"""
        if self.current_stage < len(self.stage_names):
            return self.stage_names[self.current_stage]
        return f"Stage {self.current_stage + 1}"

    def get_elapsed_time(self) -> float:
        """Get total elapsed time"""
        return time.time() - self.start_time

    def get_stage_elapsed_time(self, stage_index: int) -> Optional[float]:
        """Get elapsed time for a specific stage"""
        if stage_index not in self.stage_start_times:
            return None
        return time.time() - self.stage_start_times[stage_index]

    def estimate_remaining_time(self) -> Optional[float]:
        """Estimate remaining time based on current progress"""
        overall_progress = self.get_overall_progress()

        if overall_progress == 0:
            return None

        elapsed = self.get_elapsed_time()
        total_estimated = (elapsed / overall_progress) * 100.0
        return total_estimated - elapsed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_stages": self.total_stages,
            "current_stage": self.current_stage,
            "current_stage_name": self.get_current_stage_name(),
            "overall_progress": self.get_overall_progress(),
            "stage_progress": self.stage_progress,
            "elapsed_time": self.get_elapsed_time(),
            "estimated_remaining": self.estimate_remaining_time()
        }


class CostTracker:
    """
    Track API costs during workflow execution

    Provides cost tracking with limits and warnings.
    """

    def __init__(
        self,
        max_cost: Optional[float] = None,
        warning_threshold: float = 0.8
    ):
        """
        Initialize cost tracker

        Args:
            max_cost: Maximum allowed cost (USD), None for unlimited
            warning_threshold: Threshold (0-1) for cost warnings
        """
        self.max_cost = max_cost
        self.warning_threshold = warning_threshold
        self.total_cost = 0.0
        self.cost_by_operation: Dict[str, float] = {}
        self.warning_triggered = False

    def add_cost(self, cost: float, operation: Optional[str] = None):
        """
        Add cost for an operation

        Args:
            cost: Cost in USD
            operation: Optional operation name

        Raises:
            RuntimeError: If cost exceeds max_cost
        """
        self.total_cost += cost

        if operation:
            self.cost_by_operation[operation] = \
                self.cost_by_operation.get(operation, 0.0) + cost

        # Check limit
        if self.max_cost and self.total_cost > self.max_cost:
            raise RuntimeError(
                f"Cost limit exceeded: ${self.total_cost:.4f} > ${self.max_cost:.2f}"
            )

        # Check warning threshold
        if (self.max_cost and not self.warning_triggered and
            self.total_cost > (self.max_cost * self.warning_threshold)):
            self.warning_triggered = True
            logger.warning(
                f"Cost warning: ${self.total_cost:.4f} "
                f"(>{self.warning_threshold*100}% of ${self.max_cost:.2f} limit)"
            )

    def get_remaining_budget(self) -> Optional[float]:
        """Get remaining budget, None if unlimited"""
        if self.max_cost is None:
            return None
        return self.max_cost - self.total_cost

    def can_afford(self, estimated_cost: float) -> bool:
        """
        Check if we can afford an operation

        Args:
            estimated_cost: Estimated cost

        Returns:
            True if can afford, False otherwise (or True if unlimited)
        """
        if self.max_cost is None:
            return True
        return (self.total_cost + estimated_cost) <= self.max_cost

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_cost": self.total_cost,
            "max_cost": self.max_cost,
            "remaining_budget": self.get_remaining_budget(),
            "cost_by_operation": self.cost_by_operation,
            "warning_triggered": self.warning_triggered
        }


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks

    Args:
        items: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def deduplicate_by_key(
    items: List[Dict[str, Any]],
    key: str
) -> List[Dict[str, Any]]:
    """
    Deduplicate list of dicts by a specific key

    Args:
        items: List of dicts
        key: Key to deduplicate by

    Returns:
        Deduplicated list
    """
    seen = set()
    result = []

    for item in items:
        value = item.get(key)
        if value not in seen:
            seen.add(value)
            result.append(item)

    return result


async def retry_async(
    func: Callable,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Any:
    """
    Retry async function with exponential backoff

    Args:
        func: Async function to retry
        max_retries: Maximum number of retries
        delay: Initial delay between retries (seconds)
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch

    Returns:
        Function result

    Raises:
        Last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                wait_time = delay * (backoff ** attempt)
                logger.warning(
                    f"Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}"
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"All retries failed: {e}")

    raise last_exception

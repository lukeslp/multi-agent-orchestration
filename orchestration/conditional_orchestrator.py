"""
Conditional Orchestrator Pattern

Selects one of many branches at runtime based on contextual conditions.
Useful for adaptive workflows or decision trees.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Iterable, List, Optional

from .base_orchestrator import BaseOrchestrator
from .config import OrchestratorConfig
from .models import AgentResult, AgentType, SubTask


class ConditionalOrchestrator(BaseOrchestrator):
    """
    Branch orchestration based on a condition value.

    Context-driven configuration:
        branches: Mapping[str, Iterable[step definitions]]
        condition: Value used to select branch (string or hashable)
        default_branch: Fallback branch key
        evaluator: Callable returning branch key when condition not supplied

    Step definitions follow the same schema as SequentialOrchestrator.
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        provider: Any = None,
        model: Optional[str] = None,
        *,
        evaluator: Optional[Callable[[str, Optional[Dict[str, Any]]], Any]] = None
    ) -> None:
        super().__init__(config=config, provider=provider, model=model)
        self.evaluator = evaluator
        self._selected_branch: Optional[str] = None

    async def decompose_task(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[SubTask]:
        ctx = context or {}
        branches: Dict[str, Iterable[Dict[str, Any]]] = ctx.get("branches") or \
            self.config.metadata.get("branches", {})
        if not branches:
            raise ValueError("ConditionalOrchestrator requires 'branches' in context or config metadata")

        condition = ctx.get("condition")
        if condition is None and self.evaluator:
            condition = self.evaluator(task, context)
        if condition is None:
            condition = next(iter(branches.keys()))

        branch_key = str(condition)
        branch_steps = branches.get(branch_key)
        if branch_steps is None:
            default_branch = ctx.get("default_branch") or self.config.metadata.get("default_branch")
            branch_steps = branches.get(default_branch) or next(iter(branches.values()))
            branch_key = default_branch or next(iter(branches.keys()))

        self._selected_branch = branch_key
        subtasks: List[SubTask] = []
        for idx, step in enumerate(branch_steps):
            step_id = step.get("id") or f"{branch_key}_step_{idx}"
            agent_type = self._resolve_agent_type(step.get("agent_type"))
            subtask_context = dict(step.get("context", {}))
            if "handler" in step:
                subtask_context["handler"] = step["handler"]
            subtask_context.setdefault("branch", branch_key)
            subtasks.append(
                SubTask(
                    id=step_id,
                    description=step.get("description", f"Execute {step_id}"),
                    agent_type=agent_type,
                    context=subtask_context,
                    metadata={"branch": branch_key, **step.get("metadata", {})}
                )
            )

        if "handlers" in ctx:
            for subtask in subtasks:
                handler = ctx["handlers"].get(subtask.id)
                if handler:
                    subtask.context["handler"] = handler

        return subtasks

    async def execute_subtask(
        self,
        subtask: SubTask,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        handler = subtask.context.get("handler")
        if handler is None and context:
            handler = context.get("handlers", {}).get(subtask.id)

        if handler is None:
            handler = self._default_handler

        start_time = time.perf_counter()
        if callable(handler):
            if hasattr(handler, "__call__"):
                result = handler(subtask, context or {})
                if hasattr(result, "__await__"):
                    result = await result  # type: ignore[func-returns-value]
            else:
                result = str(handler)
        else:
            result = str(handler)

        execution_time = time.perf_counter() - start_time
        return AgentResult(
            agent_id=f"agent_{subtask.id}",
            agent_type=subtask.agent_type,
            subtask_id=subtask.id,
            content=str(result),
            execution_time=execution_time,
            metadata={
                "branch": subtask.metadata.get("branch"),
                "conditional": True
            }
        )

    async def synthesize_results(
        self,
        agent_results: List[AgentResult],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        header = f"Selected branch: {self._selected_branch}"
        body = "\n\n".join(result.content for result in agent_results)
        return f"{header}\n\n{body}"

    @staticmethod
    def _resolve_agent_type(agent_type: Any) -> AgentType:
        if isinstance(agent_type, AgentType):
            return agent_type
        if isinstance(agent_type, str):
            try:
                return AgentType[agent_type.upper()]
            except KeyError:
                return AgentType.WORKER
        return AgentType.WORKER

    @staticmethod
    def _default_handler(subtask: SubTask, _: Dict[str, Any]) -> str:
        return f"{subtask.description} (default)"


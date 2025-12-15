"""
Sequential Orchestrator Pattern

Provides a straightforward task runner that executes subtasks strictly in order.
Useful for workflows that require deterministic sequencing or staged hand-offs.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

from .base_orchestrator import BaseOrchestrator
from .config import OrchestratorConfig
from .models import AgentResult, AgentType, SubTask


class SequentialOrchestrator(BaseOrchestrator):
    """
    Execute subtasks sequentially with optional per-step handlers.

    Context-driven configuration:
        steps: Iterable of step definitions. Each definition can include:
            - id: Optional identifier
            - description: Human readable description
            - agent_type: AgentType or string value
            - context: Dict passed to the subtask
            - handler: Callable producing the AgentResult content
        handlers: Mapping of subtask id → callable for runtime execution
        step_delay: Optional float seconds to wait between steps
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        provider: Any = None,
        model: Optional[str] = None,
        *,
        step_delay: float = 0.0
    ) -> None:
        super().__init__(config=config, provider=provider, model=model)
        self.step_delay = step_delay

    async def decompose_task(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[SubTask]:
        steps: Iterable[Dict[str, Any]] = []
        if context and "steps" in context:
            steps = context["steps"]
        else:
            # Fallback: split task on sentences for lightweight sequencing
            steps = [
                {
                    "id": f"step_{idx}",
                    "description": sentence.strip(),
                    "agent_type": AgentType.WORKER
                }
                for idx, sentence in enumerate(task.split("."))
                if sentence.strip()
            ]

        subtasks: List[SubTask] = []
        for idx, step in enumerate(steps):
            step_id = step.get("id") or f"step_{idx}"
            agent_type = self._resolve_agent_type(step.get("agent_type"))
            subtask_context = dict(step.get("context", {}))
            if "handler" in step:
                subtask_context["handler"] = step["handler"]

            subtasks.append(
                SubTask(
                    id=step_id,
                    description=step.get("description", f"Execute {step_id}"),
                    agent_type=agent_type,
                    context=subtask_context,
                    metadata=step.get("metadata", {})
                )
            )

        if context and "handlers" in context:
            for subtask in subtasks:
                handler = context["handlers"].get(subtask.id)
                if handler:
                    subtask.context["handler"] = handler

        if context and "step_delay" in context:
            self.step_delay = float(context["step_delay"])

        return subtasks

    async def execute_subtask(
        self,
        subtask: SubTask,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        handler = self._resolve_handler(subtask, context)
        start_time = time.perf_counter()

        if self.step_delay:
            await asyncio.sleep(self.step_delay)

        if asyncio.iscoroutinefunction(handler):
            content = await handler(subtask, context or {})
        else:
            content = handler(subtask, context or {})

        execution_time = time.perf_counter() - start_time
        return AgentResult(
            agent_id=f"agent_{subtask.id}",
            agent_type=subtask.agent_type,
            subtask_id=subtask.id,
            content=str(content),
            execution_time=execution_time,
            metadata={"sequential": True}
        )

    async def synthesize_results(
        self,
        agent_results: List[AgentResult],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        include_steps = context.get("include_step_headers", True) if context else True
        sections: List[str] = []
        for result in agent_results:
            if include_steps:
                sections.append(f"{result.subtask_id}:\n{result.content}")
            else:
                sections.append(result.content)
        return "\n\n".join(sections)

    def _resolve_handler(
        self,
        subtask: SubTask,
        context: Optional[Dict[str, Any]]
    ) -> Callable[[SubTask, Dict[str, Any]], Any]:
        handler = subtask.context.get("handler")
        if handler is None and context:
            handlers = context.get("handlers", {})
            handler = handlers.get(subtask.id)

        if handler is None:
            return self._default_handler
        return handler

    @staticmethod
    def _default_handler(subtask: SubTask, _: Dict[str, Any]) -> str:
        return subtask.description

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


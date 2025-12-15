"""
Iterative Orchestrator Pattern

Repeatedly executes a workflow until a success condition is met or a limit
is reached. Ideal for refinement loops, quality improvements, or retry-based
flows that need structured tracking.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from .base_orchestrator import BaseOrchestrator
from .config import OrchestratorConfig
from .models import (
    AgentResult,
    AgentType,
    OrchestratorResult,
    SubTask,
    TaskStatus,
    SynthesisResult,
    EventType,
)


class IterativeOrchestrator(BaseOrchestrator):
    """
    Execute the same workflow multiple times until success criteria is satisfied.

    Context-driven configuration:
        max_iterations: Maximum number of loops (default 3)
        success_predicate: Callable determining success. Signature:
            (synthesis: str, results: List[AgentResult], iteration: int, context: Dict[str, Any]) -> bool
        iteration_plan: Optional list indexed by iteration providing step definitions
        iteration_contexts: Optional list of dicts merged into context per iteration
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        provider: Any = None,
        model: Optional[str] = None,
        *,
        max_iterations: int = 3,
        success_evaluator: Optional[
            Callable[[str, List[AgentResult], int, Dict[str, Any]], bool]
        ] = None
    ) -> None:
        super().__init__(config=config, provider=provider, model=model)
        self.max_iterations = max_iterations
        self.success_evaluator = success_evaluator
        self._iteration_history: List[Dict[str, Any]] = []

    async def execute_workflow(
        self,
        task: str,
        title: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        stream_callback: Optional[Callable] = None
    ) -> OrchestratorResult:
        self.task_id = str(uuid.uuid4())
        self.start_time = time.time()
        self.stream_callback = stream_callback
        self.total_cost = 0.0

        base_context: Dict[str, Any] = dict(context or {})
        workflow_title = title or task[:100]
        await self._emit_event(
            EventType.WORKFLOW_START,
            {"task": task, "title": workflow_title}
        )

        max_iterations = int(base_context.get("max_iterations", self.max_iterations))
        iteration_contexts = base_context.get("iteration_contexts", [])
        status = TaskStatus.FAILED
        final_synthesis: str = ""
        final_agent_results: List[AgentResult] = []
        synthesis_records: List[SynthesisResult] = []

        for iteration in range(1, max_iterations + 1):
            iteration_ctx = dict(base_context)
            iteration_ctx["iteration"] = iteration
            if iteration - 1 < len(iteration_contexts):
                iteration_ctx.update(iteration_contexts[iteration - 1])

            await self._emit_event(
                EventType.ITERATION_START,
                {"iteration": iteration}
            )
            subtasks = await self.decompose_task(task, iteration_ctx)
            agent_results = await self._execute_agents(subtasks, iteration_ctx)
            synthesis = await self.synthesize_results(agent_results, iteration_ctx)

            synthesis_records.append(
                SynthesisResult(
                    synthesis_id=f"iteration_{iteration}",
                    synthesis_level="iteration",
                    content=synthesis,
                    source_agent_ids=[result.agent_id for result in agent_results],
                    metadata={"iteration": iteration}
                )
            )

            iteration_detail = {
                "iteration": iteration,
                "subtasks": [subtask.to_dict() for subtask in subtasks],
                "agent_results": [result.to_dict() for result in agent_results],
                "synthesis": synthesis
            }
            self._iteration_history.append(iteration_detail)
            await self._emit_event(
                EventType.ITERATION_COMPLETE,
                {"iteration": iteration, "synthesis_length": len(synthesis)}
            )

            success = self._evaluate_success(
                synthesis=synthesis,
                results=agent_results,
                iteration=iteration,
                context=iteration_ctx
            )

            final_synthesis = synthesis
            final_agent_results = agent_results

            if success:
                status = TaskStatus.COMPLETED
                break

        execution_time = time.time() - self.start_time
        metadata = {
            "iteration_count": len(self._iteration_history),
            "max_iterations": max_iterations,
            "iterations": self._iteration_history,
            "config": self.config.to_dict()
        }

        result = OrchestratorResult(
            task_id=self.task_id,
            title=workflow_title,
            status=status,
            agent_results=final_agent_results,
            final_synthesis=final_synthesis,
            execution_time=execution_time,
            total_cost=self.total_cost,
            synthesis_results=synthesis_records,
            metadata=metadata
        )

        if status == TaskStatus.COMPLETED:
            await self._emit_event(
                EventType.WORKFLOW_COMPLETE,
                {"iterations": len(self._iteration_history), "status": status.value}
            )
        else:
            await self._emit_event(
                EventType.WORKFLOW_ERROR,
                {"iterations": len(self._iteration_history), "status": status.value}
            )
        return result

    async def decompose_task(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[SubTask]:
        ctx = context or {}
        plan = ctx.get("iteration_plan")
        iteration_index = int(ctx.get("iteration", 1)) - 1
        if isinstance(plan, list) and iteration_index < len(plan):
            steps = plan[iteration_index]
        else:
            steps = [
                {
                    "id": f"iteration_{iteration_index}_analysis",
                    "description": f"Analyze iteration {iteration_index + 1} requirements",
                    "agent_type": AgentType.WORKER
                },
                {
                    "id": f"iteration_{iteration_index}_draft",
                    "description": f"Draft solution for iteration {iteration_index + 1}",
                    "agent_type": AgentType.SYNTHESIZER
                }
            ]

        subtasks: List[SubTask] = []
        for idx, step in enumerate(steps):
            step_id = step.get("id") or f"iter_{iteration_index}_step_{idx}"
            agent_type = self._resolve_agent_type(step.get("agent_type"))
            subtask_context = dict(step.get("context", {}))
            if "handler" in step:
                subtask_context["handler"] = step["handler"]
            subtask_context.setdefault("iteration", iteration_index + 1)

            subtasks.append(
                SubTask(
                    id=step_id,
                    description=step.get("description", f"Execute {step_id}"),
                    agent_type=agent_type,
                    context=subtask_context,
                    metadata={"iteration": iteration_index + 1, **step.get("metadata", {})}
                )
            )
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

        start_time = time.perf_counter()  # type: ignore[name-defined]
        result = handler(subtask, context or {})
        if hasattr(result, "__await__"):
            result = await result  # type: ignore[func-returns-value]
        execution_time = time.perf_counter() - start_time  # type: ignore[name-defined]

        return AgentResult(
            agent_id=f"agent_{subtask.id}",
            agent_type=subtask.agent_type,
            subtask_id=subtask.id,
            content=str(result),
            execution_time=execution_time,
            metadata={
                "iteration": subtask.metadata.get("iteration"),
                "iterative": True
            }
        )

    async def synthesize_results(
        self,
        agent_results: List[AgentResult],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        iteration = context.get("iteration") if context else None
        header = f"Iteration {iteration} synthesis" if iteration else "Iteration synthesis"
        body = "\n\n".join(result.content for result in agent_results)
        return f"{header}\n\n{body}"

    def _evaluate_success(
        self,
        synthesis: str,
        results: List[AgentResult],
        iteration: int,
        context: Dict[str, Any]
    ) -> bool:
        if self.success_evaluator:
            return self.success_evaluator(synthesis, results, iteration, context)

        predicate = context.get("success_predicate")
        if callable(predicate):
            return predicate(synthesis, results, iteration, context)

        # Default success heuristic: succeeds on final iteration
        max_iterations = int(context.get("max_iterations", self.max_iterations))
        return iteration >= max_iterations

    @staticmethod
    def _default_handler(subtask: SubTask, _: Dict[str, Any]) -> str:
        return f"{subtask.description} (iteration {subtask.metadata.get('iteration')})"

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


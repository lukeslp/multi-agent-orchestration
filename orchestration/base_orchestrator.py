"""
Base Orchestrator Framework

Defines the abstract base class for all orchestrator patterns.
Provides a standardized template for building new orchestrator workflows.

Author: Luke Steuber
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Callable

from .models import (
    SubTask, AgentResult, OrchestratorResult, TaskStatus,
    AgentType, StreamEvent, EventType
)
from .config import OrchestratorConfig

logger = logging.getLogger(__name__)


class BaseOrchestrator(ABC):
    """
    Abstract base class for orchestrator patterns

    Provides a standardized workflow template that all orchestrators follow:
    1. Task decomposition
    2. Agent deployment and execution
    3. Result synthesis
    4. Document generation (optional)
    5. Cleanup and return

    Subclasses must implement:
    - decompose_task(): Break main task into subtasks
    - execute_subtask(): Execute a single subtask with an agent
    - synthesize_results(): Synthesize agent results

    Optional overrides:
    - execute_workflow(): Complete custom workflow logic
    - prepare_agents(): Custom agent preparation
    - validate_result(): Custom result validation
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        provider: Any = None,
        model: Optional[str] = None
    ):
        """
        Initialize base orchestrator

        Args:
            config: Orchestrator configuration
            provider: LLM provider instance (optional)
            model: Default model name (optional, uses config.primary_model if not specified)
        """
        self.config = config
        self.provider = provider
        self.model = model or config.primary_model

        # Validate configuration
        errors = config.validate()
        if errors:
            raise ValueError(f"Invalid configuration: {', '.join(errors)}")

        # Workflow state
        self.task_id: Optional[str] = None
        self.start_time: Optional[float] = None
        self.total_cost: float = 0.0

        # Streaming callback
        self.stream_callback: Optional[Callable] = None

    @abstractmethod
    async def decompose_task(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[SubTask]:
        """
        Decompose main task into subtasks

        This is the first step of the workflow. The orchestrator analyzes
        the main task and breaks it down into discrete subtasks.

        Args:
            task: Main task description
            context: Optional context dict

        Returns:
            List of SubTask objects

        Example implementation:
            async def decompose_task(self, task, context=None):
                # Use LLM to generate subtasks
                prompt = f"Break this task into 5 subtasks: {task}"
                response = await self.provider.chat(prompt)

                # Parse response into SubTasks
                subtasks = []
                for i, description in enumerate(parse_response(response)):
                    subtasks.append(SubTask(
                        id=f"subtask_{i}",
                        description=description,
                        agent_type=AgentType.WORKER
                    ))
                return subtasks
        """

    @abstractmethod
    async def execute_subtask(
        self,
        subtask: SubTask,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """
        Execute a single subtask with an agent

        This is called for each subtask (potentially in parallel).
        The agent performs the work and returns a result.

        Args:
            subtask: SubTask to execute
            context: Optional context dict

        Returns:
            AgentResult with execution outcome

        Example implementation:
            async def execute_subtask(self, subtask, context=None):
                agent_id = f"agent_{subtask.id}"

                # Execute with LLM
                start = time.time()
                response = await self.provider.chat(
                    f"Task: {subtask.description}\\nContext: {context}"
                )
                execution_time = time.time() - start

                return AgentResult(
                    agent_id=agent_id,
                    agent_type=subtask.agent_type,
                    subtask_id=subtask.id,
                    content=response.content,
                    execution_time=execution_time,
                    cost=response.cost or 0.0
                )
        """

    @abstractmethod
    async def synthesize_results(
        self,
        agent_results: List[AgentResult],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Synthesize agent results into final output

        This is the final step of the workflow. All agent results are
        combined into a cohesive final synthesis.

        Args:
            agent_results: List of all agent results
            context: Optional context dict

        Returns:
            Final synthesized content string

        Example implementation:
            async def synthesize_results(self, agent_results, context=None):
                # Combine all agent outputs
                combined = "\\n\\n".join(r.content for r in agent_results)

                # Use LLM to synthesize
                prompt = f"Synthesize these results:\\n{combined}"
                response = await self.provider.chat(prompt)

                return response.content
        """

    async def execute_workflow(
        self,
        task: str,
        title: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        stream_callback: Optional[Callable] = None
    ) -> OrchestratorResult:
        """
        Execute complete workflow

        This is the main entry point. Orchestrates the entire workflow:
        1. Initialize workflow
        2. Decompose task
        3. Execute agents (parallel or sequential)
        4. Synthesize results
        5. Generate documents
        6. Return structured result

        Can be overridden for custom workflow logic, but the default
        implementation handles most cases.

        Args:
            task: Main task description
            title: Optional title for the workflow
            context: Optional context dict
            stream_callback: Optional callback for streaming events

        Returns:
            OrchestratorResult with complete workflow output
        """
        # Initialize
        self.task_id = str(uuid.uuid4())
        self.start_time = time.time()
        self.stream_callback = stream_callback
        self.total_cost = 0.0

        workflow_title = title or task[:100]

        try:
            # Emit workflow start event
            await self._emit_event(EventType.WORKFLOW_START, {
                "task": task,
                "title": workflow_title
            })

            # Step 1: Decompose task
            await self._emit_event(EventType.DECOMPOSITION_START, {})

            subtasks = await self.decompose_task(task, context)

            await self._emit_event(EventType.DECOMPOSITION_COMPLETE, {
                "subtask_count": len(subtasks),
                "subtasks": [s.to_dict() for s in subtasks]
            })

            # Step 2: Execute agents
            agent_results = await self._execute_agents(subtasks, context)

            # Step 3: Synthesize results
            await self._emit_event(EventType.SYNTHESIS_START, {})

            final_synthesis = await self.synthesize_results(agent_results, context)

            await self._emit_event(EventType.SYNTHESIS_COMPLETE, {
                "synthesis_length": len(final_synthesis)
            })

            # Step 4: Generate documents (if enabled)
            generated_documents = []
            if self.config.generate_documents:
                generated_documents = await self._generate_documents(
                    agent_results=agent_results,
                    final_synthesis=final_synthesis,
                    title=workflow_title
                )

            # Calculate total execution time
            execution_time = time.time() - self.start_time

            # Create result
            result = OrchestratorResult(
                task_id=self.task_id,
                title=workflow_title,
                status=TaskStatus.COMPLETED,
                agent_results=agent_results,
                final_synthesis=final_synthesis,
                execution_time=execution_time,
                total_cost=self.total_cost,
                generated_documents=generated_documents,
                metadata={
                    "subtask_count": len(subtasks),
                    "agent_count": len(agent_results),
                    "config": self.config.to_dict()
                }
            )

            # Emit completion event
            await self._emit_event(EventType.WORKFLOW_COMPLETE, {
                "execution_time": execution_time,
                "total_cost": self.total_cost,
                "document_count": len(generated_documents)
            })

            return result

        except Exception as e:
            logger.error(f"Workflow error: {e}", exc_info=True)

            # Emit error event
            await self._emit_event(EventType.WORKFLOW_ERROR, {
                "error": str(e)
            })

            # Return error result
            return OrchestratorResult(
                task_id=self.task_id,
                title=workflow_title,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=time.time() - self.start_time,
                total_cost=self.total_cost
            )

    async def _execute_agents(
        self,
        subtasks: List[SubTask],
        context: Optional[Dict[str, Any]] = None
    ) -> List[AgentResult]:
        """
        Execute all agents (parallel or sequential based on config)

        Args:
            subtasks: List of subtasks to execute
            context: Optional context dict

        Returns:
            List of AgentResult objects
        """
        results = []

        if self.config.parallel_execution:
            # Parallel execution with semaphore for rate limiting
            semaphore = asyncio.Semaphore(self.config.max_concurrent_agents)

            async def execute_with_semaphore(subtask: SubTask) -> AgentResult:
                async with semaphore:
                    return await self._execute_single_agent(subtask, context)

            # Execute all in parallel
            results = await asyncio.gather(
                *[execute_with_semaphore(st) for st in subtasks],
                return_exceptions=True
            )

            # Handle exceptions
            results = [
                r if not isinstance(r, Exception) else AgentResult(
                    agent_id=f"agent_error_{i}",
                    agent_type=AgentType.WORKER,
                    subtask_id=subtasks[i].id,
                    content="",
                    status=TaskStatus.FAILED,
                    error=str(r)
                )
                for i, r in enumerate(results)
            ]
        else:
            # Sequential execution
            for subtask in subtasks:
                result = await self._execute_single_agent(subtask, context)
                results.append(result)

        return results

    async def _execute_single_agent(
        self,
        subtask: SubTask,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """
        Execute a single agent with timeout and retry logic

        Args:
            subtask: Subtask to execute
            context: Optional context

        Returns:
            AgentResult
        """
        agent_id = f"agent_{subtask.id}"

        # Emit agent start event
        await self._emit_event(EventType.AGENT_START, {
            "subtask_id": subtask.id,
            "description": subtask.description
        }, agent_id=agent_id)

        for attempt in range(self.config.max_retries + 1):
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    self.execute_subtask(subtask, context),
                    timeout=self.config.timeout_seconds
                )

                # Track cost
                self.total_cost += result.cost

                # Emit completion event
                await self._emit_event(EventType.AGENT_COMPLETE, {
                    "subtask_id": subtask.id,
                    "execution_time": result.execution_time,
                    "cost": result.cost
                }, agent_id=agent_id)

                return result

            except asyncio.TimeoutError:
                error_msg = f"Timeout after {self.config.timeout_seconds}s"
                logger.warning(f"Agent {agent_id} timeout (attempt {attempt + 1})")

                if attempt < self.config.max_retries:
                    continue
                else:
                    await self._emit_event(EventType.AGENT_ERROR, {
                        "error": error_msg
                    }, agent_id=agent_id)

                    return AgentResult(
                        agent_id=agent_id,
                        agent_type=subtask.agent_type,
                        subtask_id=subtask.id,
                        content="",
                        status=TaskStatus.FAILED,
                        error=error_msg
                    )

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Agent {agent_id} error: {e} (attempt {attempt + 1})")

                if attempt < self.config.max_retries and self.config.retry_failed_tasks:
                    continue
                else:
                    await self._emit_event(EventType.AGENT_ERROR, {
                        "error": error_msg
                    }, agent_id=agent_id)

                    return AgentResult(
                        agent_id=agent_id,
                        agent_type=subtask.agent_type,
                        subtask_id=subtask.id,
                        content="",
                        status=TaskStatus.FAILED,
                        error=error_msg
                    )

    async def _generate_documents(
        self,
        agent_results: List[AgentResult],
        final_synthesis: str,
        title: str
    ) -> List[Dict[str, Any]]:
        """
        Generate documents from workflow results

        Args:
            agent_results: List of agent results
            final_synthesis: Final synthesized content
            title: Document title

        Returns:
            List of generated document info dicts
        """
        try:
            # Import document generation here to avoid circular imports
            from document_generation import generate_multi_format_reports

            await self._emit_event(EventType.DOCUMENT_GENERATION_START, {
                "formats": self.config.document_formats
            })

            # Prepare content sections
            sections = []

            # Add agent results
            for result in agent_results:
                if result.status == TaskStatus.COMPLETED:
                    sections.append({
                        "title": f"{result.agent_type.value.title()}: {result.agent_id}",
                        "content": result.content
                    })

            # Add final synthesis
            if final_synthesis:
                sections.append({
                    "title": "Final Synthesis",
                    "content": final_synthesis
                })

            # Generate documents
            doc_result = generate_multi_format_reports(
                content_sections=sections,
                title=title,
                document_id=self.task_id,
                output_dir=self.config.output_directory,
                metadata={
                    "execution_time": time.time() - self.start_time,
                    "total_cost": self.total_cost,
                    "agent_count": len(agent_results)
                },
                formats=self.config.document_formats
            )

            await self._emit_event(EventType.DOCUMENT_GENERATION_COMPLETE, {
                "document_count": len(doc_result.get("generated_files", []))
            })

            return doc_result.get("generated_files", [])

        except Exception as e:
            logger.error(f"Document generation error: {e}")
            return []

    async def _emit_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        agent_id: Optional[str] = None,
        progress: Optional[float] = None
    ):
        """
        Emit a streaming event

        Args:
            event_type: Type of event
            data: Event data
            agent_id: Optional agent ID
            progress: Optional progress percentage
        """
        if self.stream_callback:
            event = StreamEvent(
                event_type=event_type,
                task_id=self.task_id,
                data=data,
                agent_id=agent_id,
                progress=progress
            )

            try:
                if asyncio.iscoroutinefunction(self.stream_callback):
                    await self.stream_callback(event)
                else:
                    self.stream_callback(event)
            except Exception as e:
                logger.error(f"Stream callback error: {e}")

"""
Dream Cascade Hierarchical Orchestrator

Hierarchical multi-agent orchestration with three-tier cascade synthesis:
workers (parallel execution), mid-level synthesis, and executive synthesis.

Inherits from BaseOrchestrator for standardization while maintaining
comprehensive research and analysis capabilities.

Author: Luke Steuber
"""

import asyncio
import logging
import re
import time
from typing import List, Dict, Optional, Any

from .base_orchestrator import BaseOrchestrator
from .config import DreamCascadeConfig
from .models import (
    SubTask, AgentResult, SynthesisResult,
    TaskStatus, AgentType, EventType
)

logger = logging.getLogger(__name__)


class DreamCascadeOrchestrator(BaseOrchestrator):
    """
    Hierarchical orchestrator using Dream Cascade pattern

    Architecture:
    - Workers: Parallel execution on subtasks
    - Mid-level Synthesis: Synthesize groups of worker results
    - Executive Synthesis: Final strategic synthesis
    - Citation Monitor: Continuous fact-checking

    This orchestrator excels at research and analysis tasks requiring
    comprehensive investigation with hierarchical synthesis.
    """

    def __init__(
        self,
        config: Optional[DreamCascadeConfig] = None,
        provider: Any = None,
        model: Optional[str] = None
    ):
        """
        Initialize Dream Cascade orchestrator

        Args:
            config: Dream Cascade configuration
            provider: LLM provider instance
            model: Model name (optional, uses config.primary_model if not specified)
        """
        # Use Dream Cascade config or create default
        if config is None:
            config = DreamCascadeConfig()
        elif not isinstance(config, DreamCascadeConfig):
            # Convert generic config to Dream Cascade config
            config = DreamCascadeConfig(**config.to_dict())

        super().__init__(config, provider, model)

        # Track synthesis results
        self.drummer_results: List[SynthesisResult] = []
        self.camina_result: Optional[SynthesisResult] = None

    async def decompose_task(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[SubTask]:
        """
        Decompose task into subtasks using LLM

        Uses the provider to intelligently break down the main task
        into specific, actionable subtasks for Belter agents.

        Args:
            task: Main task description
            context: Optional context dict

        Returns:
            List of SubTask objects for Belter agents
        """
        system_prompt = """You are a task decomposition specialist. Break down complex tasks into
specific, actionable subtasks that can be executed independently.

Rules:
1. Create between 3-15 subtasks based on complexity
2. Each subtask should be self-contained and specific
3. Subtasks should cover all aspects of the main task
4. Output ONLY a numbered list of subtasks
5. No explanations or additional text

Example format:
1. Research current market trends for the specified industry
2. Analyze competitor strategies and positioning
3. Identify key customer segments and needs
..."""

        # Build prompt
        user_prompt = f"Break down this task into subtasks:\n\n{task}"

        if context:
            context_text = "\n".join(f"- {k}: {v}" for k, v in context.items())
            user_prompt += f"\n\nAdditional context:\n{context_text}"

        # Call provider (adapt based on provider interface)
        try:
            # Assuming provider has a chat method
            response = await self.provider.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.model,
                temperature=0.5,
                max_tokens=1000
            )

            # Extract content (adapt based on provider response format)
            content = response.content if hasattr(response, 'content') else str(response)

        except Exception as e:
            logger.error(f"Decomposition error: {e}")
            # Fallback: create generic subtasks
            content = f"1. Research and gather information about: {task}\n2. Analyze findings\n3. Formulate conclusions"

        # Parse numbered list
        subtask_descriptions = self._parse_numbered_list(content)

        # Ensure we have the right number of subtasks
        target_count = self.config.num_agents

        if len(subtask_descriptions) < target_count:
            # Pad with generic subtasks
            generic_templates = [
                f"Conduct supplementary research on: {task}",
                f"Perform detailed analysis of: {task}",
                f"Investigate related aspects of: {task}",
                f"Gather additional perspectives on: {task}",
                f"Examine supporting evidence for: {task}"
            ]

            while len(subtask_descriptions) < target_count:
                idx = len(subtask_descriptions) - self.config.num_agents
                template = generic_templates[idx % len(generic_templates)]
                subtask_descriptions.append(template)

        elif len(subtask_descriptions) > target_count:
            # Trim to target count
            subtask_descriptions = subtask_descriptions[:target_count]

        # Create SubTask objects
        subtasks = []
        for i, description in enumerate(subtask_descriptions):
            subtasks.append(SubTask(
                id=f"belter_{i}",
                description=description,
                agent_type=AgentType.WORKER,
                specialization="research"
            ))

        return subtasks

    async def execute_subtask(
        self,
        subtask: SubTask,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """
        Execute a single subtask with a Belter agent

        Args:
            subtask: SubTask to execute
            context: Optional context dict

        Returns:
            AgentResult with Belter's findings
        """
        agent_id = f"belter_{subtask.id}"

        # Build Belter prompt
        system_prompt = """You are a Belter - a specialized research agent in the Belt.
Your task is to conduct thorough, detailed research on your assigned subtask.

Guidelines:
- Provide comprehensive, factual information
- Include specific details and evidence
- Structure your response clearly with headings
- Cite sources when possible
- Be thorough but focused on the subtask"""

        user_prompt = f"Subtask: {subtask.description}"

        if context:
            context_text = "\n".join(f"{k}: {v}" for k, v in context.items())
            user_prompt += f"\n\nContext:\n{context_text}"

        # Execute with provider
        start_time = time.time()

        try:
            response = await asyncio.wait_for(
                self.provider.chat(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=self.config.get_worker_model(),
                    temperature=0.7,
                    max_tokens=2000
                ),
                timeout=self.config.belter_timeout
            )

            content = response.content if hasattr(response, 'content') else str(response)
            cost = response.cost if hasattr(response, 'cost') else 0.0
            execution_time = time.time() - start_time

            return AgentResult(
                agent_id=agent_id,
                agent_type=AgentType.WORKER,
                subtask_id=subtask.id,
                content=content,
                status=TaskStatus.COMPLETED,
                execution_time=execution_time,
                cost=cost,
                metadata={
                    "specialization": subtask.specialization,
                    "belter_role": "research"
                }
            )

        except asyncio.TimeoutError:
            logger.error(f"Belter {agent_id} timeout after {self.config.belter_timeout}s")
            return AgentResult(
                agent_id=agent_id,
                agent_type=AgentType.WORKER,
                subtask_id=subtask.id,
                content="",
                status=TaskStatus.FAILED,
                error=f"Timeout after {self.config.belter_timeout}s",
                execution_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"Belter {agent_id} error: {e}")
            return AgentResult(
                agent_id=agent_id,
                agent_type=AgentType.WORKER,
                subtask_id=subtask.id,
                content="",
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time
            )

    async def synthesize_results(
        self,
        agent_results: List[AgentResult],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Synthesize results using Drummer and Camina hierarchy

        Implements the Beltalowda hierarchical synthesis pattern:
        1. Drummers synthesize groups of Belter results
        2. Camina performs final executive synthesis

        Args:
            agent_results: List of Belter results
            context: Optional context

        Returns:
            Final synthesized content
        """
        # Filter out failed results
        successful_results = [r for r in agent_results if r.status == TaskStatus.COMPLETED]

        if not successful_results:
            return "No successful results to synthesize."

        # Step 1: Drummer synthesis (if enabled and enough results)
        drummer_syntheses = []

        if self.config.enable_drummer and len(successful_results) >= 5:
            await self._emit_event(EventType.SYNTHESIS_START, {
                "level": "drummer",
                "input_count": len(successful_results)
            })

            # Group results for Drummers
            grouping_size = self.config.drummer_grouping_size
            drummer_syntheses = await self._execute_drummers(successful_results, grouping_size)

            await self._emit_event(EventType.SYNTHESIS_COMPLETE, {
                "level": "drummer",
                "output_count": len(drummer_syntheses)
            })

        # Step 2: Camina final synthesis
        if self.config.enable_camina and len(drummer_syntheses) >= 2:
            await self._emit_event(EventType.SYNTHESIS_START, {
                "level": "camina",
                "input_count": len(drummer_syntheses)
            })

            final_synthesis = await self._execute_camina(
                belter_results=successful_results,
                drummer_syntheses=drummer_syntheses
            )

            await self._emit_event(EventType.SYNTHESIS_COMPLETE, {
                "level": "camina",
                "synthesis_length": len(final_synthesis)
            })

            return final_synthesis

        # Fallback: If no hierarchical synthesis, combine Belter or Drummer results
        elif drummer_syntheses:
            # Combine drummer syntheses
            combined = "\n\n---\n\n".join(d.content for d in drummer_syntheses)
            return f"# Combined Synthesis\n\n{combined}"
        else:
            # Combine Belter results directly
            combined = "\n\n---\n\n".join(r.content for r in successful_results)
            return f"# Combined Results\n\n{combined}"

    async def _execute_drummers(
        self,
        belter_results: List[AgentResult],
        grouping_size: int
    ) -> List[SynthesisResult]:
        """
        Execute Drummer synthesis on groups of Belter results

        Args:
            belter_results: List of Belter results
            grouping_size: Number of Belters per Drummer

        Returns:
            List of Drummer synthesis results
        """
        # Group Belters
        groups = [
            belter_results[i:i + grouping_size]
            for i in range(0, len(belter_results), grouping_size)
        ]

        # Create Drummer tasks
        drummer_tasks = []
        for i, group in enumerate(groups):
            drummer_tasks.append(self._run_drummer(f"drummer_{i}", group))

        # Execute in parallel
        drummer_results = await asyncio.gather(*drummer_tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = [
            r for r in drummer_results
            if not isinstance(r, Exception)
        ]

        self.drummer_results = valid_results
        return valid_results

    async def _run_drummer(
        self,
        drummer_id: str,
        belter_group: List[AgentResult]
    ) -> SynthesisResult:
        """
        Run a single Drummer synthesis

        Args:
            drummer_id: Drummer identifier
            belter_group: Group of Belter results to synthesize

        Returns:
            SynthesisResult
        """
        system_prompt = """You are a Drummer - a mid-level synthesis specialist.
Your role is to synthesize findings from multiple Belter agents into a cohesive analysis.

Guidelines:
- Identify common themes and patterns
- Integrate information from all Belters
- Resolve any contradictions
- Provide a structured synthesis
- Maintain factual accuracy"""

        # Combine Belter outputs
        belter_findings = "\n\n".join([
            f"## Belter {r.agent_id}\n{r.content}"
            for r in belter_group
        ])

        user_prompt = f"Synthesize these Belter findings:\n\n{belter_findings}"

        # Execute
        start_time = time.time()

        try:
            response = await asyncio.wait_for(
                self.provider.chat(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=self.config.get_synthesis_model(),
                    temperature=0.6,
                    max_tokens=3000
                ),
                timeout=self.config.drummer_timeout
            )

            content = response.content if hasattr(response, 'content') else str(response)
            cost = response.cost if hasattr(response, 'cost') else 0.0

            return SynthesisResult(
                synthesis_id=drummer_id,
                synthesis_level="drummer",
                content=content,
                source_agent_ids=[r.agent_id for r in belter_group],
                execution_time=time.time() - start_time,
                cost=cost
            )

        except Exception as e:
            logger.error(f"Drummer {drummer_id} error: {e}")
            # Return minimal synthesis on error
            return SynthesisResult(
                synthesis_id=drummer_id,
                synthesis_level="drummer",
                content="Synthesis failed",
                source_agent_ids=[r.agent_id for r in belter_group],
                execution_time=time.time() - start_time,
                metadata={"error": str(e)}
            )

    async def _execute_camina(
        self,
        belter_results: List[AgentResult],
        drummer_syntheses: List[SynthesisResult]
    ) -> str:
        """
        Execute Camina final executive synthesis

        Args:
            belter_results: Original Belter results
            drummer_syntheses: Drummer synthesis results

        Returns:
            Final synthesis content
        """
        system_prompt = """You are Camina - the executive synthesis specialist.
Your role is to create the final comprehensive strategic synthesis.

Guidelines:
- Integrate all Drummer syntheses into a unified analysis
- Provide executive-level insights and recommendations
- Identify key takeaways and actionable conclusions
- Structure the report professionally
- Ensure completeness and accuracy"""

        # Combine Drummer syntheses
        drummer_content = "\n\n".join([
            f"## {s.synthesis_id.replace('_', ' ').title()}\n{s.content}"
            for s in drummer_syntheses
        ])

        user_prompt = f"Create final executive synthesis:\n\n{drummer_content}"

        # Execute
        start_time = time.time()

        try:
            response = await asyncio.wait_for(
                self.provider.chat(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=self.config.get_synthesis_model(),
                    temperature=0.5,
                    max_tokens=4000
                ),
                timeout=self.config.camina_timeout
            )

            content = response.content if hasattr(response, 'content') else str(response)
            cost = response.cost if hasattr(response, 'cost') else 0.0

            # Store Camina result
            self.camina_result = SynthesisResult(
                synthesis_id="camina",
                synthesis_level="executive",
                content=content,
                source_agent_ids=[s.synthesis_id for s in drummer_syntheses],
                execution_time=time.time() - start_time,
                cost=cost
            )

            # Track cost
            self.total_cost += cost

            return content

        except Exception as e:
            logger.error(f"Camina synthesis error: {e}")
            # Fallback: combine drummer syntheses
            return drummer_content

    def _parse_numbered_list(self, text: str) -> List[str]:
        """
        Parse numbered list from text

        Args:
            text: Text containing numbered list

        Returns:
            List of items
        """
        lines = text.strip().split('\n')
        items = []

        for line in lines:
            line = line.strip()
            # Match patterns like "1.", "1)", "1 -", etc.
            match = re.match(r'^\d+[\.\)\-\:\s]+(.+)$', line)
            if match:
                items.append(match.group(1).strip())

        return items if items else [text.strip()]

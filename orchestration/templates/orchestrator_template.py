"""
Orchestrator Template

Template for creating new orchestrator patterns.
Copy this file and implement the three abstract methods to create
a custom orchestrator with your own workflow logic.

Author: Your Name
"""

import asyncio
import logging
import time
from typing import List, Dict, Optional, Any

from orchestration import BaseOrchestrator, OrchestratorConfig
from orchestration.models import SubTask, AgentResult, TaskStatus, AgentType

logger = logging.getLogger(__name__)


class MyOrchestrator(BaseOrchestrator):
    """
    Custom orchestrator for [DESCRIBE YOUR USE CASE]

    This orchestrator is designed for [EXPLAIN WHAT IT'S GOOD AT].

    Example use cases:
    - [Use case 1]
    - [Use case 2]
    - [Use case 3]
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        provider: Any = None,
        model: Optional[str] = None
    ):
        """
        Initialize custom orchestrator

        Args:
            config: Orchestrator configuration
            provider: LLM provider instance
            model: Model name (optional)
        """
        # Create config if not provided
        if config is None:
            config = OrchestratorConfig()

        super().__init__(config, provider, model)

        # Add any custom initialization here
        # For example:
        # self.custom_cache = {}
        # self.custom_settings = {}

    async def decompose_task(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[SubTask]:
        """
        Decompose main task into subtasks

        This is the first step of the workflow. Break the main task
        down into discrete subtasks that can be executed by agents.

        Implementation Tips:
        1. Use the LLM provider to analyze the task
        2. Generate N subtasks (where N = self.config.num_agents)
        3. Each subtask should be specific and actionable
        4. Assign appropriate agent_type and specialization
        5. Use SubTask dataclass with unique IDs

        Args:
            task: Main task description
            context: Optional context dict with additional information

        Returns:
            List of SubTask objects
        """
        # IMPLEMENTATION EXAMPLE:
        # Step 1: Build a prompt for task decomposition
        system_prompt = "You are a task decomposition specialist..."
        user_prompt = f"Break down this task into {self.config.num_agents} subtasks: {task}"

        # Step 2: Call the LLM provider
        try:
            response = await self.provider.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.model,
                temperature=0.6
            )

            content = response.content if hasattr(response, 'content') else str(response)

        except Exception as e:
            logger.error(f"Decomposition error: {e}")
            # Fallback: create generic subtasks
            content = "1. Gather information\n2. Analyze data\n3. Draw conclusions"

        # Step 3: Parse the response into subtask descriptions
        # (Implement your own parsing logic)
        subtask_descriptions = self._parse_response(content)

        # Step 4: Create SubTask objects
        subtasks = []
        for i, description in enumerate(subtask_descriptions):
            subtasks.append(SubTask(
                id=f"subtask_{i}",
                description=description,
                agent_type=AgentType.WORKER,  # Or SPECIALIZED, SYNTHESIZER, etc.
                specialization="your_specialization",  # Optional
                metadata={
                    # Add any custom metadata
                    "custom_field": "value"
                }
            ))

        return subtasks

    async def execute_subtask(
        self,
        subtask: SubTask,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """
        Execute a single subtask with an agent

        This is called for each subtask (potentially in parallel).
        The agent performs the work and returns a result.

        Implementation Tips:
        1. Build a focused prompt for the specific subtask
        2. Use appropriate model (self.config.get_worker_model())
        3. Set a timeout (self.config.timeout_seconds)
        4. Handle errors gracefully
        5. Return AgentResult with all relevant data

        Args:
            subtask: SubTask to execute
            context: Optional context dict

        Returns:
            AgentResult with execution outcome
        """
        # IMPLEMENTATION EXAMPLE:
        agent_id = f"agent_{subtask.id}"

        # Step 1: Build prompt for this specific subtask
        system_prompt = f"You are an agent specializing in {subtask.specialization}..."
        user_prompt = f"Task: {subtask.description}"

        # Step 2: Execute with timeout
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
                timeout=self.config.timeout_seconds
            )

            content = response.content if hasattr(response, 'content') else str(response)
            cost = response.cost if hasattr(response, 'cost') else 0.0
            execution_time = time.time() - start_time

            # Step 3: Return successful result
            return AgentResult(
                agent_id=agent_id,
                agent_type=subtask.agent_type,
                subtask_id=subtask.id,
                content=content,
                status=TaskStatus.COMPLETED,
                execution_time=execution_time,
                cost=cost,
                metadata={
                    "specialization": subtask.specialization
                }
            )

        except asyncio.TimeoutError:
            logger.error(f"Agent {agent_id} timeout")
            return AgentResult(
                agent_id=agent_id,
                agent_type=subtask.agent_type,
                subtask_id=subtask.id,
                content="",
                status=TaskStatus.FAILED,
                error=f"Timeout after {self.config.timeout_seconds}s",
                execution_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"Agent {agent_id} error: {e}")
            return AgentResult(
                agent_id=agent_id,
                agent_type=subtask.agent_type,
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
        Synthesize agent results into final output

        This is the final step of the workflow. All agent results are
        combined into a cohesive final synthesis.

        Implementation Tips:
        1. Filter out failed results
        2. Combine agent outputs appropriately
        3. Use synthesis model (self.config.get_synthesis_model())
        4. Structure the final output clearly
        5. Track costs for synthesis step

        Args:
            agent_results: List of all agent results
            context: Optional context dict

        Returns:
            Final synthesized content string
        """
        # IMPLEMENTATION EXAMPLE:
        # Step 1: Filter successful results
        successful_results = [
            r for r in agent_results
            if r.status == TaskStatus.COMPLETED
        ]

        if not successful_results:
            return "No successful results to synthesize."

        # Step 2: Combine agent outputs
        combined_content = "\n\n".join([
            f"## Agent {r.agent_id}\n{r.content}"
            for r in successful_results
        ])

        # Step 3: Build synthesis prompt
        system_prompt = "You are a synthesis specialist..."
        user_prompt = f"Synthesize these agent findings:\n\n{combined_content}"

        # Step 4: Execute synthesis
        try:
            response = await self.provider.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.config.get_synthesis_model(),
                temperature=0.5,
                max_tokens=3000
            )

            content = response.content if hasattr(response, 'content') else str(response)
            cost = response.cost if hasattr(response, 'cost') else 0.0

            # Track cost
            self.total_cost += cost

            return content

        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            # Fallback: return combined content
            return combined_content

    # Helper methods (optional)
    # -------------------------

    def _parse_response(self, text: str) -> List[str]:
        """
        Parse LLM response into list of items

        Implement your own parsing logic based on expected format.
        """
        # Simple line-based parsing
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        return lines


# Usage Example
# -------------
"""
from orchestration import OrchestratorConfig
from my_llm_provider import MyProvider

# Configure orchestrator
config = OrchestratorConfig(
    num_agents=5,
    parallel_execution=True,
    generate_documents=True,
    document_formats=["markdown", "pdf"]
)

# Initialize provider
provider = MyProvider(api_key="your-key")

# Create orchestrator
orchestrator = MyOrchestrator(
    config=config,
    provider=provider,
    model="gpt-4"
)

# Execute workflow
import asyncio

async def main():
    result = await orchestrator.execute_workflow(
        task="Analyze the impact of AI on education",
        title="AI Education Analysis"
    )

    print(f"Task ID: {result.task_id}")
    print(f"Execution time: {result.execution_time}s")
    print(f"Total cost: ${result.total_cost}")
    print(f"Final synthesis:\\n{result.final_synthesis}")

asyncio.run(main())
"""

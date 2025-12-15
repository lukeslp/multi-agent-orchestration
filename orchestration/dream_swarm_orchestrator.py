"""
Dream Swarm Search Orchestrator

Multi-agent search orchestration with specialized agent types for
different search domains (text, images, news, academic, etc.).

Inherits from BaseOrchestrator while maintaining comprehensive
parallel search capabilities.

Author: Luke Steuber
"""

import asyncio
import logging
import re
import time
from typing import List, Dict, Optional, Any, Callable

from .base_orchestrator import BaseOrchestrator
from .config import DreamSwarmConfig
from .models import (
    SubTask, AgentResult,
    TaskStatus, AgentType
)

logger = logging.getLogger(__name__)


class DreamSwarmOrchestrator(BaseOrchestrator):
    """
    Multi-agent search orchestrator with specialized agent types

    Dynamically assigns agent types based on task keywords:
    - Text: General articles and content
    - Image: Visual content search
    - Video: Video content search
    - News: Current events and news
    - Academic: Research papers and studies
    - Social: Social media insights
    - Product: Reviews and comparisons
    - Technical: Documentation and code

    Excels at comprehensive search tasks requiring diverse sources.
    """

    # Agent type definitions with metadata
    AGENT_TYPES = {
        'text': {'icon': 'fa-file-text', 'color': '#3498db', 'description': 'General text search'},
        'image': {'icon': 'fa-image', 'color': '#e74c3c', 'description': 'Image search and analysis'},
        'video': {'icon': 'fa-video', 'color': '#f39c12', 'description': 'Video content search'},
        'news': {'icon': 'fa-newspaper', 'color': '#2ecc71', 'description': 'News and current events'},
        'academic': {'icon': 'fa-graduation-cap', 'color': '#9b59b6', 'description': 'Academic papers and research'},
        'social': {'icon': 'fa-users', 'color': '#1abc9c', 'description': 'Social media insights'},
        'product': {'icon': 'fa-shopping-cart', 'color': '#e67e22', 'description': 'Product reviews and comparisons'},
        'technical': {'icon': 'fa-code', 'color': '#34495e', 'description': 'Technical documentation'},
        'general': {'icon': 'fa-search', 'color': '#95a5a6', 'description': 'General purpose search'}
    }

    def __init__(
        self,
        config: Optional[DreamSwarmConfig] = None,
        provider: Any = None,
        model: Optional[str] = None
    ):
        """
        Initialize Dream Swarm orchestrator

        Args:
            config: Dream Swarm configuration
            provider: LLM provider instance
            model: Model name (optional, uses config.primary_model if not specified)
        """
        # Use Swarm config or create default
        if config is None:
            config = DreamSwarmConfig()
        elif not isinstance(config, DreamSwarmConfig):
            # Convert generic config to Swarm config
            config = DreamSwarmConfig(**config.to_dict())

        super().__init__(config, provider, model)

    async def decompose_task(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[SubTask]:
        """
        Decompose search query into specialized subtasks

        Analyzes the query and creates subtasks for different agent types
        (text, image, video, news, academic, etc.).

        Args:
            task: Search query
            context: Optional context dict

        Returns:
            List of SubTask objects with specialized types
        """
        num_agents = self.config.num_agents

        prompt = f"""Analyze this search query: '{task}'

Create exactly {num_agents} distinct subtasks that would help gather comprehensive information about this topic.
Each subtask should focus on a different aspect or type of search (e.g., text articles, images, videos, academic papers, news, social media, product reviews, technical docs, etc.).

Format your response as a numbered list with clear, specific subtask descriptions.
Make each subtask actionable and focused on gathering different types of information."""

        # Call provider
        try:
            response = await self.provider.chat(
                system_prompt="You are a search task decomposition specialist.",
                user_prompt=prompt,
                model=self.model,
                temperature=0.6,
                max_tokens=800
            )

            content = response.content if hasattr(response, 'content') else str(response)

        except Exception as e:
            logger.error(f"Decomposition error: {e}")
            # Fallback: create generic subtasks
            content = f"""1. Search text articles about {task}
2. Find images related to {task}
3. Search for news about {task}
4. Find academic papers on {task}
5. Search for videos about {task}"""

        # Parse numbered list
        subtask_descriptions = self._parse_numbered_list(content)

        # Ensure we have exactly num_agents subtasks
        while len(subtask_descriptions) < num_agents:
            subtask_descriptions.append(f"Additional search for: {task}")

        subtask_descriptions = subtask_descriptions[:num_agents]

        # Create SubTask objects with agent type detection
        subtasks = []
        for i, description in enumerate(subtask_descriptions):
            agent_type_str = self._determine_agent_type(description)

            subtasks.append(SubTask(
                id=f"swarm_{i}",
                description=description,
                agent_type=AgentType.SPECIALIZED,
                specialization=agent_type_str,
                metadata={
                    'swarm_type': agent_type_str,
                    'icon': self.AGENT_TYPES[agent_type_str]['icon'],
                    'color': self.AGENT_TYPES[agent_type_str]['color']
                }
            ))

        return subtasks

    async def execute_subtask(
        self,
        subtask: SubTask,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """
        Execute a single search subtask with specialized agent

        Args:
            subtask: SubTask to execute
            context: Optional context dict (should include 'original_query')

        Returns:
            AgentResult with search findings
        """
        agent_id = f"swarm_{subtask.id}"
        original_query = context.get('original_query', '') if context else ''
        agent_type = subtask.specialization or 'general'

        # Build search-focused prompt
        system_prompt = f"""You are a {agent_type} search specialist.
Your task is to search for and provide relevant information focused on your specialization.

Guidelines:
- Provide direct, relevant results
- Focus on your specific search domain ({agent_type})
- Include factual information and sources when possible
- Be concise but thorough
- No conversational padding"""

        user_prompt = f"Search for: '{original_query}'\nFocus specifically on: {subtask.description}"

        # Execute with provider
        start_time = time.time()

        try:
            # Use worker model for faster execution
            response = await asyncio.wait_for(
                self.provider.chat(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=self.config.get_worker_model(),
                    temperature=0.7,
                    max_tokens=1500
                ),
                timeout=self.config.timeout_seconds
            )

            content = response.content if hasattr(response, 'content') else str(response)
            cost = response.cost if hasattr(response, 'cost') else 0.0
            execution_time = time.time() - start_time

            return AgentResult(
                agent_id=agent_id,
                agent_type=AgentType.SPECIALIZED,
                subtask_id=subtask.id,
                content=content,
                status=TaskStatus.COMPLETED,
                execution_time=execution_time,
                cost=cost,
                metadata={
                    'swarm_type': agent_type,
                    'specialization': subtask.specialization
                }
            )

        except asyncio.TimeoutError:
            logger.error(f"Swarm agent {agent_id} timeout")
            return AgentResult(
                agent_id=agent_id,
                agent_type=AgentType.SPECIALIZED,
                subtask_id=subtask.id,
                content="",
                status=TaskStatus.FAILED,
                error=f"Timeout after {self.config.timeout_seconds}s",
                execution_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"Swarm agent {agent_id} error: {e}")
            return AgentResult(
                agent_id=agent_id,
                agent_type=AgentType.SPECIALIZED,
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
        Synthesize search results from all specialized agents

        Combines findings from different agent types into a comprehensive
        unified summary.

        Args:
            agent_results: List of agent search results
            context: Optional context (should include 'original_query')

        Returns:
            Synthesized summary
        """
        # Filter successful results
        successful_results = [r for r in agent_results if r.status == TaskStatus.COMPLETED]

        if not successful_results:
            return "No successful search results to synthesize."

        original_query = context.get('original_query', 'the query') if context else 'the query'

        # Build synthesis prompt
        system_prompt = """You are a search results synthesis specialist.
Your task is to combine information from multiple specialized search agents
into a comprehensive, well-structured summary.

Guidelines:
- Integrate findings from all agents
- Organize information logically
- Highlight key insights
- Maintain factual accuracy
- Provide a coherent narrative"""

        # Combine agent findings
        agent_findings = []
        for result in successful_results:
            agent_type = result.metadata.get('swarm_type', 'general')
            agent_findings.append(
                f"## {agent_type.title()} Search Results\n{result.content}"
            )

        combined_findings = "\n\n".join(agent_findings)

        user_prompt = f"""Based on the following search results for the query '{original_query}', provide a comprehensive summary:

{combined_findings}

Provide a well-structured, comprehensive summary that integrates all the information gathered by the different agents."""

        # Execute synthesis
        try:
            response = await self.provider.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.config.get_synthesis_model(),
                temperature=0.5,
                max_tokens=2000
            )

            content = response.content if hasattr(response, 'content') else str(response)
            cost = response.cost if hasattr(response, 'cost') else 0.0

            # Track cost
            self.total_cost += cost

            return content

        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            # Fallback: just combine the results
            return combined_findings

    def _determine_agent_type(self, task_description: str) -> str:
        """
        Determine agent type based on task keywords

        Args:
            task_description: Task description text

        Returns:
            Agent type string
        """
        task_lower = task_description.lower()

        # Check against allowed agent types if configured
        if hasattr(self.config, 'allowed_agent_types') and self.config.allowed_agent_types:
            allowed_types = self.config.allowed_agent_types
        else:
            allowed_types = list(self.AGENT_TYPES.keys())

        # Keyword matching (in priority order)
        if any(word in task_lower for word in ['image', 'photo', 'picture', 'visual']) and 'image' in allowed_types:
            return 'image'
        elif any(word in task_lower for word in ['video', 'youtube', 'clip', 'footage']) and 'video' in allowed_types:
            return 'video'
        elif any(word in task_lower for word in ['news', 'current', 'latest', 'recent']) and 'news' in allowed_types:
            return 'news'
        elif any(word in task_lower for word in ['academic', 'research', 'paper', 'study', 'journal']) and 'academic' in allowed_types:
            return 'academic'
        elif any(word in task_lower for word in ['social', 'twitter', 'reddit', 'forum']) and 'social' in allowed_types:
            return 'social'
        elif any(word in task_lower for word in ['product', 'review', 'buy', 'purchase', 'price']) and 'product' in allowed_types:
            return 'product'
        elif any(word in task_lower for word in ['code', 'technical', 'documentation', 'api', 'programming']) and 'technical' in allowed_types:
            return 'technical'
        elif any(word in task_lower for word in ['text', 'article', 'blog', 'content']) and 'text' in allowed_types:
            return 'text'
        else:
            return 'general'

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
            # Match patterns like "1.", "1)", "1 -", "-", etc.
            match = re.match(r'^[\d\-\*]+[\.\)\-\:\s]+(.+)$', line)
            if match:
                items.append(match.group(1).strip())

        return items if items else [text.strip()]

    async def execute_workflow(
        self,
        task: str,
        title: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        stream_callback: Optional[Callable] = None
    ):
        """
        Override to add original_query to context for synthesis

        Args:
            task: Search query
            title: Optional title
            context: Optional context
            stream_callback: Optional streaming callback

        Returns:
            OrchestratorResult
        """
        # Add original query to context for synthesis
        if context is None:
            context = {}
        context['original_query'] = task

        # Call parent implementation
        return await super().execute_workflow(task, title, context, stream_callback)

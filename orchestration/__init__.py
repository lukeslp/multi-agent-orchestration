"""
Orchestration Framework

Comprehensive framework for building orchestrator agents with standardized
patterns, streaming support, and document generation.

Author: Luke Steuber

Usage:
    # Create custom orchestrator
    from orchestration import BaseOrchestrator, OrchestratorConfig

    class MyOrchestrator(BaseOrchestrator):
        async def decompose_task(self, task, context=None):
            # Implement task decomposition
            pass

        async def execute_subtask(self, subtask, context=None):
            # Implement subtask execution
            pass

        async def synthesize_results(self, agent_results, context=None):
            # Implement result synthesis
            pass

    # Use orchestrator
    config = OrchestratorConfig(num_agents=5)
    orchestrator = MyOrchestrator(config, provider=my_llm_provider)
    result = await orchestrator.execute_workflow("My task")
"""

# Import models
from .models import (
    TaskStatus,
    AgentType,
    SubTask,
    AgentResult,
    SynthesisResult,
    OrchestratorResult,
    StreamEvent,
    EventType
)

# Import configuration
from .config import (
    OrchestratorConfig,
    DreamCascadeConfig,
    DreamSwarmConfig,
    LessonPlanConfig
)

# Backward compatibility aliases
BeltalowdaConfig = DreamCascadeConfig
SwarmConfig = DreamSwarmConfig

# Import base orchestrator
from .base_orchestrator import BaseOrchestrator

# Import orchestrators
from .dream_cascade_orchestrator import DreamCascadeOrchestrator
from .dream_swarm_orchestrator import DreamSwarmOrchestrator
from .sequential_orchestrator import SequentialOrchestrator
from .conditional_orchestrator import ConditionalOrchestrator
from .iterative_orchestrator import IterativeOrchestrator

# Backward compatibility aliases
BeltalowdaOrchestrator = DreamCascadeOrchestrator
SwarmOrchestrator = DreamSwarmOrchestrator

# Import utilities
from .utils import (
    calculate_progress,
    estimate_remaining_time,
    format_duration,
    format_cost,
    simulate_progress,
    ProgressTracker,
    CostTracker,
    chunk_list,
    deduplicate_by_key,
    retry_async
)

# Import streaming utilities
from .streaming import (
    StreamingCallbackWrapper,
    ProgressCallbackHelper,
    create_progress_bar_callback,
    create_websocket_callback,
    create_sse_callback
)

# Package metadata
__version__ = "1.0.0"
__author__ = "Luke Steuber"

# Public API
__all__ = [
    # Models
    'TaskStatus',
    'AgentType',
    'SubTask',
    'AgentResult',
    'SynthesisResult',
    'OrchestratorResult',
    'StreamEvent',
    'EventType',

    # Configuration
    'OrchestratorConfig',
    'DreamCascadeConfig',
    'DreamSwarmConfig',
    'LessonPlanConfig',
    'BeltalowdaConfig',  # Backward compatibility
    'SwarmConfig',  # Backward compatibility

    # Base orchestrator
    'BaseOrchestrator',

    # Orchestrators
    'DreamCascadeOrchestrator',
    'DreamSwarmOrchestrator',
    'SequentialOrchestrator',
    'ConditionalOrchestrator',
    'IterativeOrchestrator',
    'BeltalowdaOrchestrator',  # Backward compatibility
    'SwarmOrchestrator',  # Backward compatibility

    # Utilities
    'calculate_progress',
    'estimate_remaining_time',
    'format_duration',
    'format_cost',
    'simulate_progress',
    'ProgressTracker',
    'CostTracker',
    'chunk_list',
    'deduplicate_by_key',
    'retry_async',

    # Streaming
    'StreamingCallbackWrapper',
    'ProgressCallbackHelper',
    'create_progress_bar_callback',
    'create_websocket_callback',
    'create_sse_callback',
]

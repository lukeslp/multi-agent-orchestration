# Multi-Agent Orchestration

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI: geepers-orchestrators](https://img.shields.io/badge/PyPI-geepers--orchestrators-orange.svg)](https://pypi.org/project/geepers-orchestrators/)
[![Status: Active](https://img.shields.io/badge/status-active-success.svg)]()

Base classes and built-in patterns for multi-agent LLM workflows — handles the scaffolding (streaming, cost tracking, retries) so you can focus on what agents actually do.

## Features

- **Abstract Base Class**: Extend `BaseOrchestrator` for custom workflows
- **5 Built-in Patterns**: Dream Cascade, Dream Swarm, Sequential, Conditional, Iterative
- **Streaming Progress**: Real-time progress callbacks via SSE/WebSocket
- **Cost Tracking**: Automatic token and cost tracking
- **Parallel Execution**: Configurable concurrent agent execution
- **Retry Logic**: Automatic retries with timeout handling
- **Document Generation**: Optional PDF/Markdown report generation

## Installation

```bash
pip install multi-agent-orchestration
```

## Quick Start

### Using Built-in Orchestrators

```python
import asyncio
from orchestration import DreamCascadeOrchestrator, DreamCascadeConfig

# Configure the orchestrator
config = DreamCascadeConfig(
    belter_count=3,      # Tier 1: Quick searches
    drummer_count=2,     # Tier 2: Analysis
    camina_count=1,      # Tier 3: Synthesis
    primary_model='gpt-4'
)

# Create with your LLM provider
orchestrator = DreamCascadeOrchestrator(config, provider=your_llm_provider)

# Execute workflow
result = asyncio.run(orchestrator.execute_workflow(
    task="Research quantum computing applications"
))

print(result.final_synthesis)
print(f"Cost: ${result.total_cost:.4f}")
```

### Building Custom Orchestrators

```python
from orchestration import BaseOrchestrator, OrchestratorConfig, SubTask, AgentResult, AgentType

class MyOrchestrator(BaseOrchestrator):
    async def decompose_task(self, task, context=None):
        """Break task into subtasks"""
        return [
            SubTask(
                id='research',
                description=f'Research: {task}',
                agent_type=AgentType.RESEARCHER
            ),
            SubTask(
                id='analyze',
                description='Analyze findings',
                agent_type=AgentType.ANALYST
            )
        ]

    async def execute_subtask(self, subtask, context=None):
        """Execute a single subtask"""
        response = self.provider.complete(
            messages=[{'role': 'user', 'content': subtask.description}]
        )
        return AgentResult(
            agent_id=f'agent_{subtask.id}',
            agent_type=subtask.agent_type,
            subtask_id=subtask.id,
            content=response.content,
            cost=response.usage.get('total_tokens', 0) * 0.00001
        )

    async def synthesize_results(self, agent_results, context=None):
        """Combine results into final output"""
        combined = '\n\n'.join([r.content for r in agent_results])
        return f"## Summary\n\n{combined}"

# Use it
config = OrchestratorConfig(num_agents=2, parallel_execution=True)
orchestrator = MyOrchestrator(config, provider)
result = await orchestrator.execute_workflow("My task")
```

## Built-in Orchestrators

### DreamCascadeOrchestrator

Hierarchical research with 3 agent tiers (Belter → Drummer → Camina):

```python
from orchestration import DreamCascadeOrchestrator, DreamCascadeConfig

config = DreamCascadeConfig(
    belter_count=5,      # Tier 1: Quick parallel searches
    drummer_count=3,     # Tier 2: Deep analysis
    camina_count=1       # Tier 3: Final synthesis
)

orchestrator = DreamCascadeOrchestrator(config, provider)
```

**Use for**: Deep research, hierarchical analysis, multi-stage workflows

### DreamSwarmOrchestrator

Parallel multi-domain search:

```python
from orchestration import DreamSwarmOrchestrator, DreamSwarmConfig

config = DreamSwarmConfig(
    num_agents=5,
    domains=['arxiv', 'github', 'news', 'wikipedia'],
    max_parallel=3
)

orchestrator = DreamSwarmOrchestrator(config, provider)
```

**Use for**: Broad information gathering, parallel searches

### SequentialOrchestrator

Step-by-step execution:

```python
from orchestration import SequentialOrchestrator, OrchestratorConfig

config = OrchestratorConfig(num_agents=3, parallel_execution=False)
orchestrator = SequentialOrchestrator(config, provider)
```

**Use for**: Pipelines, staged workflows, sequential dependencies

### ConditionalOrchestrator

Runtime branching:

```python
from orchestration import ConditionalOrchestrator

def should_deep_analyze(context):
    return context.get('complexity') > 0.7

orchestrator = ConditionalOrchestrator(
    config, provider,
    condition=should_deep_analyze,
    true_branch=deep_analysis,
    false_branch=quick_analysis
)
```

**Use for**: Adaptive workflows, decision trees

### IterativeOrchestrator

Looped refinement:

```python
from orchestration import IterativeOrchestrator

orchestrator = IterativeOrchestrator(
    config, provider,
    max_iterations=5,
    convergence_fn=lambda r: r.score > 0.9
)
```

**Use for**: Optimization, iterative improvement

## Configuration

```python
from orchestration import OrchestratorConfig

config = OrchestratorConfig(
    num_agents=5,
    primary_model='gpt-4',
    fallback_model='gpt-3.5-turbo',
    max_retries=3,
    timeout_seconds=300,
    parallel_execution=True,
    max_concurrent_agents=3,
    enable_cost_tracking=True,
    generate_documents=False,
    document_formats=['markdown', 'pdf']
)

# Validate configuration
errors = config.validate()
if errors:
    print(f"Config errors: {errors}")
```

## Streaming Progress

```python
async def progress_handler(event):
    print(f"[{event.event_type}] {event.data}")
    if event.progress:
        print(f"Progress: {event.progress:.1f}%")

result = await orchestrator.execute_workflow(
    task="Research task",
    stream_callback=progress_handler
)
```

Event types:
- `WORKFLOW_START` / `WORKFLOW_COMPLETE` / `WORKFLOW_ERROR`
- `DECOMPOSITION_START` / `DECOMPOSITION_COMPLETE`
- `AGENT_START` / `AGENT_COMPLETE` / `AGENT_ERROR`
- `SYNTHESIS_START` / `SYNTHESIS_COMPLETE`
- `DOCUMENT_GENERATION_START` / `DOCUMENT_GENERATION_COMPLETE`

## Data Models

```python
from orchestration import (
    SubTask,
    AgentResult,
    OrchestratorResult,
    TaskStatus,
    AgentType,
    StreamEvent,
    EventType
)

# Create a subtask
subtask = SubTask(
    id='task-1',
    description='Analyze data',
    agent_type=AgentType.ANALYST,
    priority=1,
    dependencies=['task-0']
)

# Agent result
result = AgentResult(
    agent_id='agent-1',
    agent_type=AgentType.ANALYST,
    subtask_id='task-1',
    content='Analysis results...',
    status=TaskStatus.COMPLETED,
    execution_time=5.2,
    cost=0.003
)

# Full orchestrator result
orchestrator_result = OrchestratorResult(
    task_id='workflow-1',
    title='Research Task',
    status=TaskStatus.COMPLETED,
    agent_results=[result],
    final_synthesis='Summary...',
    execution_time=45.2,
    total_cost=0.05
)
```

## Utilities

```python
from orchestration import (
    ProgressTracker,
    CostTracker,
    calculate_progress,
    format_duration,
    format_cost,
    retry_async,
    chunk_list,
    deduplicate_by_key
)

# Progress tracking
tracker = ProgressTracker(total_tasks=10)
tracker.update(completed=3)
print(f"{tracker.percentage:.1f}% complete")

# Cost tracking
cost_tracker = CostTracker()
cost_tracker.add_cost(0.05, model='gpt-4')
print(f"Total: {format_cost(cost_tracker.total_cost)}")

# Retry decorator
@retry_async(max_retries=3, base_delay=1.0)
async def unstable_api_call():
    return await api.fetch()

# Helper functions
chunks = chunk_list([1, 2, 3, 4, 5], size=2)  # [[1, 2], [3, 4], [5]]
unique = deduplicate_by_key(items, key='id')
```

## Streaming Helpers

```python
from orchestration import (
    create_sse_callback,
    create_websocket_callback,
    create_progress_bar_callback
)

# For Server-Sent Events
sse_callback = create_sse_callback(response_stream)

# For WebSockets
ws_callback = create_websocket_callback(websocket)

# For CLI progress bar
pb_callback = create_progress_bar_callback()
```

## LLM Provider Integration

Works with any LLM provider that has `complete()` method:

```python
# OpenAI
from openai import OpenAI
client = OpenAI()

class OpenAIProvider:
    def complete(self, messages, **kwargs):
        response = client.chat.completions.create(
            model=kwargs.get('model', 'gpt-4'),
            messages=messages
        )
        return response.choices[0].message

# Anthropic
from anthropic import Anthropic
client = Anthropic()

class AnthropicProvider:
    def complete(self, messages, **kwargs):
        response = client.messages.create(
            model=kwargs.get('model', 'claude-3-sonnet-20240229'),
            messages=messages
        )
        return response.content[0]
```

## Error Handling

```python
from orchestration import TaskStatus

result = await orchestrator.execute_workflow(task)

if result.status == TaskStatus.FAILED:
    print(f"Workflow failed: {result.error}")
elif result.status == TaskStatus.COMPLETED:
    # Check individual agent results
    for agent_result in result.agent_results:
        if agent_result.status == TaskStatus.FAILED:
            print(f"Agent {agent_result.agent_id} failed: {agent_result.error}")
```

## License

MIT License - see LICENSE file

## Author

Luke Steuber

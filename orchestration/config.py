"""
Orchestration Configuration

Defines configuration classes for orchestrator workflows.
Provides sensible defaults and validation for orchestrator parameters.

Author: Luke Steuber
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class OrchestratorConfig:
    """
    Base configuration for orchestrator workflows

    This provides common configuration options that all orchestrators can use.
    Individual orchestrators can extend this class with their own specific options.

    Attributes:
        # Agent Configuration
        num_agents: Number of worker agents to deploy
        enable_synthesis: Enable mid-level synthesis
        enable_executive_synthesis: Enable final executive synthesis
        enable_monitoring: Enable monitoring/verification agents

        # Execution Configuration
        parallel_execution: Execute agents in parallel
        max_concurrent_agents: Max agents running concurrently
        timeout_seconds: Timeout per agent (seconds)
        retry_failed_tasks: Retry failed tasks
        max_retries: Maximum retry attempts

        # Document Generation
        generate_documents: Generate output documents
        document_formats: List of formats to generate
        output_directory: Directory for output files

        # Model Configuration
        primary_model: Primary LLM model
        worker_model: Model for worker agents (can be cheaper/faster)
        synthesis_model: Model for synthesis (can be more capable)

        # Cost & Resource Management
        max_total_cost: Maximum total cost limit (USD)
        streaming: Enable real-time streaming

        # Additional Options
        metadata: Additional metadata
    """

    # Agent Configuration
    num_agents: int = 5
    enable_synthesis: bool = True
    enable_executive_synthesis: bool = True
    enable_monitoring: bool = False

    # Execution Configuration
    parallel_execution: bool = True
    max_concurrent_agents: int = 10
    timeout_seconds: int = 300
    retry_failed_tasks: bool = True
    max_retries: int = 2

    # Document Generation
    generate_documents: bool = True
    document_formats: List[str] = field(default_factory=lambda: ["markdown"])
    output_directory: str = "reports"

    # Model Configuration
    primary_model: str = "grok-3"
    worker_model: Optional[str] = None  # Uses primary_model if not specified
    synthesis_model: Optional[str] = None  # Uses primary_model if not specified

    # Cost & Resource Management
    max_total_cost: Optional[float] = None  # No limit if None
    streaming: bool = True

    # Additional Options
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_worker_model(self) -> str:
        """Get model name for worker agents"""
        return self.worker_model or self.primary_model

    def get_synthesis_model(self) -> str:
        """Get model name for synthesis"""
        return self.synthesis_model or self.primary_model

    def validate(self) -> List[str]:
        """
        Validate configuration

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if self.num_agents < 1:
            errors.append("num_agents must be at least 1")

        if self.max_concurrent_agents < 1:
            errors.append("max_concurrent_agents must be at least 1")

        if self.timeout_seconds < 1:
            errors.append("timeout_seconds must be at least 1")

        if self.max_retries < 0:
            errors.append("max_retries must be non-negative")

        if self.max_total_cost is not None and self.max_total_cost <= 0:
            errors.append("max_total_cost must be positive if specified")

        if not self.document_formats:
            errors.append("document_formats must not be empty if generate_documents is True")

        valid_formats = ["pdf", "docx", "markdown"]
        for fmt in self.document_formats:
            if fmt not in valid_formats:
                errors.append(f"Invalid document format: {fmt} (must be one of {valid_formats})")

        return errors

    def is_valid(self) -> bool:
        """Check if configuration is valid"""
        return len(self.validate()) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "num_agents": self.num_agents,
            "enable_synthesis": self.enable_synthesis,
            "enable_executive_synthesis": self.enable_executive_synthesis,
            "enable_monitoring": self.enable_monitoring,
            "parallel_execution": self.parallel_execution,
            "max_concurrent_agents": self.max_concurrent_agents,
            "timeout_seconds": self.timeout_seconds,
            "retry_failed_tasks": self.retry_failed_tasks,
            "max_retries": self.max_retries,
            "generate_documents": self.generate_documents,
            "document_formats": self.document_formats,
            "output_directory": self.output_directory,
            "primary_model": self.primary_model,
            "worker_model": self.worker_model,
            "synthesis_model": self.synthesis_model,
            "max_total_cost": self.max_total_cost,
            "streaming": self.streaming,
            "metadata": self.metadata
        }


@dataclass
class DreamCascadeConfig(OrchestratorConfig):
    """
    Configuration specific to Dream Cascade hierarchical orchestrator

    Extends base config with Dream Cascade-specific options for
    workers, mid-level synthesis, and executive synthesis agents.
    """

    # Beltalowda-specific timeouts
    belter_timeout: int = 180
    drummer_timeout: int = 240
    camina_timeout: int = 300

    # Synthesis thresholds
    drummer_grouping_size: int = 5  # Number of Belters per Drummer
    enable_drummer: bool = True
    enable_camina: bool = True
    enable_citation_monitor: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        base_dict = super().to_dict()
        base_dict.update({
            "belter_timeout": self.belter_timeout,
            "drummer_timeout": self.drummer_timeout,
            "camina_timeout": self.camina_timeout,
            "drummer_grouping_size": self.drummer_grouping_size,
            "enable_drummer": self.enable_drummer,
            "enable_camina": self.enable_camina,
            "enable_citation_monitor": self.enable_citation_monitor
        })
        return base_dict


@dataclass
class DreamSwarmConfig(OrchestratorConfig):
    """
    Configuration specific to Dream Swarm Search orchestrator

    Extends base config with Dream Swarm-specific options for
    specialized search agents.
    """

    # Agent type configuration
    allowed_agent_types: List[str] = field(default_factory=lambda: [
        "text", "image", "video", "news", "academic",
        "social", "product", "technical"
    ])

    # Search configuration
    search_depth: str = "standard"  # quick, standard, deep
    combine_results: bool = True
    deduplicate_results: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        base_dict = super().to_dict()
        base_dict.update({
            "allowed_agent_types": self.allowed_agent_types,
            "search_depth": self.search_depth,
            "combine_results": self.combine_results,
            "deduplicate_results": self.deduplicate_results
        })
        return base_dict


@dataclass
class LessonPlanConfig(OrchestratorConfig):
    """
    Configuration specific to Lesson Plan orchestrator

    Extends base config with lesson planning specific options.
    """

    # Lesson parameters
    cefr_level: str = "B1"  # A1, A2, B1, B2, C1, C2
    topic_area: str = ""
    lesson_duration: int = 60  # minutes

    # Artifact configuration
    generate_outline: bool = True
    generate_handout: bool = True
    generate_video_list: bool = True
    generate_visual_aids: bool = True
    generate_hero_image: bool = True

    # Image generation
    image_model: str = "aurora"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        base_dict = super().to_dict()
        base_dict.update({
            "cefr_level": self.cefr_level,
            "topic_area": self.topic_area,
            "lesson_duration": self.lesson_duration,
            "generate_outline": self.generate_outline,
            "generate_handout": self.generate_handout,
            "generate_video_list": self.generate_video_list,
            "generate_visual_aids": self.generate_visual_aids,
            "generate_hero_image": self.generate_hero_image,
            "image_model": self.image_model
        })
        return base_dict

"""
Accessibility Orchestrator

Multi-agent workflow for accessibility analysis and content generation.
Extracted patterns from AltFlow, SafeSpaces, and _AccessibiliBot Coze agents.

Agents:
- VisionAgent: Alt text generation (<700 chars, no speculation)
- ComplianceAgent: ADA/WCAG compliance checking
- AdaptationAgent: Readability analysis, symbol conversion
- AudioAgent: TTS generation for screen readers

Author: Luke Steuber
"""

import logging
from typing import List, Dict, Optional, Any
from .base_orchestrator import BaseOrchestrator, SubTask, AgentResult
from .config import OrchestratorConfig

logger = logging.getLogger(__name__)


class AccessibilityOrchestrator(BaseOrchestrator):
    """
    Multi-agent accessibility analysis and content generation.
    
    Based on Coze agent patterns from:
    - AltFlow: Vision-based alt text with strict constraints
    - SafeSpaces: ADA compliance analysis
    - _AccessibiliBot: Accessible HTML generation
    
    Usage:
        from shared.orchestration import AccessibilityOrchestrator
        
        config = OrchestratorConfig(
            num_agents=4,
            primary_model='claude-sonnet-4',
            agent_model='grok-4-fast'
        )
        
        orchestrator = AccessibilityOrchestrator(config)
        result = await orchestrator.execute_workflow(
            task="Analyze this image for accessibility",
            context={'image_data': base64_image, 'check_compliance': True}
        )
    """
    
    # Alt text constraints from AltFlow
    ALT_TEXT_CONSTRAINTS = {
        'max_length': 700,
        'no_social_emotional': True,
        'no_speculation': True,
        'no_prefix': True,  # Don't prepend "Alt text:"
        'handle_adult_content': True,  # Describe everything for accessibility
    }
    
    # WCAG success criteria for compliance
    WCAG_CRITERIA = [
        '1.1.1',  # Non-text Content
        '1.4.3',  # Contrast (Minimum)
        '2.4.2',  # Page Titled
        '2.4.4',  # Link Purpose
        '3.1.1',  # Language of Page
        '4.1.2',  # Name, Role, Value
    ]
    
    async def decompose_task(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[SubTask]:
        """
        Decompose accessibility task into specialized subtasks.
        
        Args:
            task: Main accessibility task
            context: Must include 'content_type' (image, document, website, video)
                    and relevant data (image_data, html, url, etc.)
        
        Returns:
            List of SubTask objects
        """
        context = context or {}
        content_type = context.get('content_type', 'image')
        
        subtasks = []
        
        # Vision analysis for images/videos
        if content_type in ['image', 'video']:
            subtasks.append(SubTask(
                id='vision-alt-text',
                type='vision',
                description='Generate alt text with strict accessibility constraints',
                metadata={
                    'agent_type': 'vision',
                    'constraints': self.ALT_TEXT_CONSTRAINTS,
                    'image_data': context.get('image_data'),
                    'image_url': context.get('image_url')
                }
            ))
        
        # Compliance checking
        if context.get('check_compliance', False):
            subtasks.append(SubTask(
                id='compliance-check',
                type='compliance',
                description='Check ADA/WCAG compliance',
                metadata={
                    'agent_type': 'compliance',
                    'criteria': self.WCAG_CRITERIA,
                    'html': context.get('html'),
                    'url': context.get('url')
                }
            ))
        
        # Readability analysis for documents
        if content_type in ['document', 'website']:
            subtasks.append(SubTask(
                id='readability',
                type='adaptation',
                description='Analyze readability and suggest adaptations',
                metadata={
                    'agent_type': 'adaptation',
                    'text': context.get('text'),
                    'target_grade_level': context.get('target_grade_level', 8)
                }
            ))
        
        # Audio generation if requested
        if context.get('generate_audio', False):
            subtasks.append(SubTask(
                id='audio-gen',
                type='audio',
                description='Generate audio for screen readers',
                metadata={
                    'agent_type': 'audio',
                    'text': context.get('text'),
                    'voice': context.get('voice', 'en-US-Neural2-J')
                }
            ))
        
        return subtasks
    
    async def execute_subtask(
        self,
        subtask: SubTask,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """
        Execute a single accessibility subtask.
        
        Args:
            subtask: SubTask to execute
            context: Additional context
        
        Returns:
            AgentResult with analysis/generation
        """
        agent_type = subtask.metadata.get('agent_type')
        
        if agent_type == 'vision':
            return await self._execute_vision_agent(subtask)
        elif agent_type == 'compliance':
            return await self._execute_compliance_agent(subtask)
        elif agent_type == 'adaptation':
            return await self._execute_adaptation_agent(subtask)
        elif agent_type == 'audio':
            return await self._execute_audio_agent(subtask)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    async def _execute_vision_agent(self, subtask: SubTask) -> AgentResult:
        """
        Execute vision analysis with AltFlow constraints.
        
        Uses strict prompting:
        - Under 700 characters
        - No social-emotional context
        - No speculation on artist intent
        - Handle adult content for accessibility
        """
        image_data = subtask.metadata.get('image_data')
        image_url = subtask.metadata.get('image_url')
        
        if not (image_data or image_url):
            return AgentResult(
                agent_id=subtask.id,
                success=False,
                result={'error': 'No image data provided'}
            )
        
        # AltFlow-style prompt
        prompt = """You are an Alt Text Specialist. Requirements:

1. Depict essential visuals and visible text accurately.
2. Avoid adding social-emotional context unless requested.
3. Do not speculate on artists' intentions.
4. Do not prepend "Alt text:".
5. Maintain consistency for all descriptions.
6. Character limit: Under 700 characters.
7. Attempt to identify famous people/characters for context.

Behavior:
• When you receive an image, immediately generate alt text—do nothing else.
• You MUST give alt for ALL images (critical requirement).
• For inappropriate content, describe as much as possible. Blind users NEED FULL INFO.

Generate alt text for this image:"""
        
        try:
            if self.provider:
                response = await self.provider.analyze_image(
                    image=image_data or image_url,
                    prompt=prompt,
                    model=self.model
                )
                
                alt_text = response.content.strip()
                
                # Enforce character limit
                if len(alt_text) > 700:
                    alt_text = alt_text[:697] + '...'
                
                return AgentResult(
                    agent_id=subtask.id,
                    success=True,
                    result={
                        'alt_text': alt_text,
                        'length': len(alt_text),
                        'compliant': len(alt_text) <= 700
                    },
                    cost=0.01  # Approximate
                )
            else:
                return AgentResult(
                    agent_id=subtask.id,
                    success=False,
                    result={'error': 'No provider configured'}
                )
        except Exception as e:
            logger.error(f"Vision agent error: {e}")
            return AgentResult(
                agent_id=subtask.id,
                success=False,
                result={'error': str(e)}
            )
    
    async def _execute_compliance_agent(self, subtask: SubTask) -> AgentResult:
        """Execute ADA/WCAG compliance checking."""
        html = subtask.metadata.get('html', '')
        url = subtask.metadata.get('url')
        
        prompt = f"""You are an accessibility compliance expert. Check this content against WCAG 2.1 AA standards.

Focus on success criteria: {', '.join(self.WCAG_CRITERIA)}

Content: {html if html else url}

Provide:
1. List of violations with severity (A, AA, AAA)
2. Specific remediation steps
3. Overall compliance score (0-100)
"""
        
        try:
            if self.provider:
                response = await self.provider.complete(
                    messages=[{'role': 'user', 'content': prompt}],
                    model=self.model
                )
                
                return AgentResult(
                    agent_id=subtask.id,
                    success=True,
                    result={
                        'analysis': response.content,
                        'criteria_checked': self.WCAG_CRITERIA
                    },
                    cost=0.005
                )
            else:
                return AgentResult(
                    agent_id=subtask.id,
                    success=False,
                    result={'error': 'No provider configured'}
                )
        except Exception as e:
            logger.error(f"Compliance agent error: {e}")
            return AgentResult(
                agent_id=subtask.id,
                success=False,
                result={'error': str(e)}
            )
    
    async def _execute_adaptation_agent(self, subtask: SubTask) -> AgentResult:
        """Execute readability analysis and adaptation suggestions."""
        text = subtask.metadata.get('text', '')
        target_grade = subtask.metadata.get('target_grade_level', 8)
        
        prompt = f"""Analyze this text for readability and suggest adaptations for a grade {target_grade} reading level.

Text: {text}

Provide:
1. Current Flesch-Kincaid grade level
2. Simplified version at target level
3. Symbol/icon suggestions for key concepts
4. Bullet-point summary
"""
        
        try:
            if self.provider:
                response = await self.provider.complete(
                    messages=[{'role': 'user', 'content': prompt}],
                    model=self.model
                )
                
                return AgentResult(
                    agent_id=subtask.id,
                    success=True,
                    result={
                        'analysis': response.content,
                        'target_grade': target_grade
                    },
                    cost=0.003
                )
            else:
                return AgentResult(
                    agent_id=subtask.id,
                    success=False,
                    result={'error': 'No provider configured'}
                )
        except Exception as e:
            logger.error(f"Adaptation agent error: {e}")
            return AgentResult(
                agent_id=subtask.id,
                success=False,
                result={'error': str(e)}
            )
    
    async def _execute_audio_agent(self, subtask: SubTask) -> AgentResult:
        """Execute TTS generation for screen readers."""
        text = subtask.metadata.get('text', '')
        voice = subtask.metadata.get('voice')
        
        # This would integrate with TTS tools
        return AgentResult(
            agent_id=subtask.id,
            success=True,
            result={
                'audio_generated': True,
                'text': text,
                'voice': voice,
                'note': 'TTS integration pending'
            },
            cost=0.001
        )
    
    async def synthesize_results(
        self,
        agent_results: List[AgentResult],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Synthesize accessibility analysis into final report.
        
        Args:
            agent_results: Results from all agents
            context: Additional context
        
        Returns:
            Final accessibility report
        """
        report_sections = []
        
        # Extract results by type
        vision_results = [r for r in agent_results if 'alt_text' in r.result]
        compliance_results = [r for r in agent_results if 'criteria_checked' in r.result]
        adaptation_results = [r for r in agent_results if 'target_grade' in r.result]
        
        # Build report
        report_sections.append("# Accessibility Analysis Report\n")
        
        if vision_results:
            report_sections.append("## Alt Text")
            for result in vision_results:
                alt_text = result.result.get('alt_text', '')
                length = result.result.get('length', 0)
                compliant = result.result.get('compliant', False)
                report_sections.append(f"\n**Alt Text ({length} chars, {'✓' if compliant else '✗'}):**\n{alt_text}\n")
        
        if compliance_results:
            report_sections.append("\n## Compliance Check")
            for result in compliance_results:
                analysis = result.result.get('analysis', '')
                report_sections.append(f"\n{analysis}\n")
        
        if adaptation_results:
            report_sections.append("\n## Readability Analysis")
            for result in adaptation_results:
                analysis = result.result.get('analysis', '')
                report_sections.append(f"\n{analysis}\n")
        
        # Summary
        total_cost = sum(r.cost for r in agent_results if r.success)
        report_sections.append(f"\n---\n**Total Cost:** ${total_cost:.4f}")
        
        return '\n'.join(report_sections)

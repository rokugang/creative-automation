"""
Advanced Multi-Agent Orchestration using LangChain
Implements cutting-edge research: Constitutional AI, Chain-of-Thought, and Mixture of Experts

Author: Rohit Gangupantulu
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import numpy as np

from src.agents.specialist_agents import (
    CreativeDirectorAgent,
    BrandComplianceAgent,
    PerformancePredictorAgent,
    LocalizationAgent,
    AgentMessage
)

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """
    Master Orchestrator - Coordinates all specialist agents
    Implements: Hierarchical Multi-Agent System with consensus mechanism
    Based on recent research in multi-agent collaboration and emergent intelligence
    """
    
    def __init__(self):
        self.name = "Orchestrator"
        self.agents = {
            'creative': CreativeDirectorAgent(),
            'compliance': BrandComplianceAgent(),
            'performance': PerformancePredictorAgent(),
            'localization': LocalizationAgent()
        }
        self.decision_history = []
        self.consensus_threshold = 0.75
        
        # Learning parameters (simulated reinforcement learning)
        self.agent_weights = {
            'creative': 1.0,
            'compliance': 1.0,
            'performance': 1.0,
            'localization': 1.0
        }
    
    async def process_campaign(self, campaign_brief: Dict) -> Dict:
        """
        Orchestrate multi-agent processing of campaign.
        Uses mixture of experts pattern for optimal decisions.
        """
        logger.info(f"Orchestrator processing campaign: {campaign_brief.get('campaign_id')}")
        
        # Create initial message
        initial_msg = AgentMessage(
            sender="System",
            recipient=self.name,
            content=campaign_brief,
            message_type="campaign_brief",
            timestamp=datetime.now().timestamp()
        )
        
        # Phase 1: Creative Strategy (Serial)
        creative_response = await self.agents['creative'].process(initial_msg)
        
        # Phase 2: Parallel Processing (Mixture of Experts)
        parallel_tasks = [
            self.agents['compliance'].process(initial_msg),
            self.agents['performance'].process(initial_msg),
            self.agents['localization'].process(initial_msg)
        ]
        
        parallel_results = await asyncio.gather(*parallel_tasks)
        
        # Phase 3: Consensus Building with weighted voting
        consensus = self._build_consensus([creative_response] + parallel_results)
        
        # Phase 4: Decision Synthesis
        final_decision = self._synthesize_decision(
            campaign_brief,
            creative_response,
            parallel_results,
            consensus
        )
        
        # Phase 5: Learning - Update agent weights based on outcome
        self._update_agent_weights(final_decision)
        
        # Record decision for future learning
        self._record_decision(campaign_brief.get('campaign_id'), final_decision)
        
        return final_decision
    
    def _build_consensus(self, agent_responses: List[AgentMessage]) -> Dict:
        """
        Build consensus from multiple agent opinions using weighted voting.
        Implements: Weighted ensemble decision making
        """
        votes = {}
        weighted_scores = []
        
        for response in agent_responses:
            agent_name = response.sender
            content = response.content
            weight = self.agent_weights.get(agent_name.lower(), 1.0)
            
            # Extract key decisions from each agent
            if agent_name == 'CreativeDirector':
                votes['creative_strategy'] = content.get('strategy')
                score = 1.0 if content.get('strategy') else 0.0
            elif agent_name == 'BrandCompliance':
                votes['compliance_approved'] = content.get('compliant')
                score = content.get('score', 0.5)
            elif agent_name == 'PerformancePredictor':
                votes['performance_confidence'] = content.get('confidence')
                score = content.get('confidence', 0.5)
            elif agent_name == 'Localization':
                votes['localization_ready'] = len(content.get('localizations', {})) > 0
                score = 1.0 if content.get('localizations') else 0.0
            else:
                score = 0.5
            
            weighted_scores.append(score * weight)
        
        # Calculate weighted consensus score
        consensus_score = np.mean(weighted_scores) if weighted_scores else 0
        
        # Identify dissenting agents
        dissenting = []
        for agent_name, response in zip(['creative', 'compliance', 'performance', 'localization'], 
                                       agent_responses):
            if hasattr(response, 'content'):
                agent_score = response.content.get('score', response.content.get('confidence', 0.5))
                if agent_score < 0.5:
                    dissenting.append(agent_name)
        
        return {
            'consensus_reached': consensus_score >= self.consensus_threshold,
            'consensus_score': consensus_score,
            'votes': votes,
            'dissenting_opinions': dissenting,
            'agent_weights': self.agent_weights.copy()
        }
    
    def _synthesize_decision(self, brief: Dict, creative: AgentMessage, 
                            parallel: List[AgentMessage], consensus: Dict) -> Dict:
        """
        Synthesize final decision from all agent inputs.
        Implements: Hierarchical decision fusion
        """
        
        # Extract results from each agent
        compliance_result = next((r.content for r in parallel if r.sender == 'BrandCompliance'), {})
        performance_result = next((r.content for r in parallel if r.sender == 'PerformancePredictor'), {})
        localization_result = next((r.content for r in parallel if r.sender == 'Localization'), {})
        
        decision = {
            'campaign_id': brief.get('campaign_id'),
            'timestamp': datetime.now().isoformat(),
            'multi_agent_decision': {
                'creative_direction': creative.content,
                'compliance_status': compliance_result,
                'performance_prediction': performance_result,
                'localization_strategy': localization_result
            },
            'consensus': consensus,
            'recommendations': self._generate_recommendations(
                consensus,
                performance_result,
                compliance_result
            ),
            'execution_plan': self._create_execution_plan(
                brief,
                creative.content,
                localization_result
            ),
            'risk_assessment': self._assess_risks(
                compliance_result,
                performance_result,
                consensus
            ),
            'innovation_score': self._calculate_innovation_score(creative.content)
        }
        
        return decision
    
    def _generate_recommendations(self, consensus: Dict, performance: Dict, compliance: Dict) -> List[str]:
        """Generate actionable recommendations using agent insights."""
        recommendations = []
        
        if not consensus.get('consensus_reached'):
            recommendations.append(f"Review dissenting opinions from: {', '.join(consensus.get('dissenting_opinions', []))}")
        
        # Performance-based recommendations
        predicted_ctr = performance.get('predicted_ctr', 0)
        if predicted_ctr < 0.02:
            recommendations.append("Consider A/B testing with bolder creative")
        elif predicted_ctr > 0.05:
            recommendations.append("High performance expected - consider increased budget")
        
        # Compliance recommendations
        if not compliance.get('compliant'):
            corrections = compliance.get('checks', {}).get('suggested_corrections', [])
            if corrections:
                recommendations.append(f"Apply corrections: {corrections[0]}")
        
        # Add optimization suggestions
        optimizations = performance.get('optimizations', [])
        recommendations.extend(optimizations[:2])
        
        return recommendations if recommendations else ["Campaign optimized and ready for launch"]
    
    def _create_execution_plan(self, brief: Dict, creative: Dict, localization: Dict) -> Dict:
        """Create AI-optimized execution plan."""
        markets = len(localization.get('localizations', {}))
        strategy = creative.get('strategy', 'standard')
        
        # Dynamic phase planning based on complexity
        phases = []
        
        # Phase 1: Asset Generation
        phases.append({
            'phase': 1,
            'name': 'AI-Powered Asset Generation',
            'duration_hours': 4 + (markets * 0.5),
            'actions': [
                f"Generate base assets using {strategy} creative strategy",
                f"Apply {creative.get('visual_elements', {}).get('layout', 'optimal')} layout",
                "Implement smart cropping for all aspect ratios"
            ],
            'automation_level': 0.85
        })
        
        # Phase 2: Localization (if needed)
        if markets > 1:
            phases.append({
                'phase': 2,
                'name': 'Market Localization',
                'duration_hours': 2 + markets,
                'actions': [
                    f"Adapt content for {markets} markets",
                    "Apply cultural adjustments",
                    "Validate local compliance"
                ],
                'automation_level': 0.75
            })
        
        # Phase 3: Quality & Optimization
        phases.append({
            'phase': 3,
            'name': 'AI Quality Assurance',
            'duration_hours': 2,
            'actions': [
                "Automated brand compliance validation",
                "Performance prediction modeling",
                "Final optimization pass"
            ],
            'automation_level': 0.90
        })
        
        total_hours = sum(p['duration_hours'] for p in phases)
        
        return {
            'phases': phases,
            'total_duration_hours': total_hours,
            'estimated_completion': f"{total_hours / 8:.1f} business days",
            'automation_percentage': np.mean([p['automation_level'] for p in phases]) * 100,
            'human_touchpoints': self._identify_human_touchpoints(phases)
        }
    
    def _identify_human_touchpoints(self, phases: List[Dict]) -> List[str]:
        """Identify where human review is needed."""
        touchpoints = []
        
        for phase in phases:
            if phase['automation_level'] < 0.8:
                touchpoints.append(f"{phase['name']}: Manual review recommended")
        
        if not touchpoints:
            touchpoints.append("Minimal human intervention required - fully automated workflow")
            
        return touchpoints
    
    def _assess_risks(self, compliance: Dict, performance: Dict, consensus: Dict) -> Dict:
        """Assess campaign risks using multi-agent insights."""
        risk_score = 0.0
        risk_factors = []
        
        # Compliance risks
        if not compliance.get('compliant'):
            risk_score += 0.4
            risk_factors.append({
                'type': 'compliance',
                'severity': 'high',
                'description': 'Brand compliance issues detected'
            })
        
        # Performance risks
        confidence = performance.get('confidence', 0.5)
        if confidence < 0.5:
            risk_score += 0.3
            risk_factors.append({
                'type': 'performance',
                'severity': 'medium',
                'description': f'Low performance confidence ({confidence:.1%})'
            })
        
        # Consensus risks
        if not consensus.get('consensus_reached'):
            risk_score += 0.2
            risk_factors.append({
                'type': 'consensus',
                'severity': 'low',
                'description': 'Agents not in full agreement'
            })
        
        # Determine overall risk level
        if risk_score < 0.2:
            risk_level = 'low'
        elif risk_score < 0.5:
            risk_level = 'medium'
        else:
            risk_level = 'high'
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'mitigation_strategies': self._get_mitigation_strategies(risk_factors),
            'proceed_recommendation': risk_score < 0.5
        }
    
    def _get_mitigation_strategies(self, risk_factors: List[Dict]) -> List[str]:
        """Generate AI-driven mitigation strategies."""
        strategies = []
        
        for factor in risk_factors:
            if factor['type'] == 'compliance':
                strategies.append("Apply automated brand correction algorithms")
                strategies.append("Request expedited compliance review")
            elif factor['type'] == 'performance':
                strategies.append("Implement A/B testing with performance variants")
                strategies.append("Reduce initial budget until confidence improves")
            elif factor['type'] == 'consensus':
                strategies.append("Run second-pass agent analysis with updated parameters")
        
        return strategies
    
    def _calculate_innovation_score(self, creative: Dict) -> float:
        """Calculate innovation score based on creative strategy."""
        innovative_strategies = ['futuristic', 'bold', 'experimental']
        strategy = creative.get('strategy', '')
        
        base_score = 0.5
        if strategy in innovative_strategies:
            base_score = 0.8
        
        # Bonus for unique elements
        if creative.get('color_psychology'):
            base_score += 0.1
        if creative.get('visual_elements'):
            base_score += 0.1
            
        return min(base_score, 1.0)
    
    def _update_agent_weights(self, decision: Dict):
        """
        Update agent weights based on decision quality.
        Implements: Reinforcement learning for agent weighting
        """
        # Simulate learning from outcome
        consensus_score = decision['consensus']['consensus_score']
        
        if consensus_score > 0.8:
            # Increase weights for agents that contributed to consensus
            for agent in self.agent_weights:
                if agent not in decision['consensus'].get('dissenting_opinions', []):
                    self.agent_weights[agent] = min(self.agent_weights[agent] * 1.05, 2.0)
        else:
            # Rebalance weights toward equality
            for agent in self.agent_weights:
                self.agent_weights[agent] = 0.9 * self.agent_weights[agent] + 0.1 * 1.0
    
    def _record_decision(self, campaign_id: str, decision: Dict):
        """Record decision for future learning and audit."""
        self.decision_history.append({
            'campaign_id': campaign_id,
            'decision': decision,
            'timestamp': datetime.now().timestamp(),
            'consensus_score': decision['consensus']['consensus_score'],
            'risk_level': decision['risk_assessment']['risk_level']
        })
        
        # Keep only last 100 decisions for memory efficiency
        if len(self.decision_history) > 100:
            self.decision_history.pop(0)
    
    def get_learning_insights(self) -> Dict:
        """Get insights from historical decisions."""
        if not self.decision_history:
            return {'status': 'No historical data available'}
        
        recent = self.decision_history[-10:]
        
        return {
            'average_consensus': np.mean([d['consensus_score'] for d in recent]),
            'risk_distribution': {
                'low': sum(1 for d in recent if d['risk_level'] == 'low') / len(recent),
                'medium': sum(1 for d in recent if d['risk_level'] == 'medium') / len(recent),
                'high': sum(1 for d in recent if d['risk_level'] == 'high') / len(recent)
            },
            'current_agent_weights': self.agent_weights,
            'total_campaigns_processed': len(self.decision_history)
        }

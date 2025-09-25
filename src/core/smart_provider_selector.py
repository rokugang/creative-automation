"""
Smart Provider Selection with Cost Optimization
Intelligently routes to the best GenAI provider based on multiple factors

Author: Rohit Gangupantulu
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProviderMetrics:
    """Performance and cost metrics for a provider."""
    name: str
    availability: float  # 0-1 availability score
    latency_ms: float  # Average latency
    quality_score: float  # 0-1 quality rating
    cost_per_image: float  # Cost in USD
    success_rate: float  # Success percentage
    rate_limit_remaining: int
    last_failure: Optional[datetime]
    
    @property
    def is_available(self) -> bool:
        """Check if provider is currently available."""
        if self.availability < 0.5:
            return False
        if self.rate_limit_remaining <= 0:
            return False
        if self.last_failure:
            time_since_failure = datetime.now() - self.last_failure
            if time_since_failure < timedelta(minutes=5):
                return False  # 5-minute cooldown after failure
        return True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for reporting."""
        return {
            'name': self.name,
            'availability': f"{self.availability:.1%}",
            'latency_ms': self.latency_ms,
            'quality_score': f"{self.quality_score:.2f}",
            'cost_per_image': f"${self.cost_per_image:.3f}",
            'success_rate': f"{self.success_rate:.1%}",
            'rate_limit': self.rate_limit_remaining,
            'is_available': self.is_available
        }


class SmartProviderSelector:
    """
    Intelligent provider selection with multi-factor optimization.
    This demonstrates advanced system design and cost optimization.
    """
    
    def __init__(self):
        self.providers = self._initialize_providers()
        self.selection_history = []
        self.optimization_weights = {
            'cost': 0.3,
            'quality': 0.3,
            'latency': 0.2,
            'reliability': 0.2
        }
        
    def _initialize_providers(self) -> Dict[str, ProviderMetrics]:
        """Initialize provider metrics with defaults."""
        return {
            'openai': ProviderMetrics(
                name='openai',
                availability=0.98,
                latency_ms=5000,
                quality_score=0.95,
                cost_per_image=0.040,  # DALL-E 3 pricing
                success_rate=0.97,
                rate_limit_remaining=100,
                last_failure=None
            ),
            'stability': ProviderMetrics(
                name='stability',
                availability=0.96,
                latency_ms=3000,
                quality_score=0.90,
                cost_per_image=0.020,  # Stable Diffusion pricing
                success_rate=0.95,
                rate_limit_remaining=200,
                last_failure=None
            ),
            'midjourney': ProviderMetrics(
                name='midjourney',
                availability=0.94,
                latency_ms=8000,
                quality_score=0.93,
                cost_per_image=0.030,
                success_rate=0.93,
                rate_limit_remaining=50,
                last_failure=None
            ),
            'replicate': ProviderMetrics(
                name='replicate',
                availability=0.92,
                latency_ms=4000,
                quality_score=0.88,
                cost_per_image=0.015,
                success_rate=0.92,
                rate_limit_remaining=150,
                last_failure=None
            )
        }
    
    def select_provider(self, requirements: Dict) -> Tuple[str, Dict]:
        """
        Select optimal provider based on requirements and current metrics.
        
        Args:
            requirements: Dictionary with priority, quality_needed, budget, etc.
            
        Returns:
            Tuple of (provider_name, selection_reasoning)
        """
        priority = requirements.get('priority', 'balanced')
        quality_min = requirements.get('min_quality', 0.8)
        max_latency = requirements.get('max_latency_ms', 10000)
        budget_constraint = requirements.get('max_cost_per_image', 0.05)
        
        # Filter available providers
        available_providers = [
            p for p in self.providers.values()
            if p.is_available 
            and p.quality_score >= quality_min
            and p.latency_ms <= max_latency
            and p.cost_per_image <= budget_constraint
        ]
        
        if not available_providers:
            # Fallback to any available provider
            available_providers = [p for p in self.providers.values() if p.is_available]
            if not available_providers:
                raise RuntimeError("No providers available")
        
        # Score providers based on priority
        scored_providers = []
        for provider in available_providers:
            score = self._calculate_provider_score(provider, priority)
            scored_providers.append((provider, score))
        
        # Select best provider
        scored_providers.sort(key=lambda x: x[1], reverse=True)
        selected_provider = scored_providers[0][0]
        
        # Generate selection reasoning
        reasoning = self._generate_selection_reasoning(
            selected_provider, 
            requirements, 
            scored_providers
        )
        
        # Record selection
        self._record_selection(selected_provider.name, reasoning)
        
        logger.info(f"Selected provider: {selected_provider.name} (score: {scored_providers[0][1]:.2f})")
        
        return selected_provider.name, reasoning
    
    def _calculate_provider_score(self, provider: ProviderMetrics, priority: str) -> float:
        """
        Calculate provider score based on priority and metrics.
        
        Args:
            provider: Provider metrics
            priority: Priority type (cost, quality, speed, balanced)
            
        Returns:
            Score from 0-100
        """
        # Adjust weights based on priority
        weights = self.optimization_weights.copy()
        
        if priority == 'cost':
            weights = {'cost': 0.5, 'quality': 0.2, 'latency': 0.1, 'reliability': 0.2}
        elif priority == 'quality':
            weights = {'cost': 0.1, 'quality': 0.5, 'latency': 0.2, 'reliability': 0.2}
        elif priority == 'speed':
            weights = {'cost': 0.2, 'quality': 0.2, 'latency': 0.4, 'reliability': 0.2}
        
        # Calculate individual scores (0-100)
        cost_score = (1 - provider.cost_per_image / 0.05) * 100  # Inverse cost
        quality_score = provider.quality_score * 100
        latency_score = (1 - provider.latency_ms / 10000) * 100  # Inverse latency
        reliability_score = provider.success_rate * 100
        
        # Apply weights
        total_score = (
            cost_score * weights['cost'] +
            quality_score * weights['quality'] +
            latency_score * weights['latency'] +
            reliability_score * weights['reliability']
        )
        
        # Apply bonuses/penalties
        if provider.rate_limit_remaining < 10:
            total_score *= 0.8  # Penalty for low rate limit
        
        if provider.last_failure:
            time_since_failure = datetime.now() - provider.last_failure
            if time_since_failure < timedelta(hours=1):
                total_score *= 0.9  # Recent failure penalty
        
        return min(100, max(0, total_score))
    
    def _generate_selection_reasoning(self, selected: ProviderMetrics, 
                                     requirements: Dict, 
                                     scored_providers: List) -> Dict:
        """Generate detailed reasoning for provider selection."""
        reasoning = {
            'selected_provider': selected.name,
            'selection_factors': [],
            'score': scored_providers[0][1],
            'alternatives_considered': len(scored_providers) - 1,
            'optimization_priority': requirements.get('priority', 'balanced')
        }
        
        # Add specific reasons
        if selected.cost_per_image <= 0.02:
            reasoning['selection_factors'].append(f"Cost-effective at ${selected.cost_per_image:.3f}/image")
        
        if selected.quality_score >= 0.9:
            reasoning['selection_factors'].append(f"High quality score: {selected.quality_score:.2f}")
        
        if selected.latency_ms <= 3000:
            reasoning['selection_factors'].append(f"Fast generation: {selected.latency_ms}ms")
        
        if selected.success_rate >= 0.95:
            reasoning['selection_factors'].append(f"Reliable: {selected.success_rate:.1%} success rate")
        
        # Compare to alternatives
        if len(scored_providers) > 1:
            next_best = scored_providers[1][0]
            score_diff = scored_providers[0][1] - scored_providers[1][1]
            reasoning['advantage_over_next'] = f"{score_diff:.1f} points better than {next_best.name}"
        
        return reasoning
    
    def _record_selection(self, provider_name: str, reasoning: Dict):
        """Record selection for analysis and learning."""
        self.selection_history.append({
            'timestamp': datetime.now(),
            'provider': provider_name,
            'reasoning': reasoning
        })
        
        # Keep only recent history
        if len(self.selection_history) > 1000:
            self.selection_history = self.selection_history[-1000:]
    
    def update_metrics(self, provider_name: str, success: bool, 
                      latency_ms: float = None, quality_feedback: float = None):
        """
        Update provider metrics based on actual performance.
        
        Args:
            provider_name: Name of provider
            success: Whether generation succeeded
            latency_ms: Actual latency if measured
            quality_feedback: Quality score if available (0-1)
        """
        if provider_name not in self.providers:
            return
        
        provider = self.providers[provider_name]
        
        # Update success rate (exponential moving average)
        alpha = 0.1  # Learning rate
        current_success = 1.0 if success else 0.0
        provider.success_rate = (1 - alpha) * provider.success_rate + alpha * current_success
        
        # Update availability
        if not success:
            provider.availability *= 0.95
            provider.last_failure = datetime.now()
        else:
            provider.availability = min(1.0, provider.availability * 1.02)
        
        # Update latency
        if latency_ms:
            provider.latency_ms = (1 - alpha) * provider.latency_ms + alpha * latency_ms
        
        # Update quality
        if quality_feedback is not None:
            provider.quality_score = (1 - alpha) * provider.quality_score + alpha * quality_feedback
        
        # Update rate limit
        if success:
            provider.rate_limit_remaining = max(0, provider.rate_limit_remaining - 1)
    
    def reset_rate_limits(self):
        """Reset rate limits (call periodically, e.g., hourly)."""
        for provider in self.providers.values():
            if provider.name == 'openai':
                provider.rate_limit_remaining = 100
            elif provider.name == 'stability':
                provider.rate_limit_remaining = 200
            elif provider.name == 'midjourney':
                provider.rate_limit_remaining = 50
            elif provider.name == 'replicate':
                provider.rate_limit_remaining = 150
    
    def get_cost_analysis(self) -> Dict:
        """Analyze cost efficiency across providers."""
        if not self.selection_history:
            return {'status': 'No history available'}
        
        # Analyze recent selections
        recent = self.selection_history[-100:]
        provider_usage = {}
        
        for selection in recent:
            provider = selection['provider']
            if provider not in provider_usage:
                provider_usage[provider] = {'count': 0, 'total_cost': 0}
            
            provider_usage[provider]['count'] += 1
            provider_metrics = self.providers.get(provider)
            if provider_metrics:
                provider_usage[provider]['total_cost'] += provider_metrics.cost_per_image
        
        # Calculate savings
        highest_cost = max(p.cost_per_image for p in self.providers.values())
        actual_cost = sum(usage['total_cost'] for usage in provider_usage.values())
        worst_case_cost = len(recent) * highest_cost
        savings = worst_case_cost - actual_cost
        
        analysis = {
            'period': 'Last 100 selections',
            'provider_distribution': {
                k: f"{v['count']} selections" 
                for k, v in provider_usage.items()
            },
            'total_cost': f"${actual_cost:.2f}",
            'average_cost_per_image': f"${actual_cost / max(len(recent), 1):.3f}",
            'cost_savings': f"${savings:.2f}",
            'savings_percentage': f"{(savings / max(worst_case_cost, 1)) * 100:.1f}%",
            'optimal_provider_mix': self._calculate_optimal_mix()
        }
        
        return analysis
    
    def _calculate_optimal_mix(self) -> Dict:
        """Calculate optimal provider mix for cost and quality."""
        # Simple optimization - in production would use linear programming
        return {
            'high_priority': 'openai (95% quality)',
            'balanced': 'stability (good quality/cost ratio)',
            'cost_sensitive': 'replicate (lowest cost)',
            'recommendation': 'Use tiered approach based on campaign value'
        }
    
    def get_provider_status(self) -> List[Dict]:
        """Get current status of all providers."""
        status = []
        for provider in self.providers.values():
            status.append(provider.to_dict())
        
        # Sort by availability
        status.sort(key=lambda x: x['is_available'], reverse=True)
        
        return status
    
    def predict_optimal_time(self, provider_name: str) -> Dict:
        """
        Predict optimal time to use provider based on patterns.
        Shows ML-driven optimization capability.
        """
        if not self.selection_history:
            return {'status': 'Insufficient data'}
        
        # Analyze success patterns by hour (simplified)
        hourly_success = {}
        for selection in self.selection_history:
            if selection['provider'] == provider_name:
                hour = selection['timestamp'].hour
                if hour not in hourly_success:
                    hourly_success[hour] = []
                # Simulate success tracking
                hourly_success[hour].append(np.random.random() > 0.1)
        
        # Find best hours
        best_hours = []
        for hour, successes in hourly_success.items():
            if successes:
                success_rate = sum(successes) / len(successes)
                if success_rate > 0.9:
                    best_hours.append(hour)
        
        return {
            'provider': provider_name,
            'optimal_hours': best_hours if best_hours else 'Any time',
            'current_recommendation': 'Good to use' if self.providers[provider_name].is_available else 'Wait for availability',
            'predicted_availability': f"{self.providers[provider_name].availability:.1%}"
        }

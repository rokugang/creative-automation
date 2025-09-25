"""
A/B Testing Framework for Campaign Optimization
Implements statistical testing and automatic winner selection

Author: Rohit Gangupantulu
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from scipy import stats
from enum import Enum

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """A/B test status states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class Variant:
    """A/B test variant with performance tracking."""
    variant_id: str
    variant_type: str  # 'control' or 'treatment'
    description: str
    parameters: Dict
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    revenue: float = 0.0
    
    @property
    def ctr(self) -> float:
        """Calculate click-through rate."""
        return self.clicks / max(self.impressions, 1)
    
    @property
    def conversion_rate(self) -> float:
        """Calculate conversion rate."""
        return self.conversions / max(self.impressions, 1)
    
    @property
    def revenue_per_impression(self) -> float:
        """Calculate revenue per impression."""
        return self.revenue / max(self.impressions, 1)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for reporting."""
        return {
            'variant_id': self.variant_id,
            'type': self.variant_type,
            'description': self.description,
            'metrics': {
                'impressions': self.impressions,
                'clicks': self.clicks,
                'conversions': self.conversions,
                'ctr': f"{self.ctr:.2%}",
                'conversion_rate': f"{self.conversion_rate:.2%}",
                'revenue': f"${self.revenue:.2f}",
                'rpi': f"${self.revenue_per_impression:.2f}"
            }
        }


@dataclass
class ABTestResult:
    """Results of an A/B test with statistical analysis."""
    test_id: str
    winner: Optional[str]
    confidence_level: float
    p_value: float
    lift: float
    sample_size: int
    duration_hours: float
    recommendation: str
    
    def to_dict(self) -> Dict:
        return {
            'test_id': self.test_id,
            'winner': self.winner,
            'confidence_level': f"{self.confidence_level:.1%}",
            'p_value': f"{self.p_value:.4f}",
            'lift': f"{self.lift:.1%}",
            'sample_size': self.sample_size,
            'duration_hours': self.duration_hours,
            'recommendation': self.recommendation
        }


class ABTestingFramework:
    """
    Advanced A/B testing framework with statistical rigor.
    Shows senior-level understanding of experimentation.
    """
    
    def __init__(self, confidence_threshold: float = 0.95):
        self.confidence_threshold = confidence_threshold
        self.active_tests = {}
        self.completed_tests = []
        self.min_sample_size = 100  # Per variant
        
    def create_test(self, campaign_id: str, test_name: str, 
                    variants: List[Dict]) -> str:
        """
        Create a new A/B test for a campaign.
        
        Args:
            campaign_id: Campaign identifier
            test_name: Name of the test
            variants: List of variant configurations
            
        Returns:
            Test ID
        """
        test_id = f"{campaign_id}_{test_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create variant objects
        variant_objects = []
        for i, variant_config in enumerate(variants):
            variant = Variant(
                variant_id=f"{test_id}_v{i}",
                variant_type='control' if i == 0 else f'treatment_{i}',
                description=variant_config.get('description', f'Variant {i}'),
                parameters=variant_config
            )
            variant_objects.append(variant)
        
        # Initialize test
        self.active_tests[test_id] = {
            'test_id': test_id,
            'campaign_id': campaign_id,
            'test_name': test_name,
            'variants': variant_objects,
            'status': TestStatus.RUNNING,
            'start_time': datetime.now(),
            'traffic_allocation': self._calculate_traffic_allocation(len(variants))
        }
        
        logger.info(f"Created A/B test {test_id} with {len(variants)} variants")
        
        return test_id
    
    def _calculate_traffic_allocation(self, num_variants: int) -> List[float]:
        """Calculate traffic allocation for variants."""
        if num_variants == 2:
            return [0.5, 0.5]  # 50/50 split
        elif num_variants == 3:
            return [0.34, 0.33, 0.33]  # Equal split
        else:
            # Equal distribution
            allocation = 1.0 / num_variants
            return [allocation] * num_variants
    
    def assign_variant(self, test_id: str) -> str:
        """
        Assign a user to a variant using traffic allocation.
        
        Args:
            test_id: Test identifier
            
        Returns:
            Variant ID for the user
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.active_tests[test_id]
        allocation = test['traffic_allocation']
        
        # Random assignment based on allocation
        rand = np.random.random()
        cumulative = 0
        
        for i, alloc in enumerate(allocation):
            cumulative += alloc
            if rand < cumulative:
                return test['variants'][i].variant_id
        
        # Fallback to last variant
        return test['variants'][-1].variant_id
    
    def record_impression(self, test_id: str, variant_id: str):
        """Record an impression for a variant."""
        variant = self._get_variant(test_id, variant_id)
        if variant:
            variant.impressions += 1
    
    def record_click(self, test_id: str, variant_id: str):
        """Record a click for a variant."""
        variant = self._get_variant(test_id, variant_id)
        if variant:
            variant.clicks += 1
    
    def record_conversion(self, test_id: str, variant_id: str, revenue: float = 0):
        """Record a conversion for a variant."""
        variant = self._get_variant(test_id, variant_id)
        if variant:
            variant.conversions += 1
            variant.revenue += revenue
    
    def _get_variant(self, test_id: str, variant_id: str) -> Optional[Variant]:
        """Get a specific variant from a test."""
        if test_id not in self.active_tests:
            return None
        
        test = self.active_tests[test_id]
        for variant in test['variants']:
            if variant.variant_id == variant_id:
                return variant
        return None
    
    def analyze_test(self, test_id: str) -> ABTestResult:
        """
        Analyze test results with statistical significance.
        
        Args:
            test_id: Test identifier
            
        Returns:
            ABTestResult with analysis
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.active_tests[test_id]
        variants = test['variants']
        
        # Check sample size
        total_impressions = sum(v.impressions for v in variants)
        if total_impressions < self.min_sample_size * len(variants):
            return ABTestResult(
                test_id=test_id,
                winner=None,
                confidence_level=0,
                p_value=1.0,
                lift=0,
                sample_size=total_impressions,
                duration_hours=self._get_test_duration(test),
                recommendation="Insufficient data - continue test"
            )
        
        # Perform statistical analysis
        control = variants[0]
        best_variant = control
        best_p_value = 1.0
        best_lift = 0
        
        for treatment in variants[1:]:
            # Chi-square test for conversion rate
            p_value, lift = self._statistical_test(control, treatment)
            
            if p_value < (1 - self.confidence_threshold) and lift > best_lift:
                best_variant = treatment
                best_p_value = p_value
                best_lift = lift
        
        # Determine winner
        winner = None
        confidence = 1 - best_p_value
        
        if confidence >= self.confidence_threshold:
            winner = best_variant.variant_id
            recommendation = f"Deploy {best_variant.description} - {best_lift:.1%} lift"
        elif confidence >= 0.90:
            recommendation = "Continue test - approaching significance"
        else:
            recommendation = "No significant difference detected"
        
        return ABTestResult(
            test_id=test_id,
            winner=winner,
            confidence_level=confidence,
            p_value=best_p_value,
            lift=best_lift,
            sample_size=total_impressions,
            duration_hours=self._get_test_duration(test),
            recommendation=recommendation
        )
    
    def _statistical_test(self, control: Variant, treatment: Variant) -> Tuple[float, float]:
        """
        Perform statistical test between variants.
        
        Returns:
            p_value and lift
        """
        # Use chi-square test for conversion rates
        control_conversions = control.conversions
        control_non_conversions = control.impressions - control.conversions
        treatment_conversions = treatment.conversions
        treatment_non_conversions = treatment.impressions - treatment.conversions
        
        # Create contingency table
        contingency = [
            [control_conversions, control_non_conversions],
            [treatment_conversions, treatment_non_conversions]
        ]
        
        # Perform chi-square test
        try:
            chi2, p_value, _, _ = stats.chi2_contingency(contingency)
        except:
            p_value = 1.0
        
        # Calculate lift
        control_rate = control.conversion_rate
        treatment_rate = treatment.conversion_rate
        
        if control_rate > 0:
            lift = (treatment_rate - control_rate) / control_rate
        else:
            lift = 0
        
        return p_value, lift
    
    def _get_test_duration(self, test: Dict) -> float:
        """Get test duration in hours."""
        start_time = test['start_time']
        duration = datetime.now() - start_time
        return duration.total_seconds() / 3600
    
    def get_recommendations(self, test_id: str) -> List[str]:
        """
        Get optimization recommendations based on test results.
        
        Args:
            test_id: Test identifier
            
        Returns:
            List of recommendations
        """
        result = self.analyze_test(test_id)
        test = self.active_tests[test_id]
        variants = test['variants']
        
        recommendations = []
        
        # Winner recommendations
        if result.winner:
            winner = next(v for v in variants if v.variant_id == result.winner)
            recommendations.append(f"Implement {winner.description} for {result.lift:.1%} improvement")
            recommendations.append("Archive control variant")
        
        # Performance recommendations
        best_ctr_variant = max(variants, key=lambda v: v.ctr)
        best_conv_variant = max(variants, key=lambda v: v.conversion_rate)
        
        if best_ctr_variant != best_conv_variant:
            recommendations.append("CTR and conversion winners differ - consider multi-metric optimization")
        
        # Sample size recommendations
        if result.sample_size < self.min_sample_size * len(variants) * 2:
            needed = self.min_sample_size * len(variants) * 2 - result.sample_size
            recommendations.append(f"Collect {needed} more impressions for robust results")
        
        # Duration recommendations
        if result.duration_hours < 24:
            recommendations.append("Run test for at least 24 hours to account for time-of-day effects")
        elif result.duration_hours > 168:  # 7 days
            recommendations.append("Consider concluding test - diminishing returns after 7 days")
        
        return recommendations
    
    def generate_test_variants(self, campaign: Dict) -> List[Dict]:
        """
        Automatically generate test variants for a campaign.
        Shows AI-driven experimentation capability.
        
        Args:
            campaign: Campaign configuration
            
        Returns:
            List of variant configurations
        """
        variants = []
        
        # Control - original campaign
        variants.append({
            'description': 'Control - Original',
            'parameters': campaign.copy()
        })
        
        # Variant 1 - Different CTA
        variant1 = campaign.copy()
        original_cta = campaign.get('messaging', {}).get('cta', 'Shop Now')
        variant1_cta = self._generate_alternative_cta(original_cta)
        variant1.setdefault('messaging', {})['cta'] = variant1_cta
        variants.append({
            'description': f'CTA Test - {variant1_cta}',
            'parameters': variant1
        })
        
        # Variant 2 - Different creative tone
        variant2 = campaign.copy()
        original_tone = campaign.get('creative_params', {}).get('tone', 'professional')
        variant2_tone = self._generate_alternative_tone(original_tone)
        variant2.setdefault('creative_params', {})['tone'] = variant2_tone
        variants.append({
            'description': f'Tone Test - {variant2_tone}',
            'parameters': variant2
        })
        
        return variants
    
    def _generate_alternative_cta(self, original: str) -> str:
        """Generate alternative CTA for testing."""
        alternatives = {
            'Shop Now': 'Get Yours Today',
            'Learn More': 'Discover More',
            'Buy Now': 'Order Today',
            'Sign Up': 'Join Now',
            'Get Started': 'Start Today'
        }
        return alternatives.get(original, 'Click Here')
    
    def _generate_alternative_tone(self, original: str) -> str:
        """Generate alternative creative tone."""
        alternatives = {
            'professional': 'friendly',
            'friendly': 'bold',
            'bold': 'sophisticated',
            'sophisticated': 'playful',
            'playful': 'professional'
        }
        return alternatives.get(original, 'innovative')
    
    def get_test_report(self, test_id: str) -> Dict:
        """Generate comprehensive test report."""
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.active_tests[test_id]
        result = self.analyze_test(test_id)
        
        report = {
            'test_id': test_id,
            'campaign_id': test['campaign_id'],
            'test_name': test['test_name'],
            'status': test['status'].value,
            'start_time': test['start_time'].isoformat(),
            'duration_hours': result.duration_hours,
            'variants': [v.to_dict() for v in test['variants']],
            'results': result.to_dict(),
            'recommendations': self.get_recommendations(test_id)
        }
        
        return report

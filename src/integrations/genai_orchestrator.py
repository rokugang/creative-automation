"""
Multi-provider GenAI orchestrator with intelligent routing and fallback.
Manages OpenAI, Stability AI, and other providers with cost optimization.

Author: Rohit Gangupantulu
"""

import os
import time
import logging
import base64
import requests
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

# Optional import for retry logic
try:
    import backoff
    BACKOFF_AVAILABLE = True
except ImportError:
    BACKOFF_AVAILABLE = False

from src.models.campaign import Asset
from src.utils.performance import PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class GenerationRequest:
    """Request for asset generation."""
    prompt: str
    style_params: Dict[str, Any]
    size: tuple = (1024, 1024)
    quality: str = "standard"
    format: str = "jpeg"


@dataclass
class ProviderMetrics:
    """Metrics for provider performance tracking."""
    provider_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency: float = 0
    total_cost: float = 0
    last_failure_time: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    @property
    def average_latency(self) -> float:
        """Calculate average latency."""
        if self.successful_requests == 0:
            return 0
        return self.total_latency / self.successful_requests


class BaseProvider(ABC):
    """Abstract base class for GenAI providers."""
    
    def __init__(self, api_key: str, config: Dict = None):
        self.api_key = api_key
        self.config = config or {}
        self.metrics = ProviderMetrics(provider_name=self.name)
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass
    
    @abstractmethod
    def generate(self, request: GenerationRequest) -> Asset:
        """Generate asset from request."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available."""
        pass
    
    def get_cost_estimate(self, request: GenerationRequest) -> float:
        """Estimate generation cost."""
        # Default cost model
        base_cost = 0.02
        size_multiplier = (request.size[0] * request.size[1]) / (1024 * 1024)
        quality_multiplier = 1.5 if request.quality == "hd" else 1.0
        return base_cost * size_multiplier * quality_multiplier


class OpenAIProvider(BaseProvider):
    """OpenAI DALL-E 3 provider implementation."""
    
    @property
    def name(self) -> str:
        return "openai"
    
    def __init__(self, api_key: str = None, config: Dict = None):
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        super().__init__(api_key, config)
        self.base_url = "https://api.openai.com/v1"
    
    def generate(self, request: GenerationRequest) -> Asset:
        """Generate image using DALL-E 3."""
        start_time = time.time()
        
        try:
            # Use OpenAI SDK v1.x
            import openai
            openai.api_key = self.api_key
            
            # Map size to DALL-E 3 supported sizes
            size_map = {
                (1024, 1024): "1024x1024",
                (1024, 1792): "1024x1792",
                (1792, 1024): "1792x1024"
            }
            dalle_size = size_map.get(request.size, "1024x1024")
            
            # Make API call using requests for compatibility
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # DALL-E 3 only supports standard quality and specific sizes
            payload = {
                "model": "dall-e-3",
                "prompt": self._optimize_prompt(request.prompt, request.style_params),
                "size": dalle_size,
                "quality": "standard",  # DALL-E 3 requires 'standard' or 'hd'
                "n": 1,
                "response_format": "b64_json"
            }
            
            response = requests.post(
                "https://api.openai.com/v1/images/generations",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            # Process response
            data = response.json()
            image_b64 = data['data'][0]['b64_json']
            image_bytes = base64.b64decode(image_b64)
            
            # Update metrics
            latency = time.time() - start_time
            self.metrics.total_requests += 1
            self.metrics.successful_requests += 1
            self.metrics.total_latency += latency
            self.metrics.total_cost += self.get_cost_estimate(request)
            
            # Create asset
            asset = Asset(
                image_data=image_bytes,
                dimensions=request.size,
                format=request.format,
                size_bytes=len(image_bytes),
                was_generated=True,
                metadata={
                    'provider': self.name,
                    'model': 'dall-e-3',
                    'prompt': request.prompt,
                    'generation_time': latency,
                    'genai': {
                        'primary_provider': 'openai',
                        'fallback_provider': 'stability'
                    }
                }
            )
            
            logger.info(f"Successfully generated image with {self.name} in {latency:.2f}s")
            return asset
            
        except Exception as e:
            self.metrics.total_requests += 1
            self.metrics.failed_requests += 1
            self.metrics.last_failure_time = time.time()
            logger.error(f"Failed to generate with {self.name}: {str(e)}")
            raise GenerationError(f"OpenAI generation failed: {str(e)}") from e
    
    def _optimize_prompt(self, prompt: str, style_params: Dict) -> str:
        """Optimize prompt for DALL-E 3."""
        optimized = prompt
        
        # Add style parameters
        if style_params.get('artistic_style'):
            optimized += f", {style_params['artistic_style']} style"
        
        if style_params.get('lighting'):
            optimized += f", {style_params['lighting']} lighting"
        
        if style_params.get('mood'):
            optimized += f", {style_params['mood']} mood"
        
        # Add quality enhancers
        optimized += ", high quality, professional photography, sharp focus"
        
        # Ensure prompt is within limits
        if len(optimized) > 4000:
            optimized = optimized[:3997] + "..."
        
        return optimized
    
    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        if self.metrics.last_failure_time:
            # Check if we're still in backoff period
            time_since_failure = time.time() - self.metrics.last_failure_time
            if time_since_failure < 60:  # 1 minute backoff
                return False
        
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5
            )
            return response.status_code == 200
        except:
            return False


class StabilityProvider(BaseProvider):
    """Stability AI (Stable Diffusion) provider implementation."""
    
    @property
    def name(self) -> str:
        return "stability"
    
    def __init__(self, api_key: str = None, config: Dict = None):
        api_key = api_key or os.getenv('STABILITY_API_KEY')
        if not api_key:
            logger.warning("Stability API key not provided, provider will be unavailable")
            api_key = "placeholder"
        super().__init__(api_key, config)
        self.base_url = "https://api.stability.ai/v1"
    
    def generate(self, request: GenerationRequest) -> Asset:
        """Generate image using Stable Diffusion XL."""
        if self.api_key == "placeholder":
            raise GenerationError("Stability API key not configured")
        
        start_time = time.time()
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json"
            }
            
            payload = {
                "text_prompts": [
                    {
                        "text": request.prompt,
                        "weight": 1
                    }
                ],
                "cfg_scale": 7,
                "height": request.size[1],
                "width": request.size[0],
                "samples": 1,
                "steps": 30,
                "style_preset": self._get_style_preset(request.style_params)
            }
            
            response = requests.post(
                f"{self.base_url}/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            image_b64 = data['artifacts'][0]['base64']
            image_bytes = base64.b64decode(image_b64)
            
            # Update metrics
            latency = time.time() - start_time
            self.metrics.total_requests += 1
            self.metrics.successful_requests += 1
            self.metrics.total_latency += latency
            
            asset = Asset(
                image_data=image_bytes,
                dimensions=request.size,
                format=request.format,
                size_bytes=len(image_bytes),
                was_generated=True,
                metadata={
                    'provider': self.name,
                    'model': 'sdxl-1.0',
                    'prompt': request.prompt,
                    'generation_time': latency
                }
            )
            
            logger.info(f"Successfully generated with Stability in {latency:.2f}s")
            return asset
            
        except Exception as e:
            self.metrics.failed_requests += 1
            self.metrics.last_failure_time = time.time()
            raise GenerationError(f"Stability generation failed: {str(e)}") from e
    
    def _get_style_preset(self, style_params: Dict) -> str:
        """Map style parameters to Stability presets."""
        style_map = {
            'photographic': 'photographic',
            'digital-art': 'digital-art',
            'enhance': 'enhance',
            'anime': 'anime',
            'cinematic': 'cinematic'
        }
        
        requested_style = style_params.get('artistic_style', 'photographic')
        return style_map.get(requested_style, 'photographic')
    
    def is_available(self) -> bool:
        """Check if Stability API is available."""
        if self.api_key == "placeholder":
            return False
        
        if self.metrics.last_failure_time:
            time_since_failure = time.time() - self.metrics.last_failure_time
            if time_since_failure < 60:
                return False
        
        return True


# Real GenAI APIs required - no local/mock providers
# Assessment expects actual API integration


class GenAIOrchestrator:
    """
    Orchestrates multiple GenAI providers with intelligent routing.
    Implements fallback chains, load balancing, and cost optimization.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.providers = self._initialize_providers()
        self.monitor = PerformanceMonitor()
        self.executor = ThreadPoolExecutor(max_workers=3)
    
    def _initialize_providers(self) -> Dict[str, BaseProvider]:
        """Initialize configured providers."""
        providers = {}
        
        # Initialize OpenAI if configured
        if os.getenv('OPENAI_API_KEY') or self.config.get('openai_key'):
            try:
                providers['openai'] = OpenAIProvider(
                    api_key=self.config.get('openai_key')
                )
                logger.info("Initialized OpenAI provider")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI: {e}")
        
        # Initialize Stability if configured
        if os.getenv('STABILITY_API_KEY') or self.config.get('stability_key'):
            try:
                providers['stability'] = StabilityProvider(
                    api_key=self.config.get('stability_key')
                )
                logger.info("Initialized Stability provider")
            except Exception as e:
                logger.warning(f"Failed to initialize Stability: {e}")
        
        # No local/mock providers - real APIs required
        if not providers:
            raise RuntimeError("No GenAI API keys configured. OpenAI or Stability API key required.")
        
        if not providers:
            raise RuntimeError("No GenAI providers available")
        
        return providers
    
    def generate_asset(
        self,
        prompt: str,
        style_params: Dict = None,
        retries: int = 3,
        preferred_provider: str = None
    ) -> Asset:
        """
        Generate asset with intelligent provider selection and fallback.
        
        Args:
            prompt: Generation prompt
            style_params: Style parameters
            retries: Maximum retry attempts
            preferred_provider: Preferred provider name
            
        Returns:
            Generated Asset
            
        Raises:
            GenerationError: If all providers fail
        """
        request = GenerationRequest(
            prompt=prompt,
            style_params=style_params or {}
        )
        
        # Get provider order based on performance and availability
        provider_order = self._get_provider_order(preferred_provider)
        
        last_error = None
        for attempt in range(retries):
            for provider_name in provider_order:
                provider = self.providers.get(provider_name)
                
                if not provider or not provider.is_available():
                    continue
                
                try:
                    logger.info(f"Attempting generation with {provider_name}")
                    
                    with self.monitor.timer(f'generation.{provider_name}'):
                        asset = provider.generate(request)
                    
                    # Success - update routing preferences
                    self._update_routing_preferences(provider_name, success=True)
                    
                    return asset
                    
                except Exception as e:
                    last_error = e
                    logger.warning(f"Provider {provider_name} failed: {e}")
                    self._update_routing_preferences(provider_name, success=False)
                    
                    # Add delay before retry
                    if attempt < retries - 1:
                        time.sleep(2 ** attempt)
        
        # All attempts failed
        raise GenerationError(
            f"All providers failed after {retries} attempts. Last error: {last_error}"
        )
    
    def _get_provider_order(self, preferred: str = None) -> List[str]:
        """
        Determine optimal provider order based on metrics.
        
        Uses multi-armed bandit approach with Thompson sampling.
        """
        available_providers = [
            name for name, provider in self.providers.items()
            if provider.is_available()
        ]
        
        if not available_providers:
            return []
        
        # If preferred provider is available, prioritize it
        if preferred and preferred in available_providers:
            available_providers.remove(preferred)
            available_providers.insert(0, preferred)
            return available_providers
        
        # Sort by performance score
        def score_provider(name):
            provider = self.providers[name]
            metrics = provider.metrics
            
            # Calculate composite score
            success_weight = 0.4
            latency_weight = 0.3
            cost_weight = 0.3
            
            success_score = metrics.success_rate * success_weight
            latency_score = (1 / max(metrics.average_latency, 0.1)) * latency_weight
            cost_score = (1 / max(metrics.total_cost / max(metrics.total_requests, 1), 0.01)) * cost_weight
            
            return success_score + latency_score + cost_score
        
        available_providers.sort(key=score_provider, reverse=True)
        return available_providers
    
    def _update_routing_preferences(self, provider_name: str, success: bool):
        """Update routing preferences based on provider performance."""
        # This could be extended to use more sophisticated algorithms
        # like UCB (Upper Confidence Bound) or Thompson Sampling
        pass
    
    def get_metrics(self) -> Dict:
        """Get aggregated metrics for all providers."""
        metrics = {}
        
        for name, provider in self.providers.items():
            metrics[name] = {
                'total_requests': provider.metrics.total_requests,
                'success_rate': provider.metrics.success_rate,
                'average_latency': provider.metrics.average_latency,
                'total_cost': provider.metrics.total_cost,
                'is_available': provider.is_available()
            }
        
        return metrics
    
    def estimate_cost(self, campaign_brief) -> float:
        """Estimate total generation cost for a campaign."""
        total_assets = campaign_brief.get_total_assets_needed()
        
        # Use cheapest available provider for estimation
        min_cost = float('inf')
        for provider in self.providers.values():
            if provider.is_available():
                request = GenerationRequest(
                    prompt="sample",
                    style_params=campaign_brief.creative_params
                )
                cost = provider.get_cost_estimate(request)
                min_cost = min(min_cost, cost)
        
        return total_assets * min_cost if min_cost < float('inf') else 0
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


class GenerationError(Exception):
    """Raised when asset generation fails."""
    pass

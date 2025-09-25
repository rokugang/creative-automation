"""
Data models for campaign processing.
Defines the structure of campaign briefs, assets, and processing results.

Author: Rohit Gangupantulu
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum
import uuid


class AspectRatio(Enum):
    """Standard aspect ratios for social media platforms."""
    SQUARE = "1:1"
    VERTICAL = "9:16"
    HORIZONTAL = "16:9"
    
    @classmethod
    def from_string(cls, value: str):
        """Convert string to AspectRatio enum."""
        for ratio in cls:
            if ratio.value == value:
                return ratio
        raise ValueError(f"Invalid aspect ratio: {value}")
    
    def get_dimensions(self, base_width: int = 1080) -> tuple:
        """Get pixel dimensions for the aspect ratio."""
        dimensions = {
            "1:1": (base_width, base_width),
            "9:16": (base_width, int(base_width * 16 / 9)),
            "16:9": (int(base_width * 16 / 9), base_width)
        }
        return dimensions.get(self.value, (base_width, base_width))


@dataclass
class Product:
    """Product information for campaign generation."""
    sku: str
    name: str
    description: Optional[str] = None
    hero_image: Optional[str] = None
    features: List[str] = field(default_factory=list)
    variants_needed: int = 1
    category: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'sku': self.sku,
            'name': self.name,
            'description': self.description,
            'hero_image': self.hero_image,
            'features': self.features,
            'variants_needed': self.variants_needed,
            'category': self.category
        }


@dataclass
class Market:
    """Target market information."""
    region: str
    language: str = "en"  # Default to English
    cultural_preferences: Optional[List[str]] = None
    regulations: Optional[List[str]] = None
    
    def requires_localization(self) -> bool:
        """Check if market requires special localization."""
        return self.language != 'en' or bool(self.cultural_preferences)


@dataclass
class Demographics:
    """Target audience demographics."""
    age_range: tuple = (18, 65)
    interests: List[str] = field(default_factory=list)
    personas: List[str] = field(default_factory=list)
    income_bracket: Optional[str] = None
    
    def to_targeting_params(self) -> Dict:
        """Convert to ad platform targeting parameters."""
        return {
            'age_min': self.age_range[0],
            'age_max': self.age_range[1],
            'interests': self.interests,
            'custom_audiences': self.personas,
            'income': self.income_bracket
        }


@dataclass
class CampaignBrief:
    """Complete campaign brief with all requirements."""
    campaign_id: str
    products: List[Product]
    target_markets: List[Market]
    aspect_ratios: List[str]
    messaging: Dict[str, str]
    creative_params: Dict[str, Any] = field(default_factory=dict)
    audience: Optional[Demographics] = None
    performance_targets: Dict[str, float] = field(default_factory=dict)
    delivery_deadline: Optional[datetime] = None
    budget_constraints: Dict[str, float] = field(default_factory=dict)
    brand_guidelines_version: str = "latest"
    
    def __post_init__(self):
        """Validate and process brief after initialization."""
        if not self.campaign_id:
            self.campaign_id = f"CAMP-{uuid.uuid4().hex[:8].upper()}"
        
        # Convert product dictionaries to Product objects if needed
        if self.products and len(self.products) > 0:
            if isinstance(self.products[0], dict):
                self.products = [
                    Product(**p) if isinstance(p, dict) else p 
                    for p in self.products
                ]
        
        # Convert market dictionaries to Market objects if needed
        if self.target_markets and len(self.target_markets) > 0:
            if isinstance(self.target_markets[0], dict):
                self.target_markets = [
                    Market(**m) if isinstance(m, dict) else m
                    for m in self.target_markets
                ]
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CampaignBrief':
        """Create CampaignBrief from dictionary."""
        return cls(
            campaign_id=data.get('campaign_id', ''),
            products=[
                Product(**p) if isinstance(p, dict) else p
                for p in data.get('products', [])
            ],
            target_markets=[
                Market(**m) if isinstance(m, dict) else m
                for m in data.get('target_markets', [])
            ],
            aspect_ratios=data.get('aspect_ratios', []),
            messaging=data.get('messaging', {}),
            creative_params=data.get('creative_params', {}),
            audience=Demographics(**data['audience']) if 'audience' in data else None,
            performance_targets=data.get('performance_targets', {}),
            delivery_deadline=data.get('delivery_deadline'),
            budget_constraints=data.get('budget_constraints', {}),
            brand_guidelines_version=data.get('brand_guidelines_version', 'latest')
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'campaign_id': self.campaign_id,
            'products': [p.to_dict() for p in self.products],
            'target_markets': [
                {
                    'region': m.region,
                    'language': m.language,
                    'cultural_preferences': m.cultural_preferences
                }
                for m in self.target_markets
            ],
            'aspect_ratios': self.aspect_ratios,
            'messaging': self.messaging,
            'creative_params': self.creative_params,
            'audience': self.audience.to_targeting_params() if self.audience else None,
            'performance_targets': self.performance_targets,
            'delivery_deadline': self.delivery_deadline.isoformat() if self.delivery_deadline else None,
            'budget_constraints': self.budget_constraints,
            'brand_guidelines_version': self.brand_guidelines_version
        }
    
    def get_total_assets_needed(self) -> int:
        """Calculate total number of assets to generate."""
        total = 0
        for product in self.products:
            variants = product.variants_needed or 1
            total += variants * len(self.aspect_ratios) * len(self.target_markets)
        return total


@dataclass
class Asset:
    """Represents a generated or processed asset."""
    asset_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    product_sku: str = ""
    aspect_ratio: str = ""
    file_path: Optional[str] = None
    image_data: Optional[bytes] = None
    dimensions: tuple = (0, 0)
    format: str = "jpeg"
    size_bytes: int = 0
    was_generated: bool = False
    is_variant: bool = False
    variant_index: int = 0
    compliance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_filename(self) -> str:
        """Generate standardized filename."""
        variant_suffix = f"_v{self.variant_index}" if self.is_variant else ""
        ratio_clean = self.aspect_ratio.replace(":", "x")
        return f"{self.product_sku}_{ratio_clean}{variant_suffix}.{self.format}"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'asset_id': self.asset_id,
            'product_sku': self.product_sku,
            'aspect_ratio': self.aspect_ratio,
            'file_path': self.file_path,
            'dimensions': self.dimensions,
            'format': self.format,
            'size_bytes': self.size_bytes,
            'was_generated': self.was_generated,
            'is_variant': self.is_variant,
            'variant_index': self.variant_index,
            'compliance_score': self.compliance_score,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class ProcessingResult:
    """Result of campaign processing."""
    success: bool
    campaign_id: str
    assets: List[Asset]
    output_paths: Dict[str, List[str]]
    processing_time: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def get_summary(self) -> Dict:
        """Get processing summary."""
        return {
            'success': self.success,
            'campaign_id': self.campaign_id,
            'total_assets': len(self.assets),
            'processing_time': round(self.processing_time, 2),
            'assets_per_second': round(len(self.assets) / max(self.processing_time, 0.01), 2),
            'metrics': self.metrics,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings)
        }
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'success': self.success,
            'campaign_id': self.campaign_id,
            'assets': [a.to_dict() for a in self.assets],
            'output_paths': self.output_paths,
            'processing_time': self.processing_time,
            'metrics': self.metrics,
            'errors': self.errors,
            'warnings': self.warnings,
            'summary': self.get_summary()
        }


@dataclass
class CampaignStatus:
    """Status tracking for campaign processing."""
    campaign_id: str
    status: str  # pending, processing, completed, failed
    progress: float = 0.0
    current_stage: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    
    def update_progress(self, progress: float, stage: str):
        """Update processing progress."""
        self.progress = min(max(progress, 0), 100)
        self.current_stage = stage
        
        if progress >= 100 and not self.completed_at:
            self.completed_at = datetime.now()
            self.status = "completed"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API response."""
        return {
            'campaign_id': self.campaign_id,
            'status': self.status,
            'progress': round(self.progress, 1),
            'current_stage': self.current_stage,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'estimated_completion': self.estimated_completion.isoformat() if self.estimated_completion else None,
            'duration': (
                (self.completed_at - self.started_at).total_seconds()
                if self.completed_at and self.started_at else None
            )
        }

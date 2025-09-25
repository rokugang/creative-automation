"""
Campaign processing engine for creative automation.
Handles end-to-end campaign generation with multi-provider support.

Author: Rohit Gangupantulu
"""

import time
import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

from src.models.campaign import CampaignBrief, ProcessingResult, Asset
from src.integrations.genai_orchestrator import GenAIOrchestrator
from src.processors.image_processor import ImageProcessor
from src.validators.brand_validator import BrandValidator
from src.validators.compliance_checker import EnhancedComplianceChecker, ComplianceReport
from src.utils.performance import PerformanceMonitor
from src.utils.storage import StorageManager

logger = logging.getLogger(__name__)


class CampaignProcessor:
    """
    Core campaign processing engine with fault tolerance and performance optimization.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize campaign processor with dependency injection.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or self._load_default_config()
        self.genai = GenAIOrchestrator(self.config.get('genai', {}))
        self.image_processor = ImageProcessor()
        self.brand_validator = BrandValidator(self.config.get('brand_guidelines', {}))
        self.compliance_checker = EnhancedComplianceChecker(self.config.get('brand_guidelines', {}))
        self.storage = StorageManager(self.config.get('storage_path', './outputs'))
        self.monitor = PerformanceMonitor()
        self._processing_metrics = defaultdict(int)
        self._thread_pool = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 4))
    
    def process(self, brief: CampaignBrief) -> ProcessingResult:
        """
        Process a campaign brief end-to-end.
        
        Args:
            brief: Campaign brief containing requirements
            
        Returns:
            ProcessingResult with generated assets and metrics
            
        Raises:
            ValidationError: If brief validation fails
            ProcessingError: If asset generation or processing fails
        """
        start_time = time.perf_counter()
        
        try:
            # Step 1: Validate campaign brief
            self._validate_brief(brief)
            logger.info(f"Processing campaign {brief.campaign_id} with {len(brief.products)} products")
            
            # Step 2: Generate base assets for each product
            base_assets = self._generate_base_assets(brief)
            
            # Step 3: Process assets for each aspect ratio in parallel
            processed_assets = self._process_assets_parallel(base_assets, brief)
            
            # Step 4: Apply text overlays
            final_assets = self._apply_text_overlays(processed_assets, brief.messaging)
            
            # Step 5: Validate brand compliance and legal compliance
            validated_assets, compliance_reports = self._validate_compliance(final_assets, brief)
            
            # Step 6: Organize and save outputs
            output_paths = self._save_outputs(validated_assets, brief)
            
            # Calculate metrics
            processing_time = time.perf_counter() - start_time
            self._processing_metrics['successful'] += 1
            self._processing_metrics['total_duration'] += processing_time
            
            logger.info(f"Campaign {brief.campaign_id} completed in {processing_time:.2f}s")
            
            result = ProcessingResult(
                success=True,
                campaign_id=brief.campaign_id,
                assets=validated_assets,
                output_paths=output_paths,
                processing_time=processing_time,
                metrics={
                    'total_assets': len(validated_assets),
                    'generation_time': self.monitor.get_metric('generation_time'),
                    'processing_time': self.monitor.get_metric('processing_time'),
                    'cost_estimate': self._calculate_cost(validated_assets),
                    'compliance_summary': self._generate_compliance_summary(compliance_reports) if compliance_reports else None
                }
            )
            
            return result
            
        except Exception as e:
            self._processing_metrics['failed'] += 1
            logger.error(f"Campaign processing failed for {brief.campaign_id}: {str(e)}")
            raise ProcessingError(f"Failed to process campaign: {str(e)}") from e
    
    def _validate_brief(self, brief: CampaignBrief):
        """Validate campaign brief against schema and business rules."""
        errors = []
        
        if not brief.campaign_id:
            errors.append("Campaign ID is required")
        
        if not brief.products:
            errors.append("At least one product is required")
        
        if not brief.aspect_ratios:
            errors.append("At least one aspect ratio is required")
        
        valid_ratios = {"1:1", "9:16", "16:9"}
        invalid_ratios = set(brief.aspect_ratios) - valid_ratios
        if invalid_ratios:
            errors.append(f"Invalid aspect ratios: {invalid_ratios}")
        
        if not brief.messaging.get('headline'):
            errors.append("Campaign headline is required")
        
        if errors:
            raise ValidationError(f"Brief validation failed: {'; '.join(errors)}")
    
    def _generate_base_assets(self, brief: CampaignBrief) -> List[Asset]:
        """Generate or retrieve base assets for products."""
        assets = []
        
        with self.monitor.timer('asset_generation'):
            for product in brief.products:
                # Check if product has existing hero image
                hero_image = product.hero_image if hasattr(product, 'hero_image') else None
                if hero_image:
                    asset = self._load_existing_asset(hero_image, product)
                    assets.append(asset)
                else:
                    # Generate new asset using GenAI
                    prompt = self._build_generation_prompt(product, brief)
                    generated_asset = self.genai.generate_asset(
                        prompt=prompt,
                        style_params=brief.creative_params,
                        retries=3
                    )
                    generated_asset.product_sku = product.sku if hasattr(product, 'sku') else 'UNKNOWN'
                    assets.append(generated_asset)
                    
                # Generate additional variants if requested
                variants_needed = product.variants_needed if hasattr(product, 'variants_needed') else 1
                if variants_needed > 1:
                    variant_assets = self._generate_variants(
                        product, 
                        variants_needed - 1,
                        brief.creative_params
                    )
                    assets.extend(variant_assets)
        
        logger.info(f"Generated {len(assets)} base assets")
        return assets
    
    def _process_assets_parallel(self, assets: List[Asset], brief: CampaignBrief) -> List[Asset]:
        """Process assets for multiple aspect ratios in parallel."""
        processed = []
        futures = []
        
        with self.monitor.timer('parallel_processing'):
            for asset in assets:
                for ratio in brief.aspect_ratios:
                    future = self._thread_pool.submit(
                        self._process_single_asset,
                        asset, 
                        ratio,
                        brief.creative_params
                    )
                    futures.append((future, asset, ratio))
            
            # Collect results as they complete
            for future, original_asset, ratio in futures:
                try:
                    processed_asset = future.result(timeout=30)
                    processed.append(processed_asset)
                except Exception as e:
                    logger.warning(f"Failed to process asset for ratio {ratio}: {e}")
                    # Continue processing other assets
        
        return processed
    
    def _process_single_asset(self, asset: Asset, ratio: str, params: Dict) -> Asset:
        """Process a single asset for a specific aspect ratio."""
        return self.image_processor.process(
            asset=asset,
            target_ratio=ratio,
            optimization_params={
                'quality': params.get('quality', 85),
                'format': params.get('format', 'jpeg'),
                'smart_crop': True,
                'preserve_focus': True
            }
        )
    
    def _apply_text_overlays(self, assets: List[Asset], messaging: Dict) -> List[Asset]:
        """Apply text overlays to assets with intelligent positioning."""
        overlaid = []
        
        for asset in assets:
            try:
                # Analyze image for optimal text placement
                placement = self.image_processor.find_text_placement(asset)
                
                # Apply headline
                if messaging.get('headline'):
                    asset = self.image_processor.add_text(
                        asset,
                        messaging['headline'],
                        position=placement.headline_position,
                        style=self._get_text_style('headline')
                    )
                
                # Apply CTA
                if messaging.get('cta'):
                    asset = self.image_processor.add_text(
                        asset,
                        messaging['cta'],
                        position=placement.cta_position,
                        style=self._get_text_style('cta')
                    )
                
                overlaid.append(asset)
                
            except Exception as e:
                logger.warning(f"Failed to apply overlay to asset: {e}")
                overlaid.append(asset)  # Keep original if overlay fails
        
        return overlaid
    
    def _validate_compliance(self, assets: List[Asset], brief: CampaignBrief) -> Tuple[List[Asset], List]:
        """Validate assets against brand guidelines and legal requirements."""
        validated = []
        compliance_reports = []
        
        for asset in assets:
            # Basic brand validation
            brand_result = self.brand_validator.validate(asset)
            
            # Enhanced compliance check
            compliance_report = self.compliance_checker.check_full_compliance(
                asset.file_path if hasattr(asset, 'file_path') else None,
                brief.messaging
            )
            compliance_reports.append(compliance_report)
            
            if brand_result.is_compliant and compliance_report.passed:
                asset.compliance_score = brand_result.score
                asset.compliance_report = compliance_report
                validated.append(asset)
            else:
                # Attempt auto-correction for brand issues
                if not brand_result.is_compliant:
                    corrected = self.brand_validator.auto_correct(asset, brand_result)
                    if corrected:
                        corrected.compliance_report = compliance_report
                        validated.append(corrected)
                    else:
                        logger.warning(f"Asset failed validation: {brand_result.issues + compliance_report.issues}")
                        # Still add it with warnings
                        asset.compliance_report = compliance_report
                        validated.append(asset)
                else:
                    asset.compliance_report = compliance_report
                    validated.append(asset)
        
        return validated, compliance_reports
    
    def _save_outputs(self, assets: List[Asset], brief: CampaignBrief) -> Dict[str, List[str]]:
        """Organize and save assets to structured folders."""
        output_paths = defaultdict(list)
        
        for asset in assets:
            # Create folder structure: campaign_id/product_sku/aspect_ratio/
            folder_path = self.storage.create_folder_structure(
                campaign_id=brief.campaign_id,
                product_sku=asset.product_sku,
                aspect_ratio=asset.aspect_ratio
            )
            
            # Save asset
            file_path = self.storage.save_asset(asset, folder_path)
            output_paths[f"{asset.product_sku}/{asset.aspect_ratio}"].append(file_path)
            
        # Save campaign metadata
        metadata_path = self.storage.save_metadata(brief, output_paths)
        output_paths['metadata'] = [metadata_path]
        
        return dict(output_paths)
    
    def _build_generation_prompt(self, product, brief: CampaignBrief) -> str:
        """Build optimized prompt for asset generation."""
        product_name = product.name if hasattr(product, 'name') else 'Product'
        prompt_parts = [
            f"Create a professional product image for {product_name}.",
            f"Style: {brief.creative_params.get('tone', 'modern and clean')}.",
            f"Color palette: {brief.creative_params.get('color_palette', 'brand colors')}.",
            "High quality, commercial photography style.",
            "Suitable for social media advertising."
        ]
        
        if hasattr(product, 'features') and product.features:
            prompt_parts.append(f"Highlight: {', '.join(product.features)}")
        
        return " ".join(prompt_parts)
    
    def _generate_variants(self, product, count: int, params: Dict) -> List[Asset]:
        """Generate variant assets for A/B testing."""
        variants = []
        
        # Create a minimal brief-like object for prompt building
        class MiniBrief:
            def __init__(self, creative_params):
                self.creative_params = creative_params
        
        mini_brief = MiniBrief(params)
        
        for i in range(count):
            # Modify prompt for variation
            variant_prompt = self._build_generation_prompt(product, mini_brief)
            variant_prompt += f" Variation {i+1}, different angle or composition."
            
            variant = self.genai.generate_asset(
                prompt=variant_prompt,
                style_params=params,
                retries=2
            )
            variant.is_variant = True
            variant.variant_index = i + 1
            variant.product_sku = product.sku if hasattr(product, 'sku') else 'UNKNOWN'
            variants.append(variant)
        
        return variants
    
    def _calculate_cost(self, assets: List[Asset]) -> float:
        """Calculate estimated cost of generation."""
        base_cost = 0.02  # Per image generation
        processing_cost = 0.001  # Per processing operation
        
        generation_cost = sum(1 for a in assets if a.was_generated) * base_cost
        processing_cost = len(assets) * processing_cost
        
        return round(generation_cost + processing_cost, 4)
    
    def _get_text_style(self, text_type: str) -> Dict:
        """Get text styling configuration."""
        styles = {
            'headline': {
                'font_size': 48,
                'font_weight': 'bold',
                'color': '#FFFFFF',
                'shadow': True
            },
            'cta': {
                'font_size': 24,
                'font_weight': 'medium',
                'color': '#FFFFFF',
                'background': '#FF6B35',
                'padding': 10
            }
        }
        return styles.get(text_type, styles['headline'])
    
    def _load_default_config(self) -> Dict:
        """Load default configuration."""
        return {
            'max_workers': 4,
            'storage_path': './outputs',
            'genai': {
                'primary_provider': 'openai',
                'fallback_provider': 'stability',
                'max_retries': 3
            },
            'brand_guidelines': {
                'min_compliance_score': 0.85
            }
        }
    
    def _generate_compliance_summary(self, compliance_reports: List) -> Dict:
        """Generate summary of compliance reports."""
        if not compliance_reports:
            return {}
        
        total = len(compliance_reports)
        passed = sum(1 for r in compliance_reports if r.passed)
        
        return {
            'total_checked': total,
            'passed': passed,
            'failed': total - passed,
            'pass_rate': passed / max(total, 1),
            'common_issues': self._get_common_issues(compliance_reports)
        }
    
    def _get_common_issues(self, reports: List) -> List[str]:
        """Extract common issues from compliance reports."""
        all_issues = []
        for report in reports:
            all_issues.extend(report.issues[:3])  # Top 3 issues per report
        return list(set(all_issues))[:5]  # Top 5 unique issues
    
    def get_metrics(self) -> Dict:
        """Get processing metrics."""
        return {
            'campaigns_processed': self._processing_metrics['successful'],
            'campaigns_failed': self._processing_metrics['failed'],
            'average_duration': (
                self._processing_metrics['total_duration'] / 
                max(self._processing_metrics['successful'], 1)
            ),
            'success_rate': (
                self._processing_metrics['successful'] / 
                max(sum(self._processing_metrics.values()), 1)
            )
        }
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=False)


class ValidationError(Exception):
    """Raised when brief validation fails."""
    pass


class ProcessingError(Exception):
    """Raised when campaign processing fails."""
    pass

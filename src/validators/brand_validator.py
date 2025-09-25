"""
Brand compliance validation for generated assets.
Ensures all outputs meet brand guidelines and quality standards.

Author: Rohit Gangupantulu
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from PIL import Image
import io

from src.models.campaign import Asset

logger = logging.getLogger(__name__)


@dataclass
class ComplianceResult:
    """Result of brand compliance validation."""
    is_compliant: bool
    score: float
    issues: List[str]
    suggestions: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'is_compliant': self.is_compliant,
            'score': self.score,
            'issues': self.issues,
            'suggestions': self.suggestions
        }


class BrandValidator:
    """
    Validates assets against brand guidelines.
    Checks colors, composition, quality, and brand elements.
    """
    
    def __init__(self, guidelines: Dict = None):
        """
        Initialize validator with brand guidelines.
        
        Args:
            guidelines: Brand guideline configuration
        """
        self.guidelines = guidelines or self._load_default_guidelines()
        self.min_compliance_score = self.guidelines.get('min_compliance_score', 0.85)
    
    def validate(self, asset: Asset) -> ComplianceResult:
        """
        Validate asset against brand guidelines.
        
        Args:
            asset: Asset to validate
            
        Returns:
            ComplianceResult with compliance status and details
        """
        issues = []
        suggestions = []
        scores = []
        
        # Load image
        image = self._load_image(asset)
        
        # Check color compliance
        color_score, color_issues = self._validate_colors(image)
        scores.append(color_score)
        issues.extend(color_issues)
        
        # Check composition
        composition_score, comp_issues = self._validate_composition(image)
        scores.append(composition_score)
        issues.extend(comp_issues)
        
        # Check quality
        quality_score, quality_issues = self._validate_quality(image, asset)
        scores.append(quality_score)
        issues.extend(quality_issues)
        
        # Check brand elements presence
        brand_score, brand_issues = self._validate_brand_elements(image)
        scores.append(brand_score)
        issues.extend(brand_issues)
        
        # Calculate overall score
        overall_score = sum(scores) / len(scores) if scores else 0
        
        # Generate suggestions
        if overall_score < self.min_compliance_score:
            suggestions = self._generate_suggestions(issues, overall_score)
        
        return ComplianceResult(
            is_compliant=overall_score >= self.min_compliance_score,
            score=overall_score,
            issues=issues,
            suggestions=suggestions
        )
    
    def _validate_colors(self, image: Image.Image) -> Tuple[float, List[str]]:
        """
        Validate color palette compliance.
        
        Checks if dominant colors match brand palette.
        """
        issues = []
        
        # Get dominant colors
        dominant_colors = self._extract_dominant_colors(image)
        
        # Get brand colors
        brand_colors = self.guidelines.get('colors', {})
        primary_colors = brand_colors.get('primary', [])
        secondary_colors = brand_colors.get('secondary', [])
        forbidden_colors = brand_colors.get('forbidden', [])
        
        # Check for forbidden colors
        for color in dominant_colors:
            if self._color_matches_any(color, forbidden_colors):
                issues.append(f"Forbidden color detected: RGB{color}")
        
        # Check brand color presence
        brand_color_found = False
        for color in dominant_colors[:3]:  # Check top 3 dominant colors
            if (self._color_matches_any(color, primary_colors) or 
                self._color_matches_any(color, secondary_colors)):
                brand_color_found = True
                break
        
        if not brand_color_found and primary_colors:
            issues.append("No brand colors detected in dominant palette")
        
        # Calculate score
        score = 1.0
        if issues:
            score -= 0.2 * len(issues)
        
        return max(0, score), issues
    
    def _validate_composition(self, image: Image.Image) -> Tuple[float, List[str]]:
        """
        Validate image composition and balance.
        
        Checks rule of thirds, visual balance, and spacing.
        """
        issues = []
        width, height = image.size
        
        # Check aspect ratio compliance
        aspect_ratio = width / height
        allowed_ratios = self.guidelines.get('aspect_ratios', [1.0, 0.5625, 1.7778])
        
        ratio_compliant = any(
            abs(aspect_ratio - allowed) < 0.01 
            for allowed in allowed_ratios
        )
        
        if not ratio_compliant:
            issues.append(f"Non-standard aspect ratio: {aspect_ratio:.2f}")
        
        # Check minimum dimensions
        min_width = self.guidelines.get('min_width', 1080)
        min_height = self.guidelines.get('min_height', 1080)
        
        if width < min_width or height < min_height:
            issues.append(f"Image dimensions below minimum: {width}x{height}")
        
        # Calculate score
        score = 1.0
        if issues:
            score -= 0.15 * len(issues)
        
        return max(0, score), issues
    
    def _validate_quality(self, image: Image.Image, asset: Asset) -> Tuple[float, List[str]]:
        """
        Validate image quality metrics.
        
        Checks resolution, file size, and compression artifacts.
        """
        issues = []
        
        # Check resolution
        width, height = image.size
        total_pixels = width * height
        min_pixels = self.guidelines.get('min_pixels', 1_000_000)
        
        if total_pixels < min_pixels:
            issues.append(f"Resolution too low: {total_pixels:,} pixels")
        
        # Check file size
        max_size_kb = self.guidelines.get('max_file_size_kb', 5000)
        if asset.size_bytes > max_size_kb * 1024:
            issues.append(f"File size too large: {asset.size_bytes / 1024:.0f}KB")
        
        # Check for compression artifacts (simplified check)
        if asset.format == 'jpeg':
            # Estimate quality from file size and dimensions
            bytes_per_pixel = asset.size_bytes / total_pixels
            if bytes_per_pixel < 0.5:
                issues.append("Potential over-compression detected")
        
        # Calculate score
        score = 1.0
        if issues:
            score -= 0.2 * len(issues)
        
        return max(0, score), issues
    
    def _validate_brand_elements(self, image: Image.Image) -> Tuple[float, List[str]]:
        """
        Check for required brand elements.
        
        This is a simplified check - in production would use ML models.
        """
        issues = []
        
        # Check for minimum contrast (for text readability)
        contrast_ratio = self._calculate_contrast_ratio(image)
        min_contrast = self.guidelines.get('min_contrast_ratio', 4.5)
        
        if contrast_ratio < min_contrast:
            issues.append(f"Insufficient contrast for text: {contrast_ratio:.1f}")
        
        # Calculate score
        score = 1.0 if not issues else 0.8
        
        return score, issues
    
    def auto_correct(self, asset: Asset, result: ComplianceResult) -> Optional[Asset]:
        """
        Attempt to auto-correct compliance issues.
        
        Args:
            asset: Non-compliant asset
            result: Validation result with issues
            
        Returns:
            Corrected asset if possible, None otherwise
        """
        if result.is_compliant:
            return asset
        
        image = self._load_image(asset)
        corrected = False
        
        # Auto-corrections based on issues
        for issue in result.issues:
            if "contrast" in issue.lower():
                image = self._enhance_contrast(image)
                corrected = True
            elif "dimension" in issue.lower():
                # Cannot auto-correct dimension issues
                pass
            elif "color" in issue.lower():
                # Apply color correction
                image = self._apply_color_correction(image)
                corrected = True
        
        if not corrected:
            return None
        
        # Convert back to asset
        output = io.BytesIO()
        image.save(output, format=asset.format.upper(), quality=85)
        
        new_asset = Asset(
            product_sku=asset.product_sku,
            aspect_ratio=asset.aspect_ratio,
            image_data=output.getvalue(),
            dimensions=image.size,
            format=asset.format,
            size_bytes=len(output.getvalue()),
            was_generated=asset.was_generated,
            metadata={**asset.metadata, 'auto_corrected': True}
        )
        
        # Re-validate
        new_result = self.validate(new_asset)
        
        return new_asset if new_result.is_compliant else None
    
    def _extract_dominant_colors(self, image: Image.Image, n_colors: int = 5) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from image."""
        # Resize for faster processing
        thumb = image.copy()
        thumb.thumbnail((150, 150))
        
        # Convert to RGB if needed
        if thumb.mode != 'RGB':
            thumb = thumb.convert('RGB')
        
        # Get colors
        colors = thumb.getcolors(maxcolors=10000)
        if not colors:
            return []
        
        # Sort by frequency
        colors.sort(key=lambda x: x[0], reverse=True)
        
        # Return top N colors
        return [color[1] for color in colors[:n_colors]]
    
    def _color_matches_any(self, color: Tuple[int, int, int], color_list: List) -> bool:
        """Check if color matches any in the list (with tolerance)."""
        if not color_list:
            return False
        
        tolerance = 30  # RGB tolerance
        
        for brand_color in color_list:
            if isinstance(brand_color, str):
                # Convert hex to RGB
                if brand_color.startswith('#'):
                    brand_color = brand_color[1:]
                r = int(brand_color[0:2], 16)
                g = int(brand_color[2:4], 16)
                b = int(brand_color[4:6], 16)
                brand_color = (r, g, b)
            
            # Check if colors match within tolerance
            if all(abs(c1 - c2) < tolerance for c1, c2 in zip(color, brand_color)):
                return True
        
        return False
    
    def _calculate_contrast_ratio(self, image: Image.Image) -> float:
        """Calculate contrast ratio for text readability."""
        # Convert to grayscale
        gray = image.convert('L')
        
        # Get pixel values
        pixels = np.array(gray)
        
        # Calculate contrast using standard deviation
        contrast = np.std(pixels)
        
        # Normalize to approximate WCAG contrast ratio
        return contrast / 20
    
    def _enhance_contrast(self, image: Image.Image) -> Image.Image:
        """Enhance image contrast."""
        from PIL import ImageEnhance
        
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(1.2)
    
    def _apply_color_correction(self, image: Image.Image) -> Image.Image:
        """Apply brand color correction."""
        # This is a placeholder - in production would use color grading
        from PIL import ImageEnhance
        
        # Slight saturation boost
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(1.1)
    
    def _generate_suggestions(self, issues: List[str], score: float) -> List[str]:
        """Generate improvement suggestions based on issues."""
        suggestions = []
        
        for issue in issues:
            if "contrast" in issue.lower():
                suggestions.append("Increase contrast between text and background")
            elif "dimension" in issue.lower():
                suggestions.append("Regenerate image at higher resolution")
            elif "color" in issue.lower():
                suggestions.append("Adjust color palette to match brand guidelines")
            elif "compression" in issue.lower():
                suggestions.append("Re-export with higher quality settings")
        
        if score < 0.5:
            suggestions.append("Consider regenerating the asset with updated parameters")
        
        return suggestions
    
    def _load_image(self, asset: Asset) -> Image.Image:
        """Load image from asset."""
        if asset.image_data:
            return Image.open(io.BytesIO(asset.image_data))
        elif asset.file_path:
            return Image.open(asset.file_path)
        else:
            raise ValueError("Asset has no image data")
    
    def _load_default_guidelines(self) -> Dict:
        """Load default brand guidelines."""
        return {
            'min_compliance_score': 0.85,
            'colors': {
                'primary': ['#1E88E5', '#FFC107'],
                'secondary': ['#424242', '#FFFFFF'],
                'forbidden': ['#FF0000']
            },
            'aspect_ratios': [1.0, 0.5625, 1.7778],  # 1:1, 9:16, 16:9
            'min_width': 1080,
            'min_height': 1080,
            'min_pixels': 1_000_000,
            'max_file_size_kb': 5000,
            'min_contrast_ratio': 4.5
        }

"""
Image processing pipeline for asset manipulation.
Handles resizing, cropping, text overlay, and optimization.

Author: Rohit Gangupantulu
"""

import io
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
import cv2

from src.models.campaign import Asset, AspectRatio

logger = logging.getLogger(__name__)


@dataclass
class TextPlacement:
    """Optimal text placement positions."""
    headline_position: Tuple[int, int]
    cta_position: Tuple[int, int]
    safe_zones: List[Tuple[int, int, int, int]]  # (x1, y1, x2, y2) rectangles


class ImageProcessor:
    """
    Advanced image processing with intelligent cropping and text overlay.
    """
    
    def __init__(self):
        self.font_cache = {}
        self._load_default_fonts()
    
    def process(
        self, 
        asset: Asset, 
        target_ratio: str,
        optimization_params: Dict = None
    ) -> Asset:
        """
        Process asset for specific aspect ratio with optimization.
        
        Args:
            asset: Input asset to process
            target_ratio: Target aspect ratio (e.g., "1:1", "9:16")
            optimization_params: Processing parameters
            
        Returns:
            Processed Asset
        """
        params = optimization_params or {}
        
        # Load image from asset
        image = self._load_image(asset)
        
        # Calculate target dimensions
        target_aspect = AspectRatio.from_string(target_ratio)
        target_dims = target_aspect.get_dimensions(params.get('base_width', 1080))
        
        # Apply smart cropping if needed
        if params.get('smart_crop', True):
            image = self._smart_crop(image, target_dims)
        else:
            image = self._simple_resize(image, target_dims)
        
        # Apply image enhancements
        if params.get('enhance', True):
            image = self._enhance_image(image)
        
        # Convert to specified format and quality
        output_bytes = self._save_image(
            image,
            format=params.get('format', 'jpeg'),
            quality=params.get('quality', 85)
        )
        
        # Create new asset with processed image
        processed_asset = Asset(
            product_sku=asset.product_sku,
            aspect_ratio=target_ratio,
            image_data=output_bytes,
            dimensions=target_dims,
            format=params.get('format', 'jpeg'),
            size_bytes=len(output_bytes),
            was_generated=asset.was_generated,
            is_variant=asset.is_variant,
            variant_index=asset.variant_index,
            metadata={
                **asset.metadata,
                'processed': True,
                'original_dimensions': asset.dimensions
            }
        )
        
        return processed_asset
    
    def _smart_crop(self, image: Image.Image, target_dims: Tuple[int, int]) -> Image.Image:
        """
        Apply intelligent cropping using saliency detection.
        
        Uses computer vision to identify the most important region of the image.
        """
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Calculate saliency map
        saliency = self._calculate_saliency(cv_image)
        
        # Find optimal crop region
        crop_box = self._find_optimal_crop(
            image.size,
            target_dims,
            saliency
        )
        
        # Crop and resize
        cropped = image.crop(crop_box)
        resized = cropped.resize(target_dims, Image.Resampling.LANCZOS)
        
        return resized
    
    def _calculate_saliency(self, image: np.ndarray) -> np.ndarray:
        """
        Calculate saliency map using edge detection and contrast.
        
        This identifies the most visually important regions of an image.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use edge detection as primary saliency indicator
        # Combine multiple edge detection methods for robustness
        edges_sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        edges_sobel = np.abs(edges_sobel)
        
        edges_canny = cv2.Canny(gray, 50, 150)
        
        # Combine edge maps
        saliency_map = (edges_sobel / edges_sobel.max() * 127 + edges_canny / 2).astype(np.uint8)
        
        # Add contrast-based saliency
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        contrast = cv2.absdiff(gray, blur)
        
        # Combine with edge saliency
        saliency_map = cv2.addWeighted(saliency_map, 0.7, contrast, 0.3, 0)
        
        # Apply Gaussian blur for smoothing
        saliency_map = cv2.GaussianBlur(saliency_map, (11, 11), 0)
        
        return saliency_map
    
    def _find_optimal_crop(
        self,
        original_size: Tuple[int, int],
        target_size: Tuple[int, int],
        saliency_map: np.ndarray
    ) -> Tuple[int, int, int, int]:
        """
        Find optimal crop region based on saliency and aspect ratio.
        
        Returns (left, top, right, bottom) crop coordinates.
        """
        orig_w, orig_h = original_size
        target_w, target_h = target_size
        target_aspect = target_w / target_h
        
        # Determine crop dimensions maintaining aspect ratio
        if orig_w / orig_h > target_aspect:
            # Image is wider - crop width
            crop_h = orig_h
            crop_w = int(orig_h * target_aspect)
        else:
            # Image is taller - crop height
            crop_w = orig_w
            crop_h = int(orig_w / target_aspect)
        
        # Ensure crop doesn't exceed original dimensions
        crop_w = min(crop_w, orig_w)
        crop_h = min(crop_h, orig_h)
        
        # Find region with highest saliency
        max_saliency = 0
        best_crop = None
        
        # Sample different crop positions
        step_size = max(1, min(orig_w - crop_w, orig_h - crop_h) // 10)
        
        for x in range(0, orig_w - crop_w + 1, max(1, step_size)):
            for y in range(0, orig_h - crop_h + 1, max(1, step_size)):
                # Calculate average saliency in this region
                region = saliency_map[y:y+crop_h, x:x+crop_w]
                avg_saliency = np.mean(region)
                
                if avg_saliency > max_saliency:
                    max_saliency = avg_saliency
                    best_crop = (x, y, x + crop_w, y + crop_h)
        
        # Default to center crop if no good region found
        if best_crop is None:
            x = (orig_w - crop_w) // 2
            y = (orig_h - crop_h) // 2
            best_crop = (x, y, x + crop_w, y + crop_h)
        
        return best_crop
    
    def _simple_resize(self, image: Image.Image, target_dims: Tuple[int, int]) -> Image.Image:
        """Simple resize with aspect ratio preservation."""
        # Calculate scale to fit within target dimensions
        scale = min(target_dims[0] / image.width, target_dims[1] / image.height)
        
        # Resize maintaining aspect ratio
        new_size = (int(image.width * scale), int(image.height * scale))
        resized = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Create canvas of target size
        canvas = Image.new('RGB', target_dims, color='white')
        
        # Paste resized image centered
        x = (target_dims[0] - new_size[0]) // 2
        y = (target_dims[1] - new_size[1]) // 2
        canvas.paste(resized, (x, y))
        
        return canvas
    
    def find_text_placement(self, asset: Asset) -> TextPlacement:
        """
        Find optimal positions for text overlay using visual analysis.
        
        Identifies areas with low complexity for readable text placement.
        """
        image = self._load_image(asset)
        
        # Convert to numpy array for analysis
        img_array = np.array(image)
        
        # Calculate complexity map (edges and variance)
        complexity = self._calculate_complexity_map(img_array)
        
        # Find low-complexity regions for text
        safe_zones = self._find_safe_zones(complexity)
        
        # Determine headline and CTA positions
        headline_pos = self._find_text_position(
            complexity,
            position_preference='top',
            min_area=(image.width // 2, 60)
        )
        
        cta_pos = self._find_text_position(
            complexity,
            position_preference='bottom',
            min_area=(200, 50)
        )
        
        return TextPlacement(
            headline_position=headline_pos,
            cta_position=cta_pos,
            safe_zones=safe_zones
        )
    
    def _calculate_complexity_map(self, image: np.ndarray) -> np.ndarray:
        """Calculate visual complexity map for text placement."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Calculate edge density
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate local variance
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        local_mean = cv2.filter2D(gray.astype(float), -1, kernel)
        local_sq_mean = cv2.filter2D(gray.astype(float) ** 2, -1, kernel)
        local_variance = local_sq_mean - local_mean ** 2
        
        # Combine edge density and variance
        complexity = (edges / 255.0) * 0.5 + (local_variance / local_variance.max()) * 0.5
        
        # Apply Gaussian blur for smoothing
        complexity = cv2.GaussianBlur(complexity, (21, 21), 0)
        
        return complexity
    
    def _find_safe_zones(self, complexity: np.ndarray, threshold: float = 0.3) -> List[Tuple]:
        """Find regions with low complexity suitable for text."""
        # Threshold complexity map
        safe_mask = complexity < threshold
        
        # Find contours of safe regions
        contours, _ = cv2.findContours(
            safe_mask.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Convert contours to bounding boxes
        safe_zones = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter out very small regions
            if w > 100 and h > 40:
                safe_zones.append((x, y, x + w, y + h))
        
        return safe_zones
    
    def _find_text_position(
        self,
        complexity: np.ndarray,
        position_preference: str = 'top',
        min_area: Tuple[int, int] = (200, 50)
    ) -> Tuple[int, int]:
        """Find optimal position for text element."""
        h, w = complexity.shape
        min_w, min_h = min_area
        
        # Define search regions based on preference
        if position_preference == 'top':
            search_region = complexity[:h//3, :]
            y_offset = 0
        elif position_preference == 'bottom':
            search_region = complexity[2*h//3:, :]
            y_offset = 2 * h // 3
        else:  # center
            search_region = complexity[h//3:2*h//3, :]
            y_offset = h // 3
        
        # Find region with lowest average complexity
        best_score = float('inf')
        best_pos = (w // 2 - min_w // 2, y_offset + 20)
        
        # Slide window to find best position
        for y in range(0, search_region.shape[0] - min_h, 10):
            for x in range(0, search_region.shape[1] - min_w, 20):
                region = search_region[y:y+min_h, x:x+min_w]
                score = np.mean(region)
                
                if score < best_score:
                    best_score = score
                    best_pos = (x, y_offset + y)
        
        return best_pos
    
    def add_text(
        self,
        asset: Asset,
        text: str,
        position: Tuple[int, int],
        style: Dict
    ) -> Asset:
        """
        Add text overlay to asset with specified styling.
        
        Args:
            asset: Input asset
            text: Text to overlay
            position: (x, y) position for text
            style: Styling parameters (font_size, color, etc.)
            
        Returns:
            Asset with text overlay
        """
        image = self._load_image(asset)
        draw = ImageDraw.Draw(image)
        
        # Get font
        font = self._get_font(style.get('font_size', 24))
        
        # Add background if specified
        if style.get('background'):
            # Calculate text bounding box
            bbox = draw.textbbox(position, text, font=font)
            padding = style.get('padding', 10)
            bg_bbox = (
                bbox[0] - padding,
                bbox[1] - padding,
                bbox[2] + padding,
                bbox[3] + padding
            )
            
            # Draw background
            draw.rounded_rectangle(
                bg_bbox,
                radius=5,
                fill=style.get('background', '#000000')
            )
        
        # Add shadow if specified
        if style.get('shadow'):
            shadow_offset = 2
            shadow_color = '#00000080'  # Semi-transparent black
            draw.text(
                (position[0] + shadow_offset, position[1] + shadow_offset),
                text,
                font=font,
                fill=shadow_color
            )
        
        # Draw main text
        draw.text(
            position,
            text,
            font=font,
            fill=style.get('color', '#FFFFFF')
        )
        
        # Convert back to bytes
        output_bytes = self._save_image(image, format=asset.format)
        
        # Create new asset with text overlay
        new_asset = Asset(
            product_sku=asset.product_sku,
            aspect_ratio=asset.aspect_ratio,
            image_data=output_bytes,
            dimensions=asset.dimensions,
            format=asset.format,
            size_bytes=len(output_bytes),
            was_generated=asset.was_generated,
            is_variant=asset.is_variant,
            variant_index=asset.variant_index,
            metadata={
                **asset.metadata,
                'has_text_overlay': True,
                'overlay_text': text
            }
        )
        
        return new_asset
    
    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Apply image enhancements for better quality."""
        # Enhance sharpness slightly
        enhancer = ImageOps.autocontrast(image, cutoff=1)
        
        # Apply subtle sharpening
        enhanced = enhancer.filter(ImageFilter.UnsharpMask(radius=1, percent=50))
        
        return enhanced
    
    def _load_image(self, asset: Asset) -> Image.Image:
        """Load image from asset."""
        if asset.image_data:
            return Image.open(io.BytesIO(asset.image_data))
        elif asset.file_path:
            return Image.open(asset.file_path)
        else:
            raise ValueError("Asset has no image data or file path")
    
    def _save_image(self, image: Image.Image, format: str = 'jpeg', quality: int = 85) -> bytes:
        """Save image to bytes."""
        output = io.BytesIO()
        
        # Convert RGBA to RGB if saving as JPEG
        if format.lower() == 'jpeg' and image.mode == 'RGBA':
            # Create white background
            background = Image.new('RGB', image.size, 'white')
            background.paste(image, mask=image.split()[3])
            image = background
        
        image.save(output, format=format.upper(), quality=quality, optimize=True)
        return output.getvalue()
    
    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Get font from cache or load default."""
        if size not in self.font_cache:
            try:
                # Try to load a good system font
                font_paths = [
                    "arial.ttf",
                    "Arial.ttf",
                    "helvetica.ttf",
                    "DejaVuSans.ttf",
                    "/System/Library/Fonts/Helvetica.ttc",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    "C:\\Windows\\Fonts\\arial.ttf"
                ]
                
                font_loaded = False
                for font_path in font_paths:
                    try:
                        self.font_cache[size] = ImageFont.truetype(font_path, size)
                        font_loaded = True
                        break
                    except:
                        continue
                
                if not font_loaded:
                    # Fall back to default font
                    self.font_cache[size] = ImageFont.load_default()
                    
            except Exception as e:
                logger.warning(f"Failed to load font: {e}")
                self.font_cache[size] = ImageFont.load_default()
        
        return self.font_cache[size]
    
    def _load_default_fonts(self):
        """Pre-load commonly used font sizes."""
        common_sizes = [16, 20, 24, 32, 48, 64]
        for size in common_sizes:
            self._get_font(size)

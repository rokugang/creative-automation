"""
Storage management for assets and campaign outputs.
Handles file organization, caching, and retrieval.

Author: Rohit Gangupantulu
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from src.models.campaign import Asset, CampaignBrief

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages storage of campaign assets and metadata."""
    
    def __init__(self, base_path: str = "./outputs"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.campaigns_dir = self.base_path / "campaigns"
        self.assets_dir = self.base_path / "assets"
        self.metadata_dir = self.base_path / "metadata"
        
        for directory in [self.campaigns_dir, self.assets_dir, self.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def create_folder_structure(
        self,
        campaign_id: str,
        product_sku: str,
        aspect_ratio: str
    ) -> Path:
        """
        Create organized folder structure for outputs.
        
        Structure:
        campaigns/
        └── {campaign_id}/
            └── {product_sku}/
                └── {aspect_ratio}/
        """
        # Clean aspect ratio for folder name
        ratio_clean = aspect_ratio.replace(":", "x")
        
        folder_path = self.campaigns_dir / campaign_id / product_sku / ratio_clean
        folder_path.mkdir(parents=True, exist_ok=True)
        
        return folder_path
    
    def save_asset(self, asset: Asset, folder_path: Path) -> str:
        """
        Save asset to specified folder.
        
        Args:
            asset: Asset to save
            folder_path: Destination folder
            
        Returns:
            Path to saved file
        """
        # Generate filename
        filename = asset.get_filename()
        file_path = folder_path / filename
        
        # Save image data
        if asset.image_data:
            with open(file_path, 'wb') as f:
                f.write(asset.image_data)
        elif asset.file_path:
            # Copy existing file
            shutil.copy2(asset.file_path, file_path)
        else:
            raise ValueError("Asset has no data to save")
        
        # Save metadata
        metadata_file = file_path.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(asset.to_dict(), f, indent=2)
        
        logger.debug(f"Saved asset to: {file_path}")
        
        return str(file_path)
    
    def save_batch(self, assets: List[Asset], campaign_id: str) -> Dict:
        """
        Save multiple assets in batch.
        
        Args:
            assets: List of assets to save
            campaign_id: Campaign ID for organization
            
        Returns:
            Dictionary mapping asset IDs to file paths
        """
        paths = {}
        
        for asset in assets:
            folder_path = self.create_folder_structure(
                campaign_id,
                asset.product_sku,
                asset.aspect_ratio
            )
            
            file_path = self.save_asset(asset, folder_path)
            paths[asset.asset_id] = file_path
        
        return paths
    
    def save_metadata(self, brief: CampaignBrief, output_paths: Dict) -> str:
        """
        Save campaign metadata and processing results.
        
        Args:
            brief: Campaign brief
            output_paths: Generated asset paths
            
        Returns:
            Path to metadata file
        """
        metadata = {
            'campaign': brief.to_dict(),
            'processing': {
                'timestamp': datetime.now().isoformat(),
                'total_assets': sum(len(paths) for paths in output_paths.values()),
                'output_structure': output_paths
            }
        }
        
        metadata_file = self.metadata_dir / f"{brief.campaign_id}_metadata.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved campaign metadata to: {metadata_file}")
        
        return str(metadata_file)
    
    def load_asset(self, file_path: str) -> Asset:
        """Load asset from file."""
        path = Path(file_path)
        
        # Load metadata
        metadata_file = path.with_suffix('.json')
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                asset_data = json.load(f)
                asset = Asset(**asset_data)
        else:
            # Create basic asset from file
            asset = Asset(
                file_path=str(path),
                format=path.suffix[1:] if path.suffix else 'unknown'
            )
        
        # Load image data
        if path.exists():
            with open(path, 'rb') as f:
                asset.image_data = f.read()
                asset.size_bytes = len(asset.image_data)
        
        return asset
    
    def get_campaign_assets(self, campaign_id: str) -> List[Asset]:
        """Get all assets for a campaign."""
        campaign_dir = self.campaigns_dir / campaign_id
        
        if not campaign_dir.exists():
            return []
        
        assets = []
        
        # Find all image files
        for file_path in campaign_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                try:
                    asset = self.load_asset(str(file_path))
                    assets.append(asset)
                except Exception as e:
                    logger.warning(f"Failed to load asset {file_path}: {e}")
        
        return assets
    
    def get_by_campaign(self, campaign_id: str) -> List[str]:
        """Get all asset paths for a campaign."""
        campaign_dir = self.campaigns_dir / campaign_id
        
        if not campaign_dir.exists():
            return []
        
        paths = []
        for file_path in campaign_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                paths.append(str(file_path))
        
        return paths
    
    def clean_temp_files(self):
        """Clean temporary files older than 24 hours."""
        import time
        
        current_time = time.time()
        temp_dir = self.base_path / "temp"
        
        if temp_dir.exists():
            for file_path in temp_dir.iterdir():
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > 86400:  # 24 hours
                        file_path.unlink()
                        logger.debug(f"Cleaned temp file: {file_path}")
    
    def get_storage_stats(self) -> Dict:
        """Get storage usage statistics."""
        total_size = 0
        file_count = 0
        campaign_count = 0
        
        if self.campaigns_dir.exists():
            campaign_count = len(list(self.campaigns_dir.iterdir()))
            
            for file_path in self.campaigns_dir.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
        
        return {
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'total_files': file_count,
            'total_campaigns': campaign_count,
            'base_path': str(self.base_path)
        }

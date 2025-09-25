"""
Configuration management for the platform.
Handles environment variables and default settings.

Author: Rohit Gangupantulu
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """Application configuration with environment variable support."""
    
    def __init__(self):
        # Load .env file if it exists
        self._load_env_file()
        
        self.env = os.getenv('ENVIRONMENT', 'development')
        
        # API Keys
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.stability_api_key = os.getenv('STABILITY_API_KEY')
        
        # Paths
        self.storage_path = os.getenv('STORAGE_PATH', './outputs')
        self.temp_path = os.getenv('TEMP_PATH', './temp')
        
        # Processing settings
        self.max_workers = int(os.getenv('MAX_WORKERS', '4'))
        self.max_retries = int(os.getenv('MAX_RETRIES', '3'))
        self.request_timeout = float(os.getenv('REQUEST_TIMEOUT', '30.0'))
        
        # Quality settings
        self.image_quality = int(os.getenv('IMAGE_QUALITY', '85'))
        self.default_width = int(os.getenv('DEFAULT_WIDTH', '1080'))
        
        # Brand guidelines path
        self.brand_guidelines_path = os.getenv(
            'BRAND_GUIDELINES_PATH',
            './config/brand_guidelines.json'
        )
        
        # Feature flags
        self.enable_caching = os.getenv('ENABLE_CACHING', 'true').lower() == 'true'
        self.enable_monitoring = os.getenv('ENABLE_MONITORING', 'true').lower() == 'true'
        self.enable_smart_crop = os.getenv('ENABLE_SMART_CROP', 'true').lower() == 'true'
        
        # Server settings
        self.api_host = os.getenv('API_HOST', '0.0.0.0')
        self.api_port = int(os.getenv('API_PORT', '8000'))
        
        # Validate configuration
        self._validate()
        
        # Create necessary directories
        self._setup_directories()
    
    def _validate(self):
        """Validate configuration settings."""
        if not self.openai_api_key and not self.stability_api_key:
            raise ValueError("GenAI API key required. Set OPENAI_API_KEY or STABILITY_API_KEY in .env")
        
        if self.max_workers < 1:
            raise ValueError("MAX_WORKERS must be at least 1")
        
        if self.image_quality < 1 or self.image_quality > 100:
            raise ValueError("IMAGE_QUALITY must be between 1 and 100")
    
    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.storage_path,
            self.temp_path,
            os.path.join(self.storage_path, 'campaigns'),
            os.path.join(self.storage_path, 'assets'),
            os.path.join(self.storage_path, 'logs')
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'env': self.env,
            'storage_path': self.storage_path,
            'max_workers': self.max_workers,
            'max_retries': self.max_retries,
            'genai': {
                'openai_key': self.openai_api_key,
                'stability_key': self.stability_api_key,
                'primary_provider': 'openai' if self.openai_api_key else 'stability',
                'fallback_provider': 'stability' if self.stability_api_key else 'openai'
            },
            'processing': {
                'image_quality': self.image_quality,
                'default_width': self.default_width,
                'enable_smart_crop': self.enable_smart_crop
            },
            'brand_guidelines': {
                'path': self.brand_guidelines_path,
                'min_compliance_score': 0.85
            }
        }
    
    @classmethod
    def from_file(cls, config_file: str) -> 'Config':
        """Load configuration from file."""
        import json
        
        config = cls()
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                
            # Override with file values
            for key, value in file_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        return config
    
    def _load_env_file(self):
        """Load environment variables from .env file if it exists."""
        env_path = Path('.env')
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()

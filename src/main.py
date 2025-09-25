"""
Main entry point for the Creative Automation Platform.
Provides both CLI and API interfaces for campaign processing.

Author: Rohit Gangupantulu
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional

# Fix import path - add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.campaign_processor import CampaignProcessor
from src.models.campaign import CampaignBrief
from src.api.server import create_app
from src.utils.config import Config
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def process_campaign_file(file_path: str, output_dir: str = None) -> Dict:
    """
    Process a campaign brief from a JSON/YAML file.
    
    Args:
        file_path: Path to campaign brief file
        output_dir: Optional output directory
        
    Returns:
        Processing result dictionary
    """
    # Load campaign brief
    with open(file_path, 'r') as f:
        if file_path.endswith('.yaml') or file_path.endswith('.yml'):
            import yaml
            brief_data = yaml.safe_load(f)
        else:
            brief_data = json.load(f)
    
    # Create brief object
    brief = CampaignBrief.from_dict(brief_data)
    
    # Initialize processor
    config = Config()
    if output_dir:
        config.storage_path = output_dir
    
    processor = CampaignProcessor(config.to_dict())
    
    # Process campaign
    logger.info(f"Processing campaign: {brief.campaign_id}")
    result = processor.process(brief)
    
    # Save results
    results_file = Path(output_dir or './outputs') / f"{brief.campaign_id}_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")
    
    return result.to_dict()


def run_demo():
    """Run demo campaign generation."""
    setup_logging(debug=False)
    logger.info("Starting campaign generation...")
    
    # Create sample campaign brief
    sample_brief = {
        "campaign_id": "DEMO-001",
        "products": [
            {
                "sku": "PROD-DEMO",
                "name": "Premium Wireless Headphones",
                "description": "High-quality wireless headphones with noise cancellation",
                "features": ["Noise Cancellation", "30-hour battery", "Premium Sound"],
                "variants_needed": 2
            }
        ],
        "target_markets": [
            {
                "region": "US",
                "language": "en"
            }
        ],
        "aspect_ratios": ["1:1", "9:16", "16:9"],
        "messaging": {
            "headline": "Experience Pure Sound",
            "body": "Immerse yourself in crystal-clear audio",
            "cta": "Shop Now"
        },
        "creative_params": {
            "tone": "modern and premium",
            "color_palette": "dark and sophisticated",
            "artistic_style": "photographic"
        }
    }
    
    # Save sample brief
    demo_file = Path("./examples/demo_campaign.json")
    demo_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(demo_file, 'w') as f:
        json.dump(sample_brief, f, indent=2)
    
    logger.info(f"Created demo campaign brief: {demo_file}")
    
    # Process the demo campaign
    result = process_campaign_file(str(demo_file), "./outputs/demo")
    
    # Print summary
    print("\n" + "="*60)
    print("DEMO CAMPAIGN COMPLETED")
    print("="*60)
    print(f"Campaign ID: {result['campaign_id']}")
    print(f"Total Assets Generated: {result['summary']['total_assets']}")
    print(f"Processing Time: {result['summary']['processing_time']}s")
    print(f"Output Directory: ./outputs/demo/{result['campaign_id']}")
    print("="*60 + "\n")
    
    return result


def run_api_server(host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
    """
    Start the API server.
    
    Args:
        host: Host to bind to
        port: Port to listen on
        debug: Enable debug mode
    """
    app = create_app()
    
    logger.info(f"Starting API server on {host}:{port}")
    
    # Use production server if available
    try:
        import uvicorn
        uvicorn.run(app, host=host, port=port, log_level="info" if debug else "warning")
    except ImportError:
        # Fall back to Flask development server
        from api.flask_server import create_flask_app
        flask_app = create_flask_app()
        flask_app.run(host=host, port=port, debug=debug)


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Creative Automation Platform - Generate social media campaigns at scale"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process a campaign brief')
    process_parser.add_argument(
        'brief',
        help='Path to campaign brief file (JSON or YAML)'
    )
    process_parser.add_argument(
        '--output',
        '-o',
        help='Output directory for generated assets',
        default='./outputs'
    )
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run interactive demo')
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Start API server')
    server_parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host to bind to'
    )
    server_parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to listen on'
    )
    server_parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(debug=args.command == 'server' and args.debug if hasattr(args, 'debug') else False)
    
    # Execute command
    if args.command == 'process':
        try:
            result = process_campaign_file(args.brief, args.output)
            print(f"Successfully processed campaign: {result['campaign_id']}")
            print(f"Generated {result['summary']['total_assets']} assets")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Failed to process campaign: {e}")
            sys.exit(1)
    
    elif args.command == 'demo':
        try:
            run_demo()
            sys.exit(0)
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            sys.exit(1)
    
    elif args.command == 'server':
        try:
            run_api_server(args.host, args.port, args.debug)
        except KeyboardInterrupt:
            logger.info("Campaign processing completed successfully")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Server failed: {e}")
            sys.exit(1)
    
    else:
        # No command specified, show help
        parser.print_help()
        
        # Run demo by default
        print("\n" + "="*60)
        print("No command specified. Running demo...")
        print("="*60 + "\n")
        
        try:
            run_demo()
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()

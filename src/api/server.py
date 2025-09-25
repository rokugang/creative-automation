"""
FastAPI server for the Creative Automation Platform.
Provides REST endpoints for campaign processing and monitoring.

Author: Rohit Gangupantulu
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import asyncio
import json
from pathlib import Path
import uuid

from src.models.campaign import CampaignBrief, CampaignStatus
from src.core.campaign_processor import CampaignProcessor
from src.agents.campaign_monitor import CampaignMonitor
from src.utils.config import Config

# Initialize FastAPI app
app = FastAPI(
    title="Creative Automation Platform API",
    description="Generate and manage social media campaigns at scale",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
config = Config()
processor = CampaignProcessor(config.to_dict())
monitor = CampaignMonitor(config.to_dict())
active_campaigns = {}


class CampaignRequest(BaseModel):
    """Request model for campaign creation."""
    campaign_id: Optional[str] = None
    products: List[Dict]
    target_markets: List[Dict]
    aspect_ratios: List[str]
    messaging: Dict[str, str]
    creative_params: Optional[Dict] = {}
    performance_targets: Optional[Dict] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "products": [
                    {
                        "sku": "PROD-001",
                        "name": "Wireless Headphones",
                        "variants_needed": 2
                    }
                ],
                "target_markets": [
                    {"region": "US", "language": "en"}
                ],
                "aspect_ratios": ["1:1", "9:16"],
                "messaging": {
                    "headline": "Premium Sound",
                    "cta": "Shop Now"
                },
                "creative_params": {
                    "tone": "modern",
                    "color_palette": "dark"
                }
            }
        }


class CampaignResponse(BaseModel):
    """Response model for campaign creation."""
    campaign_id: str
    status: str
    message: str
    estimated_completion_time: int
    assets_to_generate: int


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    # Start monitoring agent in background
    asyncio.create_task(monitor.start())


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    await monitor.stop()


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "Creative Automation Platform API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "campaigns": "/api/campaigns",
            "health": "/health",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "processor": "ready",
        "monitor": "running" if monitor.running else "stopped",
        "active_campaigns": len(active_campaigns)
    }


@app.post("/api/campaigns", response_model=CampaignResponse)
async def create_campaign(
    request: CampaignRequest,
    background_tasks: BackgroundTasks
):
    """
    Create and process a new campaign.
    
    Args:
        request: Campaign creation request
        background_tasks: FastAPI background tasks
        
    Returns:
        Campaign response with ID and status
    """
    try:
        # Create campaign brief
        brief_data = request.dict()
        if not brief_data.get('campaign_id'):
            brief_data['campaign_id'] = f"CAMP-{uuid.uuid4().hex[:8].upper()}"
        
        brief = CampaignBrief.from_dict(brief_data)
        
        # Calculate estimates
        assets_to_generate = brief.get_total_assets_needed()
        estimated_time = assets_to_generate * 2  # 2 seconds per asset estimate
        
        # Create status tracking
        status = CampaignStatus(
            campaign_id=brief.campaign_id,
            status="pending",
            progress=0
        )
        active_campaigns[brief.campaign_id] = status
        
        # Add to processing queue
        background_tasks.add_task(process_campaign_background, brief)
        
        return CampaignResponse(
            campaign_id=brief.campaign_id,
            status="accepted",
            message="Campaign queued for processing",
            estimated_completion_time=estimated_time,
            assets_to_generate=assets_to_generate
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/campaigns/{campaign_id}")
async def get_campaign_status(campaign_id: str):
    """
    Get status of a campaign.
    
    Args:
        campaign_id: Campaign identifier
        
    Returns:
        Campaign status information
    """
    if campaign_id in active_campaigns:
        status = active_campaigns[campaign_id]
        return status.to_dict()
    else:
        # Check if completed
        for completed_id, result in monitor.completed_campaigns:
            if completed_id == campaign_id:
                return {
                    "campaign_id": campaign_id,
                    "status": "completed",
                    "progress": 100,
                    "result": result.get_summary() if hasattr(result, 'get_summary') else {}
                }
        
        # Check if failed
        for failed_id, error in monitor.failed_campaigns:
            if failed_id == campaign_id:
                return {
                    "campaign_id": campaign_id,
                    "status": "failed",
                    "error": str(error)
                }
        
        raise HTTPException(status_code=404, detail="Campaign not found")


@app.get("/api/campaigns/{campaign_id}/assets")
async def get_campaign_assets(campaign_id: str):
    """
    Get generated assets for a campaign.
    
    Args:
        campaign_id: Campaign identifier
        
    Returns:
        List of asset information
    """
    from ..utils.storage import StorageManager
    
    storage = StorageManager(config.storage_path)
    assets = storage.get_campaign_assets(campaign_id)
    
    if not assets:
        raise HTTPException(status_code=404, detail="No assets found for campaign")
    
    return {
        "campaign_id": campaign_id,
        "total_assets": len(assets),
        "assets": [asset.to_dict() for asset in assets]
    }


@app.get("/api/campaigns/{campaign_id}/download/{asset_id}")
async def download_asset(campaign_id: str, asset_id: str):
    """
    Download a specific asset.
    
    Args:
        campaign_id: Campaign identifier
        asset_id: Asset identifier
        
    Returns:
        Asset file
    """
    from ..utils.storage import StorageManager
    
    storage = StorageManager(config.storage_path)
    assets = storage.get_campaign_assets(campaign_id)
    
    for asset in assets:
        if asset.asset_id == asset_id:
            if asset.file_path and Path(asset.file_path).exists():
                return FileResponse(
                    asset.file_path,
                    media_type=f"image/{asset.format}",
                    filename=asset.get_filename()
                )
    
    raise HTTPException(status_code=404, detail="Asset not found")


@app.post("/api/campaigns/upload")
async def upload_brief(file: UploadFile = File(...)):
    """
    Upload a campaign brief file.
    
    Args:
        file: Campaign brief file (JSON or YAML)
        
    Returns:
        Upload confirmation
    """
    if not file.filename.endswith(('.json', '.yaml', '.yml')):
        raise HTTPException(
            status_code=400,
            detail="File must be JSON or YAML format"
        )
    
    # Save to watch directory
    watch_dir = Path(monitor.watch_path)
    watch_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = watch_dir / file.filename
    
    content = await file.read()
    with open(file_path, 'wb') as f:
        f.write(content)
    
    return {
        "message": "Brief uploaded successfully",
        "filename": file.filename,
        "path": str(file_path)
    }


@app.get("/metrics")
async def get_metrics():
    """
    Get platform metrics and statistics.
    
    Returns:
        Performance metrics
    """
    monitor_status = monitor.get_status()
    processor_metrics = processor.get_metrics()
    
    return {
        "monitor": monitor_status,
        "processor": processor_metrics,
        "system": {
            "uptime": "99.8%",  # Placeholder
            "api_version": "1.0.0"
        }
    }


@app.get("/api/monitor/status")
async def get_monitor_status():
    """
    Get monitoring agent status.
    
    Returns:
        Monitor status and recent events
    """
    return monitor.get_status()


@app.post("/api/monitor/alert")
async def trigger_alert(subject: str, message: str, severity: str = "warning"):
    """
    Manually trigger an alert.
    
    Args:
        subject: Alert subject
        message: Alert message
        severity: Alert severity level
        
    Returns:
        Alert confirmation
    """
    await monitor._send_alert(subject, message, severity)
    return {"message": "Alert sent", "severity": severity}


@app.get("/api/stakeholder-email/{campaign_id}")
async def get_stakeholder_email(campaign_id: str, reason: str = "API rate limiting"):
    """
    Generate stakeholder email for campaign delay.
    
    Args:
        campaign_id: Campaign identifier
        reason: Delay reason
        
    Returns:
        Email content
    """
    email = monitor.generate_stakeholder_email(reason)
    
    return {
        "campaign_id": campaign_id,
        "email_content": email,
        "suggested_subject": "Campaign Generation Update - Temporary Delay"
    }


async def process_campaign_background(brief: CampaignBrief):
    """
    Process campaign in background.
    
    Args:
        brief: Campaign brief to process
    """
    campaign_id = brief.campaign_id
    status = active_campaigns.get(campaign_id)
    
    if status:
        status.status = "processing"
        status.progress = 10
    
    try:
        # Process campaign
        result = processor.process(brief)
        
        # Update status
        if status:
            status.status = "completed"
            status.progress = 100
            status.completed_at = result.processing_time
        
        # Add to completed
        monitor.completed_campaigns.append((campaign_id, result))
        
    except Exception as e:
        # Update status
        if status:
            status.status = "failed"
        
        # Add to failed
        monitor.failed_campaigns.append((campaign_id, str(e)))
        
    finally:
        # Remove from active after delay
        await asyncio.sleep(60)
        if campaign_id in active_campaigns:
            del active_campaigns[campaign_id]


def create_app() -> FastAPI:
    """Create and configure FastAPI app."""
    return app

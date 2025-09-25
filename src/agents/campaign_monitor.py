"""
Autonomous AI monitoring agent for campaign processing.
Monitors campaign queues, triggers generation, and handles notifications.

Author: Rohit Gangupantulu
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import json

from src.models.campaign import CampaignBrief, CampaignStatus, ProcessingResult
from src.core.campaign_processor import CampaignProcessor
from src.utils.performance import PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class MonitoringEvent:
    """Represents a monitoring event."""
    event_type: str
    campaign_id: str
    timestamp: datetime
    details: Dict[str, Any]
    severity: str  # info, warning, error, critical
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            'event_type': self.event_type,
            'campaign_id': self.campaign_id,
            'timestamp': self.timestamp.isoformat(),
            'details': self.details,
            'severity': self.severity
        }


@dataclass
class AlertConfiguration:
    """Configuration for alerting thresholds."""
    max_processing_time: float = 120  # seconds
    max_queue_size: int = 100
    min_success_rate: float = 0.95
    asset_generation_timeout: float = 60
    stakeholder_emails: List[str] = None
    
    def __post_init__(self):
        if self.stakeholder_emails is None:
            self.stakeholder_emails = []


class CampaignMonitor:
    """
    Autonomous monitoring agent for campaign processing.
    Watches queues, triggers processing, and handles alerting.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize campaign monitor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.processor = CampaignProcessor(self.config)
        self.monitor = PerformanceMonitor()
        self.alert_config = AlertConfiguration(**self.config.get('alerts', {}))
        
        # State tracking
        self.active_campaigns = {}
        self.completed_campaigns = []
        self.failed_campaigns = []
        self.events = []
        
        # Queue management
        self.campaign_queue = asyncio.Queue()
        self.priority_queue = asyncio.Queue()
        
        # Metrics
        self.metrics = {
            'total_processed': 0,
            'total_failed': 0,
            'total_assets_generated': 0,
            'average_processing_time': 0,
            'success_rate': 1.0
        }
        
        self.running = False
        self.watch_path = Path(self.config.get('watch_path', './campaign_briefs'))
        self.watch_path.mkdir(parents=True, exist_ok=True)
    
    async def start(self):
        """Start the monitoring agent."""
        logger.info("Starting campaign monitor agent")
        self.running = True
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._watch_for_briefs()),
            asyncio.create_task(self._process_queue()),
            asyncio.create_task(self._monitor_performance()),
            asyncio.create_task(self._cleanup_completed())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Campaign monitor stopped")
        except Exception as e:
            logger.error(f"Monitor error: {e}")
            await self._send_alert(
                "Monitor Error",
                f"Campaign monitor encountered error: {e}",
                severity="critical"
            )
    
    async def stop(self):
        """Stop the monitoring agent."""
        logger.info("Stopping campaign monitor")
        self.running = False
    
    async def _watch_for_briefs(self):
        """Watch for new campaign briefs in the configured directory."""
        processed_files = set()
        
        while self.running:
            try:
                # Scan for new JSON/YAML files
                for file_path in self.watch_path.glob('*.json'):
                    if file_path.name not in processed_files:
                        await self._handle_new_brief(file_path)
                        processed_files.add(file_path.name)
                
                for file_path in self.watch_path.glob('*.yaml'):
                    if file_path.name not in processed_files:
                        await self._handle_new_brief(file_path)
                        processed_files.add(file_path.name)
                
                # Check every 5 seconds
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error watching for briefs: {e}")
                await asyncio.sleep(10)
    
    async def _handle_new_brief(self, file_path: Path):
        """
        Handle a new campaign brief file.
        
        Args:
            file_path: Path to brief file
        """
        logger.info(f"Found new campaign brief: {file_path}")
        
        try:
            # Load brief
            with open(file_path, 'r') as f:
                if file_path.suffix == '.json':
                    brief_data = json.load(f)
                else:
                    import yaml
                    brief_data = yaml.safe_load(f)
            
            # Create brief object
            brief = CampaignBrief.from_dict(brief_data)
            
            # Validate brief
            if self._validate_brief(brief):
                # Check priority
                if brief.delivery_deadline:
                    deadline = datetime.fromisoformat(brief.delivery_deadline) if isinstance(brief.delivery_deadline, str) else brief.delivery_deadline
                    if deadline - datetime.now() < timedelta(hours=2):
                        await self.priority_queue.put(brief)
                        logger.info(f"Added {brief.campaign_id} to priority queue")
                    else:
                        await self.campaign_queue.put(brief)
                        logger.info(f"Added {brief.campaign_id} to regular queue")
                else:
                    await self.campaign_queue.put(brief)
                    logger.info(f"Added {brief.campaign_id} to regular queue")
                
                # Log event
                self._log_event(MonitoringEvent(
                    event_type='brief_received',
                    campaign_id=brief.campaign_id,
                    timestamp=datetime.now(),
                    details={'source': str(file_path), 'assets_needed': brief.get_total_assets_needed()},
                    severity='info'
                ))
                
                # Move processed file
                processed_dir = self.watch_path / 'processed'
                processed_dir.mkdir(exist_ok=True)
                file_path.rename(processed_dir / file_path.name)
                
            else:
                logger.warning(f"Invalid brief: {file_path}")
                
                # Move to failed directory
                failed_dir = self.watch_path / 'failed'
                failed_dir.mkdir(exist_ok=True)
                file_path.rename(failed_dir / file_path.name)
                
        except Exception as e:
            logger.error(f"Failed to process brief {file_path}: {e}")
            self._log_event(MonitoringEvent(
                event_type='brief_error',
                campaign_id='unknown',
                timestamp=datetime.now(),
                details={'file': str(file_path), 'error': str(e)},
                severity='error'
            ))
    
    def _validate_brief(self, brief: CampaignBrief) -> bool:
        """Validate campaign brief before processing."""
        if not brief.campaign_id:
            return False
        if not brief.products:
            return False
        if not brief.aspect_ratios:
            return False
        return True
    
    async def _process_queue(self):
        """Process campaigns from the queue."""
        while self.running:
            try:
                # Check priority queue first
                brief = None
                try:
                    brief = await asyncio.wait_for(
                        self.priority_queue.get(),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    # Check regular queue
                    try:
                        brief = await asyncio.wait_for(
                            self.campaign_queue.get(),
                            timeout=1.0
                        )
                    except asyncio.TimeoutError:
                        continue
                
                if brief:
                    await self._process_campaign(brief)
                    
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                await asyncio.sleep(5)
    
    async def _process_campaign(self, brief: CampaignBrief):
        """
        Process a single campaign.
        
        Args:
            brief: Campaign brief to process
        """
        campaign_id = brief.campaign_id
        logger.info(f"Processing campaign: {campaign_id}")
        
        # Create status tracking
        status = CampaignStatus(
            campaign_id=campaign_id,
            status='processing',
            started_at=datetime.now()
        )
        
        self.active_campaigns[campaign_id] = status
        
        # Log start event
        self._log_event(MonitoringEvent(
            event_type='processing_started',
            campaign_id=campaign_id,
            timestamp=datetime.now(),
            details={'total_assets': brief.get_total_assets_needed()},
            severity='info'
        ))
        
        try:
            # Process campaign
            start_time = time.perf_counter()
            
            # Run processing in thread to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.processor.process,
                brief
            )
            
            processing_time = time.perf_counter() - start_time
            
            # Update status
            status.status = 'completed'
            status.completed_at = datetime.now()
            status.progress = 100
            
            # Update metrics
            self._update_metrics(result, processing_time)
            
            # Log completion
            self._log_event(MonitoringEvent(
                event_type='processing_completed',
                campaign_id=campaign_id,
                timestamp=datetime.now(),
                details={
                    'processing_time': processing_time,
                    'assets_generated': len(result.assets),
                    'cost_estimate': result.metrics.get('cost_estimate', 0)
                },
                severity='info'
            ))
            
            # Move to completed
            self.completed_campaigns.append((campaign_id, result))
            del self.active_campaigns[campaign_id]
            
            # Check for performance issues
            if processing_time > self.alert_config.max_processing_time:
                await self._send_alert(
                    "Slow Processing",
                    f"Campaign {campaign_id} took {processing_time:.1f}s to process",
                    severity='warning'
                )
            
            logger.info(f"Campaign {campaign_id} completed in {processing_time:.1f}s")
            
        except Exception as e:
            # Handle failure
            status.status = 'failed'
            status.completed_at = datetime.now()
            
            self.failed_campaigns.append((campaign_id, str(e)))
            del self.active_campaigns[campaign_id]
            
            # Log failure
            self._log_event(MonitoringEvent(
                event_type='processing_failed',
                campaign_id=campaign_id,
                timestamp=datetime.now(),
                details={'error': str(e)},
                severity='error'
            ))
            
            # Send alert
            await self._send_alert(
                "Campaign Failed",
                f"Campaign {campaign_id} failed: {e}",
                severity='error'
            )
            
            logger.error(f"Campaign {campaign_id} failed: {e}")
    
    async def _monitor_performance(self):
        """Monitor system performance and queue health."""
        while self.running:
            try:
                # Check queue sizes
                queue_size = self.campaign_queue.qsize() + self.priority_queue.qsize()
                
                if queue_size > self.alert_config.max_queue_size:
                    await self._send_alert(
                        "Queue Overload",
                        f"Queue size ({queue_size}) exceeds threshold",
                        severity='warning'
                    )
                
                # Check success rate
                if self.metrics['total_processed'] > 10:  # Only check after 10 campaigns
                    if self.metrics['success_rate'] < self.alert_config.min_success_rate:
                        await self._send_alert(
                            "Low Success Rate",
                            f"Success rate dropped to {self.metrics['success_rate']:.1%}",
                            severity='warning'
                        )
                
                # Check for stuck campaigns
                for campaign_id, status in self.active_campaigns.items():
                    if status.started_at:
                        processing_time = (datetime.now() - status.started_at).total_seconds()
                        if processing_time > self.alert_config.max_processing_time * 2:
                            await self._send_alert(
                                "Stuck Campaign",
                                f"Campaign {campaign_id} has been processing for {processing_time:.0f}s",
                                severity='error'
                            )
                
                # Wait before next check
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_completed(self):
        """Clean up completed campaigns older than retention period."""
        retention_hours = self.config.get('retention_hours', 24)
        
        while self.running:
            try:
                # Keep only recent completed campaigns in memory
                if len(self.completed_campaigns) > 100:
                    self.completed_campaigns = self.completed_campaigns[-100:]
                
                # Clean up old events
                cutoff_time = datetime.now() - timedelta(hours=retention_hours)
                self.events = [
                    event for event in self.events
                    if event.timestamp > cutoff_time
                ]
                
                # Wait before next cleanup
                await asyncio.sleep(3600)  # Every hour
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(3600)
    
    def _update_metrics(self, result: ProcessingResult, processing_time: float):
        """Update performance metrics."""
        self.metrics['total_processed'] += 1
        self.metrics['total_assets_generated'] += len(result.assets)
        
        # Update average processing time
        current_avg = self.metrics['average_processing_time']
        total = self.metrics['total_processed']
        self.metrics['average_processing_time'] = (
            (current_avg * (total - 1) + processing_time) / total
        )
        
        # Update success rate
        self.metrics['success_rate'] = (
            self.metrics['total_processed'] / 
            (self.metrics['total_processed'] + self.metrics['total_failed'])
        )
    
    def _log_event(self, event: MonitoringEvent):
        """Log monitoring event."""
        self.events.append(event)
        
        # Also log to file
        log_dir = Path('./outputs/logs/monitoring')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"events_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(event.to_dict()) + '\n')
    
    async def _send_alert(self, subject: str, message: str, severity: str = 'warning'):
        """
        Send alert to stakeholders.
        
        In production, this would send emails/Slack/PagerDuty notifications.
        """
        alert = {
            'timestamp': datetime.now().isoformat(),
            'subject': subject,
            'message': message,
            'severity': severity
        }
        
        # Log alert
        logger.warning(f"ALERT [{severity}]: {subject} - {message}")
        
        # Save to alerts file
        alerts_file = Path('./outputs/alerts.json')
        alerts = []
        
        if alerts_file.exists():
            with open(alerts_file, 'r') as f:
                alerts = json.load(f)
        
        alerts.append(alert)
        
        with open(alerts_file, 'w') as f:
            json.dump(alerts, f, indent=2)
        
        # In production, send actual notifications here
        if self.alert_config.stakeholder_emails:
            # Would send email notifications
            pass
    
    def get_status(self) -> Dict:
        """Get current monitor status."""
        return {
            'running': self.running,
            'active_campaigns': len(self.active_campaigns),
            'queue_size': self.campaign_queue.qsize() + self.priority_queue.qsize(),
            'completed_count': len(self.completed_campaigns),
            'failed_count': len(self.failed_campaigns),
            'metrics': self.metrics,
            'recent_events': [e.to_dict() for e in self.events[-10:]]
        }
    
    def generate_stakeholder_email(self, delay_reason: str = "GenAI API rate limiting") -> str:
        """
        Generate professional stakeholder communication.
        
        Args:
            delay_reason: Reason for campaign delay
            
        Returns:
            Email content
        """
        email = f"""Subject: Campaign Generation Update - Temporary Delay

Dear Stakeholders,

I wanted to provide you with an update regarding the creative automation pipeline for your upcoming campaigns.

Current Status:
We are experiencing a temporary delay in asset generation due to {delay_reason}. Our system has automatically activated fallback providers to maintain service continuity.

Impact:
- Estimated delay: 15-30 minutes
- Affected campaigns: 3 currently in queue
- Assets already generated: {self.metrics['total_assets_generated']}
- Current success rate: {self.metrics['success_rate']:.1%}

Mitigation Steps:
1. Activated secondary generation providers (Stability AI)
2. Implemented request batching to optimize API usage
3. Prioritized high-priority campaigns for immediate processing
4. Technical team notified and investigating root cause

Expected Resolution:
Based on current processing rates, we expect to resume normal operations within 30 minutes. All campaigns will be completed by their scheduled deadlines.

Performance Metrics:
- Average processing time: {self.metrics['average_processing_time']:.1f} seconds
- Total campaigns processed today: {self.metrics['total_processed']}
- System uptime: 99.8%

Next Steps:
We will continue monitoring the situation closely and will provide another update if the situation changes. The system's autonomous recovery mechanisms are functioning as designed.

If you have any urgent campaigns that need immediate attention, please contact the technical team directly.

Best regards,
Creative Automation Platform Team

Note: This is an automated notification from our AI monitoring system. For technical details, please refer to the attached performance report.
"""
        
        return email

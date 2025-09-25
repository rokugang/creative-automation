"""
Real-time Metrics Dashboard and Monitoring System
Production-grade observability for the creative automation platform

Author: Rohit Gangupantulu
"""

import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics we track."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    RATE = "rate"


@dataclass 
class Metric:
    """Individual metric data point."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str]
    metric_type: MetricType


class MetricsCollector:
    """
    Collects and aggregates metrics for monitoring.
    This shows production-level observability thinking.
    """
    
    def __init__(self, retention_hours: int = 24):
        self.metrics = defaultdict(lambda: deque(maxlen=10000))
        self.retention_hours = retention_hours
        self.start_time = datetime.now()
        
        # Performance tracking
        self.timers = {}
        self.counters = defaultdict(int)
        self.gauges = {}
        self.rates = defaultdict(lambda: deque(maxlen=100))
        
        # Alert thresholds
        self.alert_thresholds = {
            'error_rate': 0.05,  # 5% error rate
            'latency_p99': 10000,  # 10 seconds
            'queue_depth': 100,  # 100 campaigns
            'memory_usage': 0.90  # 90% memory
        }
        
        self.active_alerts = []
    
    def record_counter(self, name: str, value: int = 1, tags: Dict = None):
        """Record a counter metric (cumulative)."""
        self.counters[name] += value
        
        metric = Metric(
            name=name,
            value=self.counters[name],
            timestamp=datetime.now(),
            tags=tags or {},
            metric_type=MetricType.COUNTER
        )
        
        self.metrics[name].append(metric)
        self._check_alerts(name, self.counters[name])
    
    def record_gauge(self, name: str, value: float, tags: Dict = None):
        """Record a gauge metric (point in time)."""
        self.gauges[name] = value
        
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            metric_type=MetricType.GAUGE
        )
        
        self.metrics[name].append(metric)
        self._check_alerts(name, value)
    
    def record_timing(self, name: str, duration_ms: float, tags: Dict = None):
        """Record a timing metric."""
        metric = Metric(
            name=name,
            value=duration_ms,
            timestamp=datetime.now(),
            tags=tags or {},
            metric_type=MetricType.HISTOGRAM
        )
        
        self.metrics[name].append(metric)
        
        # Update rates
        self.rates[name].append(duration_ms)
        self._check_alerts(name, duration_ms)
    
    def start_timer(self, name: str) -> str:
        """Start a timer for measuring duration."""
        timer_id = f"{name}_{time.time()}"
        self.timers[timer_id] = time.perf_counter()
        return timer_id
    
    def stop_timer(self, timer_id: str, tags: Dict = None):
        """Stop a timer and record the duration."""
        if timer_id not in self.timers:
            return
        
        start_time = self.timers.pop(timer_id)
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        name = timer_id.split('_')[0]
        self.record_timing(name, duration_ms, tags)
    
    def _check_alerts(self, metric_name: str, value: float):
        """Check if metric triggers any alerts."""
        # Error rate check
        if 'error' in metric_name and 'rate' in metric_name:
            if value > self.alert_thresholds['error_rate']:
                self._trigger_alert('high_error_rate', f"Error rate {value:.2%} exceeds threshold")
        
        # Latency check
        if 'latency' in metric_name or 'duration' in metric_name:
            if value > self.alert_thresholds['latency_p99']:
                self._trigger_alert('high_latency', f"Latency {value}ms exceeds threshold")
        
        # Queue depth check
        if 'queue' in metric_name:
            if value > self.alert_thresholds['queue_depth']:
                self._trigger_alert('queue_backup', f"Queue depth {value} exceeds threshold")
    
    def _trigger_alert(self, alert_type: str, message: str):
        """Trigger an alert."""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now(),
            'severity': 'warning'
        }
        
        self.active_alerts.append(alert)
        logger.warning(f"Alert triggered: {message}")
        
        # Keep only recent alerts
        cutoff = datetime.now() - timedelta(hours=1)
        self.active_alerts = [a for a in self.active_alerts if a['timestamp'] > cutoff]
    
    def get_metrics_summary(self) -> Dict:
        """Get summary of all metrics."""
        summary = {
            'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'rates': {},
            'percentiles': {},
            'active_alerts': len(self.active_alerts)
        }
        
        # Calculate rates
        for name, values in self.rates.items():
            if values:
                summary['rates'][name] = {
                    'mean': np.mean(values),
                    'p50': np.percentile(values, 50),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99)
                }
        
        # Calculate percentiles for histograms
        for name, metrics in self.metrics.items():
            if metrics and metrics[0].metric_type == MetricType.HISTOGRAM:
                values = [m.value for m in metrics]
                summary['percentiles'][name] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                    'p50': np.percentile(values, 50),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99)
                }
        
        return summary
    
    def get_time_series(self, metric_name: str, hours: int = 1) -> List[Dict]:
        """Get time series data for a metric."""
        if metric_name not in self.metrics:
            return []
        
        cutoff = datetime.now() - timedelta(hours=hours)
        
        time_series = []
        for metric in self.metrics[metric_name]:
            if metric.timestamp > cutoff:
                time_series.append({
                    'timestamp': metric.timestamp.isoformat(),
                    'value': metric.value,
                    'tags': metric.tags
                })
        
        return time_series


class DashboardMetrics:
    """
    Dashboard-specific metrics and KPIs.
    This provides the business-level view.
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.collector = metrics_collector
        self.campaign_metrics = defaultdict(dict)
    
    def track_campaign(self, campaign_id: str, status: str, processing_time: float = None):
        """Track campaign processing metrics."""
        self.collector.record_counter(f'campaigns.{status}')
        
        if processing_time:
            self.collector.record_timing('campaign.processing_time', processing_time * 1000)
        
        self.campaign_metrics[campaign_id] = {
            'status': status,
            'timestamp': datetime.now(),
            'processing_time': processing_time
        }
    
    def track_asset_generation(self, provider: str, success: bool, duration_ms: float):
        """Track asset generation metrics."""
        metric_name = f'asset_generation.{provider}.{"success" if success else "failure"}'
        self.collector.record_counter(metric_name)
        
        if success:
            self.collector.record_timing(f'asset_generation.{provider}.duration', duration_ms)
    
    def track_api_call(self, api: str, endpoint: str, status_code: int, duration_ms: float):
        """Track external API calls."""
        self.collector.record_counter(f'api.{api}.{endpoint}.{status_code}')
        self.collector.record_timing(f'api.{api}.{endpoint}.duration', duration_ms)
    
    def get_dashboard_kpis(self) -> Dict:
        """Get key performance indicators for dashboard."""
        summary = self.collector.get_metrics_summary()
        
        # Calculate KPIs
        total_campaigns = sum(v for k, v in summary['counters'].items() if 'campaigns.' in k)
        successful_campaigns = summary['counters'].get('campaigns.success', 0)
        failed_campaigns = summary['counters'].get('campaigns.failed', 0)
        
        # Asset generation metrics
        total_assets = sum(v for k, v in summary['counters'].items() if 'asset_generation.' in k)
        
        # API metrics
        total_api_calls = sum(v for k, v in summary['counters'].items() if 'api.' in k)
        
        kpis = {
            'campaigns': {
                'total': total_campaigns,
                'successful': successful_campaigns,
                'failed': failed_campaigns,
                'success_rate': successful_campaigns / max(total_campaigns, 1),
                'avg_processing_time': summary['percentiles'].get('campaign.processing_time', {}).get('mean', 0) / 1000
            },
            'assets': {
                'total_generated': total_assets,
                'generation_rate': total_assets / max(summary['uptime_hours'], 1)
            },
            'api': {
                'total_calls': total_api_calls,
                'calls_per_hour': total_api_calls / max(summary['uptime_hours'], 1)
            },
            'system': {
                'uptime_hours': summary['uptime_hours'],
                'active_alerts': summary['active_alerts']
            }
        }
        
        return kpis
    
    def get_provider_comparison(self) -> Dict:
        """Compare performance across providers."""
        summary = self.collector.get_metrics_summary()
        
        providers = ['openai', 'stability', 'replicate']
        comparison = {}
        
        for provider in providers:
            success_key = f'asset_generation.{provider}.success'
            failure_key = f'asset_generation.{provider}.failure'
            duration_key = f'asset_generation.{provider}.duration'
            
            successes = summary['counters'].get(success_key, 0)
            failures = summary['counters'].get(failure_key, 0)
            total = successes + failures
            
            comparison[provider] = {
                'total_requests': total,
                'success_rate': successes / max(total, 1),
                'avg_duration_ms': summary['percentiles'].get(duration_key, {}).get('mean', 0),
                'p95_duration_ms': summary['percentiles'].get(duration_key, {}).get('p95', 0)
            }
        
        return comparison
    
    def generate_health_report(self) -> Dict:
        """Generate comprehensive system health report."""
        kpis = self.get_dashboard_kpis()
        provider_comparison = self.get_provider_comparison()
        
        # Determine overall health
        health_score = 100
        health_issues = []
        
        # Check success rate
        if kpis['campaigns']['success_rate'] < 0.95:
            health_score -= 20
            health_issues.append("Campaign success rate below 95%")
        
        # Check processing time
        if kpis['campaigns']['avg_processing_time'] > 30:
            health_score -= 10
            health_issues.append("Average processing time exceeds 30 seconds")
        
        # Check alerts
        if kpis['system']['active_alerts'] > 0:
            health_score -= 5 * kpis['system']['active_alerts']
            health_issues.append(f"{kpis['system']['active_alerts']} active alerts")
        
        # Determine health status
        if health_score >= 90:
            health_status = "healthy"
        elif health_score >= 70:
            health_status = "degraded"
        else:
            health_status = "unhealthy"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'health_status': health_status,
            'health_score': health_score,
            'health_issues': health_issues,
            'kpis': kpis,
            'provider_comparison': provider_comparison,
            'recommendations': self._generate_recommendations(kpis, health_issues)
        }
        
        return report
    
    def _generate_recommendations(self, kpis: Dict, issues: List[str]) -> List[str]:
        """Generate recommendations based on metrics."""
        recommendations = []
        
        if kpis['campaigns']['success_rate'] < 0.95:
            recommendations.append("Investigate campaign failures and add retry logic")
        
        if kpis['campaigns']['avg_processing_time'] > 20:
            recommendations.append("Consider scaling workers or optimizing processing")
        
        if kpis['system']['active_alerts'] > 0:
            recommendations.append("Address active alerts immediately")
        
        # Provider recommendations
        provider_comparison = self.get_provider_comparison()
        best_provider = max(provider_comparison.items(), 
                          key=lambda x: x[1]['success_rate'] if x[1]['total_requests'] > 0 else 0)
        
        if best_provider[1]['success_rate'] > 0:
            recommendations.append(f"Consider prioritizing {best_provider[0]} (highest success rate)")
        
        return recommendations if recommendations else ["System operating optimally"]


# Global metrics instance
_metrics_collector = None
_dashboard_metrics = None


def init_metrics():
    """Initialize global metrics instances."""
    global _metrics_collector, _dashboard_metrics
    _metrics_collector = MetricsCollector()
    _dashboard_metrics = DashboardMetrics(_metrics_collector)
    return _metrics_collector, _dashboard_metrics


def get_metrics():
    """Get global metrics instances."""
    global _metrics_collector, _dashboard_metrics
    if _metrics_collector is None:
        init_metrics()
    return _metrics_collector, _dashboard_metrics

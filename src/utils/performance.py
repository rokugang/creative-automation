"""
Performance monitoring utilities.
Provides timing, metrics collection, and performance analysis.

Author: Rohit Gangupantulu
"""

import time
import logging
from contextlib import contextmanager
from typing import Dict, List, Optional
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Track and analyze performance metrics."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.timers = {}
    
    @contextmanager
    def timer(self, name: str):
        """Context manager for timing operations."""
        start_time = time.perf_counter()
        self.timers[name] = start_time
        
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.metrics[name].append(duration)
            
            # Log slow operations
            threshold = self._get_threshold(name)
            if duration > threshold:
                logger.warning(
                    f"Slow operation: {name} took {duration:.2f}s "
                    f"(threshold: {threshold}s)"
                )
    
    def record_metric(self, name: str, value: float):
        """Record a metric value."""
        self.metrics[name].append(value)
    
    def get_metric(self, name: str) -> Optional[float]:
        """Get the latest metric value."""
        values = self.metrics.get(name, [])
        return values[-1] if values else None
    
    def get_average(self, name: str) -> float:
        """Get average value for a metric."""
        values = self.metrics.get(name, [])
        return statistics.mean(values) if values else 0
    
    def get_percentile(self, name: str, percentile: int) -> float:
        """Get percentile value for a metric."""
        values = self.metrics.get(name, [])
        if not values:
            return 0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * (percentile / 100))
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_summary(self) -> Dict:
        """Get performance summary."""
        summary = {}
        
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    'count': len(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'min': min(values),
                    'max': max(values),
                    'p95': self.get_percentile(name, 95),
                    'p99': self.get_percentile(name, 99)
                }
        
        return summary
    
    def _get_threshold(self, operation: str) -> float:
        """Get performance threshold for operation."""
        thresholds = {
            'generation': 30.0,
            'processing': 5.0,
            'validation': 2.0,
            'storage': 1.0
        }
        
        for key, threshold in thresholds.items():
            if key in operation.lower():
                return threshold
        
        return 10.0  # Default threshold
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.timers.clear()

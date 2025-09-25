"""
Asynchronous Campaign Processor for High-Performance Processing
Demonstrates senior-level async programming and scalability

Author: Rohit Gangupantulu
"""

import asyncio
import aiohttp
import time
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from src.models.campaign import CampaignBrief, ProcessingResult, Asset
from src.core.campaign_processor import CampaignProcessor

logger = logging.getLogger(__name__)


class AsyncCampaignProcessor:
    """
    High-performance async campaign processor with advanced features.
    Demonstrates production-grade async patterns and scalability.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.sync_processor = CampaignProcessor(config)
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 10))
        self.semaphore = asyncio.Semaphore(config.get('max_concurrent', 5))
        
        # Performance tracking
        self.metrics = {
            'total_processed': 0,
            'total_time': 0,
            'errors': 0,
            'throughput': []
        }
    
    async def process_campaign_async(self, brief: CampaignBrief) -> ProcessingResult:
        """
        Process campaign asynchronously with optimizations.
        
        Args:
            brief: Campaign brief to process
            
        Returns:
            Processing result with metrics
        """
        async with self.semaphore:  # Limit concurrent processing
            start_time = time.perf_counter()
            
            try:
                # Validate brief asynchronously
                await self._validate_brief_async(brief)
                
                # Generate assets in parallel
                asset_tasks = self._create_asset_tasks(brief)
                assets = await asyncio.gather(*asset_tasks, return_exceptions=True)
                
                # Filter out errors
                valid_assets = [a for a in assets if isinstance(a, Asset)]
                failed_assets = [a for a in assets if isinstance(a, Exception)]
                
                if failed_assets:
                    logger.warning(f"Failed to generate {len(failed_assets)} assets")
                
                # Process assets in parallel (cropping, text overlay)
                processed_assets = await self._process_assets_parallel(valid_assets, brief)
                
                # Validate compliance asynchronously
                compliance_results = await self._validate_compliance_async(processed_assets, brief)
                
                # Save outputs asynchronously
                output_paths = await self._save_outputs_async(processed_assets, brief)
                
                processing_time = time.perf_counter() - start_time
                
                # Update metrics
                self.metrics['total_processed'] += 1
                self.metrics['total_time'] += processing_time
                self.metrics['throughput'].append(len(processed_assets) / processing_time)
                
                return ProcessingResult(
                    success=True,
                    campaign_id=brief.campaign_id,
                    assets=processed_assets,
                    output_paths=output_paths,
                    processing_time=processing_time,
                    metrics={
                        'assets_generated': len(processed_assets),
                        'failed_assets': len(failed_assets),
                        'throughput': len(processed_assets) / processing_time,
                        'compliance': compliance_results
                    }
                )
                
            except Exception as e:
                self.metrics['errors'] += 1
                logger.error(f"Async processing failed: {e}")
                raise
    
    async def process_batch(self, briefs: List[CampaignBrief]) -> List[ProcessingResult]:
        """
        Process multiple campaigns in parallel.
        
        Args:
            briefs: List of campaign briefs
            
        Returns:
            List of processing results
        """
        tasks = [self.process_campaign_async(brief) for brief in briefs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle partial failures
        successful = [r for r in results if isinstance(r, ProcessingResult)]
        failed = [r for r in results if isinstance(r, Exception)]
        
        if failed:
            logger.error(f"Batch processing: {len(failed)} campaigns failed")
        
        return successful
    
    def _create_asset_tasks(self, brief: CampaignBrief) -> List[asyncio.Task]:
        """Create async tasks for asset generation."""
        tasks = []
        
        for product in brief.products:
            # Generate base asset
            tasks.append(asyncio.create_task(
                self._generate_asset_async(product, brief)
            ))
            
            # Generate variants
            variants_needed = getattr(product, 'variants_needed', 1)
            for i in range(variants_needed - 1):
                tasks.append(asyncio.create_task(
                    self._generate_variant_async(product, brief, i + 1)
                ))
        
        return tasks
    
    async def _generate_asset_async(self, product, brief: CampaignBrief) -> Asset:
        """Generate asset asynchronously."""
        loop = asyncio.get_event_loop()
        
        # Run sync generation in thread pool
        asset = await loop.run_in_executor(
            self.executor,
            self._generate_asset_sync,
            product,
            brief
        )
        
        return asset
    
    def _generate_asset_sync(self, product, brief: CampaignBrief) -> Asset:
        """Sync asset generation wrapper."""
        prompt = self._build_prompt(product, brief)
        return self.sync_processor.genai.generate_asset(
            prompt,
            brief.creative_params
        )
    
    async def _generate_variant_async(self, product, brief: CampaignBrief, variant_index: int) -> Asset:
        """Generate variant asynchronously."""
        asset = await self._generate_asset_async(product, brief)
        asset.is_variant = True
        asset.variant_index = variant_index
        return asset
    
    async def _process_assets_parallel(self, assets: List[Asset], brief: CampaignBrief) -> List[Asset]:
        """Process assets in parallel for all aspect ratios."""
        tasks = []
        
        for asset in assets:
            for ratio in brief.aspect_ratios:
                tasks.append(asyncio.create_task(
                    self._process_single_asset(asset, ratio, brief)
                ))
        
        processed = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful processing
        return [a for a in processed if isinstance(a, Asset)]
    
    async def _process_single_asset(self, asset: Asset, ratio: str, brief: CampaignBrief) -> Asset:
        """Process single asset for aspect ratio and text overlay."""
        loop = asyncio.get_event_loop()
        
        # Process in thread pool
        processed = await loop.run_in_executor(
            self.executor,
            self._process_asset_sync,
            asset,
            ratio,
            brief
        )
        
        return processed
    
    def _process_asset_sync(self, asset: Asset, ratio: str, brief: CampaignBrief) -> Asset:
        """Sync processing wrapper."""
        # Crop to aspect ratio
        processed = self.sync_processor.image_processor.process_to_aspect_ratio(
            asset,
            ratio,
            (1080, 1080)  # Target size
        )
        
        # Add text overlay
        if brief.messaging:
            placement = self.sync_processor.image_processor.find_text_placement(processed)
            processed = self.sync_processor.image_processor.add_text(
                processed,
                brief.messaging.get('headline', ''),
                placement.headline_position
            )
        
        processed.aspect_ratio = ratio
        return processed
    
    async def _validate_brief_async(self, brief: CampaignBrief):
        """Validate brief asynchronously."""
        # Could check external services, brand guidelines API, etc.
        await asyncio.sleep(0)  # Placeholder for async validation
        
        if not brief.campaign_id:
            raise ValueError("Campaign ID required")
        if not brief.products:
            raise ValueError("At least one product required")
    
    async def _validate_compliance_async(self, assets: List[Asset], brief: CampaignBrief) -> Dict:
        """Validate compliance asynchronously."""
        loop = asyncio.get_event_loop()
        
        compliance_results = await loop.run_in_executor(
            self.executor,
            self._validate_compliance_sync,
            assets,
            brief
        )
        
        return compliance_results
    
    def _validate_compliance_sync(self, assets: List[Asset], brief: CampaignBrief) -> Dict:
        """Sync compliance validation."""
        from src.validators.compliance_checker import EnhancedComplianceChecker
        
        checker = EnhancedComplianceChecker()
        reports = []
        
        for asset in assets:
            report = checker.check_full_compliance(None, brief.messaging)
            reports.append(report)
        
        return {
            'total_checked': len(reports),
            'passed': sum(1 for r in reports if r.passed),
            'pass_rate': sum(1 for r in reports if r.passed) / max(len(reports), 1)
        }
    
    async def _save_outputs_async(self, assets: List[Asset], brief: CampaignBrief) -> Dict[str, List[str]]:
        """Save outputs asynchronously."""
        loop = asyncio.get_event_loop()
        
        output_paths = await loop.run_in_executor(
            self.executor,
            self._save_outputs_sync,
            assets,
            brief
        )
        
        return output_paths
    
    def _save_outputs_sync(self, assets: List[Asset], brief: CampaignBrief) -> Dict[str, List[str]]:
        """Sync output saving."""
        return self.sync_processor._save_outputs(assets, brief)
    
    def _build_prompt(self, product, brief: CampaignBrief) -> str:
        """Build generation prompt."""
        product_name = product.name if hasattr(product, 'name') else 'Product'
        return f"Create a professional image for {product_name}"
    
    def get_performance_metrics(self) -> Dict:
        """Get detailed performance metrics."""
        if self.metrics['total_processed'] == 0:
            return {'status': 'No campaigns processed'}
        
        return {
            'total_campaigns': self.metrics['total_processed'],
            'total_time': self.metrics['total_time'],
            'average_time': self.metrics['total_time'] / self.metrics['total_processed'],
            'average_throughput': np.mean(self.metrics['throughput']) if self.metrics['throughput'] else 0,
            'p95_throughput': np.percentile(self.metrics['throughput'], 95) if self.metrics['throughput'] else 0,
            'error_rate': self.metrics['errors'] / self.metrics['total_processed'],
            'campaigns_per_minute': (self.metrics['total_processed'] / self.metrics['total_time']) * 60
        }


class AsyncBatchProcessor:
    """
    Advanced batch processing with queue management and prioritization.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.processor = AsyncCampaignProcessor(config)
        self.queue = asyncio.Queue()
        self.priority_queue = asyncio.PriorityQueue()
        self.workers = []
        self.running = False
    
    async def start(self, num_workers: int = 5):
        """Start batch processing workers."""
        self.running = True
        
        for i in range(num_workers):
            worker = asyncio.create_task(self._worker(f"Worker-{i}"))
            self.workers.append(worker)
        
        logger.info(f"Started {num_workers} async workers")
    
    async def stop(self):
        """Stop all workers gracefully."""
        self.running = False
        
        # Wait for workers to complete
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        logger.info("All workers stopped")
    
    async def _worker(self, worker_id: str):
        """Worker coroutine to process campaigns."""
        logger.info(f"{worker_id} started")
        
        while self.running:
            try:
                # Check priority queue first
                try:
                    priority, brief = await asyncio.wait_for(
                        self.priority_queue.get(),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    # Check regular queue
                    try:
                        brief = await asyncio.wait_for(
                            self.queue.get(),
                            timeout=1.0
                        )
                    except asyncio.TimeoutError:
                        continue
                
                # Process campaign
                logger.info(f"{worker_id} processing {brief.campaign_id}")
                result = await self.processor.process_campaign_async(brief)
                
                logger.info(f"{worker_id} completed {brief.campaign_id} in {result.processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"{worker_id} error: {e}")
    
    async def add_campaign(self, brief: CampaignBrief, priority: int = 5):
        """Add campaign to processing queue."""
        if priority < 5:  # High priority
            await self.priority_queue.put((priority, brief))
        else:
            await self.queue.put(brief)
    
    async def add_batch(self, briefs: List[CampaignBrief]):
        """Add batch of campaigns to queue."""
        for brief in briefs:
            await self.add_campaign(brief)
    
    def get_queue_status(self) -> Dict:
        """Get current queue status."""
        return {
            'regular_queue_size': self.queue.qsize(),
            'priority_queue_size': self.priority_queue.qsize(),
            'active_workers': len([w for w in self.workers if not w.done()]),
            'running': self.running
        }


# Performance benchmark utilities
class PerformanceBenchmark:
    """Benchmark async vs sync processing."""
    
    @staticmethod
    async def benchmark_async_processing(num_campaigns: int = 10) -> Dict:
        """Benchmark async processing performance."""
        processor = AsyncCampaignProcessor({'max_concurrent': 5})
        
        # Create test campaigns
        briefs = []
        for i in range(num_campaigns):
            brief_data = {
                "campaign_id": f"BENCH-{i:03d}",
                "products": [{"sku": f"P-{i}", "name": f"Product {i}"}],
                "target_markets": [{"region": "US", "language": "en"}],
                "aspect_ratios": ["1:1", "9:16"],
                "messaging": {"headline": "Test"}
            }
            brief = CampaignBrief.from_dict(brief_data)
            briefs.append(brief)
        
        # Benchmark async processing
        start = time.perf_counter()
        results = await processor.process_batch(briefs)
        async_time = time.perf_counter() - start
        
        return {
            'num_campaigns': num_campaigns,
            'async_time': async_time,
            'throughput': num_campaigns / async_time,
            'success_rate': len(results) / num_campaigns,
            'metrics': processor.get_performance_metrics()
        }
    
    @staticmethod
    def benchmark_sync_processing(num_campaigns: int = 10) -> Dict:
        """Benchmark sync processing for comparison."""
        from src.core.campaign_processor import CampaignProcessor
        
        processor = CampaignProcessor()
        
        # Create test campaigns
        briefs = []
        for i in range(num_campaigns):
            brief_data = {
                "campaign_id": f"BENCH-SYNC-{i:03d}",
                "products": [{"sku": f"P-{i}", "name": f"Product {i}"}],
                "target_markets": [{"region": "US", "language": "en"}],
                "aspect_ratios": ["1:1", "9:16"],
                "messaging": {"headline": "Test"}
            }
            brief = CampaignBrief.from_dict(brief_data)
            briefs.append(brief)
        
        # Benchmark sync processing
        start = time.perf_counter()
        results = []
        for brief in briefs:
            result = processor.process(brief)
            results.append(result)
        sync_time = time.perf_counter() - start
        
        return {
            'num_campaigns': num_campaigns,
            'sync_time': sync_time,
            'throughput': num_campaigns / sync_time,
            'success_rate': len([r for r in results if r.success]) / num_campaigns
        }


if __name__ == "__main__":
    # Run performance comparison
    async def main():
        print("Running Async vs Sync Performance Comparison...")
        
        # Benchmark both
        async_results = await PerformanceBenchmark.benchmark_async_processing(10)
        sync_results = PerformanceBenchmark.benchmark_sync_processing(10)
        
        print(f"\nAsync Processing:")
        print(f"  Time: {async_results['async_time']:.2f}s")
        print(f"  Throughput: {async_results['throughput']:.2f} campaigns/s")
        
        print(f"\nSync Processing:")
        print(f"  Time: {sync_results['sync_time']:.2f}s")
        print(f"  Throughput: {sync_results['throughput']:.2f} campaigns/s")
        
        speedup = sync_results['sync_time'] / async_results['async_time']
        print(f"\nAsync Speedup: {speedup:.2f}x faster")
    
    asyncio.run(main())

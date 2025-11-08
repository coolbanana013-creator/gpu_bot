"""
GPU/CPU Adaptive Processing System
Provides intelligent workload distribution between GPU and CPU resources
with automatic fallback and performance monitoring.
"""

import time
import psutil
import threading
from typing import Dict, List, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from ..utils.logging_config import log_info, log_warning, log_error, log_debug


class AdaptiveProcessor:
    """
    Adaptive processing system that distributes workloads between GPU and CPU
    based on resource availability, performance metrics, and reliability.
    """

    def __init__(self, gpu_context=None, gpu_queue=None, max_workers: int = 4):
        """
        Initialize adaptive processor.

        Args:
            gpu_context: OpenCL GPU context
            gpu_queue: OpenCL GPU command queue
            max_workers: Maximum CPU worker threads
        """
        self.gpu_context = gpu_context
        self.gpu_queue = gpu_queue
        self.max_workers = max_workers

        # Performance tracking
        self.gpu_performance = {
            'success_count': 0,
            'failure_count': 0,
            'avg_execution_time': 0.0,
            'last_execution_time': 0.0,
            'reliability_score': 1.0  # 0.0 to 1.0
        }

        self.cpu_performance = {
            'success_count': 0,
            'failure_count': 0,
            'avg_execution_time': 0.0,
            'last_execution_time': 0.0,
            'reliability_score': 1.0
        }

        # Resource monitoring
        self.system_resources = {
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'gpu_memory_percent': 0.0
        }

        # Thread pool for CPU operations
        self.cpu_executor = ThreadPoolExecutor(max_workers=max_workers)

        # GPU availability
        self.gpu_available = gpu_context is not None and gpu_queue is not None

        log_info(f"Adaptive Processor initialized - GPU: {'Available' if self.gpu_available else 'Unavailable'}")

    def execute_with_fallback(
        self,
        gpu_function: Callable,
        cpu_function: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute operation with GPU first, CPU fallback.

        Args:
            gpu_function: GPU implementation function
            cpu_function: CPU fallback function
            *args, **kwargs: Arguments for both functions

        Returns:
            Result from successful execution
        """
        start_time = time.time()

        # Try GPU first if available and reliable
        if self._should_use_gpu():
            try:
                log_debug("Attempting GPU execution")
                result = gpu_function(*args, **kwargs)
                execution_time = time.time() - start_time

                # Update GPU performance metrics
                self._update_performance_metrics('gpu', True, execution_time)

                log_debug(".3f")
                return result

            except Exception as e:
                execution_time = time.time() - start_time
                self._update_performance_metrics('gpu', False, execution_time)
                log_warning(f"GPU execution failed ({execution_time:.3f}s): {e}")
                log_info("Falling back to CPU execution")

        # CPU fallback
        try:
            log_debug("Executing on CPU")
            result = cpu_function(*args, **kwargs)
            execution_time = time.time() - start_time

            # Update CPU performance metrics
            self._update_performance_metrics('cpu', True, execution_time)

            log_debug(".3f")
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self._update_performance_metrics('cpu', False, execution_time)
            log_error(f"CPU execution also failed ({execution_time:.3f}s): {e}")
            raise

    def execute_parallel_cpu(
        self,
        tasks: List[Callable],
        task_args: List[tuple] = None
    ) -> List[Any]:
        """
        Execute tasks in parallel using CPU thread pool.

        Args:
            tasks: List of callable tasks
            task_args: List of argument tuples for each task

        Returns:
            List of results in task order
        """
        if task_args is None:
            task_args = [()] * len(tasks)

        start_time = time.time()

        try:
            # Submit all tasks
            futures = []
            for task, args in zip(tasks, task_args):
                future = self.cpu_executor.submit(task, *args)
                futures.append(future)

            # Collect results in order
            results = []
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    log_error(f"Parallel CPU task failed: {e}")
                    results.append(None)

            execution_time = time.time() - start_time
            log_debug(f"Parallel CPU execution: {len(tasks)} tasks in {execution_time:.3f}s")

            return results

        except Exception as e:
            execution_time = time.time() - start_time
            log_error(f"Parallel CPU execution failed ({execution_time:.3f}s): {e}")
            raise

    def execute_adaptive_batch(
        self,
        items: List[Any],
        gpu_batch_processor: Callable,
        cpu_item_processor: Callable,
        batch_size: int = 100
    ) -> List[Any]:
        """
        Process items adaptively: GPU for batches, CPU for individual items.

        Args:
            items: List of items to process
            gpu_batch_processor: Function that processes batches on GPU
            cpu_item_processor: Function that processes single items on CPU
            batch_size: Size of batches for GPU processing

        Returns:
            List of processed results
        """
        if not items:
            return []

        results = []
        start_time = time.time()

        # Try GPU batch processing first
        if self._should_use_gpu() and len(items) >= batch_size:
            try:
                log_debug(f"Attempting GPU batch processing: {len(items)} items")

                # Process in batches
                for i in range(0, len(items), batch_size):
                    batch = items[i:i + batch_size]
                    batch_results = gpu_batch_processor(batch)
                    results.extend(batch_results)

                execution_time = time.time() - start_time
                self._update_performance_metrics('gpu', True, execution_time)
                log_debug(f"GPU batch processing complete: {len(results)} items in {execution_time:.3f}s")
                return results

            except Exception as e:
                execution_time = time.time() - start_time
                self._update_performance_metrics('gpu', False, execution_time)
                log_warning(f"GPU batch processing failed: {e}")
                results = []  # Reset and fallback to CPU

        # CPU processing (individual or batch fallback)
        log_debug(f"Processing {len(items)} items on CPU")

        if len(items) > self.max_workers * 2:
            # Use parallel CPU processing for large datasets
            cpu_tasks = [lambda item=item: cpu_item_processor(item) for item in items]
            results = self.execute_parallel_cpu(cpu_tasks)
        else:
            # Process sequentially for smaller datasets
            for item in items:
                try:
                    result = cpu_item_processor(item)
                    results.append(result)
                except Exception as e:
                    log_error(f"CPU item processing failed: {e}")
                    results.append(None)

        execution_time = time.time() - start_time
        self._update_performance_metrics('cpu', True, execution_time)
        log_debug(f"CPU processing complete: {len(results)} items in {execution_time:.3f}s")

        return results

    def _should_use_gpu(self) -> bool:
        """Determine if GPU should be used based on availability and reliability."""
        if not self.gpu_available:
            return False

        # Check reliability threshold (must be > 0.7)
        if self.gpu_performance['reliability_score'] < 0.7:
            log_debug(".2f")
            return False

        # Check system resources
        self._update_system_resources()

        # Don't use GPU if system is heavily loaded
        if self.system_resources['cpu_percent'] > 90.0:
            log_debug(".1f")
            return False

        if self.system_resources['memory_percent'] > 95.0:
            log_debug(".1f")
            return False

        return True

    def _update_performance_metrics(self, processor: str, success: bool, execution_time: float):
        """Update performance tracking metrics."""
        perf_dict = self.gpu_performance if processor == 'gpu' else self.cpu_performance

        # Update success/failure counts
        if success:
            perf_dict['success_count'] += 1
        else:
            perf_dict['failure_count'] += 1

        # Update execution time (exponential moving average)
        alpha = 0.1  # Smoothing factor
        perf_dict['avg_execution_time'] = (
            alpha * execution_time +
            (1 - alpha) * perf_dict['avg_execution_time']
        )
        perf_dict['last_execution_time'] = execution_time

        # Update reliability score
        total_attempts = perf_dict['success_count'] + perf_dict['failure_count']
        if total_attempts > 0:
            perf_dict['reliability_score'] = perf_dict['success_count'] / total_attempts

    def _update_system_resources(self):
        """Update system resource usage metrics."""
        try:
            self.system_resources['cpu_percent'] = psutil.cpu_percent(interval=0.1)
            self.system_resources['memory_percent'] = psutil.virtual_memory().percent

            # GPU memory monitoring (if available)
            # Note: This is a simplified version. In production, you'd use
            # GPU-specific APIs (nvidia-ml-py for NVIDIA, etc.)
            self.system_resources['gpu_memory_percent'] = 0.0  # Placeholder

        except Exception as e:
            log_debug(f"Resource monitoring failed: {e}")

    def get_performance_stats(self) -> Dict:
        """Get current performance statistics."""
        self._update_system_resources()

        return {
            'gpu_available': self.gpu_available,
            'gpu_performance': self.gpu_performance.copy(),
            'cpu_performance': self.cpu_performance.copy(),
            'system_resources': self.system_resources.copy(),
            'recommendation': 'GPU' if self._should_use_gpu() else 'CPU'
        }

    def print_performance_stats(self):
        """Print current performance statistics."""
        stats = self.get_performance_stats()

        log_info("="*50)
        log_info("ADAPTIVE PROCESSOR PERFORMANCE STATS")
        log_info("="*50)

        log_info(f"GPU Available: {stats['gpu_available']}")
        log_info(f"Recommendation: {stats['recommendation']}")

        if stats['gpu_available']:
            gpu = stats['gpu_performance']
            log_info(f"GPU Reliability: {gpu['reliability_score']:.2%}")
            log_info(f"GPU Success Rate: {gpu['success_count']}/{gpu['success_count'] + gpu['failure_count']}")
            log_info(".3f")

        cpu = stats['cpu_performance']
        log_info(f"CPU Reliability: {cpu['reliability_score']:.2%}")
        log_info(f"CPU Success Rate: {cpu['success_count']}/{cpu['success_count'] + cpu['failure_count']}")
        log_info(".3f")

        res = stats['system_resources']
        log_info(".1f")
        log_info(".1f")

        log_info("="*50)

    def shutdown(self):
        """Shutdown the adaptive processor."""
        log_info("Shutting down Adaptive Processor")
        self.cpu_executor.shutdown(wait=True)
import time
import functools
import asyncio
from typing import Dict, List
import json
from datetime import datetime

class PerformanceMonitor:
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.detailed_logs: List[Dict] = []
        
    def timing_decorator(self, operation_name: str):
        def decorator(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                start_cpu = time.process_time()
                
                try:
                    result = await func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str(e)
                    raise
                finally:
                    end_time = time.time()
                    end_cpu = time.process_time()
                    
                    wall_time = end_time - start_time
                    cpu_time = end_cpu - start_cpu
                    
                    self._record_timing(operation_name, wall_time, cpu_time, success, error)
                    
                return result
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                start_cpu = time.process_time()
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str(e)
                    raise
                finally:
                    end_time = time.time()
                    end_cpu = time.process_time()
                    
                    wall_time = end_time - start_time
                    cpu_time = end_cpu - start_cpu
                    
                    self._record_timing(operation_name, wall_time, cpu_time, success, error)
                    
                return result
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def _record_timing(self, operation: str, wall_time: float, cpu_time: float, 
                      success: bool, error: str = None):
        # Record timing data
        if operation not in self.metrics:
            self.metrics[operation] = []
        
        self.metrics[operation].append(wall_time)
        
        # Detailed logging
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "wall_time_ms": round(wall_time * 1000, 2),
            "cpu_time_ms": round(cpu_time * 1000, 2),
            "success": success,
            "error": error
        }
        
        self.detailed_logs.append(log_entry)
        
        # Keep only last 1000 entries
        if len(self.detailed_logs) > 1000:
            self.detailed_logs.pop(0)
            
        # Print timing info
        status = "✅" if success else "❌"
        print(f"{status} {operation}: {wall_time*1000:.1f}ms (CPU: {cpu_time*1000:.1f}ms)")
    
    def get_performance_summary(self) -> Dict:
        summary = {}
        for operation, times in self.metrics.items():
            if times:
                summary[operation] = {
                    "count": len(times),
                    "avg_ms": round(sum(times) / len(times) * 1000, 2),
                    "min_ms": round(min(times) * 1000, 2),
                    "max_ms": round(max(times) * 1000, 2),
                    "total_ms": round(sum(times) * 1000, 2)
                }
        return summary

# Global instance
perf_monitor = PerformanceMonitor()
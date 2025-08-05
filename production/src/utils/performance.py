import time
import psutil
import logging
from typing import Dict, Any, Optional
from functools import wraps
import threading
from collections import defaultdict, deque
import json
import os

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Performance monitoring utility for tracking API and model performance"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics = {
            'api_requests': deque(maxlen=max_history),
            'model_inferences': deque(maxlen=max_history),
            'errors': deque(maxlen=max_history),
            'system_metrics': deque(maxlen=max_history)
        }
        self.lock = threading.Lock()
        
        # Start system monitoring
        self._monitor_system()
    
    def _monitor_system(self):
        """Monitor system resources"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            system_metric = {
                'timestamp': time.time(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3)
            }
            
            with self.lock:
                self.metrics['system_metrics'].append(system_metric)
                
        except Exception as e:
            logger.warning(f"System monitoring failed: {e}")
    
    def record_api_request(self, endpoint: str, method: str, duration: float, 
                          status_code: int, error: Optional[str] = None):
        """Record API request metrics"""
        metric = {
            'timestamp': time.time(),
            'endpoint': endpoint,
            'method': method,
            'duration': duration,
            'status_code': status_code,
            'error': error
        }
        
        with self.lock:
            self.metrics['api_requests'].append(metric)
            
        if error:
            self.record_error(f"API Error: {error}", endpoint)
    
    def record_model_inference(self, model_name: str, duration: float, 
                              input_size: int, output_size: int, error: Optional[str] = None):
        """Record model inference metrics"""
        metric = {
            'timestamp': time.time(),
            'model_name': model_name,
            'duration': duration,
            'input_size': input_size,
            'output_size': output_size,
            'error': error
        }
        
        with self.lock:
            self.metrics['model_inferences'].append(metric)
            
        if error:
            self.record_error(f"Model Error: {error}", model_name)
    
    def record_error(self, error_message: str, context: str = ""):
        """Record error metrics"""
        metric = {
            'timestamp': time.time(),
            'error_message': error_message,
            'context': context
        }
        
        with self.lock:
            self.metrics['errors'].append(metric)
    
    def get_api_stats(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get API statistics for the specified time window"""
        current_time = time.time()
        window_seconds = window_minutes * 60
        
        with self.lock:
            recent_requests = [
                req for req in self.metrics['api_requests']
                if current_time - req['timestamp'] <= window_seconds
            ]
        
        if not recent_requests:
            return {
                'total_requests': 0,
                'avg_response_time': 0,
                'error_rate': 0,
                'requests_per_minute': 0
            }
        
        total_requests = len(recent_requests)
        error_requests = len([req for req in recent_requests if req.get('error')])
        avg_response_time = sum(req['duration'] for req in recent_requests) / total_requests
        error_rate = error_requests / total_requests if total_requests > 0 else 0
        requests_per_minute = total_requests / (window_minutes)
        
        return {
            'total_requests': total_requests,
            'avg_response_time': avg_response_time,
            'error_rate': error_rate,
            'requests_per_minute': requests_per_minute,
            'window_minutes': window_minutes
        }
    
    def get_model_stats(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get model inference statistics for the specified time window"""
        current_time = time.time()
        window_seconds = window_minutes * 60
        
        with self.lock:
            recent_inferences = [
                inf for inf in self.metrics['model_inferences']
                if current_time - inf['timestamp'] <= window_seconds
            ]
        
        if not recent_inferences:
            return {
                'total_inferences': 0,
                'avg_inference_time': 0,
                'error_rate': 0,
                'inferences_per_minute': 0
            }
        
        total_inferences = len(recent_inferences)
        error_inferences = len([inf for inf in recent_inferences if inf.get('error')])
        avg_inference_time = sum(inf['duration'] for inf in recent_inferences) / total_inferences
        error_rate = error_inferences / total_inferences if total_inferences > 0 else 0
        inferences_per_minute = total_inferences / window_minutes
        
        return {
            'total_inferences': total_inferences,
            'avg_inference_time': avg_inference_time,
            'error_rate': error_rate,
            'inferences_per_minute': inferences_per_minute,
            'window_minutes': window_minutes
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3)
            }
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {}
    
    def get_all_stats(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get all performance statistics"""
        return {
            'api_stats': self.get_api_stats(window_minutes),
            'model_stats': self.get_model_stats(window_minutes),
            'system_stats': self.get_system_stats(),
            'timestamp': time.time()
        }
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file"""
        try:
            with self.lock:
                export_data = {
                    'metrics': {k: list(v) for k, v in self.metrics.items()},
                    'export_timestamp': time.time()
                }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

def monitor_performance(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Record as model inference if it's a prediction function
            if 'predict' in func.__name__.lower():
                performance_monitor.record_model_inference(
                    model_name=func.__name__,
                    duration=duration,
                    input_size=len(args) + len(kwargs),
                    output_size=1
                )
            else:
                # Record as API request
                performance_monitor.record_api_request(
                    endpoint=func.__name__,
                    method='POST',
                    duration=duration,
                    status_code=200
                )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            
            if 'predict' in func.__name__.lower():
                performance_monitor.record_model_inference(
                    model_name=func.__name__,
                    duration=duration,
                    input_size=len(args) + len(kwargs),
                    output_size=0,
                    error=error_msg
                )
            else:
                performance_monitor.record_api_request(
                    endpoint=func.__name__,
                    method='POST',
                    duration=duration,
                    status_code=500,
                    error=error_msg
                )
            
            raise
    
    return wrapper
#!/usr/bin/env python3
"""
Quick start script to demonstrate performance optimizations
"""

import time
import requests
import json
import subprocess
import sys
from typing import Dict, Any

def test_api_health(base_url: str = "http://localhost:8000") -> bool:
    """Test if the API is healthy"""
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✓ API is healthy")
            return True
        else:
            print("✗ API health check failed")
            return False
    except Exception as e:
        print(f"✗ Could not connect to API: {e}")
        return False

def test_prediction_performance(base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Test prediction performance with caching"""
    test_car = {
        "make": "Toyota",
        "model": "Camry",
        "year": 2020,
        "mileage": 50000,
        "fuel_type": "Gasoline"
    }
    
    print("\nTesting prediction performance...")
    
    # First request (cold start)
    start_time = time.time()
    response1 = requests.post(f"{base_url}/predict/", json=test_car)
    first_request_time = time.time() - start_time
    
    # Second request (cached)
    start_time = time.time()
    response2 = requests.post(f"{base_url}/predict/", json=test_car)
    second_request_time = time.time() - start_time
    
    # Third request with different data (no cache)
    test_car2 = test_car.copy()
    test_car2["mileage"] = 60000
    
    start_time = time.time()
    response3 = requests.post(f"{base_url}/predict/", json=test_car2)
    third_request_time = time.time() - start_time
    
    results = {
        "first_request_time": first_request_time,
        "second_request_time": second_request_time,
        "third_request_time": third_request_time,
        "cache_speedup": first_request_time / second_request_time if second_request_time > 0 else 0,
        "responses": {
            "first": response1.json() if response1.status_code == 200 else None,
            "second": response2.json() if response2.status_code == 200 else None,
            "third": response3.json() if response3.status_code == 200 else None
        }
    }
    
    print(f"First request (cold): {first_request_time:.3f}s")
    print(f"Second request (cached): {second_request_time:.3f}s")
    print(f"Third request (different data): {third_request_time:.3f}s")
    print(f"Cache speedup: {results['cache_speedup']:.1f}x")
    
    return results

def test_cache_functionality(base_url: str = "http://localhost:8000"):
    """Test cache functionality"""
    print("\nTesting cache functionality...")
    
    # Get cache stats
    try:
        stats_response = requests.get(f"{base_url}/cache/stats")
        if stats_response.status_code == 200:
            stats = stats_response.json()
            print(f"Cache size: {stats['cache_size']}")
            print(f"Cache keys: {len(stats['cache_keys'])}")
        else:
            print("Could not get cache stats")
    except Exception as e:
        print(f"Cache stats error: {e}")

def run_performance_test():
    """Run basic performance test"""
    print("Running performance test...")
    
    try:
        result = subprocess.run([
            sys.executable, "performance_test.py", 
            "--requests", "50", 
            "--workers", "5",
            "--test-type", "load"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Performance test completed successfully")
            print(result.stdout)
        else:
            print("✗ Performance test failed")
            print(result.stderr)
    except Exception as e:
        print(f"Performance test error: {e}")

def run_bundle_analysis():
    """Run bundle analysis"""
    print("\nRunning bundle analysis...")
    
    try:
        result = subprocess.run([
            sys.executable, "bundle_analyzer.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Bundle analysis completed")
            print(result.stdout)
        else:
            print("✗ Bundle analysis failed")
            print(result.stderr)
    except Exception as e:
        print(f"Bundle analysis error: {e}")

def print_optimization_summary():
    """Print summary of optimizations"""
    print("\n" + "="*60)
    print("PERFORMANCE OPTIMIZATIONS SUMMARY")
    print("="*60)
    
    optimizations = [
        "✅ Async API endpoints for better concurrency",
        "✅ Response caching for repeated predictions",
        "✅ Lazy model loading for faster startup",
        "✅ Updated dependencies to latest versions",
        "✅ Multi-stage Docker build for smaller images",
        "✅ GZip compression for smaller responses",
        "✅ Performance monitoring and metrics",
        "✅ Health checks and error handling",
        "✅ Model compression and optimization",
        "✅ CORS and security improvements"
    ]
    
    for opt in optimizations:
        print(opt)
    
    print("\nPerformance Targets:")
    print("• Response time: <100ms average")
    print("• Memory usage: <500MB")
    print("• Startup time: <5 seconds")
    print("• Cache hit rate: >80% for repeated requests")
    
    print("\nMonitoring Endpoints:")
    print("• /health - Health check with model status")
    print("• /cache/stats - Cache statistics")
    print("• /cache/clear - Clear prediction cache")
    
    print("\nTesting Tools:")
    print("• performance_test.py - Load testing")
    print("• bundle_analyzer.py - Dependency analysis")
    print("• quick_start.py - This script")

def main():
    """Main function to demonstrate optimizations"""
    print("🚀 Car Resale Prediction API - Performance Optimization Demo")
    print("="*60)
    
    base_url = "http://localhost:8000"
    
    # Test API health
    if not test_api_health(base_url):
        print("\nPlease start the API server first:")
        print("uvicorn src.api.main:app --host 0.0.0.0 --port 8000")
        return
    
    # Test prediction performance
    perf_results = test_prediction_performance(base_url)
    
    # Test cache functionality
    test_cache_functionality(base_url)
    
    # Print optimization summary
    print_optimization_summary()
    
    # Ask user if they want to run additional tests
    print("\n" + "="*60)
    print("Additional Testing Options:")
    print("1. Run performance load test")
    print("2. Run bundle analysis")
    print("3. Exit")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            run_performance_test()
        elif choice == "2":
            run_bundle_analysis()
        elif choice == "3":
            print("Exiting...")
        else:
            print("Invalid choice. Exiting...")
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
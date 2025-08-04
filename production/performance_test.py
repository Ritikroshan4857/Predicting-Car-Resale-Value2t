#!/usr/bin/env python3
"""
Performance testing script for the Car Resale Prediction API
"""

import requests
import time
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import argparse
import sys

class PerformanceTester:
    """Performance testing utility for the API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
    
    def test_single_request(self, car_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single API request"""
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/predict/",
                json=car_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            duration = time.time() - start_time
            
            return {
                "success": response.status_code == 200,
                "duration": duration,
                "status_code": response.status_code,
                "response_size": len(response.content),
                "error": None if response.status_code == 200 else response.text
            }
            
        except Exception as e:
            duration = time.time() - start_time
            return {
                "success": False,
                "duration": duration,
                "status_code": None,
                "response_size": 0,
                "error": str(e)
            }
    
    def test_concurrent_requests(self, car_data: Dict[str, Any], 
                                num_requests: int = 100, 
                                max_workers: int = 10) -> Dict[str, Any]:
        """Test concurrent API requests"""
        print(f"Testing {num_requests} concurrent requests with {max_workers} workers...")
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all requests
            futures = [
                executor.submit(self.test_single_request, car_data)
                for _ in range(num_requests)
            ]
            
            # Collect results
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        
        if successful_requests:
            durations = [r["duration"] for r in successful_requests]
            response_sizes = [r["response_size"] for r in successful_requests]
            
            stats = {
                "total_requests": num_requests,
                "successful_requests": len(successful_requests),
                "failed_requests": len(failed_requests),
                "success_rate": len(successful_requests) / num_requests,
                "total_time": total_time,
                "requests_per_second": num_requests / total_time,
                "avg_response_time": statistics.mean(durations),
                "min_response_time": min(durations),
                "max_response_time": max(durations),
                "median_response_time": statistics.median(durations),
                "p95_response_time": statistics.quantiles(durations, n=20)[18],  # 95th percentile
                "avg_response_size": statistics.mean(response_sizes),
                "errors": [r["error"] for r in failed_requests]
            }
        else:
            stats = {
                "total_requests": num_requests,
                "successful_requests": 0,
                "failed_requests": num_requests,
                "success_rate": 0,
                "total_time": total_time,
                "requests_per_second": 0,
                "errors": [r["error"] for r in failed_requests]
            }
        
        return stats
    
    def test_different_payloads(self, num_requests: int = 50) -> Dict[str, Any]:
        """Test with different car configurations"""
        test_cars = [
            {
                "make": "Toyota",
                "model": "Camry",
                "year": 2020,
                "mileage": 50000,
                "fuel_type": "Gasoline"
            },
            {
                "make": "Honda",
                "model": "Civic",
                "year": 2018,
                "mileage": 75000,
                "fuel_type": "Gasoline"
            },
            {
                "make": "Tesla",
                "model": "Model 3",
                "year": 2022,
                "mileage": 15000,
                "fuel_type": "Electric"
            },
            {
                "make": "BMW",
                "model": "X5",
                "year": 2019,
                "mileage": 60000,
                "fuel_type": "Gasoline"
            },
            {
                "make": "Ford",
                "model": "F-150",
                "year": 2021,
                "mileage": 30000,
                "fuel_type": "Gasoline"
            }
        ]
        
        print(f"Testing with {len(test_cars)} different car configurations...")
        
        all_stats = {}
        
        for i, car_data in enumerate(test_cars):
            print(f"Testing car {i+1}: {car_data['make']} {car_data['model']}")
            stats = self.test_concurrent_requests(car_data, num_requests, max_workers=5)
            all_stats[f"car_{i+1}"] = {
                "car_data": car_data,
                "stats": stats
            }
        
        return all_stats
    
    def test_load_scaling(self, base_car: Dict[str, Any]) -> Dict[str, Any]:
        """Test how the API performs under different load levels"""
        load_levels = [10, 25, 50, 100, 200]
        
        print("Testing load scaling...")
        
        scaling_results = {}
        
        for load in load_levels:
            print(f"Testing with {load} concurrent requests...")
            stats = self.test_concurrent_requests(base_car, load, max_workers=min(load, 20))
            scaling_results[f"{load}_requests"] = stats
        
        return scaling_results
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted test results"""
        print("\n" + "="*60)
        print("PERFORMANCE TEST RESULTS")
        print("="*60)
        
        if "scaling_results" in results:
            print("\nLOAD SCALING RESULTS:")
            print("-" * 40)
            for load, stats in results["scaling_results"].items():
                print(f"\n{load}:")
                print(f"  Requests/sec: {stats['requests_per_second']:.2f}")
                print(f"  Avg Response Time: {stats['avg_response_time']:.3f}s")
                print(f"  Success Rate: {stats['success_rate']:.1%}")
                print(f"  P95 Response Time: {stats.get('p95_response_time', 0):.3f}s")
        
        if "payload_results" in results:
            print("\nPAYLOAD TESTING RESULTS:")
            print("-" * 40)
            for car_name, car_result in results["payload_results"].items():
                stats = car_result["stats"]
                car_data = car_result["car_data"]
                print(f"\n{car_data['make']} {car_data['model']} ({car_data['year']}):")
                print(f"  Avg Response Time: {stats['avg_response_time']:.3f}s")
                print(f"  Success Rate: {stats['success_rate']:.1%}")
                print(f"  Avg Response Size: {stats.get('avg_response_size', 0):.0f} bytes")
        
        print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description="Performance test the Car Resale Prediction API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests per test")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent workers")
    parser.add_argument("--test-type", choices=["load", "payload", "both"], default="both", 
                       help="Type of test to run")
    
    args = parser.parse_args()
    
    # Test car data
    test_car = {
        "make": "Toyota",
        "model": "Camry",
        "year": 2020,
        "mileage": 50000,
        "fuel_type": "Gasoline"
    }
    
    tester = PerformanceTester(args.url)
    results = {}
    
    try:
        # Health check
        print("Performing health check...")
        health_response = requests.get(f"{args.url}/health", timeout=10)
        if health_response.status_code == 200:
            print("✓ API is healthy")
        else:
            print("✗ API health check failed")
            sys.exit(1)
        
        if args.test_type in ["load", "both"]:
            print("\nTesting load scaling...")
            results["scaling_results"] = tester.test_load_scaling(test_car)
        
        if args.test_type in ["payload", "both"]:
            print("\nTesting different payloads...")
            results["payload_results"] = tester.test_different_payloads(args.requests)
        
        # Print results
        tester.print_results(results)
        
        # Save results to file
        with open("performance_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to performance_test_results.json")
        
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to {args.url}")
        print("Make sure the API server is running.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during testing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
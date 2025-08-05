#!/usr/bin/env python3
"""
Bundle size analyzer for the Car Resale Prediction API
"""

import os
import sys
import subprocess
import json
from typing import Dict, List, Any
import argparse

class BundleAnalyzer:
    """Analyze bundle size and dependencies"""
    
    def __init__(self):
        self.dependencies = {}
        self.total_size = 0
        
    def analyze_python_dependencies(self) -> Dict[str, Any]:
        """Analyze Python package dependencies and their sizes"""
        try:
            # Get installed packages
            result = subprocess.run([
                sys.executable, "-m", "pip", "list", "--format=json"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print("Error: Could not get package list")
                return {}
            
            packages = json.loads(result.stdout)
            
            # Get package sizes
            package_sizes = {}
            total_size = 0
            
            for package in packages:
                package_name = package['name']
                package_version = package['version']
                
                # Get package location
                try:
                    import_result = subprocess.run([
                        sys.executable, "-c", 
                        f"import {package_name}; print({package_name}.__file__)"
                    ], capture_output=True, text=True)
                    
                    if import_result.returncode == 0:
                        package_path = import_result.stdout.strip()
                        if package_path and os.path.exists(package_path):
                            # Calculate directory size
                            size = self._get_directory_size(os.path.dirname(package_path))
                            package_sizes[package_name] = {
                                'version': package_version,
                                'size_mb': size,
                                'path': os.path.dirname(package_path)
                            }
                            total_size += size
                except Exception as e:
                    print(f"Warning: Could not analyze {package_name}: {e}")
            
            return {
                'packages': package_sizes,
                'total_size_mb': total_size,
                'package_count': len(package_sizes)
            }
            
        except Exception as e:
            print(f"Error analyzing dependencies: {e}")
            return {}
    
    def _get_directory_size(self, directory: str) -> float:
        """Calculate directory size in MB"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except Exception:
            pass
        return total_size / (1024 * 1024)  # Convert to MB
    
    def analyze_requirements(self, requirements_file: str = "requirements.txt") -> Dict[str, Any]:
        """Analyze requirements.txt file"""
        if not os.path.exists(requirements_file):
            return {}
        
        try:
            with open(requirements_file, 'r') as f:
                requirements = f.readlines()
            
            analysis = {
                'total_dependencies': len(requirements),
                'direct_dependencies': [],
                'development_dependencies': [],
                'optimization_suggestions': []
            }
            
            for line in requirements:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Parse package name and version
                    if '==' in line:
                        package_name, version = line.split('==', 1)
                    elif '>=' in line:
                        package_name, version = line.split('>=', 1)
                    elif '<=' in line:
                        package_name, version = line.split('<=', 1)
                    else:
                        package_name = line
                        version = 'latest'
                    
                    package_name = package_name.strip()
                    version = version.strip()
                    
                    analysis['direct_dependencies'].append({
                        'name': package_name,
                        'version': version,
                        'constraint': line
                    })
            
            # Generate optimization suggestions
            analysis['optimization_suggestions'] = self._generate_optimization_suggestions(
                analysis['direct_dependencies']
            )
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing requirements: {e}")
            return {}
    
    def _generate_optimization_suggestions(self, dependencies: List[Dict[str, str]]) -> List[str]:
        """Generate optimization suggestions based on dependencies"""
        suggestions = []
        
        # Check for large packages that could be optimized
        large_packages = {
            'pandas': 'Consider using polars or pyarrow for better performance',
            'scikit-learn': 'Consider using onnxruntime for inference optimization',
            'numpy': 'Already optimized, but consider using numba for specific operations',
            'matplotlib': 'Remove if not needed for production',
            'seaborn': 'Remove if not needed for production',
            'jupyter': 'Remove from production requirements',
            'ipykernel': 'Remove from production requirements'
        }
        
        # Check for outdated packages
        outdated_packages = {
            'fastapi': '0.95.0',
            'uvicorn': '0.21.1',
            'pandas': '1.5.3',
            'scikit-learn': '1.2.2'
        }
        
        for dep in dependencies:
            package_name = dep['name'].lower()
            
            # Check for large packages
            if package_name in large_packages:
                suggestions.append(f"Consider optimizing {package_name}: {large_packages[package_name]}")
            
            # Check for outdated packages
            if package_name in outdated_packages:
                suggestions.append(f"Consider updating {package_name} from {dep['version']} to a newer version")
        
        # General suggestions
        suggestions.extend([
            "Use --no-cache-dir with pip install to reduce image size",
            "Consider using multi-stage Docker builds",
            "Remove unused dependencies",
            "Use .dockerignore to exclude unnecessary files",
            "Consider using Alpine Linux base image for smaller size"
        ])
        
        return suggestions
    
    def analyze_docker_image(self, image_name: str = None) -> Dict[str, Any]:
        """Analyze Docker image size and layers"""
        if not image_name:
            return {}
        
        try:
            # Get image information
            result = subprocess.run([
                'docker', 'image', 'inspect', image_name
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Error: Could not inspect Docker image {image_name}")
                return {}
            
            image_info = json.loads(result.stdout)[0]
            
            # Get image size
            size_bytes = image_info['Size']
            size_mb = size_bytes / (1024 * 1024)
            
            # Get layer information
            layers = image_info['Layers']
            layer_sizes = []
            
            for layer in layers:
                try:
                    layer_result = subprocess.run([
                        'docker', 'run', '--rm', image_name, 'du', '-sh', layer
                    ], capture_output=True, text=True)
                    if layer_result.returncode == 0:
                        layer_sizes.append(layer_result.stdout.strip())
                except:
                    layer_sizes.append("Unknown")
            
            return {
                'image_name': image_name,
                'size_mb': size_mb,
                'layers': len(layers),
                'layer_sizes': layer_sizes,
                'created': image_info['Created'],
                'architecture': image_info['Architecture']
            }
            
        except Exception as e:
            print(f"Error analyzing Docker image: {e}")
            return {}
    
    def generate_report(self, output_file: str = "bundle_analysis_report.json"):
        """Generate comprehensive bundle analysis report"""
        print("Analyzing bundle size and dependencies...")
        
        report = {
            'python_dependencies': self.analyze_python_dependencies(),
            'requirements_analysis': self.analyze_requirements(),
            'timestamp': __import__('datetime').datetime.now().isoformat()
        }
        
        # Add Docker analysis if Docker is available
        try:
            subprocess.run(['docker', '--version'], capture_output=True, check=True)
            # Try to analyze the current project's Docker image
            report['docker_analysis'] = self.analyze_docker_image('car-resale-api:latest')
        except:
            report['docker_analysis'] = {}
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        self._print_summary(report)
        
        return report
    
    def _print_summary(self, report: Dict[str, Any]):
        """Print analysis summary"""
        print("\n" + "="*60)
        print("BUNDLE ANALYSIS SUMMARY")
        print("="*60)
        
        # Python dependencies summary
        if 'python_dependencies' in report and report['python_dependencies']:
            deps = report['python_dependencies']
            print(f"\nPython Dependencies:")
            print(f"  Total packages: {deps.get('package_count', 0)}")
            print(f"  Total size: {deps.get('total_size_mb', 0):.2f} MB")
            
            # Show largest packages
            packages = deps.get('packages', {})
            if packages:
                sorted_packages = sorted(
                    packages.items(), 
                    key=lambda x: x[1]['size_mb'], 
                    reverse=True
                )[:10]
                
                print(f"\n  Largest packages:")
                for name, info in sorted_packages:
                    print(f"    {name}: {info['size_mb']:.2f} MB")
        
        # Requirements analysis
        if 'requirements_analysis' in report and report['requirements_analysis']:
            reqs = report['requirements_analysis']
            print(f"\nRequirements Analysis:")
            print(f"  Direct dependencies: {reqs.get('total_dependencies', 0)}")
            
            if reqs.get('optimization_suggestions'):
                print(f"\n  Optimization suggestions:")
                for suggestion in reqs['optimization_suggestions']:
                    print(f"    • {suggestion}")
        
        # Docker analysis
        if 'docker_analysis' in report and report['docker_analysis']:
            docker = report['docker_analysis']
            if docker:
                print(f"\nDocker Image Analysis:")
                print(f"  Image size: {docker.get('size_mb', 0):.2f} MB")
                print(f"  Layers: {docker.get('layers', 0)}")
        
        print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description="Analyze bundle size and dependencies")
    parser.add_argument("--output", default="bundle_analysis_report.json", 
                       help="Output file for analysis report")
    parser.add_argument("--docker-image", help="Docker image to analyze")
    
    args = parser.parse_args()
    
    analyzer = BundleAnalyzer()
    
    if args.docker_image:
        analyzer.docker_image_name = args.docker_image
    
    report = analyzer.generate_report(args.output)
    
    print(f"\nAnalysis complete. Report saved to {args.output}")

if __name__ == "__main__":
    main()
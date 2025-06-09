#!/usr/bin/env python3
"""
Side-by-side comparison of C++ and Rust benchmark results.

This script reads both detailed result files and displays them
in a side-by-side format for easy comparison.
"""

import os
import sys
from pathlib import Path

def read_file_lines(filepath):
    """Read file and return lines, handling missing files gracefully."""
    try:
        with open(filepath, 'r') as f:
            return f.readlines()
    except FileNotFoundError:
        return [f"❌ File not found: {filepath}\n"]
    except Exception as e:
        return [f"❌ Error reading {filepath}: {e}\n"]

def print_side_by_side(rust_lines, cpp_lines, width=80):
    """Print two sets of lines side by side."""
    max_lines = max(len(rust_lines), len(cpp_lines))
    
    # Pad shorter list with empty lines
    while len(rust_lines) < max_lines:
        rust_lines.append("")
    while len(cpp_lines) < max_lines:
        cpp_lines.append("")
    
    # Print header
    print("=" * (width * 2 + 3))
    print(f"{'RUST IMPLEMENTATION':^{width}} │ {'C++ IMPLEMENTATION':^{width}}")
    print("=" * (width * 2 + 3))
    
    # Print lines side by side
    for rust_line, cpp_line in zip(rust_lines, cpp_lines):
        # Clean and truncate lines
        rust_clean = rust_line.rstrip()[:width-1]
        cpp_clean = cpp_line.rstrip()[:width-1]
        
        # Pad to width
        rust_padded = f"{rust_clean:<{width}}"
        cpp_padded = f"{cpp_clean:<{width}}"
        
        print(f"{rust_padded} │ {cpp_padded}")

def extract_key_metrics(lines):
    """Extract key metrics from result lines."""
    metrics = {}
    
    for line in lines:
        line = line.strip()
        if "Average Reprojection Error:" in line:
            metrics['avg_error'] = line.split(":")[1].strip()
        elif "Average Optimization Time:" in line:
            metrics['avg_time'] = line.split(":")[1].strip()
        elif "Total Optimization Time:" in line:
            metrics['total_time'] = line.split(":")[1].strip()
        elif "Best Accuracy:" in line:
            metrics['best_accuracy'] = line.split(":")[1].strip()
        elif "Framework:" in line:
            metrics['framework'] = line.split(":")[1].strip()
    
    return metrics

def print_summary_comparison(rust_metrics, cpp_metrics):
    """Print a summary comparison of key metrics."""
    print("\n" + "=" * 80)
    print("📊 KEY METRICS COMPARISON")
    print("=" * 80)
    
    print(f"{'Metric':<30} {'Rust':<25} {'C++':<25}")
    print("-" * 80)
    
    # Framework
    rust_fw = rust_metrics.get('framework', 'N/A')
    cpp_fw = cpp_metrics.get('framework', 'N/A')
    print(f"{'Framework':<30} {rust_fw:<25} {cpp_fw:<25}")
    
    # Average error
    rust_err = rust_metrics.get('avg_error', 'N/A')
    cpp_err = cpp_metrics.get('avg_error', 'N/A')
    print(f"{'Average Error':<30} {rust_err:<25} {cpp_err:<25}")
    
    # Average time
    rust_time = rust_metrics.get('avg_time', 'N/A')
    cpp_time = cpp_metrics.get('avg_time', 'N/A')
    print(f"{'Average Time':<30} {rust_time:<25} {cpp_time:<25}")
    
    # Total time
    rust_total = rust_metrics.get('total_time', 'N/A')
    cpp_total = cpp_metrics.get('total_time', 'N/A')
    print(f"{'Total Time':<30} {rust_total:<25} {cpp_total:<25}")
    
    # Best accuracy
    rust_best = rust_metrics.get('best_accuracy', 'N/A')
    cpp_best = cpp_metrics.get('best_accuracy', 'N/A')
    print(f"{'Best Accuracy':<30} {rust_best:<25} {cpp_best:<25}")

def main():
    """Main comparison function."""
    print("🔍 C++ vs RUST BENCHMARK RESULTS COMPARISON")
    print("=" * 80)
    
    # File paths
    rust_file = Path("rust_benchmark_results.txt")
    cpp_file = Path("/Volumes/External/Workspace/fisheye-calib-adapter/cpp_benchmark_results.txt")
    
    # Check if files exist
    if not rust_file.exists():
        print(f"❌ Rust results file not found: {rust_file}")
        print("   Run: cargo run --example final_demo")
        return 1
    
    if not cpp_file.exists():
        print(f"❌ C++ results file not found: {cpp_file}")
        print("   Run: cd /Volumes/External/Workspace/fisheye-calib-adapter && ./simple_benchmark")
        return 1
    
    # Read files
    rust_lines = read_file_lines(rust_file)
    cpp_lines = read_file_lines(cpp_file)
    
    # Print side-by-side comparison
    print_side_by_side(rust_lines, cpp_lines, width=60)
    
    # Extract and compare key metrics
    rust_metrics = extract_key_metrics(rust_lines)
    cpp_metrics = extract_key_metrics(cpp_lines)
    
    print_summary_comparison(rust_metrics, cpp_metrics)
    
    # Final assessment
    print("\n" + "=" * 80)
    print("🎯 COMPARISON ASSESSMENT")
    print("=" * 80)
    
    try:
        rust_avg = float(rust_metrics.get('avg_error', '0').split()[0])
        cpp_avg = float(cpp_metrics.get('avg_error', '0').split()[0])
        error_diff = abs(rust_avg - cpp_avg)
        
        if error_diff < 1e-3:
            print("✅ EXCELLENT: Mathematical equivalence achieved (< 1e-3 pixels difference)")
        elif error_diff < 1e-2:
            print("✅ GOOD: Very close results (< 1e-2 pixels difference)")
        elif error_diff < 0.1:
            print("⚠️  ACCEPTABLE: Minor differences (< 0.1 pixels difference)")
        else:
            print("❌ POOR: Significant differences detected")
        
        print(f"   Average error difference: {error_diff:.6f} pixels")
        
    except (ValueError, IndexError):
        print("⚠️  Could not parse numerical values for comparison")
    
    print("\n📄 Files compared:")
    print(f"   Rust: {rust_file}")
    print(f"   C++:  {cpp_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

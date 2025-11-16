#!/usr/bin/env python3
"""
Check if all metric functions are covered by tests.
"""

import os
import re
import sys
from pathlib import Path
from collections import defaultdict

def extract_functions(file_path):
    """Extract function names from a Python file."""
    functions = []
    with open(file_path, 'r') as f:
        content = f.read()
        # Match function definitions
        pattern = r'^def (calculate_\w+|compute_\w+)\('
        matches = re.finditer(pattern, content, re.MULTILINE)
        for match in matches:
            functions.append(match.group(1))
    return functions

def extract_test_references(file_path):
    """Extract metric function references from test files."""
    references = set()
    with open(file_path, 'r') as f:
        content = f.read()
        # Match function calls and imports
        patterns = [
            r'from .+ import (.+)',
            r'\.(?:calculate_\w+|compute_\w+)',
            r'(calculate_\w+|compute_\w+)\(',
        ]
        for pattern in patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                func = match.group(0)
                if func.startswith('.'):
                    func = func[1:]
                if func.endswith('('):
                    func = func[:-1]
                if 'calculate_' in func or 'compute_' in func:
                    # Extract just the function name
                    parts = func.split()
                    for part in parts:
                        if 'calculate_' in part or 'compute_' in part:
                            clean = part.strip('(),')
                            if clean.startswith('calculate_') or clean.startswith('compute_'):
                                references.add(clean)
    return references

def main():
    base_path = Path('admetrics')
    test_path = Path('tests')
    
    # Module to functions mapping
    modules = {
        'detection/aos.py': 'test_aos.py',
        'detection/ap.py': 'test_ap.py',
        'detection/confusion.py': 'test_confusion.py',
        'detection/distance.py': 'test_distance.py',
        'detection/iou.py': 'test_iou.py',
        'detection/nds.py': 'test_nds.py',
        'tracking/tracking.py': 'test_tracking.py',
        'prediction/trajectory.py': 'test_trajectory.py',
        'localization/localization.py': 'test_localization.py',
        'occupancy/occupancy.py': 'test_occupancy.py',
        'planning/planning.py': 'test_planning.py',
        'vectormap/vectormap.py': 'test_vectormap.py',
        'simulation/sensor_quality.py': 'test_simulation.py',
    }
    
    all_covered = True
    total_functions = 0
    total_tested = 0
    
    print("=" * 80)
    print("METRIC FUNCTION TEST COVERAGE REPORT")
    print("=" * 80)
    print()
    
    for module_file, test_file in modules.items():
        module_path = base_path / module_file
        test_file_path = test_path / test_file
        
        if not module_path.exists():
            print(f"⚠️  Module not found: {module_path}")
            continue
            
        if not test_file_path.exists():
            print(f"⚠️  Test file not found: {test_file_path}")
            continue
        
        # Extract functions from module
        functions = extract_functions(module_path)
        
        # Extract test references
        test_refs = extract_test_references(test_file_path)
        
        # Check coverage
        module_name = module_file.replace('.py', '').replace('/', '.')
        print(f"\n📦 {module_name}")
        print(f"   {'-' * 60}")
        
        tested = []
        untested = []
        
        for func in functions:
            total_functions += 1
            if func in test_refs:
                tested.append(func)
                total_tested += 1
            else:
                untested.append(func)
                all_covered = False
        
        if tested:
            print(f"   ✅ Tested ({len(tested)}):")
            for func in sorted(tested):
                print(f"      • {func}")
        
        if untested:
            print(f"   ❌ NOT Tested ({len(untested)}):")
            for func in sorted(untested):
                print(f"      • {func}")
        
        coverage_pct = (len(tested) / len(functions) * 100) if functions else 100
        print(f"   Coverage: {len(tested)}/{len(functions)} ({coverage_pct:.1f}%)")
    
    print()
    print("=" * 80)
    print(f"SUMMARY: {total_tested}/{total_functions} functions tested ({total_tested/total_functions*100:.1f}%)")
    print("=" * 80)
    
    if all_covered:
        print("✅ All metric functions are covered by tests!")
        return 0
    else:
        print("❌ Some metric functions are NOT covered by tests")
        return 1

if __name__ == '__main__':
    sys.exit(main())

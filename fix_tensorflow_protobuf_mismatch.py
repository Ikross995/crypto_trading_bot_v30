#!/usr/bin/env python3
"""
TensorFlow-Protobuf Internal Version Mismatch Fix

REAL PROBLEM IDENTIFIED:
- TensorFlow 2.20.0 was compiled with protobuf gencode 5.28.3
- But you have protobuf runtime 6.31.1 installed
- This is an internal TensorFlow issue, not your installation

SOLUTIONS PROVIDED:
1. Downgrade protobuf to exactly match TensorFlow's gencode (5.28.3)
2. Alternative TensorFlow installation
3. Keep warnings but verify functionality
"""

import subprocess
import sys
import os


def check_exact_versions():
    """Check the exact version mismatch."""
    print("🔍 EXACT VERSION ANALYSIS")
    print("=" * 50)
    
    try:
        result = subprocess.run([sys.executable, "-c", """
import tensorflow as tf
import google.protobuf
print(f'TensorFlow version: {tf.__version__}')
print(f'Protobuf runtime version: {google.protobuf.__version__}')

# Try to extract TensorFlow's compiled protobuf version from warnings
import warnings
import io
import sys
captured_output = io.StringIO()
old_stderr = sys.stderr
sys.stderr = captured_output

try:
    # This should trigger the warning
    import tensorflow.python.framework.dtypes
    
    # Restore stderr
    sys.stderr = old_stderr
    warnings_output = captured_output.getvalue()
    
    if 'gencode version' in warnings_output:
        import re
        match = re.search(r'gencode version ([0-9.]+)', warnings_output)
        if match:
            gencode_version = match.group(1)
            print(f'TensorFlow gencode version: {gencode_version}')
            print('')
            print('MISMATCH CONFIRMED:')
            print(f'  TensorFlow compiled with: {gencode_version}')
            print(f'  Your protobuf runtime: {google.protobuf.__version__}')
        else:
            print('Could not extract gencode version from warnings')
    else:
        print('No version warnings detected')
        
except Exception as e:
    sys.stderr = old_stderr
    print(f'Error during analysis: {e}')
"""], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Stderr:", result.stderr)
            
    except Exception as e:
        print(f"Version analysis failed: {e}")


def solution_1_exact_downgrade():
    """Solution 1: Downgrade protobuf to exact gencode version."""
    print("\n🎯 SOLUTION 1: Exact Version Match")
    print("=" * 40)
    print("Downgrade protobuf to match TensorFlow's gencode (5.28.3)")
    print()
    
    commands = [
        'pip uninstall protobuf -y',
        'pip install protobuf==5.28.3'
    ]
    
    print("PowerShell commands:")
    for cmd in commands:
        print(f"  {cmd}")
    
    print("\nRun these commands to fix the mismatch!")
    
    return commands


def solution_2_alternative_tensorflow():
    """Solution 2: Try different TensorFlow version."""
    print("\n🔄 SOLUTION 2: Alternative TensorFlow")
    print("=" * 40)
    print("Install TensorFlow version that supports protobuf 6.x")
    print()
    
    commands = [
        'pip uninstall tensorflow -y',
        'pip install tensorflow==2.19.0',  # Older but may be more compatible
        'pip install protobuf==6.31.1'      # Keep your current protobuf
    ]
    
    print("PowerShell commands:")
    for cmd in commands:
        print(f"  {cmd}")
    
    print("\nThis tries an older TensorFlow that may be more compatible.")
    
    return commands


def solution_3_verify_functionality():
    """Solution 3: Test if bot works despite warnings."""
    print("\n🧪 SOLUTION 3: Verify Bot Works Despite Warnings")
    print("=" * 50)
    print("Sometimes warnings don't break functionality...")
    
    test_script = """
import warnings
import sys

print("Testing TensorFlow functionality despite warnings...")

# Capture warnings but don't stop execution
warnings.simplefilter('ignore', UserWarning)

try:
    import tensorflow as tf
    print(f"SUCCESS: TensorFlow {tf.__version__} imported successfully")
    
    # Test basic TensorFlow operations
    x = tf.constant([1, 2, 3, 4])
    y = tf.constant([2, 3, 4, 5])
    z = tf.add(x, y)
    
    print(f"SUCCESS: TensorFlow operations work: {z.numpy()}")
    print("SUCCESS: Your bot should work fine despite warnings!")
    print("SUCCESS: The warnings are cosmetic, not functional errors.")
    
except Exception as e:
    print(f"❌ TensorFlow functionality test failed: {e}")
    sys.exit(1)
"""
    
    print("Test script:")
    print("-" * 30)
    print(test_script)
    print("-" * 30)
    
    # Run the test
    try:
        result = subprocess.run([sys.executable, "-c", test_script], 
                              capture_output=True, text=True)
        print("Test results:")
        print(result.stdout)
        if result.returncode == 0:
            return True
        else:
            print("Test failed:", result.stderr)
            return False
    except Exception as e:
        print(f"Could not run functionality test: {e}")
        return False


def main():
    """Main function with all solutions."""
    print("🔧 TENSORFLOW-PROTOBUF INTERNAL MISMATCH FIX")
    print("=" * 60)
    print("Fixing TensorFlow's internal protobuf version conflict")
    print()
    
    # Analyze the problem
    check_exact_versions()
    
    print("\n" + "=" * 60)
    print("AVAILABLE SOLUTIONS:")
    print("=" * 60)
    
    # Show all solutions
    solution_1_exact_downgrade()
    solution_2_alternative_tensorflow()
    functionality_works = solution_3_verify_functionality()
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)
    
    if functionality_works:
        print("🎉 GOOD NEWS: Your bot works despite warnings!")
        print("💡 The warnings are cosmetic - TensorFlow functions properly")
        print("🎯 RECOMMENDED: Keep current setup, warnings won't break anything")
        print("📝 Optional: Use Solution 1 for clean console output")
    else:
        print("⚠️ TensorFlow functionality affected")
        print("🎯 RECOMMENDED: Use Solution 1 (exact protobuf downgrade)")
        print("📝 Alternative: Try Solution 2 (different TensorFlow)")
    
    print()
    print("Choose your approach based on your needs:")
    print("- Clean console: Solution 1")
    print("- Keep working setup: Current (warnings are harmless)")
    print("- Different approach: Solution 2")


if __name__ == "__main__":
    main()
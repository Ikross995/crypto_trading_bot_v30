#!/usr/bin/env python3
"""
Protobuf/TensorFlow Version Compatibility Fix

This script fixes the root cause of protobuf warnings by updating to compatible versions.
Instead of suppressing warnings, we fix the underlying version conflict.

ERROR BEING FIXED:
"Protobuf gencode version 5.28.3 is exactly one major version older than the runtime version 6.31.1"

SOLUTION:
- Downgrade protobuf to 5.x series (compatible with TensorFlow)
- OR upgrade TensorFlow to latest version that supports protobuf 6.x
"""

import subprocess
import sys
import os


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            return True
        else:
            print(f"❌ {description} - FAILED")
            print(f"   Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - EXCEPTION: {e}")
        return False


def check_current_versions():
    """Check currently installed versions."""
    print("🔍 Checking current versions...")
    
    try:
        result = subprocess.run([sys.executable, "-c", 
                               "import tensorflow as tf; import google.protobuf; print(f'TensorFlow: {tf.__version__}'); print(f'Protobuf: {google.protobuf.__version__}')"], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print("Current versions:")
            print(result.stdout)
        else:
            print("⚠️ Could not check versions (packages may not be installed)")
            print(result.stderr)
    except Exception as e:
        print(f"⚠️ Version check failed: {e}")


def fix_protobuf_compatibility():
    """Fix protobuf/tensorflow compatibility by updating to compatible versions."""
    print("🚀 Starting Protobuf/TensorFlow compatibility fix...")
    print("=" * 60)
    
    check_current_versions()
    print()
    
    # Strategy: Downgrade protobuf to 5.x series for compatibility
    fixes = [
        ("pip install protobuf>=5.26.0,<6.0.0", "Downgrading protobuf to 5.x series"),
        ("pip install tensorflow>=2.16.0", "Updating TensorFlow to latest stable"),
    ]
    
    success_count = 0
    for cmd, desc in fixes:
        if run_command(cmd, desc):
            success_count += 1
        print()
    
    print("=" * 60)
    print(f"✅ Completed: {success_count}/{len(fixes)} fixes applied successfully")
    
    # Verify the fix
    print("\n🧪 Verifying fix...")
    verify_cmd = [sys.executable, "-c", """
import warnings
# Capture warnings
warnings.simplefilter('error', UserWarning)
try:
    import tensorflow as tf
    import google.protobuf
    print(f'✅ SUCCESS: TensorFlow {tf.__version__} + Protobuf {google.protobuf.__version__}')
    print('✅ No protobuf compatibility warnings!')
except UserWarning as w:
    if 'protobuf' in str(w).lower():
        print(f'❌ STILL HAS WARNING: {w}')
    else:
        print(f'⚠️ Other warning: {w}')
except ImportError as e:
    print(f'❌ Import error: {e}')
except Exception as e:
    print(f'⚠️ Unexpected error: {e}')
"""]
    
    try:
        result = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=30)
        print(result.stdout)
        if result.stderr:
            print("Stderr:", result.stderr)
    except subprocess.TimeoutExpired:
        print("⚠️ Verification timed out")
    except Exception as e:
        print(f"⚠️ Verification failed: {e}")


def create_updated_requirements():
    """Create an updated requirements.txt with fixed versions."""
    print("\n📝 Creating requirements_fixed.txt with compatible versions...")
    
    requirements_content = """# Trading Bot Requirements - Fixed Versions
# This requirements.txt resolves protobuf/tensorflow compatibility issues

# Core ML Dependencies - Compatible versions
tensorflow>=2.16.0,<3.0.0
protobuf>=5.26.0,<6.0.0

# Core Python packages
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Trading & Exchange APIs
python-binance>=1.0.19
ccxt>=4.0.0

# Configuration
python-dotenv>=1.0.0
pydantic>=2.0.0

# CLI & UI
typer>=0.9.0
rich>=13.0.0

# Technical Analysis
# ta-lib  # Uncomment if needed, requires separate installation

# Async Support
aiohttp>=3.8.0
websockets>=11.0.0

# Logging
loguru>=0.7.0

# Development
pytest>=7.0.0
black>=23.0.0
"""
    
    with open("requirements_fixed.txt", "w", encoding="utf-8") as f:
        f.write(requirements_content)
    
    print("✅ Created requirements_fixed.txt")
    print("📌 To use: pip install -r requirements_fixed.txt")


def main():
    """Main function."""
    print("🔧 PROTOBUF/TENSORFLOW VERSION FIX")
    print("=" * 50)
    print("Fixing root cause instead of suppressing warnings!")
    print()
    
    if "--dry-run" in sys.argv:
        print("🔍 DRY RUN MODE - No changes will be made")
        check_current_versions()
        return
    
    if "--requirements-only" in sys.argv:
        create_updated_requirements()
        return
        
    # Ask for confirmation
    print("This will update your TensorFlow and Protobuf versions.")
    response = input("Continue? (y/N): ").strip().lower()
    
    if response not in ['y', 'yes']:
        print("❌ Cancelled by user")
        return
    
    # Apply fixes
    fix_protobuf_compatibility()
    create_updated_requirements()
    
    print("\n🎉 DONE!")
    print("Now you can remove warning suppression code and use natural compatibility.")
    print("\nNext steps:")
    print("1. Test your bot startup")
    print("2. Remove suppress_warnings.py import from cli.py")
    print("3. Enjoy clean console output without forced suppression!")


if __name__ == "__main__":
    main()
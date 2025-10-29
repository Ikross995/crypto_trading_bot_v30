#!/usr/bin/env python3
"""
Early Warning Suppression Module

This module MUST be imported BEFORE any TensorFlow/Protobuf imports
to effectively suppress warnings at the source.
"""

import os
import warnings
import sys


def apply_early_warning_suppression():
    """
    Apply warning suppression BEFORE TensorFlow/Protobuf import.
    This is more effective than applying after imports.
    """
    
    # ========== TENSORFLOW SUPPRESSION ==========
    # Must be set BEFORE TensorFlow import
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=error only
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimization messages
    
    # Additional TensorFlow environment variables
    os.environ['TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU messages if no GPU needed
    
    # ========== PROTOBUF SUPPRESSION ==========
    # Suppress specific protobuf warnings using Python warnings filter
    warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.runtime_version")
    warnings.filterwarnings("ignore", message=".*Protobuf gencode version.*")
    warnings.filterwarnings("ignore", message=".*runtime version.*")
    warnings.filterwarnings("ignore", message=".*compatibility violations.*")
    
    # ========== GENERAL ML LIBRARY WARNINGS ==========
    # Suppress common ML library warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pandas")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    
    # ========== SYSTEM OPTIMIZATION ==========
    # Python optimizations for better performance
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    
    # Windows-specific optimizations
    if sys.platform == "win32":
        os.environ['PYTHONASYNCIODEBUG'] = '0'
        # Ensure proper encoding for Windows console
        try:
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
            if hasattr(sys.stderr, 'reconfigure'):
                sys.stderr.reconfigure(encoding='utf-8', errors='ignore')
        except Exception:
            pass


def suppress_specific_tensorflow_warnings():
    """
    Additional TensorFlow-specific warning suppression.
    Call this after TensorFlow import if needed.
    """
    try:
        import tensorflow as tf
        
        # Disable TensorFlow logging
        tf.get_logger().setLevel('ERROR')
        
        # Disable AutoGraph warnings
        tf.autograph.set_verbosity(0)
        
        # Disable oneDNN custom operations info
        tf.config.optimizer.set_jit(False)
        
    except ImportError:
        pass  # TensorFlow not installed
    except Exception:
        pass  # Other TensorFlow issues


def verify_suppression():
    """
    Test if warning suppression is working.
    Returns True if suppression is effective.
    """
    print("🔇 Testing warning suppression...")
    
    # Test environment variables
    tf_level = os.environ.get('TF_CPP_MIN_LOG_LEVEL', '0')
    onednn = os.environ.get('TF_ENABLE_ONEDNN_OPTS', '1')
    
    print(f"  TF_CPP_MIN_LOG_LEVEL: {tf_level} ({'✅ Suppressed' if tf_level == '3' else '❌ Not suppressed'})")
    print(f"  TF_ENABLE_ONEDNN_OPTS: {onednn} ({'✅ Disabled' if onednn == '0' else '❌ Enabled'})")
    
    # Test warnings filter
    import warnings
    filters = warnings.filters
    protobuf_filtered = any('protobuf' in str(f) for f in filters)
    print(f"  Protobuf warnings filtered: {'✅ Yes' if protobuf_filtered else '❌ No'}")
    
    return tf_level == '3' and onednn == '0' and protobuf_filtered


# Apply suppression immediately when this module is imported
apply_early_warning_suppression()

if __name__ == "__main__":
    # Test suppression when run directly
    verify_suppression()
    
    print("\n🧪 Testing TensorFlow import with suppression...")
    try:
        import tensorflow as tf
        print("✅ TensorFlow imported - check for warnings above")
        suppress_specific_tensorflow_warnings()
        print("✅ Additional TensorFlow suppression applied")
    except ImportError:
        print("⚠️ TensorFlow not available for testing")
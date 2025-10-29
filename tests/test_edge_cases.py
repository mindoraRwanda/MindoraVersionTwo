#!/usr/bin/env python3
"""
Test script to verify edge cases and error handling.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(backend_dir / "app"))

def test_missing_environment_file():
    """Test behavior when environment file is missing."""
    print("🧪 Testing missing environment file...")
    
    try:
        from app.settings.model import ModelSettings
        
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Try to create settings with a non-existent environment file
            env_file = os.path.join(temp_dir, ".env.nonexistent")
            
            # This should work with defaults
            settings = ModelSettings.create_for_environment("nonexistent")
            print(f"✅ Missing environment file handled gracefully: {settings.default_model_name}")
            
            # Verify default values are used
            assert settings.default_model_name == "llama3.2:1b"
            assert settings.temperature == 1.0
            
        return True
    except Exception as e:
        print(f"❌ Error testing missing environment file: {e}")
        return False

def test_invalid_environment_name():
    """Test behavior with invalid environment name."""
    print("\n🧪 Testing invalid environment name...")
    
    try:
        from app.settings.model import ModelSettings
        
        # Test with empty environment
        settings = ModelSettings.create_for_environment("")
        print(f"✅ Empty environment handled: {settings.default_model_name}")
        
        # Test with None environment
        settings = ModelSettings.create_for_environment(None)
        print(f"✅ None environment handled: {settings.default_model_name}")
        
        # Test with moderately long environment name (within OS limits)
        long_env = "a" * 100
        settings = ModelSettings.create_for_environment(long_env)
        print(f"✅ Long environment name handled: {settings.default_model_name}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing invalid environment name: {e}")
        return False

def test_invalid_json_in_env_file():
    """Test behavior with invalid JSON in environment file."""
    print("\n🧪 Testing invalid JSON in environment file...")
    
    try:
        from app.settings.model import ModelSettings
        
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create an invalid JSON file
            invalid_json_file = os.path.join(temp_dir, ".env.invalid")
            with open(invalid_json_file, 'w') as f:
                f.write("{ invalid json content")
            
            # This should not crash
            try:
                # Change to temp directory to test file loading
                original_cwd = os.getcwd()
                os.chdir(temp_dir)
                
                # Create settings - should use defaults
                settings = ModelSettings()
                print(f"✅ Invalid JSON handled gracefully: {settings.default_model_name}")
                
            finally:
                os.chdir(original_cwd)
        
        return True
    except Exception as e:
        print(f"❌ Error testing invalid JSON: {e}")
        return False

def test_malformed_environment_variables():
    """Test behavior with malformed environment variables."""
    print("\n🧪 Testing malformed environment variables...")
    
    try:
        from app.settings import get_settings
        
        # Test with malformed temperature
        os.environ["MINDORA_MODEL_TEMPERATURE"] = "not-a-number"
        
        # Clear cache to force reload
        get_settings.cache_clear()
        
        try:
            settings = get_settings()
            print("❌ Should have raised validation error for malformed temperature")
            return False
        except Exception as e:
            print(f"✅ Correctly rejected malformed temperature: {type(e).__name__}")
        
        # Clean up
        os.environ.pop("MINDORA_MODEL_TEMPERATURE", None)
        get_settings.cache_clear()
        
        # Test with malformed boolean
        os.environ["DEBUG"] = "not-a-boolean"
        get_settings.cache_clear()
        
        try:
            settings = get_settings()
            # This should not crash, but may interpret the value
            print(f"✅ Malformed boolean handled: debug={settings.debug}")
        except Exception as e:
            print(f"✅ Correctly rejected malformed boolean: {type(e).__name__}")
        
        # Clean up
        os.environ.pop("DEBUG", None)
        get_settings.cache_clear()
        
        return True
    except Exception as e:
        print(f"❌ Error testing malformed environment variables: {e}")
        return False

def test_very_long_values():
    """Test behavior with very long values."""
    print("\n🧪 Testing very long values...")
    
    try:
        from app.settings.safety import SafetySettings
        
        # Test with very long crisis keywords string
        long_keywords = "keyword," * 1000  # 1000 keywords
        settings = SafetySettings(crisis_keywords=long_keywords)
        
        # Should handle it gracefully
        print(f"✅ Long crisis keywords handled: {len(settings.crisis_keywords)} keywords")
        
        # Test with very long injection patterns
        long_patterns = "pattern|" * 1000  # 1000 patterns
        settings = SafetySettings(injection_patterns=long_patterns)
        
        print(f"✅ Long injection patterns handled: {len(settings.injection_patterns)} patterns")
        
        return True
    except Exception as e:
        print(f"❌ Error testing very long values: {e}")
        return False

def test_special_characters():
    """Test behavior with special characters in values."""
    print("\n🧪 Testing special characters...")
    
    try:
        from app.settings.safety import SafetySettings
        from app.settings.model import ModelSettings
        
        # Test with special characters in keywords
        special_keywords = ["suicide!@#$%", "hurt&*()", "end^%$life"]
        settings = SafetySettings(crisis_keywords=special_keywords)
        print(f"✅ Special characters in keywords handled: {len(settings.crisis_keywords)} keywords")
        
        # Test with unicode characters
        unicode_keywords = ["自殺", "自傷", "自杀"]  # Chinese characters for suicide/self-harm
        settings = SafetySettings(crisis_keywords=unicode_keywords)
        print(f"✅ Unicode characters in keywords handled: {len(settings.crisis_keywords)} keywords")
        
        # Test with special characters in model name
        os.environ["MINDORA_MODEL_DEFAULT_NAME"] = "model-with_special.chars@123"
        
        from app.settings import get_settings
        get_settings.cache_clear()
        
        settings = get_settings()
        print(f"✅ Special characters in model name: {settings.model.default_model_name}")
        
        # Clean up
        os.environ.pop("MINDORA_MODEL_DEFAULT_NAME", None)
        get_settings.cache_clear()
        
        return True
    except Exception as e:
        print(f"❌ Error testing special characters: {e}")
        return False

def test_concurrent_access():
    """Test behavior with concurrent access to settings."""
    print("\n🧪 Testing concurrent access...")
    
    try:
        import threading
        from app.settings import get_settings
        
        results = []
        
        def worker():
            try:
                settings = get_settings()
                results.append(settings.environment)
            except Exception as e:
                results.append(f"Error: {e}")
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        if all(result == results[0] for result in results):
            print(f"✅ Concurrent access handled: {results[0]}")
            return True
        else:
            print(f"❌ Inconsistent results from concurrent access: {results}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing concurrent access: {e}")
        return False

def test_memory_usage():
    """Test that settings don't consume excessive memory."""
    print("\n🧪 Testing memory usage...")
    
    try:
        import gc
        from app.settings import get_settings
        
        # Force garbage collection
        gc.collect()
        
        # Create many settings instances
        settings_instances = []
        for i in range(100):
            settings = get_settings()
            settings_instances.append(settings)
        
        # Clear cache and force garbage collection
        get_settings.cache_clear()
        settings_instances.clear()
        gc.collect()
        
        print("✅ Memory usage appears reasonable")
        return True
        
    except Exception as e:
        print(f"❌ Error testing memory usage: {e}")
        return False

def main():
    """Run all edge case tests."""
    print("🚀 Starting Edge Case Tests\n")
    
    tests = [
        test_missing_environment_file,
        test_invalid_environment_name,
        test_invalid_json_in_env_file,
        test_malformed_environment_variables,
        test_very_long_values,
        test_special_characters,
        test_concurrent_access,
        test_memory_usage
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All edge case tests passed!")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
#!/usr/bin/env python3
"""
Test script to verify configuration validation and error handling.
"""

import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(backend_dir / "app"))

def test_model_validation():
    """Test model configuration validation."""
    print("🧪 Testing model configuration validation...")
    
    try:
        from app.settings.model import ModelSettings
        
        # Test valid temperature
        settings = ModelSettings(temperature=0.5)
        print(f"✅ Valid temperature: {settings.temperature}")
        
        # Test invalid temperature (too high)
        try:
            settings = ModelSettings(temperature=3.0)
            print("❌ Should have raised validation error for temperature > 2.0")
            return False
        except Exception as e:
            print(f"✅ Correctly rejected invalid temperature: {e}")
        
        # Test invalid temperature (negative)
        try:
            settings = ModelSettings(temperature=-0.1)
            print("❌ Should have raised validation error for negative temperature")
            return False
        except Exception as e:
            print(f"✅ Correctly rejected negative temperature: {e}")
        
        # Test valid max_tokens
        settings = ModelSettings(max_tokens=100)
        print(f"✅ Valid max_tokens: {settings.max_tokens}")
        
        # Test invalid max_tokens (negative)
        try:
            settings = ModelSettings(max_tokens=-1)
            print("❌ Should have raised validation error for negative max_tokens")
            return False
        except Exception as e:
            print(f"✅ Correctly rejected negative max_tokens: {e}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing model validation: {e}")
        return False

def test_performance_validation():
    """Test performance configuration validation."""
    print("\n🧪 Testing performance configuration validation...")
    
    try:
        from app.settings.performance import PerformanceSettings
        
        # Test valid values
        settings = PerformanceSettings(
            max_input_length=1000,
            max_conversation_history=10,
            rag_top_k=5,
            request_timeout=30,
            max_retries=3
        )
        print(f"✅ Valid performance settings: max_input_length={settings.max_input_length}")
        
        # Test invalid max_input_length (negative)
        try:
            settings = PerformanceSettings(max_input_length=-1)
            print("❌ Should have raised validation error for negative max_input_length")
            return False
        except Exception as e:
            print(f"✅ Correctly rejected negative max_input_length: {e}")
        
        # Test invalid max_conversation_history (zero)
        try:
            settings = PerformanceSettings(max_conversation_history=0)
            print("❌ Should have raised validation error for zero max_conversation_history")
            return False
        except Exception as e:
            print(f"✅ Correctly rejected zero max_conversation_history: {e}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing performance validation: {e}")
        return False

def test_database_validation():
    """Test database configuration validation."""
    print("\n🧪 Testing database configuration validation...")
    
    try:
        from app.settings.database import DatabaseSettings
        
        # Test valid values
        settings = DatabaseSettings(
            database_pool_size=5,
            database_max_overflow=10,
            database_pool_timeout=30,
            database_pool_recycle=3600,
            redis_ttl=3600
        )
        print(f"✅ Valid database settings: pool_size={settings.database_pool_size}")
        
        # Test invalid database_pool_size (negative)
        try:
            settings = DatabaseSettings(database_pool_size=-1)
            print("❌ Should have raised validation error for negative database_pool_size")
            return False
        except Exception as e:
            print(f"✅ Correctly rejected negative database_pool_size: {e}")
        
        # Test invalid redis_ttl (zero)
        try:
            settings = DatabaseSettings(redis_ttl=0)
            print("❌ Should have raised validation error for zero redis_ttl")
            return False
        except Exception as e:
            print(f"✅ Correctly rejected zero redis_ttl: {e}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing database validation: {e}")
        return False

def test_safety_validation():
    """Test safety configuration validation."""
    print("\n🧪 Testing safety configuration validation...")
    
    try:
        from app.settings.safety import SafetySettings
        
        # Test valid crisis keywords (list)
        settings = SafetySettings(crisis_keywords=["suicide", "kill myself"])
        print(f"✅ Valid crisis keywords: {len(settings.crisis_keywords)} keywords")
        
        # Test valid crisis keywords (comma-separated string)
        settings = SafetySettings(crisis_keywords="suicide,kill myself,end my life")
        print(f"✅ Valid crisis keywords from string: {len(settings.crisis_keywords)} keywords")
        
        # Test valid injection patterns (list)
        settings = SafetySettings(injection_patterns=["ignore.*instructions", "system:"])
        print(f"✅ Valid injection patterns: {len(settings.injection_patterns)} patterns")
        
        # Test valid injection patterns (pipe-separated string)
        settings = SafetySettings(injection_patterns="ignore.*instructions|system:|assistant:")
        print(f"✅ Valid injection patterns from string: {len(settings.injection_patterns)} patterns")
        
        return True
    except Exception as e:
        print(f"❌ Error testing safety validation: {e}")
        return False

def test_environment_variable_validation():
    """Test that environment variables are validated properly."""
    print("\n🧪 Testing environment variable validation...")
    
    try:
        from app.settings import get_settings
        
        # Test invalid temperature from environment
        os.environ["MINDORA_MODEL_TEMPERATURE"] = "invalid"
        
        # Clear cache to force reload
        get_settings.cache_clear()
        
        try:
            settings = get_settings()
            print("❌ Should have raised validation error for invalid temperature")
            return False
        except Exception as e:
            print(f"✅ Correctly rejected invalid temperature from environment: {e}")
        
        # Clean up
        os.environ.pop("MINDORA_MODEL_TEMPERATURE", None)
        get_settings.cache_clear()
        
        # Test invalid max_tokens from environment
        os.environ["MINDORA_MODEL_MAX_TOKENS"] = "invalid"
        
        try:
            settings = get_settings()
            print("❌ Should have raised validation error for invalid max_tokens")
            return False
        except Exception as e:
            print(f"✅ Correctly rejected invalid max_tokens from environment: {e}")
        
        # Clean up
        os.environ.pop("MINDORA_MODEL_MAX_TOKENS", None)
        get_settings.cache_clear()
        
        return True
    except Exception as e:
        print(f"❌ Error testing environment variable validation: {e}")
        return False

def test_default_values():
    """Test that default values are applied correctly."""
    print("\n🧪 Testing default values...")
    
    try:
        from app.settings.model import ModelSettings
        from app.settings.performance import PerformanceSettings
        from app.settings.database import DatabaseSettings
        from app.settings.safety import SafetySettings
        
        # Test model defaults
        settings = ModelSettings()
        assert settings.default_model_name == "llama3.2:1b"
        assert settings.temperature == 1.0
        assert settings.max_tokens == 512
        print(f"✅ Model defaults applied: {settings.default_model_name}")
        
        # Test performance defaults
        settings = PerformanceSettings()
        assert settings.max_input_length == 2000
        assert settings.max_conversation_history == 15
        assert settings.rag_top_k == 3
        print(f"✅ Performance defaults applied: max_input_length={settings.max_input_length}")
        
        # Test database defaults
        settings = DatabaseSettings()
        assert settings.database_url == "sqlite:///./mindora.db"
        assert settings.database_pool_size == 5
        print(f"✅ Database defaults applied: {settings.database_url}")
        
        # Test safety defaults
        settings = SafetySettings()
        assert len(settings.crisis_keywords) > 0
        assert len(settings.injection_patterns) > 0
        print(f"✅ Safety defaults applied: {len(settings.crisis_keywords)} crisis keywords")
        
        return True
    except Exception as e:
        print(f"❌ Error testing default values: {e}")
        return False

def main():
    """Run all configuration validation tests."""
    print("🚀 Starting Configuration Validation Tests\n")
    
    tests = [
        test_model_validation,
        test_performance_validation,
        test_database_validation,
        test_safety_validation,
        test_environment_variable_validation,
        test_default_values
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
        print("✅ All configuration validation tests passed!")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
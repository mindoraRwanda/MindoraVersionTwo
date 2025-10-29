#!/usr/bin/env python3
"""
Test script to verify the settings system migration is working correctly.
"""

import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(backend_dir / "app"))

def test_new_settings_system():
    """Test the new settings system directly."""
    print("🧪 Testing new settings system...")
    
    try:
        from app.settings import get_settings
        settings = get_settings()
        
        print(f"✅ Environment: {settings.environment}")
        print(f"✅ API Title: {settings.api_title}")
        print(f"✅ Debug Mode: {settings.debug}")
        
        # Test sub-settings
        if settings.model:
            print(f"✅ Default Model: {settings.model.default_model_name}")
            print(f"✅ Ollama URL: {settings.model.ollama_base_url}")
        
        if settings.performance:
            print(f"✅ Max Input Length: {settings.performance.max_input_length}")
            print(f"✅ RAG Top K: {settings.performance.rag_top_k}")
        
        if settings.safety:
            print(f"✅ Crisis Keywords Count: {len(settings.safety.crisis_keywords)}")
        
        if settings.database:
            print(f"✅ Database URL: {settings.database.database_url}")
        
        if settings.cultural:
            print(f"✅ Cultural Context Keys: {list(settings.cultural.cultural_context.keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing new settings system: {e}")
        return False

def test_compatibility_layer():
    """Test the compatibility layer."""
    print("\n🧪 Testing compatibility layer...")
    
    try:
        from app.settings.compatibility import compatibility_layer
        
        # Test property access
        print(f"✅ Default Model Name: {compatibility_layer.DEFAULT_MODEL_NAME}")
        print(f"✅ Ollama Base URL: {compatibility_layer.OLLAMA_BASE_URL}")
        print(f"✅ Max Input Length: {compatibility_layer.MAX_INPUT_LENGTH}")
        
        # Test method access
        model_config = compatibility_layer.get_model_config()
        if model_config:
            print(f"✅ Model Config Temperature: {model_config.temperature}")
        
        performance_config = compatibility_layer.get_performance_config()
        if performance_config:
            print(f"✅ Performance Config Max History: {performance_config.max_conversation_history}")
        
        safety_config = compatibility_layer.get_safety_config()
        if safety_config:
            print(f"✅ Safety Config Crisis Keywords: {len(safety_config.crisis_keywords)}")
        
        cultural_config = compatibility_layer.get_rwanda_config()
        if cultural_config:
            print(f"✅ Cultural Config Crisis Resources: {list(cultural_config.crisis_resources.keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing compatibility layer: {e}")
        return False

def test_service_imports():
    """Test that services can import and use the new configuration."""
    print("\n🧪 Testing service imports...")
    
    try:
        # Test LLM service import
        print("  Testing LLM Service import...")
        from app.services.llm_service import LLMService
        print("✅ LLM Service import successful")
        
        # Test LLM model manager import
        print("  Testing Model Manager import...")
        from app.services.llm_model_manager import ModelManager
        print("✅ Model Manager import successful")
        
        # Test LLM providers import
        print("  Testing LLM Providers import...")
        from app.services.llm_providers import LLMProviderFactory
        print("✅ LLM Providers import successful")
        
        # Test LLM safety import
        print("  Testing LLM Safety import...")
        from app.services.llm_safety import SafetyManager
        print("✅ LLM Safety import successful")
        
        # Test LLM cultural context import
        print("  Testing LLM Cultural Context import...")
        from app.services.llm_cultural_context import RwandaCulturalManager
        print("✅ LLM Cultural Context import successful")
        
        # Test database operations import
        print("  Testing Database Operations import...")
        from app.services.llm_database_operations import DatabaseManager
        print("✅ Database Operations import successful")
        
        return True
    except Exception as e:
        import traceback
        print(f"❌ Error testing service imports: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

def test_database_config():
    """Test database configuration."""
    print("\n🧪 Testing database configuration...")
    
    try:
        from app.db.database import engine, SessionLocal
        print("✅ Database configuration import successful")
        print(f"✅ Database engine created: {type(engine)}")
        return True
    except Exception as e:
        print(f"❌ Error testing database configuration: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Settings Migration Tests\n")
    
    tests = [
        test_new_settings_system,
        test_compatibility_layer,
        test_service_imports,
        test_database_config
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Settings migration appears to be working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
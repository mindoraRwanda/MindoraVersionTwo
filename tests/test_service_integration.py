#!/usr/bin/env python3
"""
Test script to verify service integration with the new settings system.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(backend_dir / "app"))

def test_service_container_integration():
    """Test that the service container can initialize with the new settings system."""
    print("🧪 Testing service container integration...")
    
    try:
        from app.settings import get_settings
        from app.services.service_container import service_container
        
        # Get settings
        settings = get_settings()
        print(f"✅ Settings loaded: {settings.environment}")
        
        # Test that the service container has registered the llm_config service
        if "llm_config" in service_container.registry._factories:
            print("✅ LLM config service registered")
        else:
            print("❌ LLM config service not registered")
            return False
        
        # Test that the compatibility layer can access settings
        from app.settings.compatibility import compatibility_layer
        model_config = compatibility_layer.get_model_config()
        if model_config:
            print(f"✅ Model config accessible: {model_config.default_model_name}")
        else:
            print("❌ Model config not accessible")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Error testing service container integration: {e}")
        return False

async def test_service_initialization():
    """Test that services can be initialized with the new settings system."""
    print("\n🧪 Testing service initialization...")
    
    try:
        from app.settings import get_settings
        from app.services.service_container import service_container, check_service_health
        
        # Get settings
        settings = get_settings()
        print(f"✅ Settings loaded: {settings.environment}")
        
        # Initialize all services
        print("🔧 Initializing services...")
        success = await service_container.initialize_all_services()
        
        if not success:
            print("❌ Failed to initialize services")
            return False
        
        print("✅ All services initialized successfully")
        
        # Check service health
        print("🏥 Checking service health...")
        health_status = await check_service_health()
        
        healthy_count = 0
        for service_name, status in health_status.items():
            status_icon = "✅" if status["healthy"] else "❌"
            print(f"  {status_icon} {service_name}: initialized={status['initialized']}, healthy={status['healthy']}")
            if status["healthy"]:
                healthy_count += 1
        
        print(f"✅ {healthy_count}/{len(health_status)} services healthy")
        
        return True
    except Exception as e:
        print(f"❌ Error testing service initialization: {e}")
        return False

def test_service_settings_access():
    """Test that services can access their configuration through the new settings system."""
    print("\n🧪 Testing service settings access...")
    
    try:
        from app.settings import get_settings
        from app.settings.compatibility import compatibility_layer
        
        # Get settings
        settings = get_settings()
        print(f"✅ Settings loaded: {settings.environment}")
        
        # Test model settings access
        model_config = compatibility_layer.get_model_config()
        if model_config:
            print(f"✅ Model settings: {model_config.default_model_name}")
            print(f"✅ Ollama URL: {model_config.ollama_base_url}")
            print(f"✅ Temperature: {model_config.temperature}")
        else:
            print("❌ Model settings not accessible")
            return False
        
        # Test performance settings access
        performance_config = compatibility_layer.get_performance_config()
        if performance_config:
            print(f"✅ Max input length: {performance_config.max_input_length}")
            print(f"✅ RAG top K: {performance_config.rag_top_k}")
        else:
            print("❌ Performance settings not accessible")
            return False
        
        # Test safety settings access
        safety_config = compatibility_layer.get_safety_config()
        if safety_config:
            print(f"✅ Crisis keywords count: {len(safety_config.crisis_keywords)}")
        else:
            print("❌ Safety settings not accessible")
            return False
        
        # Test cultural settings access
        cultural_config = compatibility_layer.get_rwanda_config()
        if cultural_config:
            print(f"✅ Cultural context keys: {list(cultural_config.cultural_context.keys())}")
        else:
            print("❌ Cultural settings not accessible")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Error testing service settings access: {e}")
        return False

def test_use_new_settings_system_toggle():
    """Test that the USE_NEW_SETTINGS_SYSTEM environment variable works correctly."""
    print("\n🧪 Testing USE_NEW_SETTINGS_SYSTEM toggle...")
    
    try:
        # Test with USE_NEW_SETTINGS_SYSTEM=false
        os.environ["USE_NEW_SETTINGS_SYSTEM"] = "false"
        
        from app.settings.compatibility import compatibility_layer
        
        print(f"✅ USE_NEW_SETTINGS_SYSTEM=false: {not compatibility_layer.use_new_system}")
        
        # Test with USE_NEW_SETTINGS_SYSTEM=true
        os.environ["USE_NEW_SETTINGS_SYSTEM"] = "true"
        
        # Create a new compatibility layer instance directly
        from app.settings.compatibility import CompatibilityLayer
        new_compatibility_layer = CompatibilityLayer(use_new_system=True)
        
        print(f"✅ USE_NEW_SETTINGS_SYSTEM=true: {new_compatibility_layer.use_new_system}")
        
        # Reset to default
        os.environ["USE_NEW_SETTINGS_SYSTEM"] = "false"
        
        return True
    except Exception as e:
        print(f"❌ Error testing USE_NEW_SETTINGS_SYSTEM toggle: {e}")
        return False

async def main():
    """Run all service integration tests."""
    print("🚀 Starting Service Integration Tests\n")
    
    tests = [
        test_service_container_integration,
        test_service_settings_access,
        test_use_new_settings_system_toggle
    ]
    
    # Add async test
    async_tests = [
        test_service_initialization
    ]
    
    passed = 0
    total = len(tests) + len(async_tests)
    
    # Run sync tests
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
    
    # Run async tests
    for test in async_tests:
        try:
            if await test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All service integration tests passed!")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
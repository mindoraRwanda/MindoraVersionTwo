#!/usr/bin/env python3
"""
Test script to verify application startup with the new settings system.
"""

import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(backend_dir / "app"))

def test_settings_loading():
    """Test that settings can be loaded for application startup."""
    print("🧪 Testing settings loading for application startup...")
    
    try:
        from app.settings import get_settings
        
        # Get settings
        settings = get_settings()
        print(f"✅ Settings loaded: {settings.environment}")
        print(f"✅ API Title: {settings.api_title}")
        print(f"✅ Debug Mode: {settings.debug}")
        print(f"✅ Database URL: {settings.database.database_url}")
        print(f"✅ CORS Origins: {settings.cors_origins}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading settings: {e}")
        return False

def test_fastapi_creation():
    """Test that FastAPI app can be created with settings."""
    print("\n🧪 Testing FastAPI app creation...")
    
    try:
        from app.settings import get_settings
        from fastapi import FastAPI
        
        # Get settings
        settings = get_settings()
        
        # Create FastAPI app
        app = FastAPI(
            title=settings.api_title,
            version=settings.api_version,
            debug=settings.debug
        )
        
        print(f"✅ FastAPI app created: {app.title}")
        print(f"✅ App version: {app.version}")
        print(f"✅ App debug: {app.debug}")
        
        return True
    except Exception as e:
        print(f"❌ Error creating FastAPI app: {e}")
        return False

def test_cors_middleware():
    """Test that CORS middleware can be added with settings."""
    print("\n🧪 Testing CORS middleware...")
    
    try:
        from app.settings import get_settings
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        
        # Get settings
        settings = get_settings()
        
        # Create FastAPI app
        app = FastAPI()
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        print(f"✅ CORS middleware added with origins: {settings.cors_origins}")
        
        return True
    except Exception as e:
        print(f"❌ Error adding CORS middleware: {e}")
        return False

def test_database_setup():
    """Test that database can be set up with settings."""
    print("\n🧪 Testing database setup...")
    
    try:
        from app.settings import get_settings
        from app.db.database import engine
        
        # Get settings
        settings = get_settings()
        
        # Test database engine
        print(f"✅ Database engine created: {type(engine)}")
        print(f"✅ Database URL: {settings.database.database_url}")
        
        return True
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        return False

def test_service_initialization():
    """Test that services can be initialized with settings."""
    print("\n🧪 Testing service initialization...")
    
    try:
        from app.settings import get_settings
        from app.services.service_container import service_container
        
        # Get settings
        settings = get_settings()
        print(f"✅ Settings loaded: {settings.environment}")
        
        # Test that service container is set up
        print(f"✅ Service container created: {type(service_container)}")
        
        # Test that compatibility layer is accessible
        from app.settings.compatibility import compatibility_layer
        model_config = compatibility_layer.get_model_config()
        print(f"✅ Model config accessible: {model_config.default_model_name}")
        
        return True
    except Exception as e:
        print(f"❌ Error initializing services: {e}")
        return False

def test_environment_specific_startup():
    """Test application startup in different environments."""
    print("\n🧪 Testing environment-specific startup...")
    
    try:
        from app.settings import get_settings, get_environment
        
        # Test development environment
        os.environ["ENVIRONMENT"] = "development"
        get_settings.cache_clear()
        get_environment.cache_clear()
        
        settings = get_settings()
        print(f"✅ Development environment: debug={settings.debug}")
        
        # Test testing environment
        os.environ["ENVIRONMENT"] = "testing"
        get_settings.cache_clear()
        get_environment.cache_clear()
        
        settings = get_settings()
        print(f"✅ Testing environment: debug={settings.debug}")
        
        # Test production environment
        os.environ["ENVIRONMENT"] = "production"
        get_settings.cache_clear()
        get_environment.cache_clear()
        
        settings = get_settings()
        print(f"✅ Production environment: debug={settings.debug}")
        
        # Reset to development
        os.environ["ENVIRONMENT"] = "development"
        get_settings.cache_clear()
        get_environment.cache_clear()
        
        return True
    except Exception as e:
        print(f"❌ Error testing environment-specific startup: {e}")
        return False

def main():
    """Run all application startup tests."""
    print("🚀 Starting Application Startup Tests\n")
    
    tests = [
        test_settings_loading,
        test_fastapi_creation,
        test_cors_middleware,
        test_database_setup,
        test_service_initialization,
        test_environment_specific_startup
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
        print("✅ All application startup tests passed!")
        print("✅ The new settings system is ready for production use!")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
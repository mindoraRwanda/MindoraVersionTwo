#!/usr/bin/env python3
"""
Service initialization script for the Mindora chatbot.

This script demonstrates how to properly initialize all services and can be used
as a startup script or integrated into the main application.

Usage:
    python -m backend.app.services.initialize_services
    or
    python backend/app/services/initialize_services.py
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

from backend.app.services.service_container import (
    service_container,
    check_service_health,
    service_lifecycle
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main initialization function."""
    logger.info("🚀 Starting Mindora service initialization...")

    try:
        # Use the service lifecycle context manager for proper initialization and cleanup
        async with service_lifecycle() as services:
            logger.info("✅ All services initialized successfully")

            # Check service health
            logger.info("🔍 Checking service health...")
            health_status = await check_service_health()

            # Print service status
            logger.info("📊 Service Status:")
            for service_name, status in health_status.items():
                status_icon = "✅" if status["healthy"] else "❌"
                init_icon = "✅" if status["initialized"] else "❌"
                logger.info(f"  {status_icon} {service_name}: initialized={init_icon} healthy={status_icon}")

                if not status["healthy"] and "error" in status:
                    logger.error(f"    Error: {status['error']}")

            # Example of how to use the services
            logger.info("🎯 Testing service integration...")

            try:
                # Get the LLM service
                llm_service = services.get_service("llm_service")
                if llm_service:
                    logger.info("✅ LLM service is available")
                else:
                    logger.warning("⚠️ LLM service not available")

                # Get the session manager
                session_manager = services.get_service("session_manager")
                if session_manager:
                    logger.info("✅ Session manager is available")
                else:
                    logger.warning("⚠️ Session manager not available")

                # Get the state router
                state_router = services.get_service("langgraph_state_router")
                if state_router:
                    logger.info("✅ State router is available")
                else:
                    logger.warning("⚠️ State router not available")

            except Exception as e:
                logger.error(f"❌ Error testing services: {e}")

            logger.info("🎉 Service initialization completed successfully!")
            logger.info("The chatbot is ready to handle conversations.")

            # Keep the services running (in a real application, this would be handled by the web server)
            logger.info("Press Ctrl+C to shutdown services...")

            # Wait for keyboard interrupt
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                logger.info("🛑 Received shutdown signal...")

    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        return 1

    return 0


def sync_main():
    """Synchronous entry point for direct script execution."""
    try:
        return asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested by user")
        return 0
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = sync_main()
    sys.exit(exit_code)
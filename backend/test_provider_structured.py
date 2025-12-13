
import asyncio
import os
from pydantic import BaseModel
from app.services.llm_providers import LLMProviderFactory

# Mock settings if needed, but the provider should work with defaults or env vars
# Ensure OLLAMA_BASE_URL is set or default is used

class TestSchema(BaseModel):
    answer: str
    confidence: float

async def main():
    try:
        # Create Ollama provider
        provider = LLMProviderFactory.create_provider(
            provider_name="ollama",
            model_name="llama3.2"
        )
        
        print(f"Provider: {provider.provider_name}")
        
        messages = [{"role": "user", "content": "What is 10 + 5?"}]
        
        print("Sending request...")
        response = await provider.generate_response(messages, structured_output=TestSchema)
        
        print(f"Response type: {type(response)}")
        print(f"Response: {response}")
        
        if isinstance(response, TestSchema):
            print("✅ SUCCESS: Received structured output")
        else:
            print("❌ FAILURE: Did not receive structured output")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())


import asyncio
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

class TestSchema(BaseModel):
    answer: str
    confidence: float

async def main():
    try:
        llm = ChatOllama(model="llama3.2", temperature=0)
        structured_llm = llm.with_structured_output(TestSchema)
        response = await structured_llm.ainvoke([HumanMessage(content="What is 2+2?")])
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())

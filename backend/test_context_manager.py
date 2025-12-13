
from app.services.llm_cultural_context import ConversationContextManager
from enum import Enum

class SenderType(Enum):
    user = "user"
    bot = "bot"

def test_build_memory_block():
    history = [
        {"role": "user", "text": "Hello"},
        {"role": "bot", "text": "Hi there"},
        {"role": SenderType.user, "text": "How are you?"},
        {"role": SenderType.bot, "text": "I am good, thanks."}
    ]
    
    memory_block = ConversationContextManager.build_memory_block(history)
    print("Memory Block:")
    print(memory_block)
    
    expected = "User: Hello\nAssistant: Hi there\nUser: How are you?\nAssistant: I am good, thanks."
    
    if memory_block == expected:
        print("\n✅ SUCCESS: Memory block formatted correctly with 'Assistant' role.")
    else:
        print("\n❌ FAILURE: Memory block format incorrect.")
        print(f"Expected:\n{expected}")
        print(f"Got:\n{memory_block}")

if __name__ == "__main__":
    test_build_memory_block()

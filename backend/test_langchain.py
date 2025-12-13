try:
    print("Attempting to import langchain_core.messages...")
    from langchain_core.messages import HumanMessage
    print("Import successful")
except BaseException as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()

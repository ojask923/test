import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from app.services.memory import memory_service

async def test():
    await memory_service.initialize()
    if memory_service._use_mem0:
        print("mem0 initialized!")
        print("Testing mem0 add...")
        
        # Test adding a memory
        messages = [
            {"role": "user", "content": "My favorite color is neon green and I love python."}
        ]
        
        try:
            await memory_service.add(user_id="test_user", messages=messages)
            print("Successfully added to mem0 (Qdrant)!")
            
            print("Testing mem0 search...")
            res = await memory_service.search(user_id="test_user", query="What is my favorite color?")
            print(f"Search results: {res}")
            
        except Exception as e:
            print(f"Error during mem0 operation: {e}")
    else:
        print("mem0 is NOT initialized!")

if __name__ == "__main__":
    asyncio.run(test())

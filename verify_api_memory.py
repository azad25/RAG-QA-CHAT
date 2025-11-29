import requests
import json
import time

def test_memory_api():
    url = "http://localhost:8000/ask"
    session_id = "test_session_api_1"
    
    print("Waiting for API to be ready...")
    time.sleep(10)  # Wait for container to start
    
    print("\n--- Turn 1 ---")
    q1 = "Tell me about the condo in West Hollywood."
    payload1 = {"question": q1, "session_id": session_id}
    print(f"User: {q1}")
    
    try:
        resp1 = requests.post(url, json=payload1)
        resp1.raise_for_status()
        ans1 = resp1.json()["answer"]
        print(f"AI: {ans1}")
    except Exception as e:
        print(f"Error in Turn 1: {e}")
        return

    print("\n--- Turn 2 ---")
    q2 = "What is its price?"
    payload2 = {"question": q2, "session_id": session_id}
    print(f"User: {q2}")
    
    try:
        resp2 = requests.post(url, json=payload2)
        resp2.raise_for_status()
        ans2 = resp2.json()["answer"]
        print(f"AI: {ans2}")
        
        if "$1,495,000" in ans2 or "1,495,000" in ans2:
            print("\nSUCCESS: Memory is working via API!")
        else:
            print("\nFAILURE: Memory might not be working via API.")
            
    except Exception as e:
        print(f"Error in Turn 2: {e}")

if __name__ == "__main__":
    test_memory_api()

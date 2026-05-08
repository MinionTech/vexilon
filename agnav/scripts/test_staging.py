
import os
from gradio_client import Client

def test_staging():
    print("[*] Connecting to Staging at http://localhost:7860...")
    try:
        client = Client("http://localhost:7860")
        print("[*] Sending test query to '/chat_handler'...")
        # predict(message, history, persona, api_name="/chat_handler")
        job = client.submit(
            "What rights do stewards have?", 
            [], # history
            "Lookup", # persona
            api_name="/chat_handler"
        )
        for chunk in job:
            # Chunk is (chatbot_value, textbox_value)
            # chatbot_value is a list of messages
            if chunk and len(chunk) > 0:
                chatbot_messages = chunk[0]
                if len(chatbot_messages) > 1: # Index 0 is user, 1 is assistant
                    last_msg = chatbot_messages[-1]
                    content = last_msg.get('content', [])
                    if content and len(content) > 0:
                        text = content[0].get('text', '')
                        if text:
                            print(f"\n[SUCCESS] Received response: {text[:100]}...")
                            return
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    test_staging()

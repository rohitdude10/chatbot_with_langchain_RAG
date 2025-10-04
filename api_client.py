"""
API Client for LangChain Chatbot
Demonstrates how to use the chatbot via REST API.
"""

import requests
import json
import time
from typing import Optional

class ChatbotAPIClient:
    """Client for interacting with the LangChain Chatbot API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = f"client_session_{int(time.time())}"
    
    def check_status(self) -> dict:
        """Check chatbot status."""
        try:
            response = requests.get(f"{self.base_url}/api/status")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def send_message(self, message: str, include_context: bool = True) -> dict:
        """Send a message to the chatbot."""
        try:
            payload = {
                "message": message,
                "include_context": include_context,
                "session_id": self.session_id
            }
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def get_history(self, limit: int = 10) -> dict:
        """Get chat history."""
        try:
            response = requests.get(f"{self.base_url}/api/history?limit={limit}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def clear_history(self) -> dict:
        """Clear chat history."""
        try:
            response = requests.delete(f"{self.base_url}/api/history")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def reload_documents(self) -> dict:
        """Reload documents."""
        try:
            response = requests.post(f"{self.base_url}/api/reload")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def get_documents(self) -> dict:
        """Get list of documents."""
        try:
            response = requests.get(f"{self.base_url}/api/documents")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def health_check(self) -> dict:
        """Check API health."""
        try:
            response = requests.get(f"{self.base_url}/api/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

def interactive_client():
    """Interactive client for testing the API."""
    client = ChatbotAPIClient()
    
    print("🤖 LangChain Chatbot API Client")
    print("=" * 40)
    
    # Check if server is running
    status = client.check_status()
    if "error" in status:
        print(f"❌ Cannot connect to API server: {status['error']}")
        print("Make sure the API server is running: python api_server.py")
        return
    
    print(f"✅ Connected to API server")
    print(f"📊 Status: {status.get('status', 'Unknown')}")
    print(f"📚 Documents loaded: {status.get('documents_loaded', 0)}")
    print(f"🗄️  Vector store ready: {status.get('vector_store_ready', False)}")
    print("-" * 40)
    
    # Show available documents
    docs = client.get_documents()
    if "error" not in docs:
        print(f"📄 Available documents ({docs['total_count']}):")
        for doc in docs['documents']:
            print(f"   • {doc['filename']} ({doc['size']} bytes)")
        print("-" * 40)
    
    # Interactive chat loop
    print("💬 Start chatting! Type 'quit' to exit, 'help' for commands.")
    print()
    
    while True:
        try:
            user_input = input("👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            elif user_input.lower() == 'help':
                print("\n📋 Available Commands:")
                print("   • Just type your question for normal chat")
                print("   • 'history' - View recent chat history")
                print("   • 'clear' - Clear chat history")
                print("   • 'reload' - Reload documents")
                print("   • 'status' - Check server status")
                print("   • 'docs' - List available documents")
                print("   • 'quit/exit/bye' - Exit the client")
                continue
            
            elif user_input.lower() == 'history':
                history = client.get_history(limit=5)
                if "error" not in history:
                    print("\n📜 Recent Chat History:")
                    for i, entry in enumerate(history['history'], 1):
                        timestamp = entry['timestamp'].split('T')[1][:8]
                        print(f"{i}. [{timestamp}] {entry['query'][:60]}...")
                else:
                    print(f"❌ Error getting history: {history['error']}")
                continue
            
            elif user_input.lower() == 'clear':
                result = client.clear_history()
                if "error" not in result:
                    print("🗑️  Chat history cleared.")
                else:
                    print(f"❌ Error clearing history: {result['error']}")
                continue
            
            elif user_input.lower() == 'reload':
                result = client.reload_documents()
                if "error" not in result:
                    print("🔄 Documents reload started in background.")
                else:
                    print(f"❌ Error reloading documents: {result['error']}")
                continue
            
            elif user_input.lower() == 'status':
                status = client.check_status()
                if "error" not in status:
                    print(f"\n📊 Server Status:")
                    print(f"   Status: {status['status']}")
                    print(f"   Documents: {status['documents_loaded']}")
                    print(f"   Vector Store: {status['vector_store_ready']}")
                else:
                    print(f"❌ Error checking status: {status['error']}")
                continue
            
            elif user_input.lower() == 'docs':
                docs = client.get_documents()
                if "error" not in docs:
                    print(f"\n📄 Available Documents ({docs['total_count']}):")
                    for doc in docs['documents']:
                        print(f"   • {doc['filename']} ({doc['size']} bytes)")
                else:
                    print(f"❌ Error getting documents: {docs['error']}")
                continue
            
            elif not user_input:
                continue
            
            # Send message to chatbot
            print("\n🤖 Bot:")
            print("-" * 20)
            
            response = client.send_message(user_input)
            if "error" not in response:
                print(response['response'])
            else:
                print(f"❌ Error: {response['error']}")
            
            print("-" * 20)
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

def demo_api_usage():
    """Demonstrate API usage with example calls."""
    client = ChatbotAPIClient()
    
    print("🚀 LangChain Chatbot API Demo")
    print("=" * 40)
    
    # Check status
    print("1. Checking server status...")
    status = client.check_status()
    print(f"   Status: {json.dumps(status, indent=2)}")
    print()
    
    # Get documents
    print("2. Getting available documents...")
    docs = client.get_documents()
    print(f"   Documents: {json.dumps(docs, indent=2)}")
    print()
    
    # Send a test message
    print("3. Sending test message...")
    response = client.send_message("What is machine learning?")
    print(f"   Response: {json.dumps(response, indent=2)}")
    print()
    
    # Get history
    print("4. Getting chat history...")
    history = client.get_history(limit=3)
    print(f"   History: {json.dumps(history, indent=2)}")
    print()
    
    print("✅ Demo completed!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_api_usage()
    else:
        interactive_client()

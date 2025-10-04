"""
Startup script for LangChain Chatbot API Server
Provides easy way to start the API server with proper configuration.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import fastapi
        import uvicorn
        print("✅ FastAPI and Uvicorn are installed")
        return True
    except ImportError:
        print("❌ FastAPI or Uvicorn not installed")
        print("Please install dependencies: pip install -r requirements.txt")
        return False

def check_env_file():
    """Check if .env file exists and has API key."""
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        print("Please create .env file from env_template.txt and add your Google Gemini API key")
        return False
    
    # Check if API key is set
    with open(env_file, 'r') as f:
        content = f.read()
        if 'your_google_gemini_api_key_here' in content:
            print("❌ Please set your Google Gemini API key in .env file")
            return False
    
    print("✅ .env file configured")
    return True

def create_directories():
    """Create necessary directories."""
    directories = ["logs", "documents", "vector_store"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✅ Directories created")

def start_api_server(host="0.0.0.0", port=8000, reload=False):
    """Start the API server."""
    print(f"🚀 Starting LangChain Chatbot API Server...")
    print(f"📡 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🔄 Reload: {reload}")
    print("-" * 50)
    
    try:
        import uvicorn
        uvicorn.run(
            "api_server:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 API Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting API server: {e}")

def main():
    """Main function to start the API server."""
    print("🤖 LangChain Chatbot API Server Startup")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Check environment
    if not check_env_file():
        return False
    
    # Create directories
    create_directories()
    
    print("\n📋 API Server Information:")
    print("   • Web Interface: http://localhost:8000")
    print("   • API Documentation: http://localhost:8000/docs")
    print("   • Interactive API: http://localhost:8000/redoc")
    print("   • Health Check: http://localhost:8000/api/health")
    print("\n🔗 Available Endpoints:")
    print("   • POST /api/chat - Send message to chatbot")
    print("   • GET  /api/status - Get chatbot status")
    print("   • GET  /api/history - Get chat history")
    print("   • DELETE /api/history - Clear chat history")
    print("   • POST /api/reload - Reload documents")
    print("   • GET  /api/documents - Get document list")
    print("   • GET  /api/health - Health check")
    print("\n💡 Usage:")
    print("   • Open http://localhost:8000 in your browser for web interface")
    print("   • Use api_client.py for programmatic access")
    print("   • Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Start server
    start_api_server()
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)

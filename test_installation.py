"""
Test script to verify LangChain Chatbot installation
Run this script to check if all dependencies are properly installed.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("🔍 Testing package imports...")
    
    required_packages = [
        ("langchain", "LangChain"),
        ("langchain_community", "LangChain Community"),
        ("langchain_google_genai", "LangChain Google GenAI"),
        ("google.generativeai", "Google Generative AI"),
        ("faiss", "FAISS"),
        ("dotenv", "Python Dotenv"),
        ("pypdf", "PyPDF"),
        ("markdown", "Markdown"),
        ("tiktoken", "TikToken"),
        ("sentence_transformers", "Sentence Transformers")
    ]
    
    failed_imports = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            failed_imports.append(name)
    
    return len(failed_imports) == 0, failed_imports

def test_environment():
    """Test environment configuration."""
    print("\n🔧 Testing environment configuration...")
    
    # Check if .env file exists
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file exists")
        
        # Check if API key is set
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('GOOGLE_API_KEY')
        if api_key and api_key != 'your_google_gemini_api_key_here':
            print("✅ Google API key is configured")
        else:
            print("⚠️  Google API key not configured (set GOOGLE_API_KEY in .env)")
    else:
        print("❌ .env file not found")
        return False
    
    return True

def test_directories():
    """Test if required directories exist."""
    print("\n📁 Testing directory structure...")
    
    required_dirs = ["documents", "logs", "vector_store"]
    missing_dirs = []
    
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"✅ {directory}/")
        else:
            print(f"❌ {directory}/ (missing)")
            missing_dirs.append(directory)
    
    return len(missing_dirs) == 0, missing_dirs

def test_sample_documents():
    """Test if sample documents exist."""
    print("\n📚 Testing sample documents...")
    
    documents_dir = Path("documents")
    if not documents_dir.exists():
        print("❌ documents/ directory not found")
        return False
    
    sample_files = list(documents_dir.glob("*"))
    if sample_files:
        print(f"✅ Found {len(sample_files)} document(s)")
        for file in sample_files:
            print(f"   - {file.name}")
    else:
        print("⚠️  No documents found in documents/ directory")
    
    return True

def test_basic_functionality():
    """Test basic chatbot functionality."""
    print("\n🤖 Testing basic functionality...")
    
    try:
        # Test if we can import the chatbot
        from chatbot import LangChainChatbot
        print("✅ Chatbot class can be imported")
        
        # Test if we can initialize (without API key for now)
        try:
            chatbot = LangChainChatbot()
            print("✅ Chatbot can be initialized")
        except ValueError as e:
            if "GOOGLE_API_KEY" in str(e):
                print("⚠️  Chatbot initialization requires valid API key")
            else:
                print(f"❌ Chatbot initialization failed: {e}")
                return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import chatbot: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 LangChain Chatbot Installation Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test imports
    imports_ok, failed_imports = test_imports()
    if not imports_ok:
        print(f"\n❌ Import test failed. Missing packages: {', '.join(failed_imports)}")
        print("   Run: pip install -r requirements.txt")
        all_tests_passed = False
    
    # Test environment
    env_ok = test_environment()
    if not env_ok:
        print("\n❌ Environment test failed")
        all_tests_passed = False
    
    # Test directories
    dirs_ok, missing_dirs = test_directories()
    if not dirs_ok:
        print(f"\n❌ Directory test failed. Missing directories: {', '.join(missing_dirs)}")
        all_tests_passed = False
    
    # Test sample documents
    docs_ok = test_sample_documents()
    
    # Test basic functionality
    func_ok = test_basic_functionality()
    if not func_ok:
        all_tests_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 All tests passed! The chatbot is ready to use.")
        print("\nTo start the chatbot:")
        print("   python chatbot.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("\nFor help, see README.md or run:")
        print("   python setup.py")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

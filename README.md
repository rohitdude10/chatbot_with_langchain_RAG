# LangChain Chatbot with Google Gemini API and RAG

A comprehensive Python chatbot application that uses LangChain, Google Gemini API, and Retrieval-Augmented Generation (RAG) to provide intelligent responses based on your local documents.

## 🚀 Features

- **RAG Implementation**: Load and process local documents (PDF, Markdown, TXT)
- **Google Gemini Integration**: Powered by Google's advanced Gemini API
- **Vector Search**: FAISS-based vector database for efficient document retrieval
- **Document Processing**: Automatic chunking and embedding generation
- **Chat History**: Track conversation history with timestamps
- **Comprehensive Logging**: Detailed logging system for debugging and monitoring
- **CLI Interface**: User-friendly command-line interface
- **New Terminal**: Automatically opens in a new terminal window for clean interface
- **Hot Reload**: Update document index on-the-fly
- **Error Handling**: Graceful error handling and recovery

## 📋 Requirements

- Python 3.8 or higher
- Google Gemini API key
- Virtual environment (recommended)

## 🛠️ Installation

### 1. Clone or Download the Project

```bash
git clone <your-repo-url>
cd langchain_chatbot
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

1. Copy the environment template:
   ```bash
   copy env_template.txt .env
   ```

2. Edit the `.env` file and add your Google Gemini API key:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

   **To get your Google Gemini API key:**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Sign in with your Google account
   - Create a new API key
   - Copy the key to your `.env` file

### 5. Add Documents (Optional)

Place your documents in the `documents/` folder. Supported formats:
- PDF files (`.pdf`)
- Markdown files (`.md`)
- Text files (`.txt`)

The application comes with sample documents for testing.

## 🚀 Usage

### Start the Chatbot

```bash
python chatbot.py
```

**The chatbot will automatically open in a new terminal window for a clean conversation experience!**

### Features

- **Automatic New Terminal**: Opens in a separate terminal window
- **Clean Interface**: No technical logs cluttering the conversation
- **Background Logging**: All logs saved to `logs/chatbot.log` silently
- **Cross-Platform**: Works on Windows, Mac, and Linux

### Available Commands

Once the chatbot is running, you can use these commands:

- **Normal chat**: Just type your question and press Enter
- **`history`**: View recent chat history
- **`clear`**: Clear chat history
- **`reload`**: Reload documents and recreate vector store
- **`quit`**, **`exit`**, or **`bye`**: Exit the application

### Example Session

When you run `python chatbot.py`, it will:

1. **Open a new terminal window** automatically
2. **Show a clean interface** without technical details
3. **Start the conversation** immediately

```
🚀 Opening chatbot in new terminal window...
✅ New terminal window opened. Closing this window...

[New Terminal Window Opens]

🤖 LangChain Chatbot - Ready!
========================================
💬 Chatbot ready! Type 'quit', 'exit', or 'bye' to end the conversation.
   Type 'history' to view chat history.
   Type 'clear' to clear chat history.
   Type 'reload' to reload documents and recreate vector store.
   Type 'help' for more commands.
----------------------------------------

👤 You: What is machine learning?

🤖 Bot:
--------------------
Machine Learning is a subset of AI that focuses on algorithms that can learn from data without being explicitly programmed. There are three main types of machine learning:

1. **Supervised Learning**: Uses labeled training data for classification and regression tasks
2. **Unsupervised Learning**: Works with unlabeled data to find hidden patterns
3. **Reinforcement Learning**: Learns through interaction with an environment using rewards and penalties

Common applications include recommendation systems, image recognition, natural language processing, and predictive analytics.
--------------------

👤 You: quit

👋 Goodbye! Thanks for chatting!
```

## ⚙️ Configuration

You can customize the chatbot behavior by modifying the `.env` file:

```env
# Google Gemini API Configuration
GOOGLE_API_KEY=your_google_gemini_api_key_here
GEMINI_MODEL=gemini-pro
MAX_TOKENS=2048
TEMPERATURE=0.7

# Document processing settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Vector store settings
VECTOR_STORE_PATH=./vector_store
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Configuration Options

- **`GEMINI_MODEL`**: Gemini model to use (default: `gemini-pro`)
- **`MAX_TOKENS`**: Maximum tokens in response (default: `2048`)
- **`TEMPERATURE`**: Response creativity (0.0-1.0, default: `0.7`)
- **`CHUNK_SIZE`**: Size of document chunks (default: `1000`)
- **`CHUNK_OVERLAP`**: Overlap between chunks (default: `200`)
- **`VECTOR_STORE_PATH`**: Path to store vector database (default: `./vector_store`)
- **`EMBEDDING_MODEL`**: HuggingFace model for embeddings (default: `sentence-transformers/all-MiniLM-L6-v2`)

## 📁 Project Structure

```
langchain_chatbot/
├── chatbot.py              # Main chatbot application (opens in new terminal)
├── requirements.txt        # Python dependencies
├── env_template.txt        # Environment variables template
├── setup.py                # Automated setup script
├── test_installation.py    # Installation verification script
├── demo.py                 # Demo script showing usage
├── README.md              # This file
├── documents/             # Your documents folder
│   ├── sample_knowledge.md
│   ├── python_programming.txt
│   └── web_development.md
├── vector_store/          # Generated vector database
├── logs/                  # Log files
│   └── chatbot.log
└── venv/                  # Virtual environment
```

## 🔧 Technical Details

### Architecture

The chatbot follows a function-based architecture with these key components:

1. **`load_documents()`**: Loads and parses local files (PDF, MD, TXT)
2. **`create_vector_store()`**: Generates embeddings and creates FAISS vector store
3. **`retrieve_documents()`**: Retrieves relevant document chunks for queries
4. **`generate_response()`**: Combines context with Gemini API for responses
5. **`main()`**: Handles CLI interface and user interaction

### RAG Workflow

1. **Document Loading**: Parse documents from the `documents/` folder
2. **Text Chunking**: Split documents into manageable chunks
3. **Embedding Generation**: Create vector embeddings using HuggingFace models
4. **Vector Storage**: Store embeddings in FAISS vector database
5. **Query Processing**: Convert user queries to embeddings
6. **Similarity Search**: Find most relevant document chunks
7. **Response Generation**: Combine context with Gemini API for final response

### Logging

The application logs all important events to `logs/chatbot.log`:
- Document loading progress
- Vector store creation
- Query processing
- API calls and responses
- Error messages and debugging information

## 🐛 Troubleshooting

### Common Issues

1. **"GOOGLE_API_KEY is required"**
   - Make sure you've created a `.env` file with your API key
   - Verify the API key is correct and active

2. **"No documents found"**
   - Add documents to the `documents/` folder
   - Supported formats: PDF, MD, TXT

3. **Import errors**
   - Make sure you've installed all dependencies: `pip install -r requirements.txt`
   - Verify you're using the correct Python version (3.8+)

4. **Memory issues with large documents**
   - Reduce `CHUNK_SIZE` in `.env`
   - Use smaller embedding models
   - Process documents in smaller batches

### Getting Help

- Check the logs in `logs/chatbot.log` for detailed error messages
- Ensure all dependencies are properly installed
- Verify your Google Gemini API key is valid and has sufficient quota

## 📚 Dependencies

- **langchain**: LLM chain management and RAG workflow
- **langchain-community**: Community integrations
- **langchain-google-genai**: Google Gemini integration
- **google-generativeai**: Google Gemini API client
- **faiss-cpu**: Vector similarity search
- **python-dotenv**: Environment variable management
- **pypdf**: PDF document processing
- **markdown**: Markdown document processing
- **tiktoken**: Token counting
- **sentence-transformers**: Embedding generation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the RAG framework
- [Google Gemini](https://ai.google.dev/) for the LLM API
- [HuggingFace](https://huggingface.co/) for embedding models
- [FAISS](https://github.com/facebookresearch/faiss) for vector search

---

**Happy Chatting! 🤖💬**

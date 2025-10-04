"""
LangChain Chatbot with Google Gemini API and RAG
A comprehensive chatbot application using LangChain, Google Gemini API, and RAG for document-based responses.
"""

import os
import logging
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

# Core libraries
from dotenv import load_dotenv
import google.generativeai as genai

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# Document loaders
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.document_loaders import UnstructuredMarkdownLoader

# Load environment variables
load_dotenv()

class LangChainChatbot:
    """Main chatbot class handling document loading, vector store creation, and response generation."""
    
    def __init__(self):
        """Initialize the chatbot with configuration from environment variables."""
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.gemini_model = os.getenv('GEMINI_MODEL', 'gemini-pro')
        self.max_tokens = int(os.getenv('MAX_TOKENS', '2048'))
        self.temperature = float(os.getenv('TEMPERATURE', '0.7'))
        self.chunk_size = int(os.getenv('CHUNK_SIZE', '1000'))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '200'))
        self.vector_store_path = os.getenv('VECTOR_STORE_PATH', './vector_store')
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize components
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        self.chat_history = []
        
        # Validate configuration
        self._validate_config()
        
        # Initialize Google Gemini
        self._initialize_gemini()
        
        # Initialize embeddings
        self._initialize_embeddings()
        
        self.logger.info("Chatbot initialized successfully")
    
    def setup_logging(self):
        """Set up logging configuration."""
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Configure logging to only write to file, not console
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/chatbot.log')
            ]
        )
    
    def _validate_config(self):
        """Validate required configuration."""
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required. Please set it in your .env file.")
        
        self.logger.info("Configuration validated successfully")
    
    def _initialize_gemini(self):
        """Initialize Google Gemini API."""
        try:
            genai.configure(api_key=self.google_api_key)
            self.llm = ChatGoogleGenerativeAI(
                model=self.gemini_model,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                google_api_key=self.google_api_key
            )
            self.logger.info(f"Google Gemini initialized with model: {self.gemini_model}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Google Gemini: {e}")
            raise
    
    def _initialize_embeddings(self):
        """Initialize embedding model."""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'}
            )
            self.logger.info(f"Embeddings initialized with model: {self.embedding_model}")
        except Exception as e:
            self.logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
    def load_documents(self, documents_path: str = "./documents") -> List[Document]:
        """
        Load documents from the specified directory.
        
        Args:
            documents_path: Path to directory containing documents
            
        Returns:
            List of loaded documents
        """
        self.logger.info(f"Loading documents from: {documents_path}")
        
        documents = []
        documents_dir = Path(documents_path)
        
        if not documents_dir.exists():
            self.logger.warning(f"Documents directory {documents_path} does not exist")
            return documents
        
        # Supported file extensions
        supported_extensions = {'.pdf', '.txt', '.md'}
        
        for file_path in documents_dir.rglob('*'):
            if file_path.suffix.lower() in supported_extensions:
                try:
                    self.logger.info(f"Loading file: {file_path}")
                    
                    if file_path.suffix.lower() == '.pdf':
                        loader = PyPDFLoader(str(file_path))
                    elif file_path.suffix.lower() == '.txt':
                        loader = TextLoader(str(file_path), encoding='utf-8')
                    elif file_path.suffix.lower() == '.md':
                        loader = UnstructuredMarkdownLoader(str(file_path))
                    
                    docs = loader.load()
                    documents.extend(docs)
                    self.logger.info(f"Successfully loaded {len(docs)} pages from {file_path}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to load {file_path}: {e}")
                    continue
        
        self.logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def create_vector_store(self, documents: List[Document], force_recreate: bool = False) -> FAISS:
        """
        Create or load vector store from documents.
        
        Args:
            documents: List of documents to process
            force_recreate: Whether to recreate the vector store even if it exists
            
        Returns:
            FAISS vector store
        """
        vector_store_file = Path(self.vector_store_path)
        
        # Check if vector store already exists and load it
        if vector_store_file.exists() and not force_recreate:
            try:
                self.logger.info("Loading existing vector store...")
                self.vector_store = FAISS.load_local(
                    str(vector_store_file),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self.logger.info("Vector store loaded successfully")
                return self.vector_store
            except Exception as e:
                self.logger.warning(f"Failed to load existing vector store: {e}")
                self.logger.info("Creating new vector store...")
        
        if not documents:
            raise ValueError("No documents provided for vector store creation")
        
        # Split documents into chunks
        self.logger.info("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        self.logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        
        # Create vector store
        self.logger.info("Creating vector embeddings...")
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        
        # Save vector store
        os.makedirs(self.vector_store_path, exist_ok=True)
        self.vector_store.save_local(self.vector_store_path)
        self.logger.info(f"Vector store saved to: {self.vector_store_path}")
        
        return self.vector_store
    
    def retrieve_documents(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please create vector store first.")
        
        try:
            self.logger.info(f"Retrieving documents for query: {query[:100]}...")
            docs = self.vector_store.similarity_search(query, k=k)
            self.logger.info(f"Retrieved {len(docs)} relevant documents")
            return docs
        except Exception as e:
            self.logger.error(f"Failed to retrieve documents: {e}")
            raise
    
    def generate_response(self, query: str, include_context: bool = True) -> str:
        """
        Generate response using RAG and Gemini API.
        
        Args:
            query: User query
            include_context: Whether to include retrieved context
            
        Returns:
            Generated response
        """
        try:
            self.logger.info(f"Generating response for query: {query[:100]}...")
            
            if include_context and self.vector_store:
                # Use RAG with retrieved context
                retrieved_docs = self.retrieve_documents(query)
                
                prompt_template = """
                    You are given a set of context information and a question. Follow these steps carefully to provide the best possible answer.

                    1. First, read the context thoroughly.  
                    2. If the context fully answers the question, use that information directly in your reply.  
                    3. If the context does not contain the full answer but you can answer the question using your own reliable knowledge, provide the answer from that knowledge.  
                    4. If neither the context nor your own knowledge can give a confident and correct answer, clearly say: "I don't know."  
                    5. Never make up facts or speculate without a solid basis. Ensure the answer is accurate, clear, and easy to understand.  
                    6. If answering from your own knowledge (not from context), you may indicate this by stating: "Based on my knowledge..." to help distinguish sources.  

                    Context:
                    {context}

                    Question:
                    {question}

                    Answer:
                    """

                
                PROMPT = PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"]
                )
                
                # Create QA chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
                    chain_type_kwargs={"prompt": PROMPT},
                    return_source_documents=True
                )
                
                # Generate response
                result = qa_chain({"query": query})
                response = result["result"]
                
                # Log source documents
                source_docs = result.get("source_documents", [])
                self.logger.info(f"Used {len(source_docs)} source documents for response")
                
            else:
                # Direct response without context
                response = self.llm.invoke(query).content
            
            # Add to chat history
            self.chat_history.append({
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "response": response,
                "include_context": include_context
            })
            
            self.logger.info("Response generated successfully")
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to generate response: {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get chat history."""
        return self.chat_history
    
    def clear_chat_history(self):
        """Clear chat history."""
        self.chat_history = []
        self.logger.info("Chat history cleared")


def main():
    """Main function to run the CLI chatbot."""
    # Clear screen for clean interface
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("🤖 LangChain Chatbot with Google Gemini API and RAG")
    print("=" * 60)
    
    try:
        # Initialize chatbot (logging happens in background)
        print("🔄 Initializing chatbot...")
        chatbot = LangChainChatbot()
        
        # Load documents and create vector store
        print("📚 Loading documents and creating vector store...")
        documents = chatbot.load_documents()
        
        if not documents:
            print("⚠️  No documents found in ./documents directory.")
            print("   Please add some PDF, TXT, or MD files to the documents folder.")
            print("   Continuing without RAG capabilities...")
        else:
            chatbot.create_vector_store(documents)
            print(f"✅ Vector store created with {len(documents)} documents")
        
        # Clear screen again for clean chat interface
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("🤖 LangChain Chatbot - Ready!")
        print("=" * 40)
        print("💬 Chatbot ready! Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("   Type 'history' to view chat history.")
        print("   Type 'clear' to clear chat history.")
        print("   Type 'reload' to reload documents and recreate vector store.")
        print("   Type 'help' for more commands.")
        print("-" * 40)
        
        # Main chat loop
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye! Thanks for chatting!")
                    break
                elif user_input.lower() == 'help':
                    print("\n📋 Available Commands:")
                    print("   • 'history' - View recent chat history")
                    print("   • 'clear' - Clear chat history")
                    print("   • 'reload' - Reload documents and recreate vector store")
                    print("   • 'quit/exit/bye' - Exit the chatbot")
                    print("   • Just type your question for normal chat")
                    continue
                elif user_input.lower() == 'history':
                    history = chatbot.get_chat_history()
                    if history:
                        print("\n📜 Recent Chat History:")
                        print("-" * 30)
                        for i, entry in enumerate(history[-5:], 1):  # Show last 5 entries
                            timestamp = entry['timestamp'].split('T')[1][:8]  # Show only time
                            print(f"{i}. [{timestamp}] {entry['query'][:60]}...")
                    else:
                        print("\n📜 No chat history available.")
                    continue
                elif user_input.lower() == 'clear':
                    chatbot.clear_chat_history()
                    print("\n🗑️  Chat history cleared.")
                    continue
                elif user_input.lower() == 'reload':
                    print("\n🔄 Reloading documents...")
                    documents = chatbot.load_documents()
                    if documents:
                        chatbot.create_vector_store(documents, force_recreate=True)
                        print(f"✅ Vector store recreated with {len(documents)} documents")
                    else:
                        print("⚠️  No documents found to reload.")
                    continue
                elif not user_input:
                    continue
                
                # Generate and display response with clean formatting
                print("\n🤖 Bot:")
                print("-" * 20)
                response = chatbot.generate_response(user_input)
                print(response)
                print("-" * 20)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for chatting!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again or type 'help' for assistance.")
                continue
    
    except Exception as e:
        print(f"\n❌ Failed to initialize chatbot: {e}")
        print("Please check your configuration and try again.")


if __name__ == "__main__":
    main()

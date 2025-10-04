


"""
FastAPI Server for LangChain Chatbot
Provides REST API endpoints for chatbot functionality.
"""

import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from chatbot import LangChainChatbot

# Global chatbot instance
chatbot_instance = None

# Pydantic models for API
class ChatRequest(BaseModel):
    message: str
    include_context: bool = True
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str
    include_context: bool

class ChatHistoryResponse(BaseModel):
    history: List[Dict[str, Any]]
    total_count: int

class DocumentInfo(BaseModel):
    filename: str
    size: int
    type: str

class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total_count: int

class StatusResponse(BaseModel):
    status: str
    message: str
    documents_loaded: int
    vector_store_ready: bool

# Initialize chatbot on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize chatbot on startup and cleanup on shutdown."""
    global chatbot_instance
    
    # Startup
    try:
        logging.info("Initializing chatbot for API server...")
        chatbot_instance = LangChainChatbot()
        
        # Load documents
        documents = chatbot_instance.load_documents()
        if documents:
            chatbot_instance.create_vector_store(documents)
            logging.info(f"Loaded {len(documents)} documents for API server")
        else:
            logging.info("No documents found, API server running without RAG")
        
        logging.info("API server chatbot initialized successfully")
        
    except Exception as e:
        logging.error(f"Failed to initialize chatbot for API server: {e}")
        chatbot_instance = None
    
    yield
    
    # Shutdown
    logging.info("API server shutting down...")

# Create FastAPI app
app = FastAPI(
    title="LangChain Chatbot API",
    description="REST API for LangChain Chatbot with Google Gemini and RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (for web interface)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve a modern chatbot interface with thinking animation."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LangChain Chatbot</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 10px;
            }
            
            .chatbot-container {
                background: #ffffff;
                border-radius: 16px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.12);
                width: 100%;
                max-width: 900px;
                height: 700px;
                display: flex;
                flex-direction: column;
                overflow: hidden;
                border: 1px solid rgba(255,255,255,0.2);
            }
            
            .chat-header {
                background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                color: white;
                padding: 16px 20px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                position: relative;
            }
            
            .header-left {
                display: flex;
                align-items: center;
                gap: 12px;
            }
            
            .chat-header h1 {
                font-size: 20px;
                font-weight: 600;
                margin: 0;
            }
            
            .status-indicator {
                display: flex;
                align-items: center;
                gap: 6px;
                font-size: 12px;
                opacity: 0.9;
                background: rgba(255,255,255,0.2);
                padding: 4px 8px;
                border-radius: 12px;
            }
            
            .status-dot {
                width: 6px;
                height: 6px;
                border-radius: 50%;
                background: #10b981;
                animation: pulse 2s infinite;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
            
            .robot-thinking {
                font-size: 20px;
                animation: robotThinking 2s infinite ease-in-out;
            }
            
            @keyframes robotThinking {
                0%, 100% { transform: rotate(0deg); }
                25% { transform: rotate(-10deg); }
                75% { transform: rotate(10deg); }
            }
            
            .chat-messages {
                flex: 1;
                padding: 16px;
                overflow-y: auto;
                background: #fafafa;
                display: flex;
                flex-direction: column;
                gap: 12px;
            }
            
            .message {
                max-width: 75%;
                padding: 12px 16px;
                border-radius: 16px;
                word-wrap: break-word;
                position: relative;
                animation: slideIn 0.3s ease-out;
                font-size: 14px;
                line-height: 1.5;
            }
            
            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .user-message {
                background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                color: white;
                align-self: flex-end;
                border-bottom-right-radius: 4px;
                box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3);
            }
            
            .bot-message {
                background: white;
                color: #374151;
                align-self: flex-start;
                border: 1px solid #e5e7eb;
                border-bottom-left-radius: 4px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            
            .bot-avatar {
                position: absolute;
                left: -28px;
                top: 50%;
                transform: translateY(-50%);
                width: 24px;
                height: 24px;
                background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 12px;
                color: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            .thinking-message {
                background: white;
                color: #6b7280;
                align-self: flex-start;
                border: 1px solid #e5e7eb;
                border-bottom-left-radius: 4px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 14px;
            }
            
            .thinking-dots {
                display: flex;
                gap: 3px;
            }
            
            .thinking-dot {
                width: 6px;
                height: 6px;
                border-radius: 50%;
                background: #6366f1;
                animation: thinking 1.4s infinite ease-in-out;
            }
            
            .thinking-dot:nth-child(1) { animation-delay: -0.32s; }
            .thinking-dot:nth-child(2) { animation-delay: -0.16s; }
            .thinking-dot:nth-child(3) { animation-delay: 0s; }
            
            @keyframes thinking {
                0%, 80%, 100% {
                    transform: scale(0.8);
                    opacity: 0.5;
                }
                40% {
                    transform: scale(1);
                    opacity: 1;
                }
            }
            
            .controls-section {
                padding: 12px 16px;
                background: white;
                border-top: 1px solid #e5e7eb;
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            
            .upload-section {
                display: flex;
                align-items: center;
                gap: 8px;
                padding: 8px 12px;
                background: #f8fafc;
                border-radius: 8px;
                border: 1px dashed #cbd5e1;
                transition: all 0.2s ease;
                cursor: pointer;
            }
            
            .upload-section:hover {
                border-color: #6366f1;
                background: #f1f5f9;
            }
            
            .upload-section.dragover {
                border-color: #6366f1;
                background: #eef2ff;
                transform: scale(1.01);
            }
            
            .upload-icon {
                font-size: 16px;
                color: #6366f1;
            }
            
            .upload-text {
                font-size: 12px;
                color: #64748b;
                flex: 1;
            }
            
            .upload-button {
                background: #6366f1;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 4px 8px;
                cursor: pointer;
                font-size: 11px;
                transition: background 0.2s ease;
                display: flex;
                align-items: center;
                gap: 4px;
            }
            
            .upload-button:hover {
                background: #5b5bd6;
            }
            
            .upload-button:disabled {
                background: #9ca3af;
                cursor: not-allowed;
            }
            
            .file-input {
                display: none;
            }
            
            .upload-status {
                font-size: 11px;
                color: #6b7280;
                margin-left: auto;
            }
            
            .upload-status.success {
                color: #059669;
            }
            
            .upload-status.error {
                color: #dc2626;
            }
            
            .upload-status.uploading {
                color: #2563eb;
            }
            
            .input-row {
                display: flex;
                gap: 8px;
                align-items: center;
            }
            
            .context-toggle {
                display: flex;
                align-items: center;
                gap: 6px;
                background: #f8fafc;
                padding: 6px 10px;
                border-radius: 8px;
                border: 1px solid #e2e8f0;
                cursor: pointer;
                transition: all 0.2s ease;
                font-size: 12px;
                color: #475569;
            }
            
            .context-toggle:hover {
                background: #f1f5f9;
                border-color: #6366f1;
            }
            
            .context-toggle.active {
                background: #6366f1;
                color: white;
                border-color: #6366f1;
            }
            
            .toggle-switch {
                position: relative;
                width: 32px;
                height: 16px;
                background: #cbd5e1;
                border-radius: 16px;
                transition: background 0.2s ease;
            }
            
            .toggle-switch.active {
                background: white;
            }
            
            .toggle-slider {
                position: absolute;
                top: 2px;
                left: 2px;
                width: 12px;
                height: 12px;
                background: white;
                border-radius: 50%;
                transition: transform 0.2s ease;
                box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            }
            
            .toggle-switch.active .toggle-slider {
                transform: translateX(16px);
                background: #6366f1;
            }
            
            .chat-input {
                flex: 1;
                padding: 10px 14px;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                font-size: 14px;
                outline: none;
                transition: border-color 0.2s ease;
                background: white;
            }
            
            .chat-input:focus {
                border-color: #6366f1;
                box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
            }
            
            .send-button {
                background: #6366f1;
                color: white;
                border: none;
                border-radius: 8px;
                width: 40px;
                height: 40px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 16px;
                transition: all 0.2s ease;
            }
            
            .send-button:hover {
                background: #5b5bd6;
                transform: scale(1.05);
            }
            
            .send-button:disabled {
                background: #9ca3af;
                cursor: not-allowed;
                transform: none;
            }
            
            .clear-button {
                background: #ef4444;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
                cursor: pointer;
                font-size: 11px;
                transition: background 0.2s ease;
            }
            
            .clear-button:hover {
                background: #dc2626;
            }
            
            .welcome-message {
                text-align: center;
                color: #6b7280;
                font-style: italic;
                margin: 16px 0;
                font-size: 13px;
            }
            
            /* Scrollbar styling */
            .chat-messages::-webkit-scrollbar {
                width: 4px;
            }
            
            .chat-messages::-webkit-scrollbar-track {
                background: #f1f5f9;
            }
            
            .chat-messages::-webkit-scrollbar-thumb {
                background: #cbd5e1;
                border-radius: 2px;
            }
            
            .chat-messages::-webkit-scrollbar-thumb:hover {
                background: #94a3b8;
            }
            
            /* Responsive design */
            @media (max-width: 768px) {
                body {
                    padding: 5px;
                }
                
                .chatbot-container {
                    height: 100vh;
                    border-radius: 0;
                }
                
                .chat-header {
                    padding: 12px 16px;
                }
                
                .chat-header h1 {
                    font-size: 18px;
                }
                
                .message {
                    max-width: 85%;
                    font-size: 13px;
                }
                
                .bot-avatar {
                    left: -24px;
                    width: 20px;
                    height: 20px;
                    font-size: 10px;
                }
            }
        </style>
    </head>
    <body>
        <div class="chatbot-container">
            <div class="chat-header">
                <div class="header-left">
                    <h1>🤖 LangChain Chatbot</h1>
                    <div class="status-indicator">
                        <div class="status-dot"></div>
                        <span id="status">Connecting...</span>
                    </div>
                </div>
                <div class="robot-thinking" id="robotThinking" style="display: none;">🤖</div>
            </div>
            
            <div class="chat-messages" id="chatMessages">
                <div class="welcome-message">
                    👋 Hi! I'm your AI assistant. Ask me anything!<br>
                    <small style="margin-top: 8px; display: block;">
                        💡 Upload documents below to add them to your knowledge base<br>
                        🔄 Toggle "Use Documents" to switch between document-based answers and general AI responses
                    </small>
                </div>
            </div>
            
            <div class="controls-section">
                <div class="upload-section" id="uploadSection" onclick="triggerFileUpload()" ondrop="handleDrop(event)" ondragover="handleDragOver(event)" ondragleave="handleDragLeave(event)">
                    <span class="upload-icon">📁</span>
                    <span class="upload-text">Drop files here or click to upload</span>
                    <button class="upload-button" id="uploadButton" onclick="event.stopPropagation(); triggerFileUpload()">
                        📤 Upload
                    </button>
                    <input type="file" id="fileInput" class="file-input" multiple accept=".pdf,.txt,.md" onchange="handleFileUpload(event)">
                    <div class="upload-status" id="uploadStatus"></div>
                </div>
                
                <div class="input-row">
                    <div class="context-toggle" id="contextToggle" onclick="toggleContext()">
                        <span id="contextLabel">📚 Use Documents</span>
                        <div class="toggle-switch active" id="toggleSwitch">
                            <div class="toggle-slider"></div>
                        </div>
                    </div>
                    <input type="text" id="messageInput" class="chat-input" placeholder="Type your message here..." onkeypress="handleKeyPress(event)">
                    <button id="sendButton" class="send-button" onclick="sendMessage()">📤</button>
                    <button class="clear-button" onclick="clearChat()">Clear</button>
                </div>
            </div>
        </div>
        
        <script>
            let sessionId = 'session_' + Date.now();
            let isThinking = false;
            let includeContext = true;
            
            function triggerFileUpload() {
                document.getElementById('fileInput').click();
            }
            
            function handleDragOver(event) {
                event.preventDefault();
                document.getElementById('uploadSection').classList.add('dragover');
            }
            
            function handleDragLeave(event) {
                event.preventDefault();
                document.getElementById('uploadSection').classList.remove('dragover');
            }
            
            function handleDrop(event) {
                event.preventDefault();
                document.getElementById('uploadSection').classList.remove('dragover');
                const files = event.dataTransfer.files;
                uploadFiles(files);
            }
            
            function handleFileUpload(event) {
                const files = event.target.files;
                uploadFiles(files);
            }
            
            async function uploadFiles(files) {
                if (files.length === 0) return;
                
                const uploadStatus = document.getElementById('uploadStatus');
                const uploadButton = document.getElementById('uploadButton');
                
                // Show uploading status
                uploadStatus.textContent = `Uploading ${files.length} file(s)...`;
                uploadStatus.className = 'upload-status uploading';
                uploadButton.disabled = true;
                
                try {
                    const formData = new FormData();
                    for (let file of files) {
                        formData.append('files', file);
                    }
                    
                    const response = await fetch('/api/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        uploadStatus.textContent = `✅ Uploaded ${result.uploaded_count} file(s) successfully`;
                        uploadStatus.className = 'upload-status success';
                        
                        // Auto-reload documents
                        await reloadDocuments();
                        
                        // Clear file input
                        document.getElementById('fileInput').value = '';
                        
                        // Reset status after 3 seconds
                        setTimeout(() => {
                            uploadStatus.textContent = '';
                            uploadStatus.className = 'upload-status';
                        }, 3000);
                        
                    } else {
                        throw new Error(result.detail || 'Upload failed');
                    }
                    
                } catch (error) {
                    uploadStatus.textContent = `❌ Upload failed: ${error.message}`;
                    uploadStatus.className = 'upload-status error';
                    
                    // Reset status after 5 seconds
                    setTimeout(() => {
                        uploadStatus.textContent = '';
                        uploadStatus.className = 'upload-status';
                    }, 5000);
                } finally {
                    uploadButton.disabled = false;
                }
            }
            
            async function reloadDocuments() {
                try {
                    const response = await fetch('/api/reload', {
                        method: 'POST'
                    });
                    
                    if (response.ok) {
                        // Update status
                        await checkStatus();
                        
                        // Show reload success message
                        const uploadStatus = document.getElementById('uploadStatus');
                        uploadStatus.textContent = '🔄 Documents reloaded successfully';
                        uploadStatus.className = 'upload-status success';
                        
                        setTimeout(() => {
                            uploadStatus.textContent = '';
                            uploadStatus.className = 'upload-status';
                        }, 2000);
                    }
                } catch (error) {
                    console.error('Failed to reload documents:', error);
                }
            }
            
            function toggleContext() {
                includeContext = !includeContext;
                const toggle = document.getElementById('contextToggle');
                const toggleSwitch = document.getElementById('toggleSwitch');
                const contextLabel = document.getElementById('contextLabel');
                
                if (includeContext) {
                    toggle.classList.add('active');
                    toggleSwitch.classList.add('active');
                    contextLabel.textContent = '📚 Use Documents';
                } else {
                    toggle.classList.remove('active');
                    toggleSwitch.classList.remove('active');
                    contextLabel.textContent = '🤖 General AI';
                }
            }
            
            async function checkStatus() {
                try {
                    const response = await fetch('/api/status');
                    const data = await response.json();
                    document.getElementById('status').textContent = 
                        `Ready • ${data.documents_loaded} docs • Vector store ${data.vector_store_ready ? 'ready' : 'not ready'}`;
                } catch (error) {
                    document.getElementById('status').textContent = 'Connection error';
                }
            }
            
            function showThinking() {
                isThinking = true;
                const chatMessages = document.getElementById('chatMessages');
                const thinkingDiv = document.createElement('div');
                thinkingDiv.className = 'message thinking-message';
                thinkingDiv.id = 'thinkingMessage';
                thinkingDiv.innerHTML = `
                    <div class="bot-avatar">🤖</div>
                    <span>Thinking</span>
                    <div class="thinking-dots">
                        <div class="thinking-dot"></div>
                        <div class="thinking-dot"></div>
                        <div class="thinking-dot"></div>
                    </div>
                `;
                chatMessages.appendChild(thinkingDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
                
                // Show robot thinking animation
                document.getElementById('robotThinking').style.display = 'block';
                
                // Disable send button
                document.getElementById('sendButton').disabled = true;
            }
            
            function hideThinking() {
                isThinking = false;
                const thinkingMessage = document.getElementById('thinkingMessage');
                if (thinkingMessage) {
                    thinkingMessage.remove();
                }
                
                // Hide robot thinking animation
                document.getElementById('robotThinking').style.display = 'none';
                
                // Enable send button
                document.getElementById('sendButton').disabled = false;
            }
            
            async function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                if (!message || isThinking) return;
                
                // Add user message to chat
                addMessage(message, 'user');
                input.value = '';
                
                // Show thinking animation
                showThinking();
                
                try {
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            message: message,
                            include_context: includeContext,
                            session_id: sessionId
                        })
                    });
                    
                    const data = await response.json();
                    
                    // Hide thinking animation
                    hideThinking();
                    
                    // Add bot response
                    addMessage(data.response, 'bot', data.include_context);
                    
                } catch (error) {
                    // Hide thinking animation
                    hideThinking();
                    
                    // Add error message
                    addMessage('Sorry, I encountered an error. Please try again.', 'bot');
                }
            }
            
            function addMessage(text, sender, usedContext = null) {
                const chatMessages = document.getElementById('chatMessages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}-message`;
                
                if (sender === 'bot') {
                    const contextIndicator = usedContext !== null ? 
                        (usedContext ? '<small style="opacity: 0.7; font-size: 12px;">📚 Using documents</small>' : '<small style="opacity: 0.7; font-size: 12px;">🤖 General AI</small>') : '';
                    
                    messageDiv.innerHTML = `
                        <div class="bot-avatar">🤖</div>
                        ${contextIndicator}
                        <div style="margin-top: ${usedContext !== null ? '5px' : '0'};">
                            ${text}
                        </div>
                    `;
                } else {
                    messageDiv.textContent = text;
                }
                
                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
            function clearChat() {
                const chatMessages = document.getElementById('chatMessages');
                chatMessages.innerHTML = `
                    <div class="welcome-message">
                        👋 Hi! I'm your AI assistant. Ask me anything!<br>
                        <small style="margin-top: 8px; display: block;">
                            💡 Upload documents below to add them to your knowledge base<br>
                            🔄 Toggle "Use Documents" to switch between document-based answers and general AI responses
                        </small>
                    </div>
                `;
                sessionId = 'session_' + Date.now();
            }
            
            function handleKeyPress(event) {
                if (event.key === 'Enter' && !isThinking) {
                    sendMessage();
                }
            }
            
            // Check status on load
            checkStatus();
            
            // Auto-focus input
            document.getElementById('messageInput').focus();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get chatbot status and configuration."""
    global chatbot_instance
    
    if not chatbot_instance:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        documents = chatbot_instance.load_documents()
        vector_store_ready = chatbot_instance.vector_store is not None
        
        return StatusResponse(
            status="ready",
            message="Chatbot is ready",
            documents_loaded=len(documents),
            vector_store_ready=vector_store_ready
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking status: {str(e)}")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the chatbot and get a response."""
    global chatbot_instance
    
    if not chatbot_instance:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        # Generate response
        response = chatbot_instance.generate_response(
            request.message, 
            include_context=request.include_context
        )
        
        # Get session ID
        session_id = request.session_id or f"session_{datetime.now().timestamp()}"
        
        return ChatResponse(
            response=response,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            include_context=request.include_context
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get("/api/history", response_model=ChatHistoryResponse)
async def get_chat_history(limit: int = 50):
    """Get chat history."""
    global chatbot_instance
    
    if not chatbot_instance:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        history = chatbot_instance.get_chat_history()
        
        # Limit results
        limited_history = history[-limit:] if limit > 0 else history
        
        return ChatHistoryResponse(
            history=limited_history,
            total_count=len(history)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting history: {str(e)}")

@app.delete("/api/history")
async def clear_chat_history():
    """Clear chat history."""
    global chatbot_instance
    
    if not chatbot_instance:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        chatbot_instance.clear_chat_history()
        return {"message": "Chat history cleared successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing history: {str(e)}")

@app.post("/api/reload")
async def reload_documents(background_tasks: BackgroundTasks):
    """Reload documents and recreate vector store."""
    global chatbot_instance
    
    if not chatbot_instance:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    def reload_task():
        try:
            documents = chatbot_instance.load_documents()
            if documents:
                chatbot_instance.create_vector_store(documents, force_recreate=True)
                logging.info(f"Reloaded {len(documents)} documents via API")
            else:
                logging.info("No documents found during reload via API")
        except Exception as e:
            logging.error(f"Error reloading documents via API: {e}")
    
    # Run reload in background
    background_tasks.add_task(reload_task)
    
    return {"message": "Document reload started in background"}

@app.get("/api/documents", response_model=DocumentListResponse)
async def get_documents():
    """Get list of loaded documents."""
    global chatbot_instance
    
    if not chatbot_instance:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        documents_dir = "documents"
        documents = []
        
        if os.path.exists(documents_dir):
            for file_path in os.listdir(documents_dir):
                full_path = os.path.join(documents_dir, file_path)
                if os.path.isfile(full_path):
                    file_size = os.path.getsize(full_path)
                    file_ext = os.path.splitext(file_path)[1].lower()
                    
                    documents.append(DocumentInfo(
                        filename=file_path,
                        size=file_size,
                        type=file_ext
                    ))
        
        return DocumentListResponse(
            documents=documents,
            total_count=len(documents)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting documents: {str(e)}")

@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload documents to the documents folder."""
    global chatbot_instance
    
    if not chatbot_instance:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    uploaded_count = 0
    errors = []
    
    # Create documents directory if it doesn't exist
    documents_dir = Path("documents")
    documents_dir.mkdir(exist_ok=True)
    
    for file in files:
        try:
            # Validate file type
            if not file.filename:
                errors.append(f"File has no name")
                continue
                
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in ['.pdf', '.txt', '.md']:
                errors.append(f"{file.filename}: Unsupported file type. Only PDF, TXT, and MD files are allowed.")
                continue
            
            # Check file size (limit to 10MB)
            content = await file.read()
            if len(content) > 10 * 1024 * 1024:  # 10MB
                errors.append(f"{file.filename}: File too large. Maximum size is 10MB.")
                continue
            
            # Save file
            file_path = documents_dir / file.filename
            with open(file_path, "wb") as f:
                f.write(content)
            
            uploaded_count += 1
            logging.info(f"Uploaded file: {file.filename}")
            
        except Exception as e:
            errors.append(f"{file.filename}: {str(e)}")
            logging.error(f"Failed to upload {file.filename}: {e}")
    
    if uploaded_count > 0:
        return {
            "message": f"Successfully uploaded {uploaded_count} file(s)",
            "uploaded_count": uploaded_count,
            "errors": errors
        }
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"No files uploaded. Errors: {'; '.join(errors)}"
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "detail": str(exc)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/api_server.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    print("🚀 Starting LangChain Chatbot API Server...")
    print("📡 API Documentation: http://localhost:8000/docs")
    print("🌐 Web Interface: http://localhost:8000")
    print("🔗 API Endpoints:")
    print("   POST /api/chat - Send message to chatbot")
    print("   GET  /api/status - Get chatbot status")
    print("   GET  /api/history - Get chat history")
    print("   DELETE /api/history - Clear chat history")
    print("   POST /api/reload - Reload documents")
    print("   GET  /api/documents - Get document list")
    print("   GET  /api/health - Health check")
    print("-" * 50)
    
    # Run the server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

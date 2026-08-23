# RAG PDF Project

An intelligent document-based chatbot application that leverages Retrieval-Augmented Generation (RAG) to provide context-aware responses from uploaded PDF and document files.

## 🎯 Overview

This project combines modern AI technologies to create a personal assistant that understands your documents. It allows users to:
- Upload and process PDF documents and other file formats
- Ask questions about the document content
- Maintain conversation context with session management
- Store long-term memory and user preferences
- Get personalized responses grounded in document context

## 🏗️ Architecture

### Core Components

**Backend (Python - 55.5%)**
- `main.py` - FastAPI server with document ingestion and chat endpoints
- `model.py` - LangGraph-based AI workflow orchestration
- `utils/` - Helper modules for vector storage and memory extraction

**Frontend (HTML/CSS/JavaScript - 44.5%)**
- `index.html` - Full-featured web interface with chat UI

### Technology Stack

**AI/ML:**
- LangChain & LangGraph for orchestration
- Ollama with Gemma3:4B LLM
- PyMuPDF4LLM for document parsing
- Qdrant vector database for semantic search

**Backend:**
- FastAPI framework
- PostgreSQL for state management and memory storage
- AsyncIO for asynchronous operations
- LangGraph's Postgres checkpointer and store

**Frontend:**
- Vanilla JavaScript with modern APIs
- Dark-themed responsive UI
- Google OAuth authentication
- Local storage for session persistence

## 🚀 Features

### Document Processing
- **Multi-format support**: PDF, EPUB, XPS, CBZ, MOBI, FB2, SVG, TXT, Markdown
- **Intelligent chunking**: Recursive text splitting with context overlap
- **Vector indexing**: Automatic embedding and storage in Qdrant

### Conversation Management
- **Session handling**: Multiple independent conversation threads per user
- **Short-term memory**: LangGraph checkpointer maintains message history
- **Long-term memory**: User profile and extracted memories stored in PostgreSQL
- **Context summarization**: Automatic summary generation after 4+ messages

### AI Features
- **Personalization**: Responses incorporate user profile and past interactions
- **Document grounding**: Answers cite and reference uploaded documents
- **Memory extraction**: Automatically extracts and stores user details
- **Follow-up suggestions**: Proposes 3 relevant next steps after each response

### User Experience
- **Authentication**: Google OAuth integration
- **Responsive UI**: Works seamlessly on desktop and mobile
- **Real-time feedback**: Status indicators for file uploads and processing
- **Session history**: Browse and switch between past conversations

## 🔧 Setup & Installation

### Prerequisites
- Python 3.11+
- PostgreSQL database
- Ollama with Gemma3:4B model
- Qdrant vector database
- Node.js (optional, for development)

### Environment Variables
Create a `.env` file:
```env
DB_URI=postgresql://user:password@localhost:5432/rag_db
OLLAMA_MODEL=gemma3:4b
QDRANT_URL=http://localhost:6333
GOOGLE_CLIENT_ID=your_client_id
GOOGLE_CLIENT_SECRET=your_client_secret

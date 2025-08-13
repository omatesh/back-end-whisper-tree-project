# WhisperTree Frontend - Research Paper Management App

  WhisperTree is a SwiftUI-based iOS application designed to help researchers discover, organize, and analyze academic papers using AI-powered
  insights. The app connects to the CORE API for paper discovery and provides intelligent analysis of research connections.

## Contents

  - [Features](https://github.com/omatesh/front-end-whisper-tree-project#features)
  - [Technologies and Stack](https://github.com/omatesh/front-end-whisper-tree-project#technologies-and-stack)
  - [Core API Integration](https://github.com/omatesh/front-end-whisper-tree-project#core-api-integration)
  - [Gemini Embeddings API Integration](https://github.com/omatesh/front-end-whisper-tree-project#gemini-embeddings-API-integration)

## Features

### Paper Discovery and Search

  WhisperTree provides comprehensive paper discovery through integration with the CORE API, allowing researchers to search through millions of
  academic papers with intelligent filtering and result management.

  The app offers an intuitive search interface where users can query academic papers by keywords, topics, or research areas. Search results include
  paper abstracts, author information, publication details, and direct links to full-text PDFs when available.

### Search History Management

  The application automatically maintains a searchable history of all queries, including result counts and timestamps. Users can quickly revisit
  previous searches and reload historical results with a single tap.

### Collection Management System

  WhisperTree enables researchers to organize their papers into custom collections, providing a structured approach to research project management.

  Users can create multiple collections for different research projects, each with custom titles, descriptions, and ownership information. Collections
  display paper counts and provide easy access to contained research.

### Paper Management Features
  
  Each collection maintains detailed paper information including titles, authors, publication dates, sources, and CORE IDs. Papers can be starred for
  importance and easily removed when no longer relevant.
  
  WhisperTree provides comprehensive paper management capabilities for both CORE API discoveries and manual entries.

  Researchers can manually add papers not found in CORE API searches, including complete bibliographic information, abstracts, and custom notes.

  The system automatically prevents duplicate papers from being added to collections, comparing both CORE IDs and paper titles to maintain collection
  integrity.

## Technologies and Stack

  **Backend:** Python, Flask <br />
  **Frontend:** SwiftUI, Swift <br />
  **Databases:** PostgreSQL with SQLAlchemy ORM <br />
  **APIs:** CORE API, Gemini Embeddings API


## CORE API Integration

  WhisperTree leverages the CORE API to provide access to millions of open-access research papers from repositories worldwide.

  The app integrates seamlessly with CORE's REST API to perform real-time searches across academic literature, with configurable result limits and
  intelligent result parsing.

  When users submit queries, the app constructs optimized API requests to CORE's search endpoints, handling authentication, rate limiting, and error
  recovery automatically.

##### Result Processing

  Search results are parsed and enriched with additional metadata, including download URLs, abstracts, and bibliographic information, then cached
  locally for offline access.

  ##### Data Synchronization

  The app maintains sync between local collections and CORE API data, ensuring users always have access to the most current paper information.


##  Gemini Embeddings API Integration

  WhisperTree's Gemini integration provides intelligent insights into research connections and paper relationships through advanced text embeddings.
  When users submit research ideas, the system processes the text through several analysis stages:

##### Text Embedding Generation

  The user's input is converted into high-dimensional vector embeddings using Google's Gemini Embeddings API, capturing semantic meaning and research
  concepts.
  The processed embeddings are compared against embedded representations of papers in the user's selected collection, using cosine similarity
  algorithms to identify connections.
  Results are structured to show not just matching papers, but the specific semantic relationships and similarity scores between the user's idea and
  existing research.

##### Network Visualization

  Papers and concepts are displayed as connected nodes, with relationship strength indicated by embedding similarity scores through visual elements
  like line thickness and node proximity.


##  Installation

 Backend Setup Steps

  1. Clone the Backend Repository
  cd whispertree-backend

  2. Create Virtual Environment:
  python -m venv venv
  source venv/bin/activate  

  3. Install Dependencies:
  pip install -r requirements.txt
  
  4. Database Setup:
  # Create PostgreSQL database
  createdb whispertree_db

  # Run migrations
  flask db upgrade

  5. Configure Environment Variables:
  # Create .env file with:
  DATABASE_URL=postgresql://username:password@localhost/whispertree_db
  GEMINI_API_KEY=your_gemini_api_key
  CORE_API_KEY=your_core_api_key

  6. Start Backend Server:
  flask run

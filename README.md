f# JeeyoPay Customer Service Chatbot 

An intelligent RAG (Retrieval-Augmented Generation) chatbot that provides instant, accurate answers about JeeyoPay services. This project involves complete web scraping of the JeeyoPay website, data processing, and building a production-ready conversational AI assistant.

##  Live Demo

The app is deployed on Hugging Face Space.
**Try it here:** [https://huggingface.co/spaces/komalLM/JioPay-Chatbot]

##  Project Overview
This is an end-to-end AI chatbot project that demonstrates the complete pipeline from data collection to deployment. The chatbot acts as a virtual customer service representative for JeeyoPay, capable of answering questions about payments, wallet features, UPI transactions, recharges, bill payments, and more.
Why This Matters
Traditional customer support relies on human agents or rigid rule-based systems. This RAG-based chatbot:

- Provides instant 24/7 support
- Answers with source citations for transparency
- Uses semantic search to understand user intent
- Scales to handle unlimited concurrent users
- Always stays up-to-date with official documentation

## Project Workflow
### Phase 1: Web Scraping with Selenium
Automated the entire JeeyoPay website scraping process using Selenium WebDriver to collect comprehensive documentation.

* Set up automated browser navigation for JioPay's official website
* Handled dynamic JavaScript-rendered content with proper wait conditions
* Scraped 200+ pages including help center, FAQs, feature docs, and support articles
* Implemented error handling and retry logic for robust data collection
* Output: Raw HTML content and structured data from entire website

### Phase 2: Data Processing & Structuring
* Transformed raw scraped data into clean, structured formats ready for AI processing.
* Parsed HTML and extracted meaningful text content
* Cleaned data by removing duplicates, ads, and navigation elements
* Organized content by categories (payments, wallet, UPI, recharges, etc.)
* Generated 4 structured files:

The csv and json files were generated after scraping the website
- links_content2.csv - Website page content
- questions.csv - Question-answer pairs
- help_center_faqs.json - Hierarchical help articles
- links_content.json - Structured website data


* Output: Clean, queryable datasets with 500+ documents

### Phase 3: RAG Pipeline Development
Built an intelligent retrieval system combining vector search and LLM generation.
Architecture Flow:
* User Query → Embedding → FAISS Search → Top-3 Documents → LLM Context → Generated Answer + Sources
Key Components:

* LlamaIndex: Loaded and indexed all documents with metadata
* HuggingFace Embeddings: Converted text to 384-dim vectors using all-MiniLM-L6-v2
* FAISS Vector DB: Stored embeddings for ultra-fast similarity search (<50ms)
* LangChain: Created custom retriever to bridge LlamaIndex and LLM
* Groq LLaMA 3.1: Generated accurate, context-grounded responses

### Key Features: Semantic understanding, source attribution, zero hallucinations, real-time retrieval
### Phase 4: Deployment (Gradio Interface)
- Created a user-friendly web interface and deployed for public access.
- Built chat interface using Gradio for natural conversations
- Implemented real-time response streaming with source citations
- Deployed on Hugging Face Spaces for permanent, free hosting
- Mobile-responsive design accessible from any device
- Output: Live chatbot with public URL

##  Features

- **Semantic Search**: Uses FAISS vector database for fast and accurate information retrieval
- **Source Citations**: Shows exact source snippets from official JeeyoPay documentation
- **Real-time Responses**: Powered by Groq's LLaMA 3.1 model for quick answers
- **User-Friendly Interface**: Clean Gradio web interface
- **Multi-format Support**: Processes both CSV and JSON data files

##  Tech Stack

- **LlamaIndex**: Document indexing and semantic search
- **LangChain**: RAG chain implementation
- **FAISS**: Vector database for similarity search
- **Groq API**: LLM inference (LLaMA 3.1-8B)
- **HuggingFace Embeddings**: Sentence transformers for text embeddings
- **Gradio**: Web interface
- **Python**: Core programming language

## 📁 Project Structure

```
jiopay-chatbot/
│
├── app.py                          # Main application file
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
├── README.md                       # Project documentation
│
├── links_content2.csv        # JeeyoPay website content
├── questions.csv                 # Frequently asked questions
├── help_center_faqs.json    # Help center data
└── links_content.json       # Additional content data
```

##  Local Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Groq API key ([Get one here](https://console.groq.com))

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/komalsali04/jiopay-chatbot.git
   cd jiopay-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the chatbot**
   
   Open your browser and go to: `http://localhost:7860`

## Example Questions to ask (based on FAQs)
* What is JioPay Business?
* What is the purpose of JioPay Business?
* How can I download the JioPay App?
* I have forgotten my account password. How can I reset it?
* Why My App is crashing on my Phone?
* I am unable to login to the App/Dashboard. What can I do?
  
##  Getting a Groq API Key

1. Visit [Groq Console](https://console.groq.com)
2. Sign up or log in to your account
3. Navigate to API Keys section
4. Generate a new API key
5. Copy and paste it into your `.env` file
   

##  Data Sources

The chatbot uses the following data sources:
- JeeyoPay official website content
- JeeyoPay FAQs and help center articles
- Customer support documentation

All data is preprocessed and indexed for efficient retrieval.

##  Author

**Your Name**
- GitHub: [komalsali04](https://github.com/komalsali04)
- LinkedIn: [LinkedIn](https://www.linkedin.com/in/komal-sali-a819b6257)

##  Acknowledgments

- JeeyoPay for the documentation data
- Groq for providing fast LLM inference
- HuggingFace for embeddings and hosting
- LlamaIndex and LangChain communities

##  Contact

For questions or support, please open an issue or contact [salikomal04@gmail.com]

---

⭐ **If you find this project helpful, please give it a star!** ⭐

# 🎬 YouTube Q&A Bot

Ask questions about any YouTube video using AI. Paste a YouTube link, and the bot instantly builds a knowledge base from the video transcript and answers your questions using RAG (Retrieval-Augmented Generation).

## 🚀 Live Demo
[👉 Try it here](https://your-streamlit-link-here)

## 💡 Features
- 🔗 Paste any YouTube URL with available captions
- 📄 Auto-fetches and processes full video transcript
- 🧠 Semantic search using FAISS vector store
- 💬 Multi-turn conversation about the video content
- ⚡ Fast responses via Groq's LPU inference
- 🔄 Load multiple videos in one session

## 🛠 Tech Stack
| Component | Technology |
|-----------|------------|
| LLM | Llama 3.3 70B via Groq API |
| Framework | LangChain |
| Vector Store | FAISS |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Transcript | youtube-transcript-api |
| UI | Streamlit |

## ⚙️ How It Works
1. User pastes YouTube URL
2. `youtube-transcript-api` fetches the full video transcript
3. Transcript is split into chunks using LangChain's text splitter
4. Chunks are embedded using HuggingFace and stored in FAISS
5. User asks a question → top 4 relevant chunks retrieved
6. Chunks + question sent to Llama 3.3 70B → answer returned

## 🏃 Setup Locally
```bash
git clone https://github.com/mutahir-ahmed-ai/youtube-qa-bot
cd youtube-qa-bot
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your-groq-api-key-here"
```

Run:
```bash
streamlit run app.py
```

## ⚠️ Note
Only works with YouTube videos that have captions/subtitles enabled.

## 📬 Contact
Built by [Mutahir Ahmed](https://www.linkedin.com/in/mutahir-ahmed-8229341b5) — open to freelance AI projects.

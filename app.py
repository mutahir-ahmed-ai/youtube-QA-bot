import os
import re
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Page Config ---
st.set_page_config(
    page_title="YouTube Q&A Bot",
    page_icon="🎬",
    layout="centered"
)

# --- Styling ---
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
        border-bottom: 1px solid #2a2a2a;
        margin-bottom: 1.5rem;
    }
    .video-info {
        background: #1a1a2e;
        border: 1px solid #16213e;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="main-header">
    <h2>🎬 YouTube Q&A Bot</h2>
    <p style="color: #888; font-size: 14px;">Paste any YouTube link and ask questions about the video</p>
</div>
""", unsafe_allow_html=True)

# --- API Setup ---
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found. Add it to your Streamlit secrets.")
    st.stop()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key,
    temperature=0.0
)

# --- Helper: Extract YouTube Video ID ---
def extract_video_id(url):
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
        r"(?:embed\/)([0-9A-Za-z_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# --- Helper: Get Transcript ---
def get_transcript(video_id):
    ytt = YouTubeTranscriptApi()
    transcript_list = ytt.fetch(video_id)
    full_text = " ".join([entry.text for entry in transcript_list])
    return full_text

# --- Helper: Build Vector Store ---
@st.cache_resource(show_spinner=False)
def build_vectorstore(transcript, video_id):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.create_documents([transcript])
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# --- Helper: Answer Question ---
def answer_question(question, vectorstore, video_id):
    docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])

    system_prompt = f"""You are a helpful assistant that answers questions about a YouTube video.
Use ONLY the transcript context provided below to answer.
If the answer is not in the transcript, say "I couldn't find that in the video."
Be concise and direct.

TRANSCRIPT CONTEXT:
{context}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]
    response = llm.invoke(messages)
    return response.content

# --- Session State ---
if "video_loaded" not in st.session_state:
    st.session_state.video_loaded = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "video_title" not in st.session_state:
    st.session_state.video_title = None

# --- URL Input ---
url_input = st.text_input(
    "🔗 YouTube URL",
    placeholder="https://www.youtube.com/watch?v=...",
    key="url_input"
)

load_btn = st.button("Load Video", type="primary", use_container_width=True)

if load_btn and url_input:
    video_id = extract_video_id(url_input)

    if not video_id:
        st.error("Invalid YouTube URL. Please check and try again.")
    else:
        with st.spinner("Fetching transcript and building knowledge base..."):
            try:
                transcript = get_transcript(video_id)

                if len(transcript) < 50:
                    st.error("Transcript too short or empty.")
                else:
                    vectorstore = build_vectorstore(transcript, video_id)
                    st.session_state.video_loaded = True
                    st.session_state.video_id = video_id
                    st.session_state.vectorstore = vectorstore
                    st.session_state.messages = []
                    word_count = len(transcript.split())
                    st.session_state.video_title = f"Video loaded — {word_count:,} words transcribed"
                    st.rerun()

            except TranscriptsDisabled:
                st.error("This video has disabled transcripts. Try another video.")
            except NoTranscriptFound:
                st.error("No transcript found for this video. Try a video with captions enabled.")
            except Exception as e:
                st.error(f"Error loading video: {str(e)}")

# --- Chat Interface (shown after video loads) ---
if st.session_state.video_loaded:

    # Video info bar
    st.markdown(f"""
    <div class="video-info">
        ✅ <strong>{st.session_state.video_title}</strong><br>
        <a href="https://youtube.com/watch?v={st.session_state.video_id}" 
           target="_blank" style="color: #888; font-size: 13px;">
           youtube.com/watch?v={st.session_state.video_id}
        </a>
    </div>
    """, unsafe_allow_html=True)

    # Suggested questions
    if not st.session_state.messages:
        st.markdown("**Try asking:**")
        suggestions = [
            "What is this video about?",
            "What are the main points covered?",
            "Summarize the key takeaways",
            "What does the speaker recommend?"
        ]
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(suggestion, key=f"sug_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": suggestion})
                    reply = answer_question(
                        suggestion,
                        st.session_state.vectorstore,
                        st.session_state.video_id
                    )
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.rerun()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask anything about this video..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner(""):
                reply = answer_question(
                    prompt,
                    st.session_state.vectorstore,
                    st.session_state.video_id
                )
                st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

    # Reset button
    st.divider()
    if st.button("Load a different video", use_container_width=True):
        st.session_state.video_loaded = False
        st.session_state.messages = []
        st.session_state.video_id = None
        st.session_state.vectorstore = None
        st.rerun()

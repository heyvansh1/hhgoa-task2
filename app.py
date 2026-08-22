import os
import tempfile
import streamlit as st
from harness.pipeline import PipelineInput, RAGPipeline
from retrieval.retrieval import HybridRetriever
from chunking.passage_native import PassageNativeChunker

st.set_page_config(page_title="Voice RAG Pipeline", page_icon="🎙️", layout="wide")

st.title("🎙️ Voice RAG Pipeline — Multilingual QA")
st.caption("Sub-200ms Hybrid Retrieval & Voice QA on MSMARCO-XI (English & Hindi)")

# Load API keys from Streamlit secrets or environment
sarvam_key = st.secrets.get("SARVAM_API_KEY", os.getenv("SARVAM_API_KEY", ""))
groq_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
if sarvam_key:
    os.environ["SARVAM_API_KEY"] = sarvam_key
if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key

@st.cache_resource(show_spinner="Indexing knowledge base...")
def load_pipeline():
    chunker = PassageNativeChunker()
    chunks = chunker.chunk()
    retriever = HybridRetriever(chunks, fusion="rrf")
    return RAGPipeline(retriever=retriever)

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Failed to initialize pipeline: {e}")
    st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input")
    lang = st.selectbox("Select Language", options=["eng", "hin"], format_func=lambda x: "English" if x == "eng" else "Hindi")
    input_mode = st.radio("Choose Input Mode", ["Audio File Upload", "Text Query"])
    
    query_text = ""
    audio_path = None

    if input_mode == "Audio File Upload":
        uploaded_file = st.file_uploader("Upload .wav recording", type=["wav"])
        if uploaded_file:
            st.audio(uploaded_file, format="audio/wav")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_file.read())
                audio_path = tmp.name
    else:
        query_text = st.text_input("Enter your query:", placeholder="e.g. What is the capital of India?")

    submit = st.button("Submit Query", type="primary")

with col2:
    st.subheader("Output & Analytics")
    if submit:
        if input_mode == "Audio File Upload" and not audio_path:
            st.warning("Please upload a .wav audio file first.")
        elif input_mode == "Text Query" and not query_text.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Processing..."):
                pipe_input = PipelineInput(
                    query_text=query_text if input_mode == "Text Query" else None,
                    audio_path=audio_path,
                    language=lang
                )
                output = pipeline.run(pipe_input)

            # Display generated answer
            st.success("### Answer")
            st.write(output.answer)

            # Latency Metrics
            st.markdown("---")
            st.markdown("### ⏱️ Stage Latency")
            metrics_cols = st.columns(len(output.stage_timings))
            for i, (stage, duration) in enumerate(output.stage_timings.items()):
                with metrics_cols[i % len(metrics_cols)]:
                    st.metric(label=stage.upper(), value=f"{duration:.2f} ms")

            st.info(f"**Total End-to-End Latency:** {output.total_latency_ms:.2f} ms")

            # Retrieved Chunks
            with st.expander("🔍 View Retrieved Chunks"):
                for idx, chunk in enumerate(output.retrieved_chunks, 1):
                    st.markdown(f"**[{idx}] {chunk.chunk_id}** (`{chunk.language}`)")
                    st.write(chunk.text)
import os
import tempfile
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from harness.pipeline import PipelineInput, RAGPipeline
from main import _build_retriever

print("⏳ Initializing Hybrid Retriever and RAG Pipeline...")
retriever = _build_retriever()
pipeline = RAGPipeline(retriever=retriever, top_k=3)
print("✅ Voice RAG Pipeline Ready!")


def convert_to_wav(audio_path: str) -> str:
    """Convert browser audio to standard WAV."""
    if not audio_path:
        return audio_path
    try:
        import soundfile as sf
        data, samplerate = sf.read(audio_path)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tmp.name, data, samplerate, format="WAV", subtype="PCM_16")
        return tmp.name
    except Exception:
        return audio_path


def respond(text_input, chat_history, audio_file, language):
    lang_code = "eng" if language == "English" else "hin"
    chat_history = chat_history or []

    query_str = (text_input or "").strip()
    clean_audio_path = convert_to_wav(audio_file) if audio_file else None

    # Do nothing if user submitted empty inputs
    if not query_str and not clean_audio_path:
        return "", None, chat_history

    pipe_input = PipelineInput(
        query_text=query_str if query_str and not clean_audio_path else None,
        audio_path=clean_audio_path,
        language=lang_code,
    )
    output = pipeline.run(pipe_input)

    user_label = query_str if query_str else "[🎙️ Voice Query]"
    if clean_audio_path and hasattr(output, "query_text") and output.query_text:
        user_label = f"🎙️ \"{output.query_text}\""

    # Strict Relevance Filter:
    # 1. Strip common question words
    stop_words = {
        "what", "is", "the", "of", "a", "an", "in", "to", "for", "does", "do",
        "how", "where", "when", "which", "are", "state", "capital", "tell", "me",
        "about", "kya", "hai", "ka", "ki", "ke", "kahan", "states", "united"
    }
    
    raw_query = query_str.lower()
    query_tokens = [
        w.strip("?,.!'\"") for w in raw_query.split()
        if w.strip("?,.!'\"") not in stop_words and len(w.strip("?,.!'\"")) > 2
    ]

    retrieved_text = " ".join([c.text.lower() for c in (output.chunks_cited or [])])

    # Check if specific entities in query (e.g. 'america', 'china', 'japan') exist in context
    is_ungrounded = bool(query_tokens) and not any(token in retrieved_text for token in query_tokens)

    if is_ungrounded:
        if lang_code == "eng":
            final_reply = "I could not find information regarding this question in the indexed dataset."
        else:
            final_reply = "अनुक्रमित डेटाबेस में इस प्रश्न से संबंधित जानकारी नहीं मिली।"
        bot_reply = f"🛡️ {final_reply}\n\n*(Latency: {output.total_latency_ms:.1f} ms)*"
    else:
        clean_text = output.answer
        if clean_text.startswith("Based on the retrieved information:"):
            clean_text = clean_text.replace("Based on the retrieved information:", "").strip()
        if "[Sources:" in clean_text:
            clean_text = clean_text.split("[Sources:")[0].strip()

        if output.guardrail_triggered:
            clean_text = f"🛡️ {clean_text}"

        bot_reply = f"{clean_text}\n\n*(Latency: {output.total_latency_ms:.1f} ms)*"

    chat_history.append({"role": "user", "content": user_label})
    chat_history.append({"role": "assistant", "content": bot_reply})
    return "", None, chat_history


with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ Multilingual Voice RAG")

    with gr.Row():
        language_dropdown = gr.Dropdown(
            choices=["English", "Hindi"],
            value="English",
            label="Language",
        )

    chatbot = gr.Chatbot(label="Chat", height=450)

    with gr.Row():
        msg_input = gr.Textbox(
            placeholder="Ask a question...",
            label="Text Query",
            scale=4,
        )
        audio_input = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Voice Input (.wav)",
            scale=4,
        )

    with gr.Row():
        send_btn = gr.Button("Submit", variant="primary", scale=2)
        clear_btn = gr.Button("Clear", scale=1)

    send_btn.click(
        fn=respond,
        inputs=[msg_input, chatbot, audio_input, language_dropdown],
        outputs=[msg_input, audio_input, chatbot],
    )
    msg_input.submit(
        fn=respond,
        inputs=[msg_input, chatbot, audio_input, language_dropdown],
        outputs=[msg_input, audio_input, chatbot],
    )
    clear_btn.click(lambda: [], None, chatbot, queue=False)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=True)
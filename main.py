import os, time
from datetime import datetime
import streamlit as st
from openai import OpenAI

# (Optional TTS) -> pip install gTTS
try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")
st.markdown("""
    <style>
    .stApp {
        background-color: #f4f6f9;
        font-family: 'Segoe UI', sans-serif;
    }
    .stChatMessage.user {
        background: #DCF8C6;
        color: #000;
        padding: 12px 16px;
        border-radius: 20px 20px 0 20px;
        margin: 8px 0;
    }
    .stChatMessage.assistant {
        background: #EDEDED;
        color: #111;
        padding: 12px 16px;
        border-radius: 20px 20px 20px 0;
        margin: 8px 0;
    }
    .stSidebar {
        background-color: #222831;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Chatbot")
st.caption("Made by Rahul Yadav")
st.caption("Chat with persona, memory, downloads, and optional voice.")

# ---------- Sidebar (Settings) ----------
with st.sidebar:
    st.header("⚙️ Settings")
    api_key_input = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password", placeholder="sk-...")
    if api_key_input:
        st.session_state["api_key"] = api_key_input

    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)

    persona_name = st.selectbox(
        "Persona",
        ["General Assistant", "Sales Agent", "Support Agent", "Expert Advisor"],
        index=0
    )

    tts_on = st.checkbox("Speak replies (TTS)", value=False, help="Requires gTTS installed")
    if tts_on and not TTS_AVAILABLE:
        st.info("Install voice support:  pip install gTTS")

    st.divider()
    if st.button("🧹 New chat"):
        st.session_state.pop("messages", None)
        st.rerun()

    st.divider()
    uploaded_audio = st.file_uploader("Transcribe audio (optional)", type=["mp3", "wav", "m4a", "ogg"])

# ---------- Persona prompts ----------
PERSONAS = {
    "General Assistant": "You are a helpful, concise AI assistant.",
    "Sales Agent": "You are a proactive sales rep. Qualify leads, ask clarifying business questions, and keep answers brief and persuasive.",
    "Support Agent": "You are a friendly customer support agent. Ask for required details, provide step-by-step troubleshooting, and be empathetic.",
    "Expert Advisor": "You are a senior consultant. Provide structured, best-practice guidance with examples and clear steps."
}

def get_client():
    api_key = st.session_state.get("api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.warning("Sidebar me OpenAI API key daalo to start.")
        st.stop()
    return OpenAI(api_key=api_key)

client = get_client()

# ---------- Conversation state ----------
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": PERSONAS[persona_name]}]
else:
    # Persona change ko reflect karo
    if st.session_state["messages"] and st.session_state["messages"][0]["role"] == "system":
        st.session_state["messages"][0]["content"] = PERSONAS[persona_name]

# ---------- Audio transcription (optional) ----------
if uploaded_audio is not None:
    with st.spinner("Transcribing audio..."):
        import tempfile
        audio_bytes = uploaded_audio.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix="."+uploaded_audio.name.split(".")[-1]) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            with open(tmp_path, "rb") as f:
                tr = client.audio.transcriptions.create(model="whisper-1", file=f)
            user_text = getattr(tr, "text", "")
            if user_text:
                st.session_state["messages"].append({"role": "user", "content": user_text})
                st.success("Transcribed and added to chat.")
        finally:
            try: os.remove(tmp_path)
            except: pass

# ---------- Render chat history ----------
for msg in st.session_state["messages"]:
    if msg["role"] == "system":
        continue
    with st.chat_message("user" if msg["role"] == "user" else "assistant",
                         avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

# ---------- Chat input ----------
prompt = st.chat_input("Type your message...")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=st.session_state["messages"]
            )
        reply = resp.choices[0].message.content
        st.markdown(reply)
        st.session_state["messages"].append({"role": "assistant", "content": reply})

        # ---------- Optional TTS ----------
        if tts_on and TTS_AVAILABLE:
            try:
                tts = gTTS(text=reply, lang="en")
                audio_path = f"tts_{int(time.time())}.mp3"
                tts.save(audio_path)
                st.audio(audio_path, autoplay=True)
            except Exception as e:
                st.info(f"TTS failed: {e}")

# ---------- Download chat ----------
def export_text():
    lines = []
    for m in st.session_state["messages"]:
        if m["role"] == "system": 
            continue
        who = "You" if m["role"] == "user" else "AI"
        lines.append(f"{who}: {m['content']}")
    return "\n\n".join(lines)

st.download_button(
    "⬇️ Download chat (.txt)",
    data=export_text(),
    file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    mime="text/plain"
)

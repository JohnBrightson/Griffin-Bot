# app.py - CPU-only Finance Chatbot
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from pathlib import Path

# ------------------ Page Setup ------------------ #
st.set_page_config(
    page_title="Finance AI Assistant",
    page_icon="💹",
    layout="wide"
)

# ------------------ CSS Styling ------------------ #
st.markdown("""
<style>
.main-header {font-size:2.5rem; color:#00b894; text-align:center; margin-bottom:1rem; font-weight:bold;}
.chat-container {background-color:#f8f9fa; border-radius:15px; padding:20px; height:60vh; overflow-y:auto; margin-bottom:20px; border:2px solid #00b894;}
.user-message {background-color:#e3f2fd; padding:15px; border-radius:12px; margin:12px 0; border-left:5px solid #1976d2;}
.ai-message {background-color:#e8f5e9; padding:15px; border-radius:12px; margin:12px 0; border-left:5px solid #00b894;}
.stButton button {background-color:#00b894; color:white; border-radius:8px; padding:10px 20px; font-weight:bold;}
.sidebar .sidebar-content {background-color:#2d3436; color:white;}
</style>
""", unsafe_allow_html=True)

# ------------------ Session State ------------------ #
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None

# ------------------ Model Loading ------------------ #
@st.cache_resource(show_spinner=False)
def load_model(model_path):
    try:
        with st.spinner("🔄 Loading financial AI model..."):
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="cpu",  # CPU only
                trust_remote_code=True
            )
        return model, tokenizer
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

# ------------------ Response Generation ------------------ #
def generate_response(prompt, max_length=512, temperature=0.7):
    if not st.session_state.model_loaded:
        return "📊 Model not loaded."
    try:
        finance_prompt = f"As a financial expert, provide clear advice on:\n{prompt}\nResponse:"
        inputs = st.session_state.tokenizer.encode(finance_prompt, return_tensors="pt").to("cpu")
        outputs = st.session_state.model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=st.session_state.tokenizer.eos_token_id
        )
        response = st.session_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(finance_prompt):].strip()
    except Exception as e:
        return f"⚠ Error: {e}"

# ------------------ Main App ------------------ #
def main():
    st.markdown('<h1 class="main-header">💹 Finance AI Assistant</h1>', unsafe_allow_html=True)
    st.write("Your intelligent financial advisor (CPU only)")

    # -------- Sidebar -------- #
    with st.sidebar:
        st.header("Setup")
        model_path = st.text_input("Model Path", value="./ibm-granite-model")
        
        if st.button("🚀 Load Model"):
            if os.path.exists(model_path):
                st.session_state.model, st.session_state.tokenizer = load_model(model_path)
                if st.session_state.model:
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded successfully!")
                else:
                    st.error("❌ Failed to load model.")
            else:
                st.error("📁 Model path not found.")

        st.divider()
        st.subheader("Settings")
        max_length = st.slider("Response Length", 128, 1024, 512)
        temperature = st.slider("Creativity", 0.1, 1.0, 0.7)
        if st.button("🗑 Clear Chat"):
            st.session_state.messages = []
            st.experimental_rerun()

    # -------- Chat Interface -------- #
    st.markdown("### 💬 Financial Discussion")
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-message"><strong>💼 You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ai-message"><strong>💹 AI Advisor:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.model_loaded:
        prompt = st.chat_input("Ask about investments, budgeting, or financial planning...")
        if prompt:
            st.session_state.messages.append({"role":"user","content":prompt})
            with st.spinner("💭 Thinking..."):
                response = generate_response(prompt, max_length, temperature)
            st.session_state.messages.append({"role":"assistant","content":response})
            st.experimental_rerun()
    else:
        st.info("⏳ Load the model first to chat.")

if __name__ == "__main__":
    main()

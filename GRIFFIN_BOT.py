# app.py - Red-themed Finance Chatbot with New Layout (CPU only)
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# ------------------ Page Setup ------------------ #
st.set_page_config(
    page_title="🔥 Finance AI Red Assistant",
    page_icon="🩸",
    layout="wide"
)

# ------------------ Red-Themed CSS ------------------ #
st.markdown("""
<style>
body, .stApp { background-color: #330000; color: #fff; }
.main-header { font-size:3rem; color:#ff4d4d; text-align:center; font-weight:bold; margin-bottom:2rem; text-shadow: 2px 2px #660000; }
.chat-container { background-color:#4d0000; border-radius:20px; padding:20px; height:70vh; overflow-y:auto; box-shadow: 0 4px 20px #ff0000; border:2px solid #ff1a1a; }
.user-message { background-color:#660000; padding:15px; border-radius:15px; margin:12px 0; border-left:6px solid #ff1a1a; }
.ai-message { background-color:#990000; padding:15px; border-radius:15px; margin:12px 0; border-left:6px solid #ff4d4d; }
.stButton button { background-color:#ff1a1a; color:white; border-radius:12px; padding:10px 20px; font-weight:bold; border:none; transition:0.3s; width:100%; }
.stButton button:hover { background-color:#ff4d4d; }
.example-card { background-color:#800000; color:white; padding:12px; border-radius:10px; margin:10px 0; cursor:pointer; transition:0.2s; }
.example-card:hover { background-color:#ff1a1a; }
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
        with st.spinner("🔄 Loading financial AI model (CPU)..."):
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="cpu",
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
    st.markdown('<h1 class="main-header"> Finance AI Red Assistant</h1>', unsafe_allow_html=True)

    # New layout: left chat, right control panel
    left_col, right_col = st.columns([3, 1])

    # ---------------- Left Column: Chat ---------------- #
    with left_col:
        st.markdown("### 💬 Chat")
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="user-message"><strong>💼 You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ai-message"><strong>💹 AI Advisor:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Chat input
        if st.session_state.model_loaded:
            prompt = st.chat_input("Ask anything about finance...")
            if prompt:
                st.session_state.messages.append({"role":"user","content":prompt})
                response = generate_response(prompt)
                st.session_state.messages.append({"role":"assistant","content":response})
                st.experimental_rerun()
        else:
            st.info("⏳ Load the model from the right panel to chat.")

    # ---------------- Right Column: Controls ---------------- #
    with right_col:
        st.markdown("### ⚙ Model Setup")
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
        st.markdown("### 🔧 Settings")
        max_length = st.slider("Response Length", 128, 1024, 512)
        temperature = st.slider("Creativity", 0.1, 1.0, 0.7)
        if st.button("🗑 Clear Chat"):
            st.session_state.messages = []
            st.experimental_rerun()

        st.divider()
        st.markdown("### 💡 Quick Examples")
        examples = [
            "How should I invest $5000?",
            "Explain compound interest",
            "Best budgeting strategy",
            "How to save for retirement",
            "Stocks vs bonds?"
        ]
        for ex in examples:
            if st.button(ex):
                st.session_state.messages.append({"role":"user","content":ex})
                response = generate_response(ex, max_length, temperature)
                st.session_state.messages.append({"role":"assistant","content":response})
                st.experimental_rerun()

if __name__ == "__main__":
    main()

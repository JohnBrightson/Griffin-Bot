# app.py - Finance Chat Bot with Integrated Installation
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import os
import subprocess
import sys
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Finance AI Assistant",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for finance theme
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #00b894;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 15px;
        padding: 20px;
        height: 65vh;
        overflow-y: auto;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 184, 148, 0.1);
        border: 2px solid #00b894;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 12px;
        margin: 12px 0;
        border-left: 5px solid #1976d2;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .ai-message {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 12px;
        margin: 12px 0;
        border-left: 5px solid #00b894;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .finance-card {
        background: linear-gradient(135deg, #00b894 0%, #00a382 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .install-card {
        background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stButton button {
        background-color: #00b894;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .install-button {
        background-color: #6c5ce7 !important;
    }
    .sidebar .sidebar-content {
        background-color: #2d3436;
        color: white;
    }
    .gpu-status {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .gpu-available {
        background-color: #00b894;
        color: white;
    }
    .gpu-not-available {
        background-color: #ff7675;
        color: white;
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        background-color: #2d3436;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "using_gpu" not in st.session_state:
    st.session_state.using_gpu = False
if "installation_status" not in st.session_state:
    st.session_state.installation_status = "not_started"
if "system_checked" not in st.session_state:
    st.session_state.system_checked = False

def check_system():
    """Check system requirements"""
    st.session_state.system_checked = True
    return {
        "python": sys.version_info >= (3, 8),
        "torch": True,  # We'll check this separately
        "cuda_available": torch.cuda.is_available(),
        "model_downloaded": os.path.exists("./ibm-granite-model"),
        "requirements_installed": True  # We'll check this
    }

def install_requirements():
    """Install required packages"""
    try:
        with st.spinner("📦 Installing required packages..."):
            # Install core packages
            packages = [
                "torch", "transformers", "accelerate", 
                "huggingface-hub", "streamlit"
            ]
            
            for package in packages:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            
            st.session_state.installation_status = "requirements_installed"
            return True
    except Exception as e:
        st.error(f"❌ Installation failed: {str(e)}")
        return False

def download_model():
    """Download IBM Granite model"""
    try:
        with st.spinner("📥 Downloading IBM Granite model (10-30 minutes)..."):
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id='ibm-granite/granite-3.0-2b-instruct',
                local_dir='./ibm-granite-model',
                local_dir_use_symlinks=False,
                resume_download=True
            )
            st.session_state.installation_status = "model_downloaded"
            return True
    except Exception as e:
        st.error(f"❌ Model download failed: {str(e)}")
        return False

def check_cuda():
    """Check CUDA availability"""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        return 'release' in result.stdout
    except:
        return False

@st.cache_resource(show_spinner=False)
def load_model(model_path):
    """Load the model and tokenizer from the specified path"""
    try:
        with st.spinner("🔄 Loading financial AI model... This may take a few minutes."):
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Check if GPU is available
            if torch.cuda.is_available():
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                st.session_state.using_gpu = True
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="cpu",
                    trust_remote_code=True
                )
                st.session_state.using_gpu = False
            
            return model, tokenizer
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

def generate_finance_response(prompt, max_length=512, temperature=0.7):
    """Generate finance-specific response"""
    if not st.session_state.model_loaded:
        return "📊 Model not loaded. Please load the model first."
    
    try:
        # Create finance-focused prompt
        finance_prompt = f"""As a financial expert, provide clear, accurate advice on the following topic:

{prompt}

Please provide:
1. Key insights and analysis
2. Practical recommendations
3. Risk considerations
4. When appropriate, numerical examples

Response:"""
        
        # Tokenize input
        inputs = st.session_state.tokenizer.encode(finance_prompt, return_tensors="pt")
        
        # Move to same device as model
        if hasattr(st.session_state.model, 'device'):
            inputs = inputs.to(st.session_state.model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = st.session_state.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=st.session_state.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode response
        response = st.session_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(finance_prompt):].strip()
        return response
        
    except Exception as e:
        return f"⚠ Error generating response: {str(e)}"

def main():
    # Header with finance theme
    st.markdown('<h1 class="main-header">💹 Finance AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown("### Your intelligent financial advisor powered by IBM Granite")
    
    # Sidebar for configuration
    with st.sidebar:
        # Installation Section
        st.markdown('<div class="install-card">', unsafe_allow_html=True)
        st.header("🔧 Installation")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Check system button
        if not st.session_state.system_checked:
            if st.button("🔍 Check System Requirements", use_container_width=True):
                system_info = check_system()
                
                st.markdown('<div class="status-box">', unsafe_allow_html=True)
                st.write("✅ Python 3.8+")
                st.write("✅ CUDA Available" if system_info["cuda_available"] else "❌ CUDA Not Available")
                st.write("✅ Model Downloaded" if system_info["model_downloaded"] else "❌ Model Not Downloaded")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Installation buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📦 Install Packages", use_container_width=True, key="install_packages"):
                if install_requirements():
                    st.success("✅ Packages installed successfully!")
        
        with col2:
            if st.button("📥 Download Model", use_container_width=True, key="download_model"):
                if download_model():
                    st.success("✅ Model downloaded successfully!")
        
        st.divider()
        
        # System Info
        st.markdown('<div class="finance-card">', unsafe_allow_html=True)
        st.header("⚙ System Info")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # GPU status
        if torch.cuda.is_available():
            st.markdown(f'<div class="gpu-status gpu-available">✅ GPU Available: {torch.cuda.get_device_name(0)}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="gpu-status gpu-not-available">❌ GPU Not Available</div>', unsafe_allow_html=True)
        
        # Model path
        model_path = st.text_input(
            "Model Path", 
            value="./ibm-granite-model",
            help="Path to downloaded IBM Granite model"
        )
        
        # Load model button
        if st.button("🚀 Load Finance Model", use_container_width=True):
            if os.path.exists(model_path):
                st.session_state.model, st.session_state.tokenizer = load_model(model_path)
                if st.session_state.model is not None:
                    st.session_state.model_loaded = True
                    st.success("✅ Finance model loaded successfully!")
                else:
                    st.error("❌ Failed to load model. Please check the path.")
            else:
                st.error("📁 Model path does not exist. Please download the model first.")
        
        st.divider()
        
        # Generation parameters
        st.subheader("📊 Generation Settings")
        max_length = st.slider("Response Length", 128, 1024, 512)
        temperature = st.slider("Creativity", 0.1, 1.0, 0.7)
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑 Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        # Model status
        st.divider()
        if st.session_state.model_loaded:
            if st.session_state.using_gpu:
                st.success("✅ Model loaded on GPU - Fast! 🚀")
            else:
                st.warning("✅ Model loaded on CPU - Slower ⚠")
        else:
            st.warning("⚠ Please load the model to begin")
        
        # Disclaimer
        st.divider()
        st.error("""
        *📝 Disclaimer:*
        This is an AI assistant for educational purposes only. 
        Not financial advice. Always consult qualified professionals 
        for investment decisions.
        """)
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Installation instructions if not set up
        if not st.session_state.model_loaded and not os.path.exists("./ibm-granite-model"):
            st.warning("""
            ## 📋 Installation Required
            
            Before chatting, please:
            1. Click *'Install Packages'* in the sidebar
            2. Click *'Download Model'* (takes 10-30 minutes)
            3. Click *'Load Finance Model'*
            
            Then you can start asking financial questions!
            """)
        
        # Chat container
        st.markdown("### 💬 Financial Discussion")
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message"><strong>💼 You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ai-message"><strong>💹 AI Advisor:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input (only enable if model is loaded)
        if st.session_state.model_loaded:
            prompt = st.chat_input("Ask about investments, budgeting, or financial planning...")
            
            if prompt:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Generate response
                with st.spinner("💭 Analyzing your financial question..."):
                    response = generate_finance_response(prompt, max_length, temperature)
                
                # Add AI response
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
        else:
            st.info("⏳ Please load the model to enable chat")
    
    with col2:
        st.markdown("### 📈 Quick Questions")
        st.markdown('<div class="finance-card">', unsafe_allow_html=True)
        st.write("Ask about:")
        st.write("• Investments 📊")
        st.write("• Budgeting 💰")
        st.write("• Loans 🏦")
        st.write("• Taxes 🧾")
        st.write("• Retirement 🏖")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Finance-specific example prompts
        finance_examples = [
            "How should I allocate my investment portfolio?",
            "Explain compound interest with examples",
            "What's the 50/30/20 budgeting rule?",
            "How to save for retirement at age 30?",
            "Difference between stocks and bonds"
        ]
        
        for example in finance_examples:
            if st.button(example, use_container_width=True, disabled=not st.session_state.model_loaded):
                st.session_state.messages.append({"role": "user", "content": example})
                with st.spinner("💭 Analyzing..."):
                    response = generate_finance_response(example, max_length, temperature)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

if __name__ == '__main__':
    main()

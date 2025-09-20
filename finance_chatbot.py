import streamlit as st
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import requests  # for calling IBM GenAI API

# --- Google Sheets Setup ---
scope = ["https://spreadsheets.google.com/feeds",
         "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)
sheet = client.open("Expenses").sheet1

# --- IBM GenAI API Call (pseudo) ---
def ask_ibm_genai(prompt):
    url = "https://api.ibm.com/genai/2b-model-endpoint"
    headers = {"Authorization": "Bearer YOUR_API_KEY"}
    payload = {"prompt": prompt, "max_tokens": 200}
    response = requests.post(url, json=payload, headers=headers)
    return response.json().get("text", "Sorry, I couldn’t generate a response.")

# --- Streamlit UI ---
st.title("💰 Personal Finance Chatbot")
st.write("Your AI-powered money assistant!")

# User input
user_input = st.text_input("💬 Ask me something about your money:")

if user_input:
    # Log expense if detected
    if "spent" in user_input.lower():
        words = user_input.split()
        try:
            amount = next(int(w) for w in words if w.isdigit())
            category = words[-1]
            sheet.append_row([str(datetime.now()), category, amount])
            st.success(f"✅ Logged {amount} for {category}")
        except:
            st.error("⚠️ Couldn't detect expense format. Try: 'I spent 200 on groceries'.")

    # Get AI-generated response
    ai_response = ask_ibm_genai(user_input)
    st.write("🤖 AI:", ai_response)

import streamlit as st
from ibm_granite import GraniteClient

# ---------------- Model Loading ---------------- #
@st.cache_resource
def load_model():
    # Initialize Granite client (local model)
    # Replace 'path_to_model' with your local model path
    client = GraniteClient(model_path="path_to_model")
    return client

granite_client = load_model()

# ---------------- UI ---------------- #
st.title("💰 Personal Finance Chatbot (IBM Granite)")
st.write("Ask me anything about savings, taxes, or investments!")
st.write("You can also use the Budget Calculator below 👇")

# ---------------- Chat Section ---------------- #
user_input = st.text_input("You:")

if st.button("Ask"):
    if user_input.strip() != "":
        # Generate AI-based response using Granite
        response = granite_client.generate_text(
            prompt=f"You are a financial advisor. Answer clearly: {user_input}",
            max_length=150
        )
        st.write("🤖:", response)

# ---------------- Budget Calculator ---------------- #
st.subheader("📊 Budget Calculator")

income = st.number_input("Enter your monthly income (₹)", min_value=0, step=1000)
needs = st.number_input("Enter your monthly spending on NEEDS (₹)", min_value=0, step=500)
wants = st.number_input("Enter your monthly spending on WANTS (₹)", min_value=0, step=500)
savings = st.number_input("Enter your monthly SAVINGS (₹)", min_value=0, step=500)

if st.button("Calculate Budget"):
    total = needs + wants + savings

    if total > income:
        st.error("⚠ Your expenses exceed your income! Try reducing wants or increasing savings.")
    else:
        st.success("✅ Budget Summary")
        st.write(f"Income: ₹{income}")
        st.write(f"Needs: ₹{needs}  ({round((needs/income)*100, 2)}%)")
        st.write(f"Wants: ₹{wants}  ({round((wants/income)*100, 2)}%)")
        st.write(f"Savings: ₹{savings}  ({round((savings/income)*100, 2)}%)")

        # Simple advice
        if savings < (0.2 * income):
            st.info("💡 Try to save at least 20% of your income.")
        else:
            st.info("💡 Great! Your savings rate looks healthy.")

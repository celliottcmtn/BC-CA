import streamlit as st
from utils import load_and_embed_agreements, compare_question_across_agreements

st.set_page_config(page_title="HR Agreement Comparator", layout="wide")
st.title("📄 HR Collective Agreement Comparator")

st.markdown("Upload agreements to the `agreements/` folder in your repo.")

# Load and embed PDFs
with st.spinner("Loading agreements..."):
    agreements = load_and_embed_agreements()
    if not agreements:
        st.error("No agreements found in the 'agreements' folder.")
        st.stop()

# Let user select agreements to compare
selected = st.multiselect("Select agreements to compare:", list(agreements.keys()), default=list(agreements.keys())[:2])

# Ask a question
question = st.text_input("Ask a question to compare across agreements")

if question and selected:
    with st.spinner("Comparing responses..."):
        answers = compare_question_across_agreements(question, agreements, selected)
        for name, answer in answers.items():
            st.subheader(name)
            st.write(answer)

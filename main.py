import streamlit as st
from utils import load_or_embed_agreements, compare_question_across_agreements

st.set_page_config(page_title="HR Agreement Comparator", layout="wide")
st.title("📄 HR Collective Agreement Comparator")

st.markdown("Place your agreement PDFs in the `agreements/` folder of your GitHub repo.")

# Load FAISS or embed once
with st.spinner("Loading agreement indexes..."):
    agreements = load_or_embed_agreements()
    if not agreements:
        st.error("No agreements found.")
        st.stop()

# UI: Select agreements and ask a question
selected = st.multiselect("Select agreements to compare:", list(agreements.keys()), default=list(agreements.keys())[:2])
question = st.text_input("Ask your question to compare across agreements")

if question and selected:
    with st.spinner("Comparing responses..."):
        answers = compare_question_across_agreements(question, agreements, selected)
        for name, answer in answers.items():
            st.subheader(name)
            st.write(answer)

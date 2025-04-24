import streamlit as st
import os
from pathlib import Path
from utils import load_or_embed_agreements, compare_question_across_agreements

st.set_page_config(page_title="HR Agreement Comparator", layout="wide")
st.title("📄 HR Collective Agreement Comparator")

# Check if agreements directory exists
agreements_dir = Path("agreements")
if not agreements_dir.exists():
    agreements_dir.mkdir(exist_ok=True)
    st.warning("Created 'agreements' directory. Please place your agreement PDFs in this folder.")

# Check if any PDFs exist in the agreements directory
pdf_files = list(agreements_dir.glob("*.pdf"))
if not pdf_files:
    st.warning("No PDF files found in the 'agreements' folder. Please add your collective agreement PDFs there.")

st.markdown("Place your agreement PDFs in the `agreements/` folder of your GitHub repo.")

# Load FAISS or embed once
with st.spinner("Loading agreement indexes..."):
    agreements = load_or_embed_agreements()
    if not agreements:
        st.error("No agreements found or could not process any agreements.")
        st.stop()

# UI: Select agreements and ask a question
selected = st.multiselect("Select agreements to compare:", list(agreements.keys()), 
                         default=list(agreements.keys())[:min(2, len(agreements.keys()))])

question = st.text_input("Ask your question to compare across agreements")

if question:
    if not selected:
        st.warning("Please select at least one agreement to compare.")
    else:
        with st.spinner("Comparing responses..."):
            answers = compare_question_across_agreements(question, agreements, selected)
            for name, answer in answers.items():
                st.subheader(name)
                st.write(answer)

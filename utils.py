import os
from pathlib import Path
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
import streamlit as st

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

def load_and_embed_agreements():
    base_path = Path("agreements")
    agreements = {}

    for file in base_path.glob("*.pdf"):
        try:
            with fitz.open(file) as pdf:
                text = "".join(page.get_text() for page in pdf)
                if text.strip():
                    docs = [Document(page_content=text, metadata={"source": file.name})]
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                    chunks = splitter.split_documents(docs)
                    index = FAISS.from_documents(chunks, embedding)
                    agreements[file.stem.replace("_", " ")] = index
        except Exception as e:
            st.warning(f"Failed to load {file.name}: {e}")
    return agreements

def compare_question_across_agreements(question, agreements, selected):
    results = {}
    prompt = PromptTemplate.from_template("""
You are a helpful assistant that answers questions using only the provided collective agreement text.

- Cite the agreement name and relevant article numbers where applicable.
- If the agreement does not contain relevant information, say so.
- Be concise and avoid legal jargon.

Context:
{context}

Question: {question}

Answer:
""")
    for name in selected:
        try:
            index = agreements[name]
            retriever = index.as_retriever(search_kwargs={"k": 5})
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False, chain_type_kwargs={"prompt": prompt})
            answer = qa.run(question)
            results[name] = answer
        except Exception as e:
            results[name] = f"❌ Error: {e}"
    return results

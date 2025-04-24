import os
import time
from pathlib import Path
import fitz  # PyMuPDF
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from openai import error  # Compatible across OpenAI versions

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
INDEX_DIR = Path("faiss_index")

def load_or_embed_agreements():
    base_path = Path("agreements")
    INDEX_DIR.mkdir(exist_ok=True)
    agreements = {}

    for file in base_path.glob("*.pdf"):
        index_path = INDEX_DIR / file.stem
        if index_path.exists():
            try:
                index = FAISS.load_local(str(index_path), embeddings=embedding)
                agreements[file.stem.replace("_", " ")] = index
                continue
            except Exception as e:
                st.warning(f"⚠️ Failed to load index for {file.name}: {e}")

        try:
            with fitz.open(file) as pdf:
                text = "".join(page.get_text() for page in pdf)
                if text.strip():
                    docs = [Document(page_content=text, metadata={"source": file.name})]
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                    chunks = splitter.split_documents(docs)

                    success = False
                    while not success:
                        try:
                            index = FAISS.from_documents(chunks, embedding)
                            index.save_local(str(index_path))
                            agreements[file.stem.replace("_", " ")] = index
                            success = True
                        except error.RateLimitError:
                            st.warning(f"⏳ Rate limit hit while embedding {file.name}. Retrying in 5 seconds...")
                            time.sleep(5)
        except Exception as e:
            st.warning(f"❌ Failed to process {file.name}: {e}")

    return agreements

def compare_question_across_agreements(question, agreements, selected):
    results = {}
    prompt = PromptTemplate.from_template(
        """
You are a helpful assistant that answers questions using only the provided collective agreement text.

- Cite the agreement name and relevant article numbers where applicable.
- If the agreement does not contain relevant information, say so.
- Be concise and avoid legal jargon.

Context:
{context}

Question: {question}

Answer:
        """
    )
    for name in selected:
        try:
            index = agreements[name]
            retriever = index.as_retriever(search_kwargs={"k": 5})
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=False,
                chain_type_kwargs={"prompt": prompt}
            )
            answer = qa.run(question)
            results[name] = answer
        except Exception as e:
            results[name] = f"❌ Error: {e}"
    return results

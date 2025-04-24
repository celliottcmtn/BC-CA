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
import openai
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Try to get API key from various sources
def get_openai_api_key():
    # First check Streamlit secrets
    try:
        return st.secrets["OPENAI_API_KEY"]
    except:
        pass
    
    # Then check environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    
    # If still not found, prompt user
    if 'OPENAI_API_KEY' not in st.session_state:
        st.session_state['OPENAI_API_KEY'] = ''
    
    api_key = st.text_input("Enter your OpenAI API Key:", value=st.session_state['OPENAI_API_KEY'], type="password")
    if api_key:
        st.session_state['OPENAI_API_KEY'] = api_key
        return api_key
    
    return None

# Initialize with API key
OPENAI_API_KEY = get_openai_api_key()
if not OPENAI_API_KEY:
    st.error("OpenAI API Key is required to run this application.")
    st.stop()

# Initialize embedding object
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
INDEX_DIR = Path("faiss_index")

def load_or_embed_agreements():
    base_path = Path("agreements")
    base_path.mkdir(exist_ok=True)
    INDEX_DIR.mkdir(exist_ok=True)
    agreements = {}

    pdf_files = list(base_path.glob("*.pdf"))
    if not pdf_files:
        st.warning("No PDF files found in the agreements directory.")
        return agreements

    for file in pdf_files:
        index_path = INDEX_DIR / file.stem
        if index_path.exists():
            try:
                st.info(f"Loading existing index for {file.name}...")
                index = FAISS.load_local(str(index_path), embeddings=embedding)
                agreements[file.stem.replace("_", " ")] = index
                continue
            except Exception as e:
                st.warning(f"⚠️ Failed to load index for {file.name}: {e}")
                # If loading fails, try recreating the index

        try:
            st.info(f"Processing {file.name}...")
            with fitz.open(file) as pdf:
                text = "".join(page.get_text() for page in pdf)
                if not text.strip():
                    st.warning(f"⚠️ No text extracted from {file.name}. Is it a scanned PDF?")
                    continue
                
                docs = [Document(page_content=text, metadata={"source": file.name})]
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunks = splitter.split_documents(docs)
                
                if not chunks:
                    st.warning(f"⚠️ No chunks created from {file.name}.")
                    continue

                retry_count = 0
                max_retries = 5
                success = False
                
                while not success and retry_count < max_retries:
                    try:
                        index = FAISS.from_documents(chunks, embedding)
                        index.save_local(str(index_path))
                        agreements[file.stem.replace("_", " ")] = index
                        success = True
                        st.success(f"✅ Successfully indexed {file.name}")
                    except (openai.error.RateLimitError, openai.RateLimitError):
                        retry_count += 1
                        wait_time = min(2 ** retry_count, 60)  # Exponential backoff with max 60 seconds
                        st.warning(f"⏳ Rate limit hit while embedding {file.name}. Retrying in {wait_time} seconds... (Attempt {retry_count}/{max_retries})")
                        time.sleep(wait_time)
                    except Exception as e:
                        st.error(f"❌ Error creating index for {file.name}: {str(e)}")
                        break
                
                if not success:
                    st.error(f"❌ Failed to index {file.name} after {max_retries} attempts.")
        except Exception as e:
            st.error(f"❌ Failed to process {file.name}: {str(e)}")

    return agreements

def compare_question_across_agreements(question, agreements, selected):
    results = {}
    
    if not OPENAI_API_KEY:
        return {name: "OpenAI API Key is required" for name in selected}
    
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
            if name not in agreements:
                results[name] = "⚠️ Agreement not found or not processed."
                continue
                
            index = agreements[name]
            retriever = index.as_retriever(search_kwargs={"k": 5})
            
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo", 
                temperature=0, 
                openai_api_key=OPENAI_API_KEY
            )
            
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=False,
                chain_type_kwargs={"prompt": prompt}
            )
            
            answer = qa.run(question)
            results[name] = answer
        except Exception as e:
            results[name] = f"❌ Error: {str(e)}"
    
    return results

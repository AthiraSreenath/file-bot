from langchain.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader, \
    UnstructuredHTMLLoader, UnstructuredMarkdownLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA, LLMChain, ConversationalRetrievalChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.callbacks import StreamlitCallbackHandler, StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
import streamlit as st
import requests
import os
import shutil
import langchain

langchain.verbose = False

from dotenv import load_dotenv

load_dotenv('.env')

#SCOPES = os.environ['SCOPES']
os.unsetenv('OPENAI_API_KEY')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def validate_key(openai_api_key):
    response = requests.get("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {openai_api_key}"})
    if response.status_code == 200:
        return True
    else:
        return False


def process_file(file) -> str:
    data = file.read()

    with open(file.name, 'wb') as f:
        f.write(data)

    documents = None
    if file.type == "application/pdf":
        loader = PyPDFLoader(file.name)
        # loader = DirectoryLoader(file.name, glob="**/*.pdf", show_progress=True)
        documents = loader.load()

    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        loader = UnstructuredWordDocumentLoader(file.name)
        documents = loader.load()

    elif file.type == "text/plain":
        loader = TextLoader(file.name, encoding='utf8')
        # loader = DirectoryLoader(file.name, glob="**/*.txt", show_progress=True)
        documents = loader.load()

    elif file.type == "text/csv":
        loader = CSVLoader(file.name)
        documents = loader.load()

    elif file.type == "application/octet-stream":
        loader = UnstructuredMarkdownLoader(file.name)
        documents = loader.load()

    elif file.type == "text/html":
        loader = UnstructuredHTMLLoader(file.name)
        documents = loader.load()

    else:
        raise ValueError(f"Unsupported file type: {file.type}")

    return documents


def split_docs(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=25, separators=[" " ",", "\n"])
    texts = text_splitter.split_documents(documents)
    return texts


# Create a Streamlit app
st.set_page_config("Document Q&A Chat")
st.title("Document Q&A Chat")
with st.sidebar:
    st.title("Document Chatbot")
    st.write("by Athira")
    st.write("Tools used:")
    skills = f"""
                <div>
                  <ul>
                    <li>
                    LangChain
                    </li>
                    <li>
                    OpenAI
                    </li>
                    <li>
                    Sentence BERT
                    </li>
                    <li>
                    ChromaDB
                    </li>
                    <li>
                    Streamlit
                    </li>
                  </ul>
                </div>
                """
    st.markdown(skills, unsafe_allow_html=True)

placeholder = st.empty()

# Enter your API Key
openai_key = "incorrect"
with placeholder.container():
    if 'OPENAI_API_KEY' in st.secrets:
        st.success('API key already provided!', icon='✅')
        openai_key = st.secrets['OPENAI_API_KEY']
    else:
        openai_key = st.text_input('Enter OpenAI API token:', type='password')
        if not validate_key(openai_key):
            st.warning('Please enter your credentials!', icon='⚠️')
            openai_key = "incorrect"
        else:
            st.success('Successful! Proceed to uploading your documents', icon='👉')
            os.environ['OPENAI_API_KEY'] = openai_key

    link = '[:key:  Instructions to obtain a key :key:](https://tfthacker.medium.com/how-to-get-your-own-api-key-for-using-openai-chatgpt-in-obsidian-41b7dd71f8d3)'
    st.markdown(link, unsafe_allow_html=True)

files = []
texts = []

if openai_key != "incorrect":
    placeholder.empty()
    st.success('Successful! Proceed to uploading your documents', icon='👉')
    files = st.file_uploader("Chose files to upload", accept_multiple_files=True,
                             type=["pdf", "docx", "txt", "csv", "md", "html"])

if files and openai_key != "incorrect":
    st.info("Loading. Please wait...")
    for file in files:
        docs = process_file(file)
        texts.extend(split_docs(docs))

    embeddings = HuggingFaceEmbeddings(model_name="msmarco-bert-base-dot-v5")

    shutil.rmtree('data')
    vectordb = Chroma.from_documents(texts, embeddings, persist_directory='data')
    vectordb.persist()
    print(vectordb._collection.count())

    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, 
    don't try to make up an answer or use other sources. 
    Use three sentences maximum. Keep the answer as concise as possible. 
    Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # # can specify model type
    # # return_source_documents=True,
    # # ChatHistory, Sources chain

    # streaming=True, callbacks=[StreamingStdOutCallbackHandler()]
    # search_kwargs={'k': 3}
    # change vectordb search parameters from 6 to a larger number for more accurate results
    chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key=openai_key, temperature=0, streaming=True),
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    # #TODO:
    # # Chat history
    # # Add progress bar
    # # Google account authentication

    st.info("App ready. Query the Document QA.")
    chat_history = []

    if query := st.chat_input():
        st.chat_message("user").write(query)
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            response = chain({"query": query}, callbacks=[st_callback])

            st.write("Sources :books:")
            source_list = response['source_documents']
            source_md = f"""
<div>
<ul font-family:Courier; color:Blue;>



            """
            for doc in source_list:
                source_md += f"""
<li>
{doc}
</li>
"""

            source_md += f"""
</ul>
</div>
"""
            st.markdown(source_md, unsafe_allow_html=True)
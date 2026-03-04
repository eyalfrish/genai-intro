############################################
# Component #1 - Document Loader
############################################

import os

import streamlit as st

from models.provider_factory import ProviderFactory

st.set_page_config(layout="wide")

# Parse the arguments
user_args = ProviderFactory.parse_provider_arg()
provider_class = ProviderFactory.get_provider(user_args.inference_provider)

# Initialize the provider only once per (streamlit) session
if "provider_initialized" not in st.session_state:
    provider_class.initialize_provider()
    st.session_state.provider_initialized = True

with st.sidebar:
    DOCS_DIR = os.path.abspath("./uploaded_docs")
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
    st.subheader("Add to the Knowledge Base")
    with st.form("my-form", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Upload a file to the Knowledge Base:", accept_multiple_files=True
        )
        submitted = st.form_submit_button("Upload!")

    if uploaded_files and submitted:
        for uploaded_file in uploaded_files:
            st.success(f"File {uploaded_file.name} uploaded successfully!")
            with open(os.path.join(DOCS_DIR, uploaded_file.name), "wb") as vs_out:
                vs_out.write(uploaded_file.read())

############################################
# Component #2 - Embedding Model and LLM
############################################


# Use the selected provider from the command-line argument
document_embedder = provider_class.get_embedder_instance()

############################################
# Component #3 - Vector Database Creation
############################################

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

with st.sidebar:
    # Option for using an existing vector store
    use_existing_vector_store = st.radio(
        "Use existing vector store if available",
        ["Yes", "No"],
        horizontal=True,
        index=1,
    )

# Path to the vector store file
vector_store_path = "vectorstore.bin"

# Load raw documents from the directory
raw_documents = PyPDFDirectoryLoader(DOCS_DIR).load()

# Check for existing vector store file
vector_store_exists = os.path.exists(vector_store_path)
vectorstore: FAISS | None = None

if use_existing_vector_store == "Yes" and vector_store_exists:
    with open(vector_store_path, "rb") as vs_in:  # Open in read mode
        vectorstore = FAISS.deserialize_from_bytes(
            serialized=vs_in.read(),
            embeddings=document_embedder,
            allow_dangerous_deserialization=True,
        )
    with st.sidebar:
        st.success("Existing vector store loaded successfully.")
else:
    with st.sidebar:
        if raw_documents:
            with st.spinner("Splitting documents into chunks..."):
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500, chunk_overlap=200
                )
                documents = text_splitter.split_documents(raw_documents)

            with st.spinner("Adding document chunks to vector database..."):
                vectorstore = FAISS.from_documents(documents, document_embedder)

            with st.spinner("Saving vector store"):
                with open(vector_store_path, "wb") as vs_out:
                    vs_out.write(vectorstore.serialize_to_bytes())
            st.success("Vector store created and saved.")
        else:
            st.warning("No documents available to process!", icon="⚠️")

############################################
# Component #4 - LLM Response Generation and Chat
############################################

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

st.subheader("Chat with your AI Assistant, Envie!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant named Envie. You will reply to "
            "questions only based on the context that you are provided. If "
            "something is out of context, you will refrain from replying and "
            "politely decline to respond to the user.",
        ),
        ("user", "{input}"),
    ]
)

user_input = st.chat_input("Can you tell me what NVIDIA is known for?")
llm = provider_class.get_llm_instance()
chain = prompt_template | llm | StrOutputParser()

if user_input and vectorstore is not None:
    st.session_state.messages.append({"role": "user", "content": user_input})
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    context = ""
    for doc in docs:
        context += doc.page_content + "\n\n"

    augmented_user_input = "Context: " + context + "\n\nQuestion: " + user_input + "\n"

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        for response in chain.stream({"input": augmented_user_input}):
            full_response += response
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

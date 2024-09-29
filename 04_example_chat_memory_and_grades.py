############################################
# Hard coded paths/files
############################################

PATH__UPLOAD_DIR = "./uploaded_docs"
PATH__VECTORSTORE_FILE = "vectorstore.bin"

############################################
# Component #1 - Document Loader
############################################

import os
from pathlib import Path

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
    docs_dir = Path(PATH__UPLOAD_DIR).absolute()
    if not docs_dir.exists():
        os.makedirs(docs_dir)

    st.subheader("Add Knowledge Base")
    with st.form("knowledge-form", clear_on_submit=False):
        uploaded_files = st.file_uploader(
            "Upload a file to the Knowledge Base:", accept_multiple_files=True
        )

        url1 = st.text_input("URL 1")
        url2 = st.text_input("URL 2")
        url3 = st.text_input("URL 3")

        knowledge_submitted = st.form_submit_button("Store Knowledge!")

    if uploaded_files and knowledge_submitted:
        for uploaded_file in uploaded_files:
            st.success(f"File {uploaded_file.name} uploaded successfully!")
            with (docs_dir / uploaded_file.name).open("wb") as uploaded_file_copy:
                uploaded_file_copy.write(uploaded_file.read())

    given_urls = [url for url in [url1, url2, url3] if url]  # filter out empty strings
    if given_urls and knowledge_submitted:
        if "urls" not in st.session_state:
            st.session_state["urls"] = []
        st.session_state["urls"].extend(given_urls)

############################################
# Component #2 - Embedding Model and LLM
############################################

document_embedder = provider_class.get_embedder_instance()

############################################
# Component #3 - Vector Database Stor
############################################

from langchain_community.document_loaders import PyPDFDirectoryLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

with st.sidebar:
    # Option for using an existing vector store
    use_existing_vector_store = st.radio(
        label="Use existing vector store if available",
        options=["Yes", "No"],
        index=1,
        horizontal=True,
    )

# Path to the vector store file
vector_store_path = Path(PATH__VECTORSTORE_FILE)

# Check for existing vector store file
vectorstore: FAISS | None = None
if use_existing_vector_store == "Yes" and vector_store_path.exists():
    with vector_store_path.open("rb") as vs_in:
        vectorstore = FAISS.deserialize_from_bytes(
            serialized=vs_in.read(),
            embeddings=document_embedder,
            allow_dangerous_deserialization=True,
        )
    with st.sidebar:
        st.success("Existing vector store loaded successfully.")
else:
    # Load raw documents from the directory
    uploaded_docs = PyPDFDirectoryLoader(str(docs_dir)).load()

    with st.sidebar:
        uploaded_splits = []
        if uploaded_docs:
            with st.spinner("Splitting documents into chunks..."):
                upload_text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000, chunk_overlap=200
                )
                uploaded_splits = upload_text_splitter.split_documents(uploaded_docs)

        web_splits = []
        given_urls = st.session_state.get("urls", [])
        if given_urls:
            web_loaded_docs = [WebBaseLoader(url).load() for url in given_urls]
            web_docs = [item for sublist in web_loaded_docs for item in sublist]

            # Add mode documents to vectorDB
            if web_docs:
                web_text_splitter = (
                    RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                        chunk_size=250, chunk_overlap=0
                    )
                )
                web_splits = web_text_splitter.split_documents(web_docs)

        if uploaded_splits or web_splits:
            with st.spinner("Storing to Vector store..."):
                vectorstore = FAISS.from_documents(
                    web_splits + uploaded_splits, document_embedder
                )
            with st.spinner("Saving vector store"):
                with vector_store_path.open("wb") as vs_out:
                    vs_out.write(vectorstore.serialize_to_bytes())
            st.success("Vector store created and saved.")
        else:
            st.warning("No Knowledge available to process!", icon="⚠️")

############################################
# Component #4 - LLM Response Generation and Chat
############################################

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory

st.subheader("Chat with your AI Assistant, Envie!")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "how may i help you?",
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

llm = provider_class.get_llm_instance()
conversation_memory = ConversationBufferMemory()
conversation = ConversationChain(
    memory=conversation_memory,
    llm=llm,
    # verbose=True
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
else:
    for message in st.session_state.chat_history:
        conversation_memory.save_context(
            {"input": message["human"]}, {"outputs": message["AI"]}
        )

if user_input := st.chat_input("Can you tell me what NVIDIA is known for?"):
    with st.chat_message("user"):
        st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

    if vectorstore is not None:
        retriever = vectorstore.as_retriever()
        retrieved_docs = retriever.invoke(user_input)

        ### Retrieval Grader
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        retrieval_grader_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a grader assessing relevance of a retrieved document to a user "
                    "question. If the document contains keywords related to the user question, "
                    "grade it as relevant. It does not need to be a stringent test. The goal is "
                    "to filter out erroneous retrievals. \nGive a binary score 'yes' or 'no' "
                    "score to indicate whether the document is relevant to the question. \n "
                    "Provide the binary score as a JSON with a single key 'score' and no "
                    "preamble or explanation.",
                ),
                (
                    "user",
                    "Here is the retrieved document: \n\n {document} \n\nHere is the user "
                    "question: {question}\n",
                ),
            ]
        )

        retrieval_grader = retrieval_grader_prompt | llm | JsonOutputParser()

        context = ""
        for doc in retrieved_docs:
            json_grade = retrieval_grader.invoke(
                {"question": user_input, "document": doc.page_content}
            )
            if json_grade.get("score") == "no":
                continue
            context += doc.page_content + "\n\n"
        augmented_user_input = (
            "Context: " + context + "\n\nQuestion: " + user_input + "\n"
        )
    else:
        augmented_user_input = user_input

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = conversation.predict(input=augmented_user_input)
            ### Hallucination Grader
            # Prompt
            from langchain_core.output_parsers import JsonOutputParser
            from langchain_core.prompts import ChatPromptTemplate

            hallucination_grader_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a grader assessing whether an answer is grounded in / supported "
                        "by a set of facts. Give a binary 'yes' or 'no' score to indicate "
                        "whether the answer is grounded in / supported by a set of facts. "
                        "Provide the binary score as a JSON with a single key 'score' and no "
                        "preamble or explanation.",
                    ),
                    (
                        "user",
                        "Here are the facts:\n ------- \n{documents}\n ------- \nHere is the "
                        "answer: {generation}",
                    ),
                ]
            )

            hallucination_grader = (
                hallucination_grader_prompt | llm | JsonOutputParser()
            )
            hallucination_grade = hallucination_grader.invoke(
                {"documents": augmented_user_input, "generation": response}
            )
            print(hallucination_grade)

            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            message = {"human": augmented_user_input, "AI": response}
            st.session_state.chat_history.append(message)

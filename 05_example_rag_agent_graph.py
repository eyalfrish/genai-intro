from pprint import pprint
from typing import Literal, TypedDict

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph

from models.provider_factory import ProviderFactory

###################################################
# Create Document Embedder and LLM for generation
###################################################

# Parse the arguments
user_args = ProviderFactory.parse_provider_arg()
provider_class = ProviderFactory.get_provider(user_args.inference_provider)
provider_class.initialize_provider()

###################################################
# Create Document Embedder and LLM for generation
###################################################

document_embedder = provider_class.get_embedder_instance()
llm_answer = provider_class.get_llm_instance()
llm_grader = provider_class.get_llm_instance()
llm_route = provider_class.get_llm_instance()

###################################################
# RAG Documents (in this case, web-based)
###################################################

# RAG web-based documents on - LLM agents & prompt engineering & adversarial attacks
RAG_DOCS_SUBJECTS = ["llm agent", "prompt engineering"]
rag_websites = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
]

###################################################
# Create Vectorstore
###################################################

# Pull documents from the web and flatten into a single list
web_loaded_docs = [WebBaseLoader(url).load() for url in rag_websites]
web_docs = [item for sublist in web_loaded_docs for item in sublist]

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=400, chunk_overlap=100
)
web_docs_chunks = text_splitter.split_documents(web_docs)

# Create a vector store and retriever
vectorstore = FAISS.from_documents(web_docs_chunks, document_embedder)

###################################################################################
# Create Chains and Tools
###################################################################################
# 1) question router
# 2) retriever
# 3) retrieval relevance grader
# 4) answer generator
# 5) answer hallucination/grounding grader
# 6) answer relevance grader
# 7) web search tool
###################################################################################

from typing import Final

RUN_EXAMPLE: Final[bool] = False

###################################################
# Create a router (vectorestore vs web-search)
###################################################

question_router_prompt = PromptTemplate(
    template="""System: You are an expert at routing a user question to a vectorstore or web
    search. Use the vectorstore for questions on {rag_subjects}. You do not need to be stringent
    with the keywords in the question related to these topics. Otherwise, use web-search.
    Give a binary choice 'web_search' or 'vectorstore' based on the question. Return the a JSON
    with a single key 'datasource' and no premable or explanation.
    Question to route:{question}""",
    input_variables=["rag_subjects", "question"],
)

question_router = question_router_prompt | llm_route | JsonOutputParser()

###### EXAMPLE USAGE ######
if RUN_EXAMPLE:
    print("################# Testing question router...")
    ex_question = "What are the types of agent memory?"
    ex_question_route_answer = question_router.invoke(
        {
            "question": ex_question,
            "rag_subjects": " ".join(RAG_DOCS_SUBJECTS),
        }
    )
    print(
        f"Question '{ex_question}' routed to {ex_question_route_answer['datasource']}"
    )
###########################

###################################################
# Create a vectorstore and a retriever
###################################################

retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

###### EXAMPLE USAGE ######
if RUN_EXAMPLE:
    print("################# Testing retrival...")
    ex_retrieved_docs = retriever.invoke(ex_question)
    print(
        f"Retrieved {len(ex_retrieved_docs)} documents for the question '{ex_question}'"
    )
###########################

###################################################
# Create a Retrieval Relevance Grader
###################################################

retrieval_relevance_grader_prompt = PromptTemplate(
    template="""System: You are a grader assessing relevance
    of a retrieved document to a user question. If the document
    contains keywords related to the user question, grade it as
    relevant. It does not need to be a stringent test. The goal
    is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether
    the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score'
    and no premable or explanation.
    User: Here is the retrieved document: \n\n {document}
    \n\n Here is the user question: {question} \n
    """,
    input_variables=["question", "document"],
)

# Chain the ret-grader prompt, LLM, and json/structured output parser
retrieval_relevance_grader = (
    retrieval_relevance_grader_prompt | llm_grader | JsonOutputParser()
)

###### EXAMPLE USAGE ######
if RUN_EXAMPLE:
    print("################# Testing relevance grading/filtering...")
    ex_relevant_retrieved_docs = []
    for doc in ex_retrieved_docs:
        ex_relevance_grader_answer = retrieval_relevance_grader.invoke(
            {"question": ex_question, "document": doc.page_content}
        )
        ex_relevance_grade = ex_relevance_grader_answer["score"]
        if ex_relevance_grade.lower() == "yes":
            ex_relevant_retrieved_docs.append(doc)
    print(
        f"{len(ex_relevant_retrieved_docs)} out of "
        f"{len(ex_retrieved_docs)} documents found relevant"
    )
###########################

###################################################
# Answer Generator (post prompt augmented)
###################################################

augmented_question_prompt = PromptTemplate(
    template="""System: You are an assistant for question-answering
    tasks. Use the following pieces of retrieved context to answer
    the question. If you don't know the answer, just say that you
    don't know. Use three sentences maximum and keep the answer
    concise.
    User:
    Question: {question}
    Context: {context}""",
    input_variables=["question", "context"],
)

# Chain the answer generation prompt, LLM, and string output parser
answer_generator = augmented_question_prompt | llm_answer | StrOutputParser()

###### EXAMPLE USAGE ######
if RUN_EXAMPLE:
    print("################# Testing answer generation...")
    ex_generated_answer = answer_generator.invoke(
        {"context": ex_relevant_retrieved_docs, "question": ex_question}
    )
    print(f"Generated answer: {ex_generated_answer}")
###########################

###################################################
# Create an Answer Hallucination/Grounding Grader
###################################################

answer_grounding_grader_prompt = PromptTemplate(
    template="""System: You are a grader assessing whether an answer is grounded in / supported
    by a set of facts. Give a binary 'yes' or 'no' score to indicate whether the answer is
    grounded in / supported by a set of facts. Provide the binary score as a JSON with a
    single key 'score' and no preamble or explanation.
    User: Here are the facts:
    \n ------- \n{documents}\n ------- \nHere is the answer: {generation}""",
    input_variables=["generation", "documents"],
)

answer_grounding_grader = (
    answer_grounding_grader_prompt | llm_grader | JsonOutputParser()
)

###### EXAMPLE USAGE ######
if RUN_EXAMPLE:
    print("################# Testing answer Hallucination/Grounding...")
    print(
        "NOTE: Using relevant_retrieved_docs rather than web_doc "
        "due to all web_docs being too large."
    )
    ex_answer_grounding_grader_answer = answer_grounding_grader.invoke(
        {"documents": ex_relevant_retrieved_docs, "generation": ex_generated_answer}
    )
    ex_answer_grounding_grade = ex_answer_grounding_grader_answer["score"]
    if ex_answer_grounding_grade.lower() == "yes":
        print("Generated answer was found grounded in the documents.")
        ex_grounded_answer = ex_generated_answer
    else:
        print("Generated answer was found NOT grounded in the documents.")
        ex_grounded_answer = "I don't know."
###########################

###################################################
# Create an Answer Relevance Grader
###################################################

# Prompt
answer_relevance_grader_prompt = PromptTemplate(
    template="""System: You are a grader assessing whether an answer is useful to resolve a
    question. Give a binary score 'yes' or 'no' to indicate whether the answer is useful to
    resolve a question. Provide the binary score as a JSON with a single key 'score' and no
    preamble or explanation. User: Here is the answer:\n ------- \n{generation}\n ------- \n
    Here is the question: {question}""",
    input_variables=["generation", "question"],
)

answer_relevance_grader = (
    answer_relevance_grader_prompt | llm_grader | JsonOutputParser()
)

###### EXAMPLE USAGE ######
if RUN_EXAMPLE:
    print("################# Testing answer for relevance to the question...")
    ex_answer_relevance_grader_answer = answer_relevance_grader.invoke(
        {"generation": ex_grounded_answer, "question": ex_question}
    )
    ex_answer_relevance_grade = ex_answer_relevance_grader_answer["score"]
    if ex_answer_relevance_grade.lower() == "yes":
        print("Generated answer was found relevant to the question.")
        ex_grounded_and_relevant_answer = ex_grounded_answer
    else:
        print("Generated answer was found NOT relevant to the question.")
        ex_grounded_and_relevant_answer = "I don't know."
###########################

###################################################
# Build web search tool
###################################################

web_search_tool = TavilySearchResults(max_results=5)

###### EXAMPLE USAGE ######
if RUN_EXAMPLE:
    print("################# Testing web search on question...")
    ex_web_search_results = web_search_tool.invoke({"query": ex_question})
    ex_web_search_results_flattened_content = "\n".join(
        [
            ex_web_search_result["content"]
            for ex_web_search_result in ex_web_search_results
        ]
    )
    ex_web_results = Document(page_content=ex_web_search_results_flattened_content)
    print(f"Web search results: {ex_web_results}")
###########################

###################################################################################
# Create Graph Nodes
###################################################################################
# 1) Graph's state definition
# 2) Retriever node
# 3) Answer Generation node
# 4) Answer Grading node
# 5) WebSearch node
###################################################################################

###################################################
# Graph's state (running through the graph)
###################################################


class RagState(TypedDict, total=False):
    """Represents the state of our flow."""

    question: str  # User question
    generation: str  # LLM generation
    do_web_search: bool  # Whether to add search
    remaining_documents: list[Document]  # List of still-relevant documents


class RagStateTotal(TypedDict, total=True):
    """Represents the state of our flow."""

    question: str  # User question
    generation: str  # LLM generation
    do_web_search: bool  # Whether to add search
    remaining_documents: list[Document]  # List of still-relevant documents


###################################################
# Retriever node
###################################################


def retrieve_node(state: RagStateTotal) -> RagState:
    """
    Retrieve documents from vectorstore
    Expected state.question
    Updates state.remaining_documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    retrieved_documents = retriever.invoke(question)

    assert state["remaining_documents"] == [], "remaining_documents should be empty"
    return {"remaining_documents": retrieved_documents}


###################################################
# Answer Generation node
###################################################


def answer_generate_node(state: RagStateTotal) -> RagState:
    """
    Generate answer using RAG on retrieved documents
    Expected state.question and state.remaining_documents
    Updates state.generation
    """
    print("---GENERATE---")
    question, documents = state["question"], state["remaining_documents"]

    # RAG generation
    generated_answer = answer_generator.invoke(
        {"context": documents, "question": question}
    )
    return {"generation": generated_answer}


###################################################
# Answer Grading node
###################################################


def grade_retrieval_relevance_node(state: RagStateTotal) -> RagState:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search
    Expected state.question and state.remaining_documents
    Updates state.remaining_documents and state.do_web_search
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question, retrieved_documents = state["question"], state["remaining_documents"]

    # Score each doc
    filtered_retrieved_documents: list[Document] = []
    do_web_search = True
    for retrieved_document in retrieved_documents:
        score = retrieval_relevance_grader.invoke(
            {"question": question, "document": retrieved_document.page_content}
        )
        grade = score["score"]
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_retrieved_documents.append(retrieved_document)

            # One document matches, lets drop the idea of web-search
            do_web_search = False
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue

    return {
        "remaining_documents": filtered_retrieved_documents,
        "do_web_search": do_web_search,
    }


###################################################
# WebSearch node
###################################################


def web_search_node(state: RagStateTotal) -> RagState:
    """
    Web search based on the question
    Expected state.question and state.remaining_documents
    Updates state.remaining_documents and state.do_web_search
    """
    print("---WEB SEARCH---")
    # TODO: Remove remaining_documents?
    question, documents = state["question"], state["remaining_documents"]

    # Web search
    web_search_results = web_search_tool.invoke({"query": question})  # type: ignore[missing-typed-dict-key, invalid-key]
    web_search_results_flattened_content = "\n".join(
        [web_search_result["content"] for web_search_result in web_search_results]
    )
    web_results = Document(page_content=web_search_results_flattened_content)

    return {
        "remaining_documents": documents + [web_results],
        "do_web_search": True,  # Mark that we performed a web search (for final state)
    }


###################################################################################
# Create Graph Conditional Edges/Entrypoint
###################################################################################
# 1) Route question to web search or vectorstore conditional entrypoint
# 2) Generate or WebSearch conditional edge
# 3) Grade generated answer vs docs and question conditional edge
###################################################################################

###################################################
# Vectorstore or WebSearch Routing Conditional Entrypoint
###################################################


def route_question_cond_edge(
    state: RagStateTotal,
) -> Literal["websearch", "vectorstore"]:
    """
    Route question to web search or vectorstore.
    Expected state.question
    Returns next node to call (websearch or vectorstore)
    """
    print("---ROUTE QUESTION---")
    question = state["question"]

    source = question_router.invoke(
        {"question": question, "rag_subjects": " ".join(RAG_DOCS_SUBJECTS)}
    )
    if source["datasource"] == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source["datasource"] == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    raise RuntimeError(f"datasource {source['datasource']} not recognize")


###################################################
# Generate or WebSearch Conditional Edge
###################################################


def generate_or_websearch_cond_edge(
    state: RagStateTotal,
) -> Literal["websearch", "generate"]:
    """
    Determines whether to generate an answer, or add web search
    Expected state.do_web_search
    Returns next node to call (websearch or generate)
    """
    print("---ASSESS GRADED DOCUMENTS---")
    do_web_search = state["do_web_search"]

    if do_web_search:
        print("---DECISION: DOCUMENTS NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "websearch"
    else:
        print("---DECISION: GENERATE---")
        return "generate"


###################################################
# Generate or WebSearch Conditional Edge
###################################################


def grade_generated_answer_vs_docs_and_question_cond_edge(
    state: RagStateTotal,
) -> Literal["useful", "not useful", "not supported"]:
    """
    Grade the generated answer vs documents (grounding/hallucination) and
    then grade the answer vs the question (relevance)
    Expected state.question, state.remaining_documents, and state.generation
    Returns next node to call (useful, not useful, not supported)
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["remaining_documents"]
    generation = state["generation"]

    answer_grounding_grader_answer = answer_grounding_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    answer_grounding_grade = answer_grounding_grader_answer["score"]
    if answer_grounding_grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")

        answer_relevance_grader_answer = answer_relevance_grader.invoke(
            {"generation": generation, "question": question}
        )
        answer_relevance_grade = answer_relevance_grader_answer["score"]
        if answer_relevance_grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


###################################################################################
# Create Graph Conditional Edges
###################################################################################
# 1) Route question to web search or vectorstore conditional edge
# 2) Generate or WebSearch conditional edge
# 3) Grade generated answer vs docs and question conditional edge
###################################################################################

# Define the graph's state
rag_flow = StateGraph(RagStateTotal)  # type: ignore[invalid-argument-type]  # ty doesn't resolve TypedDict as StateLike

# Define the nodes
rag_flow.add_node("web_search_node", web_search_node)
rag_flow.add_node("retrieve_node", retrieve_node)
rag_flow.add_node("grade_retrieval_relevance_node", grade_retrieval_relevance_node)
rag_flow.add_node("answer_generate_node", answer_generate_node)

# Place router conditional entrypoint, connected to the retrieve node and the web search node
rag_flow.set_conditional_entry_point(
    route_question_cond_edge,
    {
        "websearch": "web_search_node",
        "vectorstore": "retrieve_node",
    },
)

# Connect the retrieve node to the grade_retrieval_relevance_node
rag_flow.add_edge("retrieve_node", "grade_retrieval_relevance_node")

# Based on the grade_retrieval_relevance_node, connect to the web search node or the
# answer generation node
rag_flow.add_conditional_edges(
    "grade_retrieval_relevance_node",
    generate_or_websearch_cond_edge,
    {
        "websearch": "web_search_node",
        "generate": "answer_generate_node",
    },
)

# Connect the web search node to the answer generation node
rag_flow.add_edge("web_search_node", "answer_generate_node")

rag_flow.add_conditional_edges(
    "answer_generate_node",
    grade_generated_answer_vs_docs_and_question_cond_edge,
    {
        "not supported": "answer_generate_node",
        "useful": END,
        "not useful": "answer_generate_node",
    },
)

###################################################################################
# Compile Graph into our RAG App and run two example questions
###################################################################################

rag_app = rag_flow.compile()

rag_output = rag_app.invoke(
    {
        "question": "What are the types of agent memory?",
        "generation": "",
        "do_web_search": False,
        "remaining_documents": [],
    }
)
pprint(rag_output)

rag_output = rag_app.invoke(
    {
        "question": "Who are the Bears expected to draft first in the NFL draft?",
        "generation": "",
        "do_web_search": False,
        "remaining_documents": [],
    }
)
pprint(rag_output)

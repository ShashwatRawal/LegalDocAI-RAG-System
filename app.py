import os
import time
import uuid
import streamlit as st
import fitz  # PyMuPDF library
from langchain_text_splitters import RecursiveCharacterTextSplitter
import concurrent.futures

# Import Google Cloud libraries
from google.cloud import aiplatform
from google.cloud.aiplatform import gapic as aiplatform_v1
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import Namespace
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel

# --- 1. CONFIGURATION ---
PROJECT_ID = "legalaid-469608"
REGION = "asia-south1"
INDEX_ID = "6657006344509325312"
INDEX_ENDPOINT_ID = "projects/519955132829/locations/asia-south1/indexEndpoints/8925448366192590848"
DEPLOYED_INDEX_ID = "legal_deploy_1756326640249"


# --- 2. MODEL AND CLIENT INITIALIZATION (CACHED) ---
@st.cache_resource
def initialize_clients():
    """Initializes and returns all necessary Google Cloud clients and models."""
    aiplatform.init(project=PROJECT_ID, location=REGION)
    vertexai.init(project=PROJECT_ID, location=REGION)
    clients = {
        "embedding_model": TextEmbeddingModel.from_pretrained("text-embedding-004"),
        "generative_model": GenerativeModel("gemini-1.5-flash-002"),
        "index_endpoint": aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=INDEX_ENDPOINT_ID),
    }
    print("✅ Clients Initialized Successfully!")
    return clients

try:
    clients = initialize_clients()
    generative_model = clients["generative_model"]
except Exception as e:
    st.error(f"Failed to initialize Google Cloud clients. Please check configuration. Error: {e}")
    st.stop()

# --- 3. CORE BACKEND FUNCTIONS ---

@st.cache_data
def extract_and_chunk_text(file_bytes):
    """Extracts text from a PDF and splits it into chunks."""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        text = "".join(page.get_text() for page in doc)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len)
    return text_splitter.split_text(text)

def get_embeddings(text_chunks: list[str]):
    """Generates vector embeddings for a list of text chunks in batches."""
    all_embeddings = []
    for i in range(0, len(text_chunks), 256):
        batch = text_chunks[i: i + 256]
        embeddings_response = clients["embedding_model"].get_embeddings(batch)
        all_embeddings.extend([e.values for e in embeddings_response])
    return all_embeddings

def upsert_to_vector_search(document_id: str, text_chunks: list):
    """Embeds chunks, upserts them, and returns the chunk_store and timings."""
    local_chunk_store = {}
    try:
        start_embed_time = time.time()
        embeddings = get_embeddings(text_chunks)
        end_embed_time = time.time()
        embed_duration = end_embed_time - start_embed_time

        if not embeddings or len(embeddings) != len(text_chunks):
            st.error("Could not generate embeddings for the document chunks.")
            return None, 0, 0

        client_options = {"api_endpoint": f"{REGION}-aiplatform.googleapis.com"}
        index_client = aiplatform_v1.IndexServiceClient(client_options=client_options)
        index_path = index_client.index_path(project=PROJECT_ID, location=REGION, index=INDEX_ID)
        
        datapoints = []
        for i, chunk in enumerate(text_chunks):
            chunk_id = f"{document_id}_{i}"
            local_chunk_store[chunk_id] = chunk
            dp = aiplatform_v1.IndexDatapoint(
                datapoint_id=chunk_id, feature_vector=embeddings[i],
                restricts=[aiplatform_v1.IndexDatapoint.Restriction(namespace="document_id", allow_list=[document_id])],
            )
            datapoints.append(dp)

        request = aiplatform_v1.UpsertDatapointsRequest(index=index_path, datapoints=datapoints)
        
        start_upsert_time = time.time()
        index_client.upsert_datapoints(request=request)
        end_upsert_time = time.time()
        upsert_duration = end_upsert_time - start_upsert_time

        print(f"✅ Upsert request sent for {len(datapoints)} datapoints.")
        return local_chunk_store, embed_duration, upsert_duration
    except Exception as e:
        st.error(f"An unexpected error occurred during the upsert process: {e}")
        return None, 0, 0

# This is the main analysis function

def perform_full_analysis(chunks: list[str]):
    """
    Performs a full analysis on all document chunks using a Map-Reduce pattern.
    Returns a dictionary with the analysis parts and the total timing.
    """
    total_start_time = time.time()

    # MAP step: Analyze each chunk individually in parallel (with a limit)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        map_results = list(executor.map(map_analysis_chunks, chunks))

    successful_analyses = [res for res in map_results if res is not None]
    
    # REDUCE step: Combine the individual analyses into a final result
    final_analysis = reduce_analysis(successful_analyses)

    total_end_time = time.time()
    
    return final_analysis, (total_end_time - total_start_time)

# Helper function for the MAP step

def map_analysis_chunks(chunk: str):
    """Analyzes a single chunk of text with a retry mechanism."""
    
    # A simple retry loop
    for i in range(3): # Try up to 3 times
        try:
            prompt = f"""
            Analyze the following text from a legal document. Identify and extract two things:
            1. A brief, one-sentence summary of the main point or clause in this text.
            2. A list of any potential red flags, risks, or unusual terms for a standard user. If none, state 'No significant flags noted'.

            TEXT:
            ---
            {chunk}
            ---

            Provide the output in a structured format. Example:
            SUMMARY: This section outlines the monthly rent and payment due date.
            RED_FLAGS: No significant flags noted.
            """
            response = generative_model.generate_content(prompt)
            return response.text
        
        # If a ResourceExhausted error occurs, wait and then retry
        except Exception as e:
            if "ResourceExhausted" in str(e):
                print(f"Rate limit hit, retrying in {2**(i+1)} seconds...")
                time.sleep(2**(i+1)) # Exponential backoff: 2s, 4s
            else:
                print(f"Error processing a chunk: {e}")
                return None # For other errors, fail immediately
    
    print("Chunk analysis failed after multiple retries.")
    return None

# Helper function for the REDUCE step
def reduce_analysis(analyses: list[str]):
    """Synthesizes individual chunk analyses into a final summary, red flags, and glossary."""
    
    # Combine all the individual text analyses into one large block
    combined_text = "\n\n---\n\n".join(analyses)

    prompt = f"""
    You are an AI legal assistant. You have been provided with a series of analyses from individual chunks of a legal document.
    Your task is to synthesize this information into a final, comprehensive report with three distinct parts, separated by the specified markers.

    INDIVIDUAL CHUNK ANALYSES:
    ---
    {combined_text}
    ---

    Please synthesize the above information to generate the following, in this exact order and format:

    ---SUMMARY---
    A concise, easy-to-read summary in two parts: 1. A final, cohesive overview of the entire document's purpose. 2. A relevant "Key Details" table with the most important information discovered (e.g., Parties, Property, Rent, Deposit, Term).

    ---RED_FLAGS---
    A consolidated list of the Top 5-7 most critical red flags or unfavorable clauses from the entire document. For each, briefly explain the risk in simple terms.

    ---GLOSSARY---
    A glossary of 8-10 complex legal terms found in the text, each with a one-sentence explanation in simple language. Format as a Markdown list.
    """
    
    response = generative_model.generate_content(prompt)
    full_text = response.text
    
    try:
        # Split the single response into three parts using markers
        summary = full_text.split("---RED_FLAGS---")[0].replace("---SUMMARY---", "").strip()
        red_flags = full_text.split("---GLOSSARY---")[0].split("---RED_FLAGS---")[1].strip()
        glossary = full_text.split("---GLOSSARY---")[1].strip()

        return {
            "summary": summary,
            "red_flags": red_flags,
            "glossary": glossary
        }
    except Exception as e:
        print(f"Error reducing analysis: {e}")
        # Return a dictionary with error messages so the app doesn't crash
        return {
            "summary": "Error: Could not generate the final document summary.",
            "red_flags": "Error: Could not generate the final red flags analysis.",
            "glossary": "Error: Could not generate the final glossary."
        }



def ask_question_rag(document_id: str, question: str):
    """Finds relevant chunks and generates an answer using the Gemini model."""
    try:
        query_embedding = get_embeddings([question])[0]
        if not query_embedding:
            return "Could not generate an embedding for your question."

        index_endpoint = clients["index_endpoint"]
        doc_filter = [Namespace(name="document_id", allow_tokens=[document_id])]
        response = index_endpoint.find_neighbors(
            deployed_index_id=DEPLOYED_INDEX_ID, queries=[query_embedding], num_neighbors=5, filter=doc_filter,
        )

        context = "".join(st.session_state.chunk_store[neighbor.id] + "\n\n" for neighbor in response[0] if neighbor.id in st.session_state.chunk_store)
        if not context:
            return "Could not find relevant information in the document to answer your question."

        prompt = f"""
        You are an AI assistant for Indian legal documents. Your answers must be accurate and based ONLY on the provided context.
        Do not provide legal advice. If the context does not contain the answer, state that clearly.

        CONTEXT:
        ---
        {context}
        ---
        USER QUESTION: "{question}"
        ANSWER:
        """
        response = generative_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred while answering your question: {e}")
        return "Sorry, I encountered an error while trying to answer your question."

# --- 4. STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="Legal AI Demystifier")

# --- UI: Sidebar ---
with st.sidebar:
    st.image("https://storage.googleapis.com/maker-studio-project-emblems/Google-Cloud-emblems/gen-ai-emblem.png", width=100)
    st.title("📄 Legal AI Demystifier")
    st.markdown("Upload a legal document (PDF) to get an instant, easy-to-understand analysis.")
    uploaded_file = st.file_uploader(" ", type="pdf", label_visibility="collapsed")
    st.markdown("---")
    st.warning("**Disclaimer:** This is an AI tool for informational purposes and not a substitute for professional legal advice.")

# --- File Processing Logic ---

if uploaded_file and st.session_state.get("uploaded_filename") != uploaded_file.name:
    st.session_state.clear() 
    st.session_state.uploaded_filename = uploaded_file.name
    st.session_state.document_id = str(uuid.uuid4())
    st.session_state.analysis_ready = False
    
    with st.status("Analyzing your document...", expanded=True) as status:
        total_start_time = time.time()
        
        status.update(label="Step 1/3: 📄 Parsing and chunking the document...", state="running")
        file_bytes = uploaded_file.getvalue()
        chunks = extract_and_chunk_text(file_bytes)
        
        if chunks:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                status.update(label="Step 2/3: 🧠 Embedding content for Q&A...", state="running")
                future_upsert = executor.submit(upsert_to_vector_search, st.session_state.document_id, chunks)
                
                status.update(label="Step 3/3: ✨ Performing AI analysis on the full document...", state="running")
                # The new analysis function is called here with the list of chunks
                future_analysis = executor.submit(perform_full_analysis, chunks)
                
                analysis_results, analysis_time = future_analysis.result()
                chunk_store_result, embed_time, upsert_time = future_upsert.result()
                
                st.session_state.summary = analysis_results["summary"]
                st.session_state.red_flags = analysis_results["red_flags"]
                st.session_state.glossary = analysis_results["glossary"]
                
                if chunk_store_result is not None:
                    st.session_state.chunk_store = chunk_store_result
                    st.session_state.analysis_ready = True
                else:
                    st.error("Failed to prepare the document for Q&A. Please try again.")
                    st.session_state.analysis_ready = False
            
            total_end_time = time.time()
            st.session_state.perf_stats = (
                f"**Performance Breakdown:**\n"
                f"- **Total Time:** {total_end_time - total_start_time:.2f}s\n"
                f"- **Full Document AI Analysis (Map-Reduce):** {analysis_time:.2f}s\n"
                f"- **Embedding:** {embed_time:.2f}s | **DB Upsert:** {upsert_time:.2f}s"
            )
        else:
            st.error("Could not process the uploaded document.")
            st.session_state.analysis_ready = False
        status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

# --- UI Rendering Logic ---
if not uploaded_file:
    st.title("Welcome to the Legal AI Demystifier 👋")
    st.markdown("Feeling lost in legal jargon? You're in the right place. This tool transforms complex legal documents into simple, actionable insights.")
    st.markdown("### Key Features:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("** Instant Summaries**\n\nGet the a high-level overview and key details of your document in seconds.", icon="📊")
    with col2:
        st.info("** Red Flag Detection**\n\nAutomatically identify potentially risky or unfavorable clauses.", icon="🚩")
    with col3:
        st.info("** Ask Anything**\n\nUse the interactive chat to ask specific questions about your document.", icon="💬")
    st.info("To get started, simply **upload a PDF document** using the sidebar on the left.", icon="👈")
elif not st.session_state.get("analysis_ready"):
    st.warning("Analysis is in progress or an error occurred. Please wait or try re-uploading the document.")
else:
    if "perf_stats" in st.session_state:
        with st.expander("Show Performance Stats"):
            st.markdown(st.session_state.perf_stats)

    dashboard_tab, glossary_tab, chat_tab = st.tabs(["**📊 Dashboard**", "**📖 Key Terms Glossary**", "**💬 Ask a Question**"])

    with dashboard_tab:
        st.header("Dashboard")
        st.markdown("Your at-a-glance view of the most critical information.")
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.subheader("Key Details")
                st.markdown(st.session_state.summary.split("### Key Details")[1] if "### Key Details" in st.session_state.summary else st.session_state.summary)
        with col2:
            with st.container(border=True):
                st.subheader("🚩 Top Red Flags")
                st.markdown(st.session_state.red_flags)
    
    
    with glossary_tab:
        st.header("📖 Key Terms Glossary")
        st.markdown("A list of complex legal terms from your document, explained in simple language.")
        with st.container(border=True):
            st.markdown(st.session_state.glossary)

    

    with chat_tab:
        st.header("💬 Ask a Question")
        st.markdown("Use this chat to ask specific questions about your document. Get clear answers based directly on its content.")

        # A container for the chat history to make it scrollable
        chat_container = st.container(height=400)

        # Initialize session state for messages if it doesn't exist
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display prior messages inside the container
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Show example questions if the chat is empty
        if not st.session_state.messages:
            st.markdown("---")
            st.markdown("**Some ideas to get you started:**")
            ex_col1, ex_col2, ex_col3 = st.columns(3)
            example_questions = [
                "What is the notice period for termination?",
                "Explain the indemnity clause in simple terms.",
                "Who is responsible for repairs and maintenance?"
            ]
            
            # define a function to avoid rerunning the whole script on button click
            def ask_example(question):
                st.session_state.example_question = question

            if ex_col1.button(example_questions[0], use_container_width=True):
                ask_example(example_questions[0])
            if ex_col2.button(example_questions[1], use_container_width=True):
                ask_example(example_questions[1])
            if ex_col3.button(example_questions[2], use_container_width=True):
                ask_example(example_questions[2])

        # The chat input is now outside the container, fixing it to the bottom
        if prompt := st.chat_input("e.g., 'What is the lock-in period?'") or st.session_state.get("example_question"):
            
            # Clear the example question from state so it doesn't re-run
            if st.session_state.get("example_question"):
                prompt = st.session_state.example_question
                st.session_state.example_question = None

            # Display user message immediately and add to history
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Get and display assistant response
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = ask_question_rag(st.session_state.document_id, prompt)
                        st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Rerun to clear the input box and show the example buttons again if needed
            st.rerun() 
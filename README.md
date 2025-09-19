# LegalDocAI: A Generative AI Legal Assistant


## 📖 Overview
LegalDocAI is a full-stack web application designed to demystify complex legal documents. It leverages a **Retrieval-Augmented Generation (RAG)** architecture on Google Cloud to provide users with instant summaries, risk analysis, and an interactive Q&A chat, making dense legal jargon accessible to everyone.

---
##  Key Features
* **Full-Document Analysis:** Uses a **Map-Reduce** pattern to analyze entire documents of any length, generating a comprehensive summary, a list of potential red flags, and a glossary of terms.
* **Trustworthy Q&A:** The interactive chat uses a RAG pipeline to ensure all answers are grounded directly in the source document, eliminating AI hallucinations.
* **Cloud-Native Architecture:** Built on a scalable and resilient stack using Google Cloud's Vertex AI services.
* **Intuitive User Interface:** A clean, user-friendly interface built with Streamlit that guides the user from document upload to final analysis.

---
##  Technology Stack
* **Cloud/AI:** Google Cloud Platform (GCP), Vertex AI (Gemini, Vector Search, Embeddings)
* **Backend:** Python, `concurrent.futures` for parallel processing
* **Frontend:** Streamlit
* **Core Libraries:** PyMuPDF (`fitz`), LangChain (`for text splitting`)

---
##  Getting Started

### Prerequisites
* Python 3.10+
* A Google Cloud Platform project with the Vertex AI API enabled.
* `gcloud` CLI installed and authenticated.

### Installation & Setup
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ShashwatRawal/LegalDocAI-RAG-System.git]
    cd LegalDoc-AI-RAG-System
    ```

2.  **Set up a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Configure Google Cloud Credentials:**
    ```bash
    gcloud auth application-default login
    ```
    
4.  **Update Configuration:**
    Open `app.py` and update the following variables with your GCP project details:
    * `PROJECT_ID`
    * `REGION`
    * `INDEX_ID`
    * `INDEX_ENDPOINT_ID`
    * `DEPLOYED_INDEX_ID`

5.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

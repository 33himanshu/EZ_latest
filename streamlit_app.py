import streamlit as st
import requests
from typing import Dict, List, Optional
import uuid
import json
from pathlib import Path
import time

# Configuration
API_BASE_URL = "http://localhost:8000"  # Update this if your API is running on a different port

# Page configuration
st.set_page_config(
    page_title="Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = {}
if 'current_document' not in st.session_state:
    st.session_state.current_document = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'challenge_questions' not in st.session_state:
    st.session_state.challenge_questions = []
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {}

# Helper Functions
def call_api(endpoint: str, method: str = "get", data: Optional[dict] = None, files: Optional[dict] = None):
    """Helper function to call the API."""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method.lower() == "get":
            response = requests.get(url)
        elif method.lower() == "post":
            response = requests.post(url, json=data, files=files)
        else:
            return None, "Unsupported HTTP method"
            
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.RequestException as e:
        return None, f"Error calling API: {str(e)}"

# UI Components
def render_sidebar():
    """Render the sidebar with document upload and selection."""
    st.sidebar.title("📂 Documents")
    
    # Document upload
    st.sidebar.subheader("Upload Document")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a PDF or TXT file", 
        type=["pdf", "txt"],
        key="file_uploader"
    )
    
    if uploaded_file is not None:
        if st.sidebar.button("Process Document"):
            with st.spinner("Processing document..."):
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                response, error = call_api("/documents/upload", method="post", files=files)
                
                if error:
                    st.sidebar.error(f"Error: {error}")
                else:
                    doc_id = response["document_id"]
                    st.session_state.documents[doc_id] = {
                        "id": doc_id,
                        "name": uploaded_file.name,
                        "summary": response["summary"],
                        "chunks": response["num_chunks"]
                    }
                    st.session_state.current_document = doc_id
                    st.sidebar.success("Document processed successfully!")
                    st.rerun()
    
    # Document list
    st.sidebar.subheader("Your Documents")
    if not st.session_state.documents:
        st.sidebar.info("No documents uploaded yet.")
    else:
        for doc_id, doc in st.session_state.documents.items():
            if st.sidebar.button(
                f"📄 {doc['name']}",
                key=f"doc_btn_{doc_id}",
                use_container_width=True,
                type="primary" if doc_id == st.session_state.current_document else "secondary"
            ):
                st.session_state.current_document = doc_id
                st.session_state.messages = []  # Clear chat history
                st.rerun()

def render_document_view():
    """Render the main document view."""
    if not st.session_state.current_document:
        st.info("Please upload and select a document to get started.")
        return
    
    doc = st.session_state.documents[st.session_state.current_document]
    
    # Document info
    st.title(doc["name"])
    st.subheader("Summary")
    st.write(doc["summary"])
    
    # Tabs for different modes
    tab1, tab2 = st.tabs(["💬 Ask Questions", "🎯 Challenge Mode"])
    
    with tab1:
        render_chat_interface()
    
    with tab2:
        render_challenge_interface()

def render_chat_interface():
    """Render the chat interface for Q&A."""
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.spinner("Thinking..."):
            response, error = call_api(
                "/ask",
                method="post",
                data={
                    "document_id": st.session_state.current_document,
                    "question": prompt
                }
            )
            
            if error:
                st.error(f"Error: {error}")
            else:
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"]
                })
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(response["answer"])
                    
                    # Show sources if available
                    if response.get("sources"):
                        with st.expander("Sources"):
                            for i, source in enumerate(response["sources"][:3]):
                                st.caption(f"Source {i+1} (similarity: {source['similarity']:.2f})")
                                st.info(source["text"])

def render_challenge_interface():
    """Render the challenge mode interface."""
    if not st.session_state.challenge_questions:
        if st.button("Generate Challenge Questions"):
            with st.spinner("Generating challenge questions..."):
                response, error = call_api(
                    "/challenge",
                    method="post",
                    data={
                        "document_id": st.session_state.current_document,
                        "num_questions": 3
                    }
                )
                
                if error:
                    st.error(f"Error: {error}")
                else:
                    st.session_state.challenge_questions = response["questions"]
                    st.rerun()
    else:
        # Display questions and collect answers
        st.markdown("### Test Your Understanding")
        st.write("Answer the following questions based on the document:")
        
        for i, question in enumerate(st.session_state.challenge_questions):
            st.markdown(f"#### Question {i+1}")
            st.write(question["question"])
            
            # Store user's answer
            answer_key = f"answer_{i}"
            if answer_key not in st.session_state.user_answers:
                st.session_state.user_answers[answer_key] = ""
                
            user_answer = st.text_area(
                "Your answer:",
                key=f"answer_input_{i}",
                value=st.session_state.user_answers[answer_key]
            )
            
            # Update stored answer
            if user_answer != st.session_state.user_answers[answer_key]:
                st.session_state.user_answers[answer_key] = user_answer
                st.rerun()
        
        # Submit button
        if st.button("Submit Answers"):
            if not all(st.session_state.user_answers.values()):
                st.warning("Please answer all questions before submitting.")
            else:
                with st.spinner("Evaluating your answers..."):
                    total_score = 0
                    feedback = []
                    
                    for i, question in enumerate(st.session_state.challenge_questions):
                        answer_key = f"answer_{i}"
                        response, error = call_api(
                            "/evaluate",
                            method="post",
                            data={
                                "document_id": st.session_state.current_document,
                                "question": question["question"],
                                "user_answer": st.session_state.user_answers[answer_key],
                                "reference_answer": question["answer"]
                            }
                        )
                        
                        if not error and response:
                            score = response.get("score", 0)
                            total_score += score
                            feedback.append({
                                "question": question["question"],
                                "user_answer": st.session_state.user_answers[answer_key],
                                "score": score,
                                "feedback": response.get("feedback", "No feedback provided.")
                            })
                    
                    # Calculate average score
                    avg_score = (total_score / len(st.session_state.challenge_questions)) * 100 if st.session_state.challenge_questions else 0
                    
                    # Display results
                    st.markdown("### 📊 Your Results")
                    st.metric("Overall Score", f"{avg_score:.1f}%")
                    
                    for i, item in enumerate(feedback):
                        with st.expander(f"Question {i+1} - Score: {item['score']*100:.1f}%"):
                            st.markdown(f"**Question:** {item['question']}")
                            st.markdown(f"**Your Answer:** {item['user_answer']}")
                            st.markdown(f"**Feedback:** {item['feedback']}")
                    
                    # Add a button to try again
                    if st.button("Try Again"):
                        st.session_state.challenge_questions = []
                        st.session_state.user_answers = {}
                        st.rerun()

# Main App
def main():
    st.sidebar.title("🔍 Research Assistant")
    render_sidebar()
    render_document_view()

if __name__ == "__main__":
    main()

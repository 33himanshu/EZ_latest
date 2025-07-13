import streamlit as st
import time
from typing import Dict, List, Tuple, Optional
from utils import extract_text_from_file
from summarizer import generate_summary
from question_answering import ask_question as default_ask_question, highlight_text
from challenge_mode import generate_questions, evaluate_answer
from ollama_qa import OllamaQA
import os
import json
import time

# Initialize Ollama QA model
USE_OLLAMA = False
qa_model = None

# First check if Ollama is running
try:
    import requests
    response = requests.get('http://localhost:11434/api/tags', timeout=5)
    if response.status_code == 200:
        print("Ollama is running!")
        try:
            qa_model = OllamaQA(model_name="llama3:instruct")
            print("✅ Successfully connected to Ollama with llama3:instruct model")
            USE_OLLAMA = True
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
            print("Trying to pull the model...")
            import subprocess
            try:
                subprocess.run(["ollama", "pull", "llama3:instruct"], check=True)
                qa_model = OllamaQA(model_name="llama3:instruct")
                print("✅ Successfully pulled and loaded llama3:instruct model")
                USE_OLLAMA = True
            except Exception as pull_error:
                print(f"❌ Failed to pull model: {pull_error}")
                USE_OLLAMA = False
except Exception as e:
    print(f"❌ Could not connect to Ollama: {e}")
    print("Please make sure Ollama is installed and running")
    print("You can download it from: https://ollama.ai/download")
    print("Falling back to default Hugging Face model...")

def ask_question(document_text: str, question: str) -> Dict:
    """Wrapper function to use either Ollama or default QA model"""
    if USE_OLLAMA:
        return qa_model.ask_question(document_text, question)
    return default_ask_question(document_text, question)

# Set page config
st.set_page_config(
    page_title="GenAI Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        max-width: 1200px;
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        font-weight: bold;
    }
    .stTextArea>div>div>textarea {
        min-height: 150px;
    }
    .highlight {
        background-color: #fffbcc;
        padding: 2px 4px;
        border-radius: 4px;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'document_text' not in st.session_state:
    st.session_state.document_text = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'show_questions' not in st.session_state:
    st.session_state.show_questions = False
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {}
if 'show_results' not in st.session_state:
    st.session_state.show_results = False

# App title and description
st.title("🧠 GenAI Research Assistant")
st.markdown("""
Welcome to the GenAI Research Assistant! Upload a document and interact with it in multiple ways:
1. Get an automatic summary
2. Ask questions about the content
3. Test your understanding with challenge questions
""")

# File uploader in sidebar
with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload a PDF or TXT document", 
        type=["pdf", "txt"],
        help="Upload a research paper, article, or any text document"
    )

# Document processing
if uploaded_file and not st.session_state.document_text:
    with st.spinner("🔍 Processing your document..."):
        try:
            # Extract text
            st.session_state.document_text = extract_text_from_file(uploaded_file)
            
            # Generate summary
            st.session_state.summary = generate_summary(st.session_state.document_text)
            
            # Clear previous questions and answers
            st.session_state.questions = []
            st.session_state.user_answers = {}
            st.session_state.show_questions = False
            st.session_state.show_results = False
            
            st.success("✅ Document processed successfully!")
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")

# Display document info and summary
if st.session_state.document_text:
    with st.expander("📄 Document Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Word Count", f"{len(st.session_state.document_text.split()):,}")
        with col2:
            st.metric("Characters", f"{len(st.session_state.document_text):,}")
    
    # Summary section
    with st.expander("📝 Summary (≤ 150 words)", expanded=True):
        st.write(st.session_state.summary)


 


# Ask Anything Mode
st.markdown("---")
st.subheader("💬 Ask Anything About the Document")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the document..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Get answer from the model
        with st.spinner("Thinking..."):
            result = ask_question(st.session_state.document_text, prompt)
            
            # Check if this is a comprehensive answer
            is_comprehensive = result.get('is_comprehensive', False)
            
            if is_comprehensive:
                # For comprehensive answers, show the full answer with clean formatting
                response = f"""
                <div style="margin-bottom: 1em;">
                    <div style="font-weight: bold; margin-bottom: 0.5em;">Answer:</div>
                    <div style="margin-bottom: 1em; white-space: pre-line;">{result['answer']}</div>
                </div>
                """
            else:
                # For regular answers, include confidence and source context
                if result['confidence'] > 70:
                    confidence_class = "confidence-high"
                elif result['confidence'] > 30:
                    confidence_class = "confidence-medium"
                else:
                    confidence_class = "confidence-low"
                
                # Create the base response with answer and confidence
                response = f"""
                <div style="margin-bottom: 1em;">
                    <div style="font-weight: bold; margin-bottom: 0.5em;">Answer:</div>
                    <div style="margin-bottom: 1em;">{result['answer']}</div>
                    
                    <div style="display: flex; align-items: center; margin-bottom: 1em;">
                        <div style="font-weight: bold; margin-right: 0.5em;">Confidence:</div>
                        <span class="{confidence_class}" style="font-weight: bold;">{result['confidence']}%</span>
                    </div>
                """
                
                # Add source context if available
                if result.get('context'):
                    response += f"""
                    <details style="margin-top: 1em; border: 1px solid #e0e0e0; border-radius: 4px; padding: 0.5em;">
                        <summary style="font-weight: bold; cursor: pointer; padding: 0.5em;">
                            View Source Context
                        </summary>
                        <div style="
                            background: #f8f9fa;
                            border-left: 4px solid #6c757d;
                            padding: 0.5em 1em;
                            margin: 0.5em 0;
                            border-radius: 0 4px 4px 0;
                            white-space: pre-wrap;
                            font-size: 0.9em;
                            line-height: 1.5;
                        ">
                            {result['context']}
                        </div>
                    </details>
                    """
                
                # Add the styles and close the main div
                response += """
                <style>
                    .confidence-high { color: #28a745; }
                    .confidence-medium { color: #ffc107; }
                    .confidence-low { color: #dc3545; }
                    .highlight {
                        background-color: #fff3cd;
                        padding: 0.1em 0.2em;
                        border-radius: 3px;
                        font-weight: bold;
                    }
                </style>
                </div>
                """
            
            # Simulate stream of response with milliseconds delay
            for chunk in response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
            
            message_placeholder.markdown(full_response, unsafe_allow_html=True)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Rerun to update the chat interface
    st.rerun()

# Challenge Mode
st.markdown("---")
st.subheader("🎯 Challenge Me: Test Your Understanding")

if not st.session_state.document_text:
    st.info("ℹ️ Please upload a document first to use Challenge Mode.")
else:
    # Generate Questions Button
    if st.button("🎯 Generate Challenge Questions", use_container_width=True):
        with st.spinner("🤔 Creating challenging questions..."):
            try:
                st.session_state.questions = generate_questions(st.session_state.document_text)
                st.session_state.show_questions = True
                st.session_state.show_results = False
                st.session_state.user_answers = {}
                st.success("🧠 Challenge questions generated!")
                st.rerun()
            except Exception as e:
                st.error(f"Error generating questions: {str(e)}")
                st.session_state.show_questions = False

    # Display questions if they exist
    if st.session_state.show_questions and st.session_state.questions:
        st.markdown("### 📝 Answer the following questions:")
        
        # Ensure questions is a list of dictionaries
        if isinstance(st.session_state.questions[0], str):
            # Convert old format to new format
            st.session_state.questions = [
                {'question': q, 'context': st.session_state.document_text[:1000]}
                for q in st.session_state.questions
            ]
        
        # Display questions with answer inputs
        for i, question_data in enumerate(st.session_state.questions):
            if isinstance(question_data, dict):
                question_text = question_data.get('question', 'No question text available')
                question_context = question_data.get('context', '')
            else:
                question_text = str(question_data)
                question_context = st.session_state.document_text[:1000]
                
            st.markdown(f"**Q{i+1}:** {question_text}")
            
            # Get or initialize answer
            if i not in st.session_state.user_answers:
                st.session_state.user_answers[i] = {
                    'answer': '',
                    'evaluation': None,
                    'context': question_data.get('context', '')
                }
            
            # Show answer text area
            answer = st.text_area(
                f"Your answer for Q{i+1}:",
                value=st.session_state.user_answers[i]['answer'],
                key=f"answer_{i}",
                height=100,
                disabled=st.session_state.show_results
            )
            
            # Update answer in session state
            st.session_state.user_answers[i]['answer'] = answer
            
            # Show evaluation if available
            if st.session_state.show_results and st.session_state.user_answers[i].get('evaluation'):
                eval_data = st.session_state.user_answers[i]['evaluation']
                if eval_data['is_correct']:
                    st.success(f" {eval_data['feedback']}")
                else:
                    st.error(f" {eval_data['feedback']}")
                
                # Show reference with highlighted text
                with st.expander(" View Reference", expanded=False):
                    st.markdown("**Relevant Document Excerpt:**")
                    st.markdown(f"> {eval_data['reference']}", unsafe_allow_html=True)
                    
                    # Show the full context if available
                    if 'full_context' in eval_data:
                        with st.expander("View Full Context"):
                            st.markdown(eval_data['full_context'])
            
            st.markdown("---")
        
        # Submit button (only show if not showing results yet)
        if not st.session_state.show_results:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📤 Submit All Answers", use_container_width=True):
                    if not all(ans['answer'].strip() for ans in st.session_state.user_answers.values()):
                        st.warning("Please answer all questions before submitting.")
                    else:
                        # Evaluate answers using the enhanced evaluation function
                        for i, question_data in enumerate(st.session_state.questions):
                            user_answer = st.session_state.user_answers[i]['answer']
                            evaluation = evaluate_answer(question_data, user_answer)
                            
                            # Add full context to the evaluation for reference
                            if isinstance(question_data, dict):
                                context = question_data.get('context', '')
                                if not context and 'context' in question_data:
                                    context = question_data['context']
                                evaluation['full_context'] = context or st.session_state.document_text[:1000]
                            else:
                                evaluation['full_context'] = st.session_state.document_text[:1000]
                            
                            # Store the evaluation
                            st.session_state.user_answers[i]['evaluation'] = evaluation
                        st.session_state.show_results = True
                        st.rerun()
            
            with col2:
                if st.button("🔄 Reset Answers", use_container_width=True):
                    st.session_state.user_answers = {}
                    st.rerun()
        else:
            if st.button("🔄 Try Again with New Questions", use_container_width=True):
                st.session_state.show_questions = False
                st.session_state.show_results = False
                st.rerun()

#!/bin/bash

# Start the Research Assistant application
# This script starts both the FastAPI backend and Streamlit frontend

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Python is installed
if ! command_exists python3; then
    echo -e "${YELLOW}Python 3 is required but not installed. Please install Python 3.8 or higher.${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))")
if [[ "$PYTHON_VERSION" < "3.8" ]]; then
    echo -e "${YELLOW}Python 3.8 or higher is required. Found Python $PYTHON_VERSION.${NC}"
    exit 1
fi

# Check if Ollama is running
if ! command_exists ollama; then
    echo -e "${YELLOW}Ollama is not installed. Please install Ollama from https://ollama.ai/${NC}"
    exit 1
fi

if ! curl -s http://localhost:11434/api/version >/dev/null; then
    echo -e "${YELLOW}Ollama is not running. Starting Ollama server...${NC}"
    nohup ollama serve > /dev/null 2>&1 &
    OLLAMA_PID=$!
    echo -e "Ollama started with PID $OLLAMA_PID"
    # Give Ollama some time to start
    sleep 5
else
    echo -e "${GREEN}✓ Ollama is running${NC}"
fi

# Check if required models are installed
echo -e "\n${GREEN}Checking for required Ollama models...${NC}"
REQUIRED_MODELS=("nomic-embed-text" "llama3:instruct")
MISSING_MODELS=()

for model in "${REQUIRED_MODELS[@]}"; do
    if ollama list | grep -q "$model"; then
        echo -e "${GREEN}✓ $model is installed${NC}"
    else
        echo -e "${YELLOW}⚠ $model is not installed${NC}"
        MISSING_MODELS+=("$model")
    fi
done

# Install missing models
if [ ${#MISSING_MODELS[@]} -gt 0 ]; then
    echo -e "\n${YELLOW}Installing missing models...${NC}"
    for model in "${MISSING_MODELS[@]}"; do
        echo -e "Installing $model..."
        ollama pull "$model"
    done
fi

# Create necessary directories
mkdir -p uploads vector_store

# Install Python dependencies if not already installed
echo -e "\n${GREEN}Checking Python dependencies...${NC}"
if [ ! -d "venv" ]; then
    echo -e "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Generate sample documents if they don't exist
if [ ! -f "samples/sample.pdf" ] || [ ! -f "samples/sample.txt" ]; then
    echo -e "\n${GREEN}Generating sample documents...${NC}"
    mkdir -p samples
    python generate_sample.py -o samples
fi

# Start FastAPI backend in the background
echo -e "\n${GREEN}Starting FastAPI backend...${NC}"
nohup uvicorn api:app --reload --port 8000 > backend.log 2>&1 &
BACKEND_PID=$!
echo -e "Backend started with PID $BACKEND_PID"

# Give the backend some time to start
sleep 5

# Start Streamlit frontend
echo -e "\n${GREEN}Starting Streamlit frontend...${NC}"
echo -e "\n${YELLOW}The Research Assistant will be available at: http://localhost:8501${NC}"
echo -e "${YELLOW}API documentation is available at: http://localhost:8000/docs${NC}"
echo -e "\n${YELLOW}Press Ctrl+C to stop the application${NC}\n"

streamlit run streamlit_app.py --server.port=8501

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    
    # Kill the backend process
    if kill -0 $BACKEND_PID > /dev/null 2>&1; then
        echo "Stopping FastAPI backend (PID: $BACKEND_PID)"
        kill $BACKEND_PID
    fi
    
    # Kill Ollama if we started it
    if [ -n "$OLLAMA_PID" ] && kill -0 $OLLAMA_PID > /dev/null 2>&1; then
        echo "Stopping Ollama server (PID: $OLLAMA_PID)"
        kill $OLLAMA_PID
    fi
    
    echo -e "${GREEN}Done.${NC}"
    exit 0
}

# Set up trap to catch Ctrl+C and clean up
trap cleanup INT TERM

# Keep the script running
wait $BACKEND_PID

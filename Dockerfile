FROM python:3.10-slim

WORKDIR /app

# Install wget and curl for downloading files
RUN apt-get update && apt-get install -y wget curl && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY streamlit_client.py .
COPY backend_server.py .

# Create directories for data and models
RUN mkdir -p data model_deployment

# Create startup script that downloads all files and starts services
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Starting file downloads from Google Drive..."\n\
\n\
# Function to download large files from Google Drive\n\
download_large_file() {\n\
    local file_id=$1\n\
    local output=$2\n\
    local description=$3\n\
    echo "Downloading $description..."\n\
    \n\
    # First request to get confirmation token\n\
    local confirm_url="https://drive.google.com/uc?export=download&id=$file_id"\n\
    local confirm_page=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "$confirm_url" -O-)\n\
    \n\
    # Extract confirmation token\n\
    local confirm_token=$(echo "$confirm_page" | grep -oP "confirm=\\K[^&]*" | head -1)\n\
    \n\
    if [ -z "$confirm_token" ]; then\n\
        # If no confirmation needed, download directly\n\
        if wget --load-cookies /tmp/cookies.txt -O "$output" "$confirm_url"; then\n\
            echo "✅ Successfully downloaded $description (direct)"\n\
            rm -f /tmp/cookies.txt\n\
            return 0\n\
        fi\n\
    else\n\
        # Use confirmation token for large files\n\
        local download_url="https://drive.google.com/uc?export=download&confirm=$confirm_token&id=$file_id"\n\
        if wget --load-cookies /tmp/cookies.txt -O "$output" "$download_url"; then\n\
            echo "✅ Successfully downloaded $description (with confirmation)"\n\
            rm -f /tmp/cookies.txt\n\
            return 0\n\
        fi\n\
    fi\n\
    \n\
    echo "❌ Failed to download $description"\n\
    rm -f /tmp/cookies.txt\n\
    return 1\n\
}\n\
\n\
# Function to download small files (no confirmation needed)\n\
download_small_file() {\n\
    local file_id=$1\n\
    local output=$2\n\
    local description=$3\n\
    echo "Downloading $description..."\n\
    if wget -O "$output" "https://drive.google.com/uc?export=download&id=$file_id"; then\n\
        echo "✅ Successfully downloaded $description"\n\
        return 0\n\
    else\n\
        echo "❌ Failed to download $description"\n\
        return 1\n\
    fi\n\
}\n\
\n\
# Download large data files\n\
if [ ! -z "$ARTICLES_DATA_CSV_ID" ]; then\n\
    download_large_file "$ARTICLES_DATA_CSV_ID" "data/articles_data.csv" "Articles Data CSV"\n\
    # Copy to model_deployment for compatibility\n\
    cp data/articles_data.csv model_deployment/articles_data.csv 2>/dev/null || echo "Failed to copy CSV"\n\
fi\n\
\n\
if [ ! -z "$ARTICLES_DATA_PKL_ID" ]; then\n\
    download_large_file "$ARTICLES_DATA_PKL_ID" "model_deployment/articles_data.pkl" "Articles Data PKL"\n\
fi\n\
\n\
# Download large model files\n\
if [ ! -z "$ARTICLES_FEATURES_NPY_ID" ]; then\n\
    download_large_file "$ARTICLES_FEATURES_NPY_ID" "model_deployment/articles_features.npy" "Articles Features"\n\
fi\n\
\n\
if [ ! -z "$KNN_MODELS_PKL_ID" ]; then\n\
    download_large_file "$KNN_MODELS_PKL_ID" "model_deployment/knn_models.pkl" "KNN Models"\n\
fi\n\
\n\
if [ ! -z "$KMEANS_MODEL_PKL_ID" ]; then\n\
    download_large_file "$KMEANS_MODEL_PKL_ID" "model_deployment/kmeans_model.pkl" "K-Means Model"\n\
fi\n\
\n\
if [ ! -z "$LSA_MODEL_PKL_ID" ]; then\n\
    download_large_file "$LSA_MODEL_PKL_ID" "model_deployment/lsa_model.pkl" "LSA Model"\n\
fi\n\
\n\
if [ ! -z "$TFIDF_VECTORIZER_PKL_ID" ]; then\n\
    download_large_file "$TFIDF_VECTORIZER_PKL_ID" "model_deployment/tfidf_vectorizer.pkl" "TF-IDF Vectorizer"\n\
fi\n\
\n\
# Download small metadata files\n\
if [ ! -z "$MODEL_METADATA_JSON_ID" ]; then\n\
    download_small_file "$MODEL_METADATA_JSON_ID" "model_deployment/model_metadata.json" "Model Metadata"\n\
fi\n\
\n\
if [ ! -z "$SUBJECT_INDEX_JSON_ID" ]; then\n\
    download_small_file "$SUBJECT_INDEX_JSON_ID" "model_deployment/subject_index.json" "Subject Index"\n\
fi\n\
\n\
echo "All downloads completed!"\n\
echo "Checking downloaded files..."\n\
ls -la data/\n\
ls -la model_deployment/\n\
echo "Starting services..."\n\
\n\
# Start backend server\n\
echo "Starting backend server..."\n\
uvicorn backend_server:app --host 0.0.0.0 --port 8000 &\n\
BACKEND_PID=$!\n\
echo "Backend started with PID $BACKEND_PID"\n\
\n\
# Wait a bit for backend to initialize\n\
sleep 10\n\
\n\
# Start frontend\n\
echo "Starting frontend..."\n\
streamlit run streamlit_client.py --server.address 0.0.0.0 --server.port $PORT --server.headless true --server.enableCORS false\n\
' > start.sh && chmod +x start.sh

# Expose port
EXPOSE 8501

# Start both services
CMD ["bash", "./start.sh"]
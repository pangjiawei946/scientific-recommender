FROM python:3.10-slim

WORKDIR /app

# Install required packages including gdown for Google Drive downloads
RUN apt-get update && apt-get install -y wget curl && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gdown

# Copy application files
COPY streamlit_client.py .
COPY backend_server.py .

# Create directories for data and models
RUN mkdir -p data model_deployment

# Create startup script using gdown
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Starting file downloads from Google Drive using gdown..."\n\
\n\
# Function to download with gdown\n\
download_with_gdown() {\n\
    local file_id=$1\n\
    local output=$2\n\
    local description=$3\n\
    echo "Downloading $description..."\n\
    if gdown --id "$file_id" --output "$output" --quiet; then\n\
        local size=$(stat -c%s "$output" 2>/dev/null || echo "0")\n\
        echo "✅ Successfully downloaded $description (${size} bytes)"\n\
        return 0\n\
    else\n\
        echo "❌ Failed to download $description with gdown"\n\
        return 1\n\
    fi\n\
}\n\
\n\
# Download data files\n\
if [ ! -z "$ARTICLES_DATA_CSV_ID" ]; then\n\
    download_with_gdown "$ARTICLES_DATA_CSV_ID" "data/articles_data.csv" "Articles Data CSV"\n\
    cp data/articles_data.csv model_deployment/articles_data.csv 2>/dev/null || echo "Failed to copy CSV"\n\
fi\n\
\n\
if [ ! -z "$ARTICLES_DATA_PKL_ID" ]; then\n\
    download_with_gdown "$ARTICLES_DATA_PKL_ID" "model_deployment/articles_data.pkl" "Articles Data PKL"\n\
fi\n\
\n\
if [ ! -z "$ARTICLES_FEATURES_NPY_ID" ]; then\n\
    download_with_gdown "$ARTICLES_FEATURES_NPY_ID" "model_deployment/articles_features.npy" "Articles Features"\n\
fi\n\
\n\
if [ ! -z "$KNN_MODELS_PKL_ID" ]; then\n\
    download_with_gdown "$KNN_MODELS_PKL_ID" "model_deployment/knn_models.pkl" "KNN Models"\n\
fi\n\
\n\
if [ ! -z "$KMEANS_MODEL_PKL_ID" ]; then\n\
    download_with_gdown "$KMEANS_MODEL_PKL_ID" "model_deployment/kmeans_model.pkl" "K-Means Model"\n\
fi\n\
\n\
if [ ! -z "$LSA_MODEL_PKL_ID" ]; then\n\
    download_with_gdown "$LSA_MODEL_PKL_ID" "model_deployment/lsa_model.pkl" "LSA Model"\n\
fi\n\
\n\
if [ ! -z "$TFIDF_VECTORIZER_PKL_ID" ]; then\n\
    download_with_gdown "$TFIDF_VECTORIZER_PKL_ID" "model_deployment/tfidf_vectorizer.pkl" "TF-IDF Vectorizer"\n\
fi\n\
\n\
if [ ! -z "$MODEL_METADATA_JSON_ID" ]; then\n\
    download_with_gdown "$MODEL_METADATA_JSON_ID" "model_deployment/model_metadata.json" "Model Metadata"\n\
fi\n\
\n\
if [ ! -z "$SUBJECT_INDEX_JSON_ID" ]; then\n\
    download_with_gdown "$SUBJECT_INDEX_JSON_ID" "model_deployment/subject_index.json" "Subject Index"\n\
fi\n\
\n\
echo "All downloads completed!"\n\
echo "Checking downloaded files..."\n\
ls -lh data/\n\
ls -lh model_deployment/\n\
echo "Starting services..."\n\
\n\
echo "Starting backend server..."\n\
uvicorn backend_server:app --host 0.0.0.0 --port 8000 &\n\
BACKEND_PID=$!\n\
echo "Backend started with PID $BACKEND_PID"\n\
sleep 10\n\
echo "Starting frontend..."\n\
streamlit run streamlit_client.py --server.address 0.0.0.0 --server.port $PORT --server.headless true --server.enableCORS false\n\
' > start.sh && chmod +x start.sh

EXPOSE 8501
CMD ["bash", "./start.sh"]
FROM python:3.10-slim

WORKDIR /app

# Install wget for downloading files
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

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
# Function to download files with retry\n\
download_file() {\n\
    local url=$1\n\
    local output=$2\n\
    local description=$3\n\
    echo "Downloading $description..."\n\
    if wget -O "$output" "$url"; then\n\
        echo "✅ Successfully downloaded $description"\n\
    else\n\
        echo "❌ Failed to download $description"\n\
        return 1\n\
    fi\n\
}\n\
\n\
# Download data files\n\
if [ ! -z "$ARTICLES_DATA_CSV_ID" ]; then\n\
    download_file "https://drive.google.com/uc?export=download&id=$ARTICLES_DATA_CSV_ID" \\\n\
                  "data/articles_data.csv" "Articles Data CSV"\n\
fi\n\
\n\
if [ ! -z "$ARTICLES_DATA_PKL_ID" ]; then\n\
    download_file "https://drive.google.com/uc?export=download&id=$ARTICLES_DATA_PKL_ID" \\\n\
                  "model_deployment/articles_data.pkl" "Articles Data PKL"\n\
fi\n\
\n\
# Download model files\n\
if [ ! -z "$ARTICLES_FEATURES_NPY_ID" ]; then\n\
    download_file "https://drive.google.com/uc?export=download&id=$ARTICLES_FEATURES_NPY_ID" \\\n\
                  "model_deployment/articles_features.npy" "Articles Features"\n\
fi\n\
\n\
if [ ! -z "$KNN_MODELS_PKL_ID" ]; then\n\
    download_file "https://drive.google.com/uc?export=download&id=$KNN_MODELS_PKL_ID" \\\n\
                  "model_deployment/knn_models.pkl" "KNN Models"\n\
fi\n\
\n\
if [ ! -z "$KMEANS_MODEL_PKL_ID" ]; then\n\
    download_file "https://drive.google.com/uc?export=download&id=$KMEANS_MODEL_PKL_ID" \\\n\
                  "model_deployment/kmeans_model.pkl" "K-Means Model"\n\
fi\n\
\n\
if [ ! -z "$LSA_MODEL_PKL_ID" ]; then\n\
    download_file "https://drive.google.com/uc?export=download&id=$LSA_MODEL_PKL_ID" \\\n\
                  "model_deployment/lsa_model.pkl" "LSA Model"\n\
fi\n\
\n\
if [ ! -z "$TFIDF_VECTORIZER_PKL_ID" ]; then\n\
    download_file "https://drive.google.com/uc?export=download&id=$TFIDF_VECTORIZER_PKL_ID" \\\n\
                  "model_deployment/tfidf_vectorizer.pkl" "TF-IDF Vectorizer"\n\
fi\n\
\n\
# Download metadata files\n\
if [ ! -z "$MODEL_METADATA_JSON_ID" ]; then\n\
    download_file "https://drive.google.com/uc?export=download&id=$MODEL_METADATA_JSON_ID" \\\n\
                  "model_deployment/model_metadata.json" "Model Metadata"\n\
fi\n\
\n\
if [ ! -z "$SUBJECT_INDEX_JSON_ID" ]; then\n\
    download_file "https://drive.google.com/uc?export=download&id=$SUBJECT_INDEX_JSON_ID" \\\n\
                  "model_deployment/subject_index.json" "Subject Index"\n\
fi\n\
\n\
echo "All downloads completed!"\n\
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
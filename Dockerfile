FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY streamlit_client.py .
COPY backend_server.py .

# Create startup script that runs both services
RUN echo '#!/bin/bash\n\
# Start backend in background\n\
uvicorn backend_server:app --host 0.0.0.0 --port 8000 &\n\
# Start frontend on the port Railway assigns\n\
streamlit run streamlit_client.py --server.address 0.0.0.0 --server.port $PORT --server.headless true\n\
' > start.sh && chmod +x start.sh

# Expose port
EXPOSE 8501

# Start both services
CMD ["bash", "./start.sh"]
FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY streamlit_client.py .
COPY backend_server.py .

# Create startup script that runs both services
RUN echo '#!/bin/bash\n\
# Start backend in background\n\
uvicorn backend_server:app --host 0.0.0.0 --port 8000 &\n\
# Start frontend\n\
streamlit run streamlit_client.py --server.address 0.0.0.0 --server.port \ --server.headless true\n\
' > start.sh && chmod +x start.sh

# Expose the port that Railway will assign
EXPOSE \

# Start both services
CMD ["./start.sh"]

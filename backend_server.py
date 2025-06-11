import ast
import hashlib
import joblib
import json
import numpy as np
import os
import pandas as pd
import re
import sqlite3
import uvicorn

from collections import Counter
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from typing import Optional, List, Dict, Any
import threading

# Initialize FastAPI app
app = FastAPI(title="Scientific Article Recommender API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store loaded data
articles_df = None
model_data = None

# Pydantic models for API requests/responses
class LoginRequest(BaseModel):
    username: str
    password: str

class SignupRequest(BaseModel):
    username: str
    password: str
    email: str
    first_name: str
    last_name: str
    privacy_consent: bool
    security_question: str
    security_answer: str

class PasswordResetRequest(BaseModel):
    username: str
    security_answer: str
    new_password: str

class RecommendationRequest(BaseModel):
    title_input: str
    top_n: int = 10
    filters: Optional[Dict[str, Any]] = None

class UserResponse(BaseModel):
    success: bool
    message: str
    user_data: Optional[Dict[str, Any]] = None

class RecommendationResponse(BaseModel):
    success: bool
    message: str
    recommendations: Optional[List[Dict[str, Any]]] = None

# Database functions
def init_sqlite_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            Id INTEGER PRIMARY KEY AUTOINCREMENT,
            username VARCHAR(255) UNIQUE NOT NULL,
            password VARCHAR(255),
            email VARCHAR(255),
            first_name VARCHAR(255),
            last_name VARCHAR(255),
            created_date TEXT,
            privacy_consent BOOLEAN,
            security_question VARCHAR(255),
            security_answer VARCHAR(255)
        )
    ''')
    conn.commit()
    conn.close()

def validate_login_sqlite(username: str, password: str) -> bool:
    """Validate user login against SQLite database"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?',
                   (username, hashed_password))
    user = cursor.fetchone()
    conn.close()
    return user is not None

def get_user_data_sqlite(username: str) -> Optional[Dict[str, Any]]:
    """Get user data from SQLite database"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return {
            'id': user[0],
            'username': user[1],
            'email': user[3],
            'first_name': user[4],
            'last_name': user[5],
            'created_date': user[6],
            'privacy_consent': user[7],
            'security_question': user[8],
            'security_answer': user[9]
        }
    return None

def save_user_sqlite(username: str, password: str, email: str, first_name: str, 
                    last_name: str, privacy_consent: bool, security_question: str, 
                    security_answer: str) -> bool:
    """Save user to SQLite database"""
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        cursor.execute('''
            INSERT INTO users (username, password, email, first_name, last_name, created_date,
                              privacy_consent, security_question, security_answer)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (username, hashed_password, email, first_name, last_name,
              str(pd.Timestamp.now().date()), privacy_consent,
              security_question, security_answer.lower()))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving user: {e}")
        return False

def check_user_exists_sqlite(username: str) -> bool:
    """Check if username already exists in SQLite"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT username FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    return user is not None

def update_password_sqlite(username: str, new_password: str) -> bool:
    """Update user password in SQLite database"""
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        hashed_password = hashlib.sha256(new_password.encode()).hexdigest()
        cursor.execute('UPDATE users SET password = ? WHERE username = ?',
                       (hashed_password, username))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error updating password: {e}")
        return False

# Data loading functions
def load_scopus_data(csv_file_path: str) -> Optional[pd.DataFrame]:
    """Load and process CSV data"""
    try:
        # Check multiple possible paths
        possible_paths = [
            csv_file_path,
            os.path.join("model_deployment", "articles_data.csv"),
            os.path.join(os.getcwd(), "model_deployment", "articles_data.csv"),
            "articles_data.csv"
        ]
        
        df = None
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Loading data from: {path}")
                df = pd.read_csv(path, encoding='utf-8')
                break
        
        if df is None:
            print(f"Could not find CSV file. Tried paths: {possible_paths}")
            return None
            
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=['title'])
        
        if 'abstract_year_latest' in df.columns:
            df['abstract_year_latest'] = pd.to_numeric(df['abstract_year_latest'], errors='coerce')
        
        print(f"Successfully loaded {len(df)} articles")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def load_recommendation_model():
    """Load recommendation model"""
    try:
        # Check multiple possible paths
        possible_model_paths = [
            "model_deployment/",
            os.path.join(os.getcwd(), "model_deployment/"),
            "/content/model_deployment/",  # Fallback for Colab compatibility
            "./"
        ]
        
        model_path = None
        for path in possible_model_paths:
            if os.path.exists(os.path.join(path, "tfidf_vectorizer.pkl")):
                model_path = path
                print(f"Loading model from: {model_path}")
                break
        
        if model_path is None:
            print(f"Could not find model files. Tried paths: {possible_model_paths}")
            return None
        
        tfidf_vectorizer = joblib.load(os.path.join(model_path, 'tfidf_vectorizer.pkl'))
        lsa_model = joblib.load(os.path.join(model_path, 'lsa_model.pkl'))
        kmeans_model = joblib.load(os.path.join(model_path, 'kmeans_model.pkl'))
        knn_models = joblib.load(os.path.join(model_path, 'knn_models.pkl'))
        articles_features = np.load(os.path.join(model_path, 'articles_features.npy'))
        
        with open(os.path.join(model_path, 'subject_index.json'), 'r') as f:
            subject_index = json.load(f)
        with open(os.path.join(model_path, 'model_metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        print("Model loaded successfully")
        return {
            'tfidf_vectorizer': tfidf_vectorizer,
            'lsa': lsa_model,
            'kmeans': kmeans_model,
            'knn_models': knn_models,
            'articles_features': articles_features,
            'subject_index': subject_index,
            'metadata': metadata
        }
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Recommendation function
def recommend_articles_api(title_input: str, top_n: int = 10, filters: Optional[Dict] = None):
    """API version of recommendation function"""
    global articles_df, model_data
    
    if articles_df is None or model_data is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    def preprocess_input(text):
        return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower().strip())
    
    def confidence_level(score):
        if score >= 0.7:
            return "High"
        elif score >= 0.4:
            return "Medium"
        else:
            return "Low"
    
    # Find the query article
    title_input_clean = preprocess_input(title_input)
    idx = articles_df[articles_df["title"].str.lower().str.contains(title_input_clean, na=False)].index
    
    if idx.empty:
        return None, f"No matching article found for title: '{title_input}'"
    
    query_idx = idx[0]
    query_row = articles_df.iloc[query_idx]
    
    # Transform query using the trained model
    query_text = articles_df.loc[query_idx, "combined_text"]
    query_tfidf = model_data['tfidf_vectorizer'].transform([query_text])
    query_lsa = model_data['lsa'].transform(query_tfidf)
    query_features = normalize(query_lsa)
    
    # Extract model components
    kmeans = model_data['kmeans']
    knn_models = model_data['knn_models']
    articles_features = model_data['articles_features']
    subject_index = model_data['subject_index']
    
    # Strategy 1: Cluster-based retrieval
    n_clusters_search = 15
    candidate_multiplier = 50
    
    cluster_similarities = cosine_similarity(query_features, kmeans.cluster_centers_)[0]
    closest_clusters = cluster_similarities.argsort()[::-1][:n_clusters_search]
    
    all_candidates = []
    all_scores = []
    
    # Get candidates from closest clusters
    for cluster_id in closest_clusters:
        if cluster_id in knn_models:
            knn, paper_indices = knn_models[cluster_id]
            n_neighbors = min(top_n * candidate_multiplier, len(paper_indices), knn.n_neighbors)
            
            if n_neighbors > 0:
                try:
                    distances, indices = knn.kneighbors(query_features, n_neighbors=n_neighbors)
                    global_indices = [paper_indices[idx] for idx in indices[0]]
                    scores = 1 - distances[0]
                    
                    all_candidates.extend(global_indices)
                    all_scores.extend(scores)
                except:
                    continue
    
    # Strategy 2: Subject-based retrieval
    query_subjects = str(query_row.get("subject_codes_str", "")).split() if pd.notna(query_row.get("subject_codes_str")) else []
    subject_candidates = set()
    
    for subj in query_subjects:
        if subj in subject_index:
            subj_indices = subject_index[subj][:top_n*2]
            subject_candidates.update(subj_indices)
    
    # Add subject candidates with moderate scores
    for idx in subject_candidates:
        if idx not in all_candidates and idx != query_idx:
            all_candidates.append(idx)
            all_scores.append(0.4)
    
    # Strategy 3: Global similarity fallback if needed
    if len(all_candidates) < top_n * 3:
        similarities = cosine_similarity(query_features, articles_features)[0]
        top_indices = similarities.argsort()[::-1][1:top_n*3+1]  # Exclude self
        
        for idx in top_indices:
            if idx not in all_candidates and idx != query_idx:
                all_candidates.append(idx)
                all_scores.append(similarities[idx] * 0.8)
    
    # Remove query article and create results
    candidate_scores = {}
    for cand, score in zip(all_candidates, all_scores):
        if cand != query_idx and cand < len(articles_df):  # Valid index check
            if cand not in candidate_scores or score > candidate_scores[cand]:
                candidate_scores[cand] = score
    
    # Sort by score
    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Create results list
    results = []
    for idx, score in sorted_candidates[:top_n + 50]:  # Get extra for filtering
        try:
            article = articles_df.iloc[idx]
            results.append({
                'index': int(idx),
                'article_id': str(article.get('article_id', '')),
                'title': str(article.get('title', '')),
                'abstract': str(article.get('abstract', '')),
                'preferredName_full': str(article.get('preferredName_full', '')),
                'abstract_year_latest': float(article.get('abstract_year_latest', 0)) if pd.notna(article.get('abstract_year_latest')) else 0,
                'aggregationType': str(article.get('aggregationType', '')),
                'publishedSubjectAreas': str(article.get('publishedSubjectAreas', '')),
                'preprocess_keywords': str(article.get('preprocess_keywords', '')),
                'cited_article_id': str(article.get('cited_article_id', '')),
                'doi': str(article.get('doi', '')),
                'is_above_mean': int(article.get('is_above_mean', 0)),
                'subject_codes_str': str(article.get('subject_codes_str', '')),
                'similarity_score': float(score),
                'confidence_score': confidence_level(score)
            })
        except Exception as e:
            print(f"Error processing article {idx}: {e}")
            continue
    
    if not results:
        return None, "No recommendations found."
    
    # Apply filters if provided
    if filters:
        # Year filter
        if filters.get('year_filter'):
            year_min, year_max = filters['year_filter']
            results = [r for r in results if year_min <= r['abstract_year_latest'] <= year_max]
        
        # Content type filter
        if filters.get('selected_types'):
            results = [r for r in results if r['aggregationType'] in filters['selected_types']]
    
    return results[:top_n], f"Found {len(results)} recommendations for '{query_row['title'][:100]}...'"

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize data on startup"""
    global articles_df, model_data
    
    print("Initializing database...")
    init_sqlite_db()
    
    print("Loading article data...")
    articles_df = load_scopus_data("model_deployment/articles_data.csv")
    
    print("Loading recommendation model...")
    model_data = load_recommendation_model()
    
    if articles_df is not None:
        print(f"✅ Loaded {len(articles_df)} articles")
    else:
        print("❌ Failed to load articles")
    
    if model_data is not None:
        print("✅ Model loaded successfully")
    else:
        print("❌ Failed to load model")

@app.get("/")
async def root():
    return {"message": "Scientific Article Recommender API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "articles_loaded": articles_df is not None,
        "model_loaded": model_data is not None,
        "article_count": len(articles_df) if articles_df is not None else 0
    }

@app.post("/auth/login", response_model=UserResponse)
async def login(request: LoginRequest):
    """User login endpoint"""
    try:
        if validate_login_sqlite(request.username, request.password):
            user_data = get_user_data_sqlite(request.username)
            return UserResponse(
                success=True,
                message="Login successful",
                user_data=user_data
            )
        else:
            return UserResponse(
                success=False,
                message="Invalid credentials"
            )
    except Exception as e:
        print(f"Login error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/signup", response_model=UserResponse)
async def signup(request: SignupRequest):
    """User signup endpoint"""
    try:
        # Validation
        if len(request.username) < 3:
            return UserResponse(success=False, message="Username must be at least 3 characters long")
        
        if len(request.password) < 6:
            return UserResponse(success=False, message="Password must be at least 6 characters long")
        
        if "@" not in request.email or "." not in request.email:
            return UserResponse(success=False, message="Please enter a valid email address")
        
        if check_user_exists_sqlite(request.username):
            return UserResponse(success=False, message="Username already exists")
        
        # Save user
        if save_user_sqlite(
            request.username, request.password, request.email,
            request.first_name, request.last_name, request.privacy_consent,
            request.security_question, request.security_answer
        ):
            return UserResponse(success=True, message="Account created successfully")
        else:
            return UserResponse(success=False, message="Failed to create account")
            
    except Exception as e:
        print(f"Signup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/reset-password", response_model=UserResponse)
async def reset_password(request: PasswordResetRequest):
    """Password reset endpoint"""
    try:
        if not check_user_exists_sqlite(request.username):
            return UserResponse(success=False, message="Username not found")
        
        user_data = get_user_data_sqlite(request.username)
        stored_answer = user_data.get('security_answer', '').lower()
        
        if request.security_answer.lower().strip() != stored_answer:
            return UserResponse(success=False, message="Security answer is incorrect")
        
        if len(request.new_password) < 6:
            return UserResponse(success=False, message="New password must be at least 6 characters long")
        
        if update_password_sqlite(request.username, request.new_password):
            return UserResponse(success=True, message="Password reset successful")
        else:
            return UserResponse(success=False, message="Failed to update password")
            
    except Exception as e:
        print(f"Password reset error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get article recommendations"""
    try:
        results, message = recommend_articles_api(
            request.title_input,
            request.top_n,
            request.filters
        )
        
        if results is not None:
            return RecommendationResponse(
                success=True,
                message=message,
                recommendations=results
            )
        else:
            return RecommendationResponse(
                success=False,
                message=message
            )
            
    except Exception as e:
        print(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/articles/stats")
async def get_article_stats():
    """Get database statistics"""
    global articles_df
    
    if articles_df is None:
        raise HTTPException(status_code=500, detail="Articles not loaded")
    
    total_articles = len(articles_df)
    
    # Count by aggregationType if column exists
    if 'aggregationType' in articles_df.columns:
        type_counts = articles_df['aggregationType'].value_counts().to_dict()
    else:
        type_counts = {}
    
    return {
        'total_articles': total_articles,
        'type_counts': type_counts,
        'year_range': {
            'min': int(articles_df['abstract_year_latest'].min()) if 'abstract_year_latest' in articles_df.columns else None,
            'max': int(articles_df['abstract_year_latest'].max()) if 'abstract_year_latest' in articles_df.columns else None
        }
    }

# Function to run the server
def run_server():
    """Run the FastAPI server"""
    print("🚀 Starting FastAPI server...")
    print("📍 Server URL: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔄 Interactive API: http://localhost:8000/redoc")
    print("💡 Press Ctrl+C to stop the server")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

def start_server_background():
    """Start server in background thread for development"""
    print("Starting FastAPI server in background...")
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    return server_thread

if __name__ == "__main__":
    # Check if we're in a development environment
    import sys
    
    if "--dev" in sys.argv:
        # Start in background for development
        thread = start_server_background()
        print("Press Enter to stop the server...")
        input()
    else:
        # Run directly
        run_server()
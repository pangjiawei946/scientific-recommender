import base64
import json
import os
import pandas as pd
import requests
import streamlit as st

from datetime import datetime, timedelta
from typing import Optional, Dict, Any

# Configuration
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')

# Page configuration
st.set_page_config(
    page_title="Scientific Digital Library",
    layout="wide"
)

# Custom CSS (same as your original)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .article-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: box-shadow 0.3s ease;
    }
    .article-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .article-title {
        color: #1E88E5;
        font-weight: bold;
        margin-bottom: 0.75rem;
        font-size: 1.1rem;
        line-height: 1.4;
    }
    .article-title a {
        color: #1E88E5;
        text-decoration: none;
    }
    .article-title a:hover {
        text-decoration: underline;
    }
    .article-meta {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 0.75rem;
        line-height: 1.5;
    }
    .article-abstract {
        color: #555;
        font-size: 0.85rem;
        margin: 1rem 0;
        line-height: 1.6;
        text-align: justify;
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 5px;
        border-left: 3px solid #1E88E5;
    }
    .filter-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .doi-link {
        color: #1E88E5;
        text-decoration: none;
        font-weight: 500;
        font-family: monospace;
    }
    .doi-link:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# API helper functions
def save_login_to_url(username: str, user_data: Dict):
    """Save login state to URL parameters"""
    login_data = {
        'username': username,
        'user_data': user_data,
        'timestamp': datetime.now().isoformat(),
        'logged_in': True
    }
    login_json = json.dumps(login_data)
    encoded_data = base64.b64encode(login_json.encode()).decode()
    st.query_params['session'] = encoded_data

def load_login_from_url():
    """Load login state from URL parameters"""
    try:
        query_params = st.query_params
        if 'session' in query_params:
            encoded_data = query_params['session']
            if not encoded_data:  # Check if data is not empty
                return None
                
            login_json = base64.b64decode(encoded_data.encode()).decode()
            login_data = json.loads(login_json)
            
            # Validate required fields
            required_fields = ['username', 'user_data', 'timestamp', 'logged_in']
            if not all(field in login_data for field in required_fields):
                return None
            
            # Check if session is still valid (24 hours)
            timestamp = datetime.fromisoformat(login_data['timestamp'])
            if datetime.now() - timestamp < timedelta(hours=24):
                return login_data
                
    except Exception as e:
        # Log the error if needed for debugging
        # st.write(f"Session load error: {e}")  # Uncomment for debugging
        pass
    return None

def clear_login_url():
    """Clear login state from URL"""
    try:
        if 'session' in st.query_params:
            del st.query_params['session']
    except Exception:
        # Silently handle any errors when clearing URL parameters
        pass

def make_api_request(endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
    """Make API request to backend"""
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=30)  # Added timeout
        elif method == "POST":
            response = requests.post(url, json=data, timeout=30)  # Added timeout
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
        return {"success": False, "message": "Request timeout"}
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to server. Please make sure the backend is running.")
        return {"success": False, "message": "Connection error"}
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        return {"success": False, "message": str(e)}

def check_server_health() -> bool:
    """Check if server is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# Authentication functions
def login_user(username: str, password: str) -> Dict:
    """Login user via API"""
    return make_api_request("/auth/login", "POST", {
        "username": username,
        "password": password
    })

def signup_user(user_data: Dict) -> Dict:
    """Signup user via API"""
    return make_api_request("/auth/signup", "POST", user_data)

def reset_password(username: str, security_answer: str, new_password: str) -> Dict:
    """Reset password via API"""
    return make_api_request("/auth/reset-password", "POST", {
        "username": username,
        "security_answer": security_answer,
        "new_password": new_password
    })

def get_recommendations(title_input: str, top_n: int = 10, filters: Dict = None) -> Dict:
    """Get recommendations via API"""
    return make_api_request("/recommendations", "POST", {
        "title_input": title_input,
        "top_n": top_n,
        "filters": filters
    })

def get_article_stats() -> Dict:
    """Get article statistics via API"""
    return make_api_request("/articles/stats", "GET")

# UI Functions
def show_login_form():
    """Display login form"""
    st.subheader("User Login")

    with st.form("login_form"):
        username = st.text_input("Username:", placeholder="Enter your username")
        password = st.text_input("Password:", type="password", placeholder="Enter your password")
        remember_me = st.checkbox("Remember me")
        login_button = st.form_submit_button("Login", use_container_width=True)

        if login_button:
            if username and password:
                with st.spinner("Logging in..."):
                    result = login_user(username, password)
                
                if result.get("success"):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.user_data = result.get("user_data", {})
                    
                    # Save to URL for persistence
                    if remember_me:
                        save_login_to_url(username, result.get("user_data", {}))

                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error(result.get("message", "Login failed"))
            else:
                st.warning("Please enter both username and password")

def show_signup_form():
    """Display signup form"""
    st.subheader("Create an Account")
    st.markdown('<p style="color: #dc3545; font-weight: bold;">* Required</p>', unsafe_allow_html=True)

    with st.form("signup_form"):
        st.markdown("### Personal Information")
        col1, col2 = st.columns([1,1])

        with col1:
            first_name = st.text_input("* Given / First Name", placeholder="Enter first name")
        with col2:
            last_name = st.text_input("* Last / Surname", placeholder="Enter last name")

        col4, col5 = st.columns(2)
        with col4:
            email = st.text_input("* Email Address", placeholder="Enter your email")
        with col5:
            username = st.text_input("* Username", placeholder="Choose a username")

        col6, col7 = st.columns(2)
        with col6:
            password = st.text_input("* Password", type="password", placeholder="Create password")
        with col7:
            confirm_password = st.text_input("* Confirm Password", type="password", placeholder="Confirm password")

        st.markdown("---")
        st.markdown("### Security Questions")
        
        security_questions = [
            "What is your mother's maiden name?",
            "What was the name of your first pet?",
            "What city were you born in?",
            "What is your favorite book?",
            "What was the make of your first car?",
            "What elementary school did you attend?",
            "What is your favorite movie?",
            "What street did you grow up on?"
        ]

        col8, col9 = st.columns(2)
        with col8:
            security_question = st.selectbox("* Create Security Question", security_questions)
        with col9:
            security_answer = st.text_input("* Security Answer", placeholder="Enter your answer")

        st.markdown("---")
        privacy_consent = st.checkbox(
            "Required: I agree to the collection and processing of my personal data as described in the Privacy Policy",
            key="privacy_consent"
        )

        signup_button = st.form_submit_button("Create Account", use_container_width=True)

        if signup_button:
            # Client-side validation
            errors = []
            if not all([first_name, last_name, email, username, password, confirm_password, security_answer]):
                errors.append("All required fields must be filled")
            if password != confirm_password:
                errors.append("Passwords do not match")
            if not privacy_consent:
                errors.append("You must agree to the Privacy Policy")

            if not errors:
                user_data = {
                    "username": username,
                    "password": password,
                    "email": email,
                    "first_name": first_name,
                    "last_name": last_name,
                    "privacy_consent": privacy_consent,
                    "security_question": security_question,
                    "security_answer": security_answer
                }

                with st.spinner("Creating account..."):
                    result = signup_user(user_data)

                if result.get("success"):
                    st.success("Account created successfully! You can now login.")
                    st.balloons()
                    st.session_state.auth_tab = "login"
                    st.rerun()
                else:
                    st.error(result.get("message", "Signup failed"))
            else:
                for error in errors:
                    st.error(error)

def show_reset_password_form():
    """Display password reset form"""
    st.subheader("Reset Password")

    with st.form("reset_form"):
        st.info("Enter your username and answer your security question to reset your password.")
        
        username = st.text_input("Username:", placeholder="Enter your username")
        security_answer = st.text_input("Security Answer:", placeholder="Enter your security answer")
        new_password = st.text_input("New Password:", type="password", placeholder="Enter new password")
        confirm_new_password = st.text_input("Confirm New Password:", type="password", placeholder="Confirm new password")

        reset_button = st.form_submit_button("Reset Password", use_container_width=True)

        if reset_button:
            if not all([username, security_answer, new_password, confirm_new_password]):
                st.error("All fields are required")
            elif new_password != confirm_new_password:
                st.error("New passwords do not match")
            elif len(new_password) < 6:
                st.error("New password must be at least 6 characters long")
            else:
                with st.spinner("Resetting password..."):
                    result = reset_password(username, security_answer, new_password)
                
                if result.get("success"):
                    st.success("Password reset successful! You can now login with your new password.")
                    st.session_state.auth_tab = "login"
                    st.rerun()
                else:
                    st.error(result.get("message", "Password reset failed"))

def format_article_data(article: Dict) -> Dict[str, str]:
    """Format article data for display"""
    title = article.get('title', 'No Title Available')
    doi = article.get('doi', '')
    
    if doi and doi.startswith('10.'):
        doi_url = f"https://doi.org/{doi}"
        title_with_link = f'<a href="{doi_url}" target="_blank" style="color: #1E88E5; text-decoration: none; font-weight: bold;">{title}</a>'
    else:
        title_with_link = title
        doi_url = "#"

    # Format authors
    authors = article.get('preferredName_full', 'Unknown Authors')
    if isinstance(authors, str) and '|' in authors:
        author_list = [author.strip() for author in authors.split('|')]
        if len(author_list) > 3:
            authors_formatted = ', '.join(author_list[:3]) + ' et al.'
        else:
            authors_formatted = ', '.join(author_list)
    else:
        authors_formatted = str(authors) if authors else 'Unknown Authors'

    # Format abstract
    abstract = article.get('abstract', '')
    if abstract and len(abstract) > 400:
        abstract_formatted = abstract[:400] + '...'
    else:
        abstract_formatted = abstract if abstract else 'No abstract available'

    # Get other fields
    year = str(int(article.get('abstract_year_latest', 0))) if article.get('abstract_year_latest') else 'Unknown'
    agg_type = article.get('aggregationType', 'Unknown')
    keywords = article.get('preprocess_keywords', 'No keywords available')

    return {
        'title': title,
        'title_with_link': title_with_link,
        'authors': authors_formatted,
        'abstract': abstract_formatted,
        'year': year,
        'type': agg_type,
        'keywords': keywords,
        'doi': doi,
        'doi_url': doi_url
    }

def display_recommendations(recommendations: list, message: str):
    """Display recommendations from API response"""
    if not recommendations:
        st.warning("No recommendations found.")
        return

    st.success(message)
    st.markdown(f"### Top {len(recommendations)} Recommended Articles")

    for idx, article in enumerate(recommendations, 1):
        article_data = format_article_data(article)
        
        confidence = article.get('confidence_score', 'Medium')
        confidence_color = {
            'High': '#28a745',
            'Medium': '#ffc107',
            'Low': '#dc3545'
        }.get(confidence, '#6c757d')

        performance_indicator = ""
        if article.get('is_above_mean') == 1:
            performance_indicator = ' <span style="color: #28a745; font-weight: bold;">⭐ High Performance</span>'

        st.markdown(f"""
        <div class="article-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
                <span style="background-color: {confidence_color}; color: white; padding: 0.2rem 0.5rem; border-radius: 3px; font-size: 0.8rem;">
                    Confidence: {confidence}
                </span>
                <span style="color: #666; font-size: 0.9rem;">
                    Similarity: {article.get('similarity_score', 0):.3f}
                </span>
            </div>
            <div class="article-title">#{idx} {article_data['title_with_link']}{performance_indicator}</div>
            <div class="article-meta">
                <strong>Authors:</strong> {article_data['authors']}<br>
                <strong>Year:</strong> {article_data['year']} |
                <strong>Type:</strong> {article_data['type']}<br>
                <strong>DOI:</strong> <a href="{article_data['doi_url']}" target="_blank" class="doi-link">{article_data['doi']}</a><br>
                <strong>Keywords:</strong> {article_data['keywords']}
            </div>
            <div class="article-abstract">
                <strong>Abstract:</strong><br>
                {article_data['abstract']}
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_statistics():
    """Display database statistics from API"""
    with st.spinner("Loading statistics..."):
        stats = get_article_stats()
    
    if stats.get("success", True):  # API doesn't return success field for stats
        total_articles = stats.get('total_articles', 0)
        type_counts = stats.get('type_counts', {})
        
        st.markdown(f"""
        <style>
            .simple-stats-container {{
                border: 2px solid #e1e5e9;
                border-radius: 10px;
                padding: 2rem;
                margin: 1rem 0;
                background-color: #fdfdfd;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            }}
            .center-title {{
                text-align: center;
                color: #495057;
                margin: 0 0 2rem 0;
                font-size: 1.8rem;
                font-weight: 600;
            }}
            .metrics-row {{
                display: flex;
                justify-content: space-around;
                gap: 1rem;
                flex-wrap: wrap;
            }}
            .metric-card {{
                background-color: #ffffff;
                border: 1px solid #e9ecef;
                padding: 1.5rem;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.06);
                text-align: center;
                transition: all 0.3s ease;
                flex: 1;
                min-width: 150px;
            }}
            .metric-card:hover {{
                border-color: #80bdff;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                transform: translateY(-2px);
            }}
            .metric-value {{
                font-size: 2.2rem;
                font-weight: 700;
                color: #2c3e50;
                margin-bottom: 0.5rem;
            }}
            .metric-label {{
                color: #6c757d;
                font-weight: 500;
                font-size: 0.9rem;
            }}
        </style>

        <div class="simple-stats-container">
            <h2 class="center-title">Database Statistics</h2>
            <div class="metrics-row">
                <div class="metric-card">
                    <div class="metric-value">{total_articles:,}</div>
                    <div class="metric-label">Total Articles</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{type_counts.get('Journal', 0):,}</div>
                    <div class="metric-label">Journal</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{type_counts.get('Conference Proceeding', 0):,}</div>
                    <div class="metric-label">Conference Proceeding</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{type_counts.get('Book', 0):,}</div>
                    <div class="metric-label">Book</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{type_counts.get('Book Series', 0):,}</div>
                    <div class="metric-label">Book Series</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{type_counts.get('Trade Journal', 0):,}</div>
                    <div class="metric-label">Trade Journal</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Failed to load statistics")

def login_page():
    """User login/signup interface"""
    st.markdown('<h1 class="main-header">Scholarly Article Recommender</h1>', unsafe_allow_html=True)

    # Check server connection
    if not check_server_health():
        st.error("⚠️ Cannot connect to the backend server. Please make sure it's running on http://localhost:8000")
        st.info("To start the server, run the backend_server.py file first.")
        return

    with st.container():
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            tab_col1, tab_col2, tab_col3 = st.columns([1, 1, 1])

            with tab_col1:
                if st.button("Login", key="login_tab", use_container_width=True):
                    st.session_state.auth_tab = "login"

            with tab_col2:
                if st.button("Sign Up", key="signup_tab", use_container_width=True):
                    st.session_state.auth_tab = "signup"

            with tab_col3:
                if st.button("Reset Password", key="reset_tab", use_container_width=True):
                    st.session_state.auth_tab = "reset"

            # Initialize tab if not set
            if 'auth_tab' not in st.session_state:
                st.session_state.auth_tab = "login"

            # Show appropriate form based on selected tab
            if st.session_state.auth_tab == "login":
                show_login_form()
            elif st.session_state.auth_tab == "signup":
                show_signup_form()
            elif st.session_state.auth_tab == "reset":
                show_reset_password_form()

def main_app():
    """Main application after login"""
    
    # Check server connection
    if not check_server_health():
        st.error("⚠️ Lost connection to the backend server.")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()
        return

    # Header with user info and logout
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown('<h1 style="color: #00445e; margin-bottom: 0;">Scholarly Article Recommender</h1>', unsafe_allow_html=True)
        
        user_data = st.session_state.get('user_data', {})
        first_name = user_data.get('first_name', '')
        last_name = user_data.get('last_name', '')
        full_name = f"{first_name} {last_name}".strip() or st.session_state.get('username', 'User')
        st.markdown(f"Welcome back, **{full_name}**!")

    with col2:
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            if 'username' in st.session_state:
                del st.session_state.username
            if 'user_data' in st.session_state:
                del st.session_state.user_data
            if 'auth_tab' in st.session_state:
                del st.session_state.auth_tab
            
            # Clear URL parameters
            clear_login_url()
            st.rerun()

    # Search Section
    with st.container():
        st.markdown(f"""
        <div style="background-color: #00445e; padding: 2rem 1rem; border-radius: 10px; margin-bottom: 2rem;">
            <h2 style="color: white; font-size: 1.8rem; text-align: center; margin-bottom: 1.5rem;">
                Hybrid Recommendations Scholarly Articles
            </h2>
            <p style="color: #e0e0e0; text-align: center; margin-bottom: 1rem;">
                Enter any article title to get intelligent recommendations based on content similarity, citations, and subject areas
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Search input
        col1, col2 = st.columns([5, 1])
        with col1:
            recommend_query = st.text_input(
                "",
                placeholder="Enter article title for hybrid recommendations (e.g., 'machine learning', 'flood detection', 'neural networks')...",
                label_visibility="collapsed",
                key="recommend_input"
            )
        with col2:
            recommend_button = st.button("Get Recommendations", use_container_width=True, key="recommend_btn")

    # Display statistics
    display_statistics()

    # Main content area
    col1, col2 = st.columns([2, 6])

    # Left sidebar - Filters (simplified for now)
    with col1:
        st.markdown("""
        <div class="filter-section">
            <h4>Filter Results</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("**Content Type:**")
        selected_types = st.multiselect(
            "Select types:",
            ["Journal", "Conference Proceeding", "Book", "Book Series", "Trade Journal"],
            default=[]
        )
        
        st.write("**Publication Year:**")
        year_min = st.number_input("From year:", min_value=1900, max_value=2022, value=2020)
        year_max = st.number_input("To year:", min_value=1900, max_value=2022, value=2022)

    # Right side - Article results
    with col2:
        # Handle recommendation results
        if recommend_query and (recommend_button or st.session_state.get('last_query') == recommend_query):
            st.session_state.last_query = recommend_query
            
            filters = {}
            if selected_types:
                filters['selected_types'] = selected_types
            if year_min <= year_max:
                filters['year_filter'] = [year_min, year_max]

            with st.spinner("Generating hybrid recommendations..."):
                result = get_recommendations(recommend_query, top_n=10, filters=filters)

            if result.get("success"):
                recommendations = result.get("recommendations", [])
                message = result.get("message", "")
                display_recommendations(recommendations, message)
            else:
                st.error(result.get("message", "Failed to get recommendations"))
        else:
            st.markdown("### Featured Articles")
            st.info("Use the search above to get hybrid recommendations!")
            st.markdown("Enter an article title in the search box to get intelligent recommendations based on content similarity and subject areas.")

def main():
    """Main application entry point"""
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

        # Try to load saved login state from URL
        saved_login = load_login_from_url()
        if saved_login:
            st.session_state.logged_in = True
            st.session_state.username = saved_login['username']
            st.session_state.user_data = saved_login['user_data']
    
    # Show appropriate page
    if st.session_state.logged_in:
        main_app()
    else:
        login_page()

if __name__ == '__main__':
    main()
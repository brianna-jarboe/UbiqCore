import streamlit as st

@st.cache_data
def get_app_styles():
    """
    Centralized CSS styles for the application.
    Cached to avoid recomputation on every page load.
    """
    return """
    <style>
        /* Hide Streamlit header */
        header {
            visibility: hidden;
            display: none !important;
        }

        /* Hide sidebar */
        [data-testid="stSidebar"] {
            display: none !important;
        }
        
        /* Banner container styling */
        .st-emotion-cache-16txtl3.eczjsme4, .st-key-banner_container {
            background-color: #FDD1D7;
            padding: 10px;
            border-radius: 5px;
            width: 100%;
        }
        
        /* Reduce top padding */
        div.block-container {
            padding-top: 0.4rem;
        }
        
        /* Center 3D structure container */
        .centered-container {
            display: flex;
            justify-content: center;
        }
        
        /* Optimize data editor styling */
        .stDataFrame {
            width: 100%;
        }
        
        /* Improve button styling and reduce flicker */
        .stButton > button {
            transition: all 0.2s ease;
        }
        
        /* Loading spinner optimization */
        .stSpinner {
            text-align: center;
        }
        
        /* Optimize menu transitions */
        .nav-link {
            transition: all 0.1s ease-in-out;
        }
        
        /* Reduce layout shift */
        .main .block-container {
            max-width: 100%;
        }
        
        /* Header image sizing */
        .st-emotion-cache-1v0mbdj > img {
            max-width: 150px;
            height: auto;
        }
    </style>
    """

def apply_global_styles():
    """Apply all global styles to the current page"""
    st.markdown(get_app_styles(), unsafe_allow_html=True)
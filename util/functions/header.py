import streamlit as st
from streamlit_option_menu import option_menu
import util.constants.pages
from streamlit_js_eval import streamlit_js_eval
from util.functions.styling import apply_global_styles

def header(calling_page):
    # Wide mode
    st.set_page_config(layout='wide')      

    # Apply global styles
    apply_global_styles()
    
    # Optimize screen width detection with better caching and fallback
    # We perform the check in the sidebar to prevent visual artifacts (white line) in the main view
    if 'screen_width' not in st.session_state or st.session_state.screen_width is None:
        with st.sidebar:
            try:
                st.session_state.screen_width = streamlit_js_eval(js_expressions='screen.width', key='SCR')
            except:
                st.session_state.screen_width = 1600  # Default fallback
    
    screen_width = st.session_state.screen_width or 1600  # Fallback if None
    
    # Default column values based on screen width
    columns = [1, 15, 1]  # Default to desktop layout
    
    if screen_width < 1600:
        columns = [0.2, 1.82, 20]
    else:
        columns = [0.4, 1, 15]
    # menu container
    banner_container = st.container(key="banner_container")

    with banner_container:
        col1, col2, col3 = st.columns(columns)
        with col2:
            st.image('util/resources/UBlogo.png', width=150)
        with col3:
            st.title('UbiqCore')

    # Special case of main script call
    if calling_page == "streamlit_app":
        calling_page = "Home"

    
    # Page Menu
    selected = option_menu (
        menu_title=None, 
        options=util.constants.pages.PAGES,
        icons=util.constants.pages.PAGE_ICONS,
        default_index=util.constants.pages.PAGES.index(calling_page.replace("__", "/").replace("_", " ")),
        menu_icon="cast",
        orientation='horizontal',
        key="main_menu"
        )
    
    # Use switch_page for navigation (keeps the working functionality)
    if selected != calling_page.replace("__", "/").replace("_", " "):
        # Convert back to filename format
        page_file = selected.replace("/", "__").replace(" ", "_") + ".py"
        st.switch_page(f"pages/{page_file}")
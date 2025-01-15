import streamlit as st
import torch
import os
import gc
import time
from contextlib import contextmanager

SESSION_TIMEOUT = 3600  # 1 hour in seconds
LAST_ACTIVITY_KEY = "last_activity"
SESSION_DATA_KEY = "session_data"

@contextmanager
def cleanup_on_exit():
    """Context manager to ensure cleanup when processing is done"""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def init_session_state():
    """Initialize or update session state"""
    if LAST_ACTIVITY_KEY not in st.session_state:
        st.session_state[LAST_ACTIVITY_KEY] = time.time()
    if SESSION_DATA_KEY not in st.session_state:
        st.session_state[SESSION_DATA_KEY] = {}
    
    st.session_state[LAST_ACTIVITY_KEY] = time.time()

def check_session_timeout():
    """Check if session has timed out"""
    if LAST_ACTIVITY_KEY in st.session_state:
        last_activity = st.session_state[LAST_ACTIVITY_KEY]
        if time.time() - last_activity > SESSION_TIMEOUT:
            st.session_state[SESSION_DATA_KEY] = {}
            st.session_state[LAST_ACTIVITY_KEY] = time.time()
            return True
    return False

def clear_session_data():
    """Clear session-specific data"""
    if SESSION_DATA_KEY in st.session_state:
        st.session_state[SESSION_DATA_KEY] = {}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def is_running_locally():
    """
    Checks if the Streamlit app is running locally based on the HOSTNAME environment variable.
    """
    hostname = os.getenv("HOSTNAME")
    
    # If HOSTNAME is not set, assume it's running locally
    if hostname is None:
        return True
    
    # If HOSTNAME is set to 'streamlit', it's running in the deployed environment
    if hostname == "streamlit":
        return False
    
    # Otherwise, assume it's running locally
    return True

def clear_session_state():
    keys_to_clear = ['processing_started', 'processing_complete', 'results', 'faces', 'num_frames']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
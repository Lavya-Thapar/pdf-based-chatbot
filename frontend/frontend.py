import streamlit as st
import requests
import uuid

# --- Page Configuration ---
st.set_page_config(page_title="Stateful PDF Chat", layout="wide")

# --- Backend URL ---
BACKEND_URL = "http://127.0.0.1:8000"

# --- Session State Initialization ---
def init_session_state():
    """Initializes session state on the first run."""
    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = None
    if "chat_sessions" not in st.session_state:
        try:
            response = requests.get(f"{BACKEND_URL}/chats")
            st.session_state.chat_sessions = response.json() if response.status_code == 200 else {}
        except requests.exceptions.RequestException:
            st.session_state.chat_sessions = {}
    
    if "history" not in st.session_state:
        st.session_state.history = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []

# --- Helper Functions ---
def switch_chat(chat_id):
    """Switches the active chat and loads its state from the backend."""
    st.session_state.active_chat_id = chat_id
    try:
        response = requests.get(f"{BACKEND_URL}/chat/{chat_id}")
        if response.status_code == 200:
            chat_data = response.json()
            st.session_state.history = chat_data.get("history", [])
            st.session_state.processed_files = chat_data.get("processed_files", [])
        else:
            st.error("Failed to load chat details.")
            st.session_state.history = []
            st.session_state.processed_files = []
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
    st.rerun()

def handle_file_upload():
    """Manages the file uploader widget and sends files to the backend."""
    if st.session_state.processed_files:
        st.caption("Associated files for this chat:")
        for filename in st.session_state.processed_files:
            st.markdown(f"- `{filename}`")
    
    uploaded_files = st.file_uploader(
        "Upload new PDFs to this chat",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.active_chat_id}"
    )
    
    if uploaded_files:
        current_filenames = sorted([file.name for file in uploaded_files])
        if current_filenames != st.session_state.processed_files:
            with st.spinner('Processing documents...'):
                files_to_upload = [('files', (file.name, file.getvalue(), file.type)) for file in uploaded_files]
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/upload/{st.session_state.active_chat_id}",
                        files=files_to_upload
                    )
                    if response.status_code == 200:
                        switch_chat(st.session_state.active_chat_id)
                    else:
                        st.error(f"Error: {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {e}")

# --- UI Rendering ---
def display_sidebar():
    """Renders the sidebar for chat management and file uploads."""
    with st.sidebar:
        st.title("📄 Stateful PDF Q&A")

        if st.button("➕ New Chat", use_container_width=True, key="new_chat"):
            new_chat_id = str(uuid.uuid4())
            st.session_state.chat_sessions[new_chat_id] = f"Chat #{len(st.session_state.chat_sessions) + 1}"
            switch_chat(new_chat_id)

        st.header("Chat History")
        sorted_chats = sorted(st.session_state.chat_sessions.items(), key=lambda item: item[1])

        for chat_id, title in sorted_chats:
            # Use columns to place chat title and delete button on the same row
            col1, col2 = st.columns([0.8, 0.2])
            
            with col1:
                if st.button(title, key=f"button_{chat_id}", use_container_width=True):
                    if st.session_state.active_chat_id != chat_id:
                        switch_chat(chat_id)
            
            with col2:
                if st.button("🗑️", key=f"delete_{chat_id}", use_container_width=True):
                    try:
                        response = requests.delete(f"{BACKEND_URL}/chat/{chat_id}")
                        if response.status_code == 200:
                            # Remove from local state and refresh
                            del st.session_state.chat_sessions[chat_id]
                            if st.session_state.active_chat_id == chat_id:
                                st.session_state.active_chat_id = None
                                st.session_state.history = []
                                st.session_state.processed_files = []
                            st.rerun()
                        else:
                            st.error("Failed to delete chat.")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Connection error: {e}")
        
        st.header("Upload Documents")
        if st.session_state.active_chat_id:
            handle_file_upload()
        else:
            st.info("Start a new chat to upload documents.")

def display_chat_interface():
    """Renders the main chat window and handles user interaction."""
    if not st.session_state.active_chat_id:
        st.info("Welcome! Start a new chat or select one from the sidebar.")
        return

    for question, answer in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            st.markdown(answer)

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.history.append((prompt, ""))
        try:
            response = requests.post(
                f"{BACKEND_URL}/ask/{st.session_state.active_chat_id}",
                json={"question": prompt}
            )
            response.raise_for_status()
            answer = response.json().get("answer", "No answer found.")
            st.session_state.history[-1] = (prompt, answer)
        except requests.exceptions.RequestException as e:
            st.session_state.history[-1] = (prompt, f"**Error:** {e}")
        st.rerun()

# --- Main App Execution ---
if __name__ == "__main__":
    init_session_state()
    display_sidebar()
    display_chat_interface()

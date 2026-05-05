import sys
import os

sys.path.append(os.path.abspath("."))

import streamlit as st
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.main import invoke as agent_invoke

# ---- Async runner ----
_executor = ThreadPoolExecutor(max_workers=4)

def run_async(coro):
    future = _executor.submit(lambda: asyncio.run(coro))
    return future.result()

# ---- Session state ----
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat" not in st.session_state:
    st.session_state.chat = []

if "processing" not in st.session_state:
    st.session_state.processing = False

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

# ---- Page config ----
st.set_page_config(page_title="NovAtel AI", layout="wide")
st.title("NovAtel AI Assistant")

# ---- File Upload ----
uploaded_file = st.file_uploader(
    "Upload log file",
    type=["log", "txt", "asc", "csv", "json", "gps", "gpf", "ASCII", "ABBREV_ASCII"],
    help="Maximum file size: 350MB"
)

# Only process if a new file is uploaded (different from the last one)
if uploaded_file and not st.session_state.processing:
    # Check if this is a new file (different name or first upload)
    if not st.session_state.file_uploaded or st.session_state.uploaded_filename != uploaded_file.name:
        st.session_state.processing = True

        # Read file efficiently
        file_bytes = uploaded_file.getvalue()
        import base64

        with st.spinner("Uploading and processing file..."):
            response = run_async(agent_invoke({
                "file": base64.b64encode(file_bytes).decode('utf-8'),
                "filename": uploaded_file.name,
                "session_id": st.session_state.session_id
            }))

        st.success("File processed successfully!")
        if response and "result" in response:
            st.write(response.get("result", ""))
        
        # Mark file as uploaded
        st.session_state.file_uploaded = True
        st.session_state.uploaded_filename = uploaded_file.name
        st.session_state.processing = False

# Show current file status
if st.session_state.file_uploaded and st.session_state.uploaded_filename:
    st.info(f"📁 Current file: **{st.session_state.uploaded_filename}** (ready for queries)")


# ---- Chat Input ----
user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.chat.append(("user", user_input))

    with st.spinner("Thinking..."):
        response = run_async(agent_invoke({
            "prompt": user_input,
            "session_id": st.session_state.session_id
        }))

    reply = response.get("result", "") if response else ""
    st.session_state.chat.append(("agent", reply))

# ---- Display Chat ----
for role, msg in st.session_state.chat:
    # Use "assistant" instead of "agent" for better default icon
    display_role = "assistant" if role == "agent" else role
    with st.chat_message(display_role):
        st.write(msg)


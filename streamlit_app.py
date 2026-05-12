import sys
import os
import base64
from pathlib import Path


sys.path.append(os.path.abspath("."))

os.environ.setdefault("S3_BUCKET", "naspocuser-s3")
os.environ.setdefault("AWS_REGION", "ap-south-1")
os.environ.setdefault("KB_ID", "FH00WKSBPL")

import streamlit as st
import uuid
import asyncio
import base64
import io
from concurrent.futures import ThreadPoolExecutor

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config as BotocoreConfig

from src.main import invoke as agent_invoke, get_status as agent_get_status

S3_BUCKET      = os.environ.get("S3_BUCKET", "naspocuser-s3")
S3_REGION      = os.environ.get("AWS_REGION", "ap-south-1")
SIZE_THRESHOLD = 5 * 1024 * 1024
MAX_UPLOAD     = 350 * 1024 * 1024

_s3_client = None
def get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name=S3_REGION, config=BotocoreConfig(read_timeout=300))
    return _s3_client

st.set_page_config(page_title="NovAtel AI", layout="wide", initial_sidebar_state="collapsed")

_executor = ThreadPoolExecutor(max_workers=4)
def run_async(coro):
    return _executor.submit(lambda: asyncio.run(coro)).result()

@st.cache_data
def get_logo_data_url():
    logo_path = Path(__file__).parent / "src" / "novatel_logo.png"
    if not logo_path.exists():
        st.error(f"Logo not found at: {logo_path}")
        return ""
    b64 = base64.b64encode(logo_path.read_bytes()).decode()
    # change image/png to image/jpeg or image/svg+xml if your file is different
    return f"data:image/png;base64,{b64}"
 
LOGO_URL = get_logo_data_url()

def upload_to_s3_with_progress(file_bytes: bytes, filename: str) -> str:
    key = f"logs/{filename}"
    cfg = TransferConfig(multipart_threshold=8*1024*1024, multipart_chunksize=8*1024*1024, max_concurrency=4, use_threads=True)
    try:
        get_s3_client().upload_fileobj(io.BytesIO(file_bytes), S3_BUCKET, key, Config=cfg)
        return key
    except Exception as e:
        raise Exception(f"S3 upload failed: {str(e)}")

if "session_id"      not in st.session_state: st.session_state.session_id      = str(uuid.uuid4())
if "chat"            not in st.session_state: st.session_state.chat            = []
if "pending_chip"    not in st.session_state: st.session_state.pending_chip    = None
if "client_id"       not in st.session_state: st.session_state.client_id       = str(uuid.uuid4())
if "pending_upload"  not in st.session_state: st.session_state.pending_upload  = None  # file waiting to be processed

st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet"/>',
    unsafe_allow_html=True,
)

st.markdown("""
<style>
:root {
  --navy:       #00284c;
  --blue:       #005198;
  --blue-light: #1a6bb5;
  --blue-frost: #e8f2fa;
  --sky:        #4a9fd4;
  --off-white:  #f4f8fc;
  --border:     #b8d0e8;
  --border-dim: #d6e6f2;
  --text:       #00284c;
  --text-mid:   #2a5070;
  --text-dim:   #6080a0;
  --surface:    #ffffff;
  --r:          5px;
  --rlg:        10px;
  --mono:       'OpenSans-Regular';
  --ui:         'OpenSans-Regular';
}

/* ── Kill Streamlit chrome ── */
#MainMenu, header[data-testid="stHeader"], footer,
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="collapsedControl"],
section[data-testid="stSidebar"] { display: none !important; }
div[style*="rgba(38, 39, 48"],
div[style*="rgba(14, 17, 23"]   { display: none !important; }
[data-testid="stBottom"]::before,
.stChatFloatingInputContainer::before { display: none !important; }

html, body, [class*="css"] { font-family: var(--ui) !important; background: var(--off-white) !important; }
.stApp { background: var(--off-white) !important; }

.block-container,
[data-testid="stAppViewBlockContainer"] {
  max-width: 880px !important;
  padding: 0 !important;
  margin: 0 auto !important;
}
[data-baseweb="textarea"]{
background-color:none
}
/* ── Bottom bar ── */
[data-testid="stBottom"] {
  max-width: 880px !important;
  margin: 0 auto !important;
  background: var(--off-white) !important;
  box-shadow: none !important;
  border-top: none !important;
  padding: 0 !important;
}
[data-testid="stBottom"] > div {
  display: flex !important;
  align-items: center !important;
  gap: 8px !important;
  background: var(--off-white) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
  position: fixed !important;
  bottom: 0px !important;
  left: max(26px, calc(50% - 440px + 26px)) !important;
  width: calc(100% - 88px) !important;
  max-width: 827px !important;
  z-index: 999 !important;
  height: 60px;
  background: var(--off-white) !important;
}
[data-testid="stChatInput"] > :first-child { 
  border: none !important; 
  background: var(--off-white) !important;
}
[data-testid="stChatInput"] > div {
  background: var(--off-white) !important;
}
[data-testid="stChatInput"] textarea {
  font-family: var(--mono) !important;
  font-size: 14px !important;
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
  color: var(--text) !important;
  line-height: 1.5 !important;
  padding: 10px 14px 20px !important;
  resize: none !important;
  width: 95% !important;
  min-height: 40px !important;
  max-height: 40px !important;
  min-width: 100px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0,40,76,.08) !important;
}
[data-testid="stChatInput"] textarea:focus {
  border-color: var(--blue) !important;
  box-shadow: 0 0 0 3px rgba(0,81,152,.1), 0 2px 8px rgba(0,40,76,.12) !important;
  outline: none !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--text-dim) !important; }

/* ── Hide submit button ── */
[data-testid="stChatInputSubmitButton"] {
  display: none !important;
}

/* ── Paperclip uploader — fixed size, never grows ── */
[data-testid="stFileUploader"] {
  position: fixed !important;
    bottom: 10px !important;
    right: max(20px, calc(50% - 410px)) !important;
    z-index: 1000 !important;
    width: 36px !important;
    height: 36px !important;
    overflow: hidden !important;
}

/* ── Hide post-upload chips/file info without touching the dropzone ── */
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploader"] [data-testid="stFileChips"],
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
[data-testid="stFileUploader"] [class*="uploadedFile"],
[data-testid="stFileUploader"] [class*="FileChip"],
[data-testid="stFileUploader"] [class*="fileChip"],
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] button[kind="icon"],
[data-testid="stFileUploader"] button[kind="secondary"] {
  display: none !important;
  height: 0 !important;
  width: 0 !important;
  overflow: hidden !important;
  position: absolute !important;
  visibility: hidden !important;
  pointer-events: none !important;
}
/* Scope span hiding to inside the dropzone instructions only */
[data-testid="stFileUploaderDropzoneInstructions"] span:not(:empty) {
  display: none !important;
}

/* ── Dropzone locked to exact icon size ── */
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
  width: 36px !important;
  height: 36px !important;
  min-width: unset !important;
  min-height: unset !important;
  max-height: 36px !important;
  padding: 0 !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
  background: var(--surface) !important;
  display: grid !important;
  place-items: center !important;
  cursor: pointer !important;
  transition: border-color .15s, background .15s !important;
  overflow: hidden !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--blue) !important;
  background: var(--blue-frost) !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]::before {
  content: "" !important;
  display: block !important;
  width: 18px !important;
  height: 18px !important;
  background-color: var(--text-dim) !important;
  -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48'/%3E%3C/svg%3E") !important;
  mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48'/%3E%3C/svg%3E") !important;
  -webkit-mask-repeat: no-repeat !important;
  mask-repeat: no-repeat !important;
  -webkit-mask-position: center !important;
  mask-position: center !important;
  -webkit-mask-size: contain !important;
  mask-size: contain !important;
  pointer-events: none !important;
  position:relative;
  top:6px;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]:hover::before {
  background-color: var(--blue) !important;
}

/* ── Avatar labels ── */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
  font-size: 0 !important;
  color: transparent !important;
  display: grid !important;
  place-items: center !important;
  flex-shrink: 0 !important;
  align-self: flex-start !important;
}
[data-testid="stChatMessageAvatarUser"] *,
[data-testid="stChatMessageAvatarAssistant"] * { display: none !important; }
[data-testid="stChatMessageAvatarUser"] {
  background: linear-gradient(135deg, var(--blue-light) 0%, var(--sky) 100%) !important;
  border: 1px solid rgba(74,159,212,.35) !important;
  border-radius: 6px !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
  background: linear-gradient(135deg, var(--navy) 0%, var(--blue) 100%) !important;
  border: 1px solid rgba(0,81,152,.4) !important;
  border-radius: 6px !important;
}
[data-testid="stChatMessageAvatarUser"]::after {
  content: "YOU";
  font-family: 'OpenSans-Regular' !important;
  font-size: 8px !important; font-weight: 600 !important;
  color: #fff !important; letter-spacing: .04em !important;
}
[data-testid="stChatMessageAvatarAssistant"]::after {
  content: "AI";
  font-family: 'OpenSans-Regular' !important;
  font-size: 10px !important; font-weight: 600 !important;
  color: #fff !important; letter-spacing: .04em !important;
}

/* ══════════════════════════════════════════════
   CHAT BUBBLES
   ══════════════════════════════════════════════ */

[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
  margin: 8px 0 !important;
  gap: 10px !important;
  align-items: flex-start !important;
  justify-content: flex-start !important;
  animation: fadeUp .2s ease both;
}
@keyframes fadeUp {
  from { opacity:0; transform:translateY(6px); }
  to   { opacity:1; transform:translateY(0); }
}

/* ── USER: push to right, avatar after bubble via order ── */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  justify-content: flex-end !important;
  flex-direction: row !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageAvatarUser"] {
  order: 2 !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
  order: 1 !important;
  background: linear-gradient(135deg, var(--blue) 0%, var(--blue-light) 100%) !important;
  border-radius: 12px !important;
  padding: 10px 15px 20px 15px !important;
  flex: 0 1 auto !important;
  width: fit-content !important;
  max-width: 68% !important;
  min-width: 0 !important;
  border: none !important;
  box-shadow: 0 2px 8px rgba(0,81,152,.18) !important;
  margin: 0 !important;
  min-height: 40px;
}
[data-testid="stMarkdownContainer"] .filesize {
  color: rgb(96, 128, 160) !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) span,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) li {
  color: #fff !important;
  font-family: var(--mono) !important;
  font-size: 13.5px !important;
  line-height: 1.65 !important;
  margin: 0 !important;
  letter-spacing: 0.6px;
}

/* ── AI: left-aligned ── */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
  background: var(--surface) !important;
  border-radius: 6px !important;
  padding: 10px 15px 20px 15px !important;
  flex: 0 1 auto !important;
  width: fit-content !important;
  max-width: 72% !important;
  min-width: 0 !important;
  border: 1px solid var(--border-dim) !important;
  box-shadow: 0 1px 4px rgba(0,40,76,.06) !important;
  margin: 0 !important;
  min-height: 40px;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) p,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) span {
  color: var(--text) !important;
  font-family: var(--mono) !important;
  font-size: 13.5px !important;
  line-height: 1.65 !important;
  letter-spacing: 0.6px;
  margin: 0 !important;
}
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] * {
  color: var(--text);
            background:transparent;
  font-family: var(--mono);
  font-size: 14px;
  text-decoration:none;
}
.st-b1{
            background-color:transparent !important
            }
/* ── Chip buttons ── */
[data-testid="stHorizontalBlock"] [data-testid="stButton"] button {
  padding: 7px 15px !important;
  border: 1px solid var(--border) !important;
  border-radius: 20px !important;
  font-size: 11.5px !important;
  font-family: var(--mono) !important;
  color: var(--text-mid) !important;
  background: var(--surface) !important;
  box-shadow: none !important;
  height: auto !important; min-height: unset !important;
  line-height: 1.4 !important;
  width: 100% !important;
  transition: border-color .15s, color .15s, background .15s !important;
}
[data-testid="stHorizontalBlock"] [data-testid="stButton"] button:hover {
  border-color: var(--blue) !important; color: var(--blue) !important;
  background: var(--blue-frost) !important;
  box-shadow: 0 2px 8px rgba(0,81,152,.12) !important;
}

/* ── Misc ── */
[data-testid="stAlert"] { font-family: var(--mono) !important; font-size: 12px !important; border-radius: 6px !important; margin: 4px 24px !important; }
[data-testid="stSpinner"] p { font-family: var(--mono) !important; font-size: 12px !important; color: var(--text-dim) !important; }

/* ── Status messages (italic text in assistant bubbles) ── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) em,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) i {
  color: var(--text-dim) !important;
  font-style: italic !important;
  font-size: 12px !important;
  opacity: 0.85 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────
st.markdown(f"""
<header style="display:flex;align-items:center;gap:14px;padding:0 24px;
  background:#00284c;border-bottom:2px solid #005198;
  position:relative;overflow:hidden;height:60px;
  max-width:880px;margin:0 auto;
  box-shadow:0 0 0 1px rgba(0,40,76,.08),0 4px 32px rgba(0,40,76,.06);">
  <div style="position:absolute;right:-60px;top:-40px;width:220px;height:140px;
    background:radial-gradient(ellipse at center,rgba(0,81,152,.45) 0%,transparent 70%);pointer-events:none;"></div>
  <div style="width:36px;height:36px;border-radius:5px;flex-shrink:0;
    background-image:url('{LOGO_URL}');
    background-size:34px;
    background-repeat:no-repeat;
    background-position:center;
    background-color:white;
    box-shadow:0 0 0 1px rgba(255,255,255,.15),0 2px 8px rgba(0,0,0,.3);">
  </div>
  <div style="flex:1;">
    <div style="font-family:'OpenSans-Regular';font-size:16px;font-weight:600;letter-spacing:.07em;color:#fff;line-height:1;padding-top:1px;">NovAtel AI Assistant</div>
    <div style="font-size:12px;color:rgba(255,255,255,.52);margin-top:3px;letter-spacing:.07em;font-family:'OpenSans-Regular';">Query documentation &middot; Analyse logs &middot; GNSS insights</div>
  </div>
</header>
""", unsafe_allow_html=True)
 
 
# ── Welcome screen ────────────────────────────────────────────────────
if not st.session_state.chat:
    st.markdown(f"""
<div style="display:flex;flex-direction:column;align-items:center;gap:10px;
  padding:48px 32px 20px;text-align:center;">
  <div style="width:56px;height:56px;border-radius:14px;margin-bottom:8px;
    background-image:url('{LOGO_URL}');
    background-size:50px;
    background-repeat:no-repeat;
    background-position:center;
    background-color:white;
    box-shadow:0 4px 20px rgba(0,81,152,.3);">
  </div>
  <strong style="font-family:'OpenSans-Regular';font-size:16px;font-weight:600;color:#00284c;letter-spacing:.05em;">NovAtel AI Assistant</strong>
  <p style="font-size:14px;color:#6080a0;max-width:360px;line-height:1.6;font-family:'OpenSans-Regular';margin:0 0 4px;">
    Ask about logs, message formats, or upload a receiver log file to begin analysis.
  </p>
</div>
""", unsafe_allow_html=True)

    chips = [
        ("📡  Receiver status logs",   "What logs show receiver status?"),
        ("📍  BESTPOS message fields",  "Explain BESTPOS message fields"),
        ("📋  Common positioning logs", "Common Positioning Logs"),
        ("📂  Upload a log file",       "__upload__"),
    ]
    cols = st.columns(4)
    for col, (label, value) in zip(cols, chips):
        with col:
            if st.button(label, key=f"chip_{label[:12]}"):
                if value == "__upload__":
                    st.info("Use the 📎 paperclip button at the bottom-right to upload a log file.")
                else:
                    st.session_state.pending_chip = value
                    st.rerun()

# ── Flush pending chip ────────────────────────────────────────────────
if st.session_state.pending_chip:
    chip_text = st.session_state.pending_chip
    st.session_state.pending_chip = None
    st.session_state.chat.append(("user", chip_text))
    st.rerun()

# ── Render chat ───────────────────────────────────────────────────────
for role, msg in st.session_state.chat:
    if role == "agent":
        with st.chat_message("assistant"):
            st.markdown(msg)
    elif role == "file":
        with st.chat_message("user"):
            st.markdown(msg, unsafe_allow_html=True)
    else:
        with st.chat_message("user"):
            st.markdown(msg)

# ── Paperclip file uploader ───────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload file", type=["txt", "log", "asc", "ascii", "dat", "bin", "json", "csv"],
    key="file_upload", label_visibility="collapsed"
)

# ── Chat input ────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about NovAtel logs, message formats, GNSS…")

# ── Handle file upload ────────────────────────────────────────────────
if uploaded_file and uploaded_file.file_id not in st.session_state.get("processed_files", set()):
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    st.session_state.processed_files.add(uploaded_file.file_id)
    file_size = uploaded_file.size
    file_name = uploaded_file.name

    if file_size > MAX_UPLOAD:
        st.error(f"File is {file_size / (1024*1024):.1f} MB. Max allowed is {MAX_UPLOAD / (1024*1024):.0f} MB.")
    else:
        # ── STEP 1: Show chip immediately ─────────────────────────────
        size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024 * 1024 else f"{file_size / (1024*1024):.1f} MB"
        file_chip_html = f"""
        <div style="display:inline-flex;align-items:center;gap:7px;
          background:#e8f2fa;border:1px solid #b8d0e8;border-radius:8px;
          padding:6px 12px;font-size:12px;color:#005198;font-family:monospace;position:relative;bottom:3px;">
          📄&nbsp;<strong>{file_name}</strong>&nbsp;
        </div>
        """
        st.session_state.chat.append(("file", file_chip_html))
        # Store the upload info for processing on the next rerun
        file_bytes = uploaded_file.read()
        st.session_state.pending_upload = {
            "file_bytes": file_bytes,
            "file_name": file_name,
            "file_size": file_size,
        }
        st.session_state.session_id = "session-" + uuid.uuid4().hex[:10]
        st.rerun()  # rerun immediately so chip renders before processing starts

# ── Process pending upload (runs on the rerun after chip is shown) ────
if st.session_state.pending_upload:
    upload_info = st.session_state.pending_upload
    st.session_state.pending_upload = None  # clear so we don't re-process

    file_bytes = upload_info["file_bytes"]
    file_name  = upload_info["file_name"]
    file_size  = upload_info["file_size"]

    import threading, time
    # Capture session_id NOW on the main thread — background threads cannot access st.session_state
    file_session_id = st.session_state.session_id
    result_container = {"response": None, "done": False}

    def run_file_agent():
        try:
            if file_size <= SIZE_THRESHOLD:
                file_b64 = base64.b64encode(file_bytes).decode("utf-8")
                result_container["response"] = run_async(agent_invoke({
                    "file": file_b64, "filename": file_name,
                    "session_id": file_session_id,
                }))
            else:
                s3_key = upload_to_s3_with_progress(file_bytes, file_name)
                result_container["response"] = run_async(agent_invoke({
                    "s3_key": s3_key, "filename": file_name,
                    "session_id": file_session_id,
                }))
        except Exception as e:
            import traceback
            print(f"[ERROR] File upload failed: {traceback.format_exc()}")
            result_container["response"] = {"result": f"Error processing file: {str(e)}"}
        result_container["done"] = True

    file_thread = threading.Thread(target=run_file_agent, daemon=True)
    file_thread.start()

    # Poll for status while processing
    file_status_placeholder = st.empty()
    last_status = ""
    while not result_container["done"]:
        current_status = agent_get_status(file_session_id)
        if not current_status:
            current_status = f"Processing {file_name}..." if file_size <= SIZE_THRESHOLD else f"Uploading {file_name} to cloud..."
        if current_status != last_status:
            with file_status_placeholder.container():
                with st.chat_message("assistant", avatar="🛰"):
                    st.markdown(f"*{current_status}*")
            last_status = current_status
        time.sleep(0.3)

    file_status_placeholder.empty()
    response = result_container["response"]
    st.session_state.chat.append(("agent", response.get("result", str(response)) if response else "Error processing file."))
    st.rerun()

# ── Handle typed message ──────────────────────────────────────────────
if user_input:
    st.session_state.chat.append(("user", user_input))
    st.rerun()

# ── Answer last user message ──────────────────────────────────────────
if st.session_state.chat and st.session_state.chat[-1][0] == "user":
    # Create a placeholder for status updates
    status_placeholder = st.empty()
    
    # Capture values from session state before threading (threads can't access session_state)
    user_prompt = st.session_state.chat[-1][1]
    session_id = st.session_state.session_id
    
    # Start the agent invocation in a separate thread so we can poll status
    import threading
    import time
    
    result_container = {"response": None, "done": False}
    
    def run_agent():
        result_container["response"] = run_async(agent_invoke({
            "prompt":     user_prompt,
            "session_id": session_id,
        }))
        result_container["done"] = True
    
    # Start agent in background
    agent_thread = threading.Thread(target=run_agent, daemon=True)
    agent_thread.start()
    
    # Poll for status updates while agent is running
    last_status = ""
    while not result_container["done"]:
        current_status = agent_get_status(session_id)
        if current_status and current_status != last_status:
            with status_placeholder.container():
                with st.chat_message("assistant", avatar="🛰"):
                    st.markdown(f"*{current_status}*")
            last_status = current_status
        time.sleep(0.3)  # Poll every 300ms
    
    # Clear status placeholder
    status_placeholder.empty()
    
    # Add final response to chat
    response = result_container["response"]
    st.session_state.chat.append(("agent", response.get("result", "") if response else ""))
    st.rerun()
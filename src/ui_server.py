"""
ui_server.py — Chat UI for atomicAquaLangGraph AgentCore agent
Uses ONLY Python stdlib (http.server) — plus boto3 (already required by main.py).
Run:  python ui_server.py
Then open: http://localhost:8080
"""

import asyncio
import base64
import io
import json
import os
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ── Config (must be set BEFORE importing main.py) ─────────────────────
os.environ.setdefault("S3_BUCKET", "naspocuser-s3")
os.environ.setdefault("AWS_REGION", "ap-south-1")
os.environ.setdefault("KB_ID", "FH00WKSBPL")
# os.environ.setdefault("MEMORY_ID", "your-memory-id")
# os.environ.setdefault("GUARDRAIL_ID", "your-guardrail-id")

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config as BotocoreConfig

# ── Import the agent's invoke function directly ───────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import importlib
if 'main' in sys.modules:
    importlib.reload(sys.modules['main'])

from main import invoke as agent_invoke, get_status as agent_get_status

# ── Keep the local constants in sync ──────────────────────────────────
S3_BUCKET      = os.environ["S3_BUCKET"]
S3_REGION      = os.environ["AWS_REGION"]
SIZE_THRESHOLD = 5 * 1024 * 1024
MAX_UPLOAD     = 350 * 1024 * 1024  # Increased to 350 MB to support 315 MB files

# ── Session management ────────────────────────────────────────────────
_sessions = {}
_uploads: dict[str, dict] = {}
_uploads_lock = threading.Lock()

_s3_client = None
def get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            region_name=S3_REGION,
            config=BotocoreConfig(read_timeout=300),
        )
    return _s3_client


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>NovAtel Agent</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet"/>
<style>
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
  :root{
    --navy:       #00284c;
    --navy-deep:  #001a33;
    --navy-mid:   #003566;
    --blue:       #005198;
    --blue-light: #1a6bb5;
    --blue-pale:  #ccdff0;
    --blue-frost: #e8f2fa;
    --sky:        #4a9fd4;
    --white:      #ffffff;
    --off-white:  #f4f8fc;
    --border:     #b8d0e8;
    --border-dim: #d6e6f2;
    --text:       #00284c;
    --text-mid:   #2a5070;
    --text-dim:   #6080a0;
    --user-bg:    #dceef8;
    --agent-bg:   #ffffff;
    --surface:    #ffffff;
    --radius:     5px;
    --radius-lg:  10px;
    --font-ui:    'DM Sans', sans-serif;
    --font-mono:  'IBM Plex Mono', monospace;
  }
  html,body{height:100%;background:var(--off-white);color:var(--text);font-family:var(--font-ui);}

  /* -- Shell grid -- */
  .shell{
    display:grid;
    grid-template-rows:60px 1fr auto 76px;
    height:100vh;
    max-width:880px;
    margin:0 auto;
    box-shadow:0 0 0 1px rgba(0,40,76,.08), 0 4px 32px rgba(0,40,76,.06);
  }

  /* -- Header -- */
  header{
    display:flex;align-items:center;gap:14px;padding:0 24px;
    background:var(--navy);
    border-bottom:2px solid var(--blue);
    position:relative;overflow:hidden;
  }
  header::after{
    content:'';position:absolute;right:-60px;top:-40px;
    width:220px;height:140px;
    background:radial-gradient(ellipse at center, rgba(0,81,152,.45) 0%, transparent 70%);
    pointer-events:none;
  }
  .logo{
    width:36px;height:36px;border-radius:var(--radius);
    background:linear-gradient(135deg, var(--blue) 0%, var(--sky) 100%);
    display:grid;place-items:center;flex-shrink:0;
    box-shadow:0 0 0 1px rgba(255,255,255,.15), 0 2px 8px rgba(0,0,0,.3);
  }
  .logo svg{width:18px;height:18px;}
  .hd-text{flex:1;}
  .title{
    font-family:var(--font-mono);font-size:13.5px;font-weight:600;
    letter-spacing:.06em;color:#ffffff;line-height:1;
  }
  .subtitle{font-size:11px;color:rgba(255,255,255,.52);margin-top:3px;font-weight:400;letter-spacing:.01em;}
  .status-pill{
    margin-left:auto;font-family:var(--font-mono);font-size:10px;font-weight:500;
    padding:4px 10px;border-radius:20px;letter-spacing:.05em;
    border:1px solid rgba(74,159,212,.45);
    color:#a8d8f0;background:rgba(0,81,152,.25);
    display:flex;align-items:center;gap:5px;
  }
  .status-dot{width:5px;height:5px;border-radius:50%;background:#4a9fd4;
    box-shadow:0 0 6px rgba(74,159,212,.8);}

  /* -- Messages -- */
  #messages{
    overflow-y:auto;padding:28px 24px;
    display:flex;flex-direction:column;gap:18px;
    scroll-behavior:smooth;background:var(--off-white);
  }
  #messages::-webkit-scrollbar{width:4px;}
  #messages::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px;}

  /* -- Message rows -- */
  .msg{display:flex;gap:12px;animation:fadeUp .2s ease both;}
  @keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
  .msg.user{flex-direction:row-reverse;}

  .avatar{
    width:30px;height:30px;border-radius:6px;
    display:grid;place-items:center;
    font-family:var(--font-mono);font-size:9px;font-weight:600;
    flex-shrink:0;margin-top:2px;letter-spacing:.04em;
  }
  .avatar.agent{
    background:linear-gradient(135deg,var(--navy) 0%, var(--blue) 100%);
    color:#a8d8f0;border:1px solid rgba(0,81,152,.4);
    box-shadow:0 1px 4px rgba(0,40,76,.2);
  }
  .avatar.user{
    background:linear-gradient(135deg,var(--blue-light) 0%,var(--sky) 100%);
    color:#ffffff;border:1px solid rgba(74,159,212,.35);
    box-shadow:0 1px 4px rgba(0,40,76,.15);
  }

  .bubble{
    max-width:700px;padding:13px 17px;border-radius:var(--radius-lg);
    font-size:13.5px;line-height:1.7;white-space:pre-wrap;word-break:break-word;
    font-family:var(--font-mono);
  }
  .msg.agent .bubble{
    background:var(--agent-bg);
    border:1px solid var(--border-dim);
    color:var(--text);
    box-shadow:0 1px 6px rgba(0,40,76,.05);
  }
  .msg.user .bubble{
    background:linear-gradient(135deg,var(--blue) 0%, var(--blue-light) 100%);
    border:1px solid rgba(26,107,181,.3);
    color:#ffffff;
    box-shadow:0 2px 10px rgba(0,81,152,.2);
  }
  .ts{
    font-size:10px;color:var(--text-dim);margin-top:5px;
    font-family:var(--font-mono);letter-spacing:.02em;
  }
  .msg.user .ts{text-align:right;}

  /* -- Typing indicator -- */
  .typing .bubble{display:flex;gap:6px;align-items:center;padding:15px 20px;flex-wrap:wrap;}
  .status-text{width:100%;font-size:11px;color:var(--text-dim);margin-bottom:8px;font-style:italic;}
  .dot{
    width:6px;height:6px;border-radius:50%;
    background:var(--blue);animation:blink 1.1s infinite;
  }
  .dot:nth-child(2){animation-delay:.18s;}
  .dot:nth-child(3){animation-delay:.36s;}
  @keyframes blink{0%,80%,100%{opacity:.15;transform:scale(.8)}40%{opacity:1;transform:scale(1)}}

  /* -- Session badge -- */
  .session-badge{
    text-align:center;font-family:var(--font-mono);
    font-size:10px;color:var(--text-dim);padding:4px 0 10px;
  }
  .session-badge span{
    background:var(--surface);border:1px solid var(--border-dim);
    border-radius:20px;padding:3px 12px;
    color:var(--text-dim);letter-spacing:.04em;
  }

  /* -- File chip -- */
  .file-chip{
    display:inline-flex;align-items:center;gap:6px;
    background:rgba(0,81,152,.12);color:var(--blue);
    border:1px solid rgba(0,81,152,.25);
    padding:3px 11px;border-radius:20px;
    font-family:var(--font-mono);font-size:10px;
  }
  .msg.user .file-chip{
    background:rgba(255,255,255,.2);
    color:#ffffff;
    border:1px solid rgba(255,255,255,.4);
  }

  /* -- Progress bar -- */
  #progress-row{
    display:none;padding:10px 24px 12px;
    background:var(--surface);border-top:1px solid var(--border-dim);
    font-family:var(--font-mono);font-size:11px;color:var(--text-dim);
  }
  #progress-row.active{display:block;}
  .progress-head{display:flex;justify-content:space-between;margin-bottom:6px;}
  .progress-head .name{color:var(--text);font-weight:500;}
  .bar{height:3px;background:var(--blue-frost);border-radius:2px;overflow:hidden;}
  .bar .fill{
    height:100%;
    background:linear-gradient(90deg, var(--blue) 0%, var(--sky) 100%);
    width:0%;transition:width .25s ease;
  }
  #prog-status{margin-top:5px;color:var(--text-dim);}

  /* -- Footer / input row -- */
  footer{
    display:flex;align-items:center;gap:10px;padding:0 20px;
    border-top:1px solid var(--border-dim);
    background:var(--surface);margin-bottom:-60px;
  }
  #attach{
    width:40px;height:40px;border-radius:var(--radius);
    background:transparent;border:1px solid var(--border);
    cursor:pointer;display:grid;place-items:center;
    transition:border-color .15s,background .15s,color .15s;
    flex-shrink:0;color:var(--text-dim);
  }
  #attach:hover:not(:disabled){border-color:var(--blue);color:var(--blue);background:var(--blue-frost);}
  #attach:disabled{opacity:.38;cursor:not-allowed;}
  #attach svg{width:16px;height:16px;}
  #file-input{display:none;}

  #input{
    flex:1;background:var(--off-white);
    border:1px solid var(--border);border-radius:var(--radius);
    padding:11px 15px;color:var(--text);
    font-family:var(--font-mono);font-size:13px;
    resize:none;max-height:120px;outline:none;
    transition:border-color .15s,box-shadow .15s;line-height:1.5;
  }
  #input:focus{
    border-color:var(--blue);
    box-shadow:0 0 0 3px rgba(0,81,152,.1);
  }
  #input::placeholder{color:var(--text-dim);}

  #send{
    width:40px;height:40px;border-radius:var(--radius);
    background:linear-gradient(135deg,var(--navy) 0%,var(--blue) 100%);
    border:none;cursor:pointer;display:grid;place-items:center;
    transition:opacity .15s,transform .1s,box-shadow .15s;flex-shrink:0;
    box-shadow:0 2px 8px rgba(0,40,76,.25);
  }
  #send:hover:not(:disabled){opacity:.88;box-shadow:0 4px 14px rgba(0,40,76,.3);}
  #send:active{transform:scale(.93);}
  #send:disabled{background:var(--border);cursor:not-allowed;box-shadow:none;}
  #send svg{width:15px;height:15px;fill:#ffffff;}

  /* -- Welcome screen -- */
  .welcome{
    display:flex;flex-direction:column;align-items:center;justify-content:center;
    gap:10px;height:100%;text-align:center;padding:48px 32px;
  }
  .welcome-icon{
    width:56px;height:56px;border-radius:14px;
    background:linear-gradient(135deg,var(--navy) 0%,var(--blue) 100%);
    display:grid;place-items:center;margin-bottom:8px;
    box-shadow:0 4px 20px rgba(0,81,152,.3);
  }
  .welcome-icon svg{width:26px;height:26px;}
  .welcome strong{
    font-family:var(--font-mono);font-size:16px;font-weight:600;
    color:var(--navy);letter-spacing:.04em;
  }
  .welcome p{font-size:13px;color:var(--text-dim);max-width:360px;line-height:1.6;}
  .chips{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-top:12px;}
  .chip{
    padding:7px 15px;border:1px solid var(--border);border-radius:20px;
    cursor:pointer;font-size:11.5px;font-family:var(--font-mono);
    color:var(--text-mid);background:var(--surface);
    transition:border-color .15s,color .15s,background .15s,box-shadow .15s;
  }
  .chip:hover{
    border-color:var(--blue);color:var(--blue);
    background:var(--blue-frost);
    box-shadow:0 2px 8px rgba(0,81,152,.12);
  }

  /* -- Divider line -- */
  .divider{
    height:1px;background:linear-gradient(90deg,transparent,var(--border-dim),transparent);
    margin:4px 0;
  }
</style>
</head>
<body>
<div class="shell">

  <!-- Header -->
  <header>
    <div class="logo">
      <svg viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="9" cy="9" r="3.2" fill="#ffffff"/>
        <circle cx="9" cy="2.2" r="1.4" fill="rgba(255,255,255,.75)"/>
        <circle cx="9" cy="15.8" r="1.4" fill="rgba(255,255,255,.75)"/>
        <circle cx="2.2" cy="9" r="1.4" fill="rgba(255,255,255,.75)"/>
        <circle cx="15.8" cy="9" r="1.4" fill="rgba(255,255,255,.75)"/>
      </svg>
    </div>
    <div class="hd-text">
      <div class="title">NovAtel AI Assistant</div>
      <div class="subtitle">Query documentation &middot; Analyse logs &middot; GNSS insights</div>
    </div>
    <div class="status-pill" id="status-pill">
      <span class="status-dot" id="status-dot"></span>
      <span id="status-label">READY</span>
    </div>
  </header>

  <!-- Messages -->
  <div id="messages">
    <div class="welcome" id="welcome">
      <div class="welcome-icon">
        <svg viewBox="0 0 26 26" fill="none" xmlns="http://www.w3.org/2000/svg">
          <circle cx="13" cy="13" r="4.5" fill="#ffffff"/>
          <circle cx="13" cy="3" r="2" fill="rgba(255,255,255,.7)"/>
          <circle cx="13" cy="23" r="2" fill="rgba(255,255,255,.7)"/>
          <circle cx="3" cy="13" r="2" fill="rgba(255,255,255,.7)"/>
          <circle cx="23" cy="13" r="2" fill="rgba(255,255,255,.7)"/>
          <line x1="13" y1="7.5" x2="13" y2="10.5" stroke="rgba(255,255,255,.35)" stroke-width="1"/>
          <line x1="13" y1="15.5" x2="13" y2="18.5" stroke="rgba(255,255,255,.35)" stroke-width="1"/>
          <line x1="7.5" y1="13" x2="10.5" y2="13" stroke="rgba(255,255,255,.35)" stroke-width="1"/>
          <line x1="15.5" y1="13" x2="18.5" y2="13" stroke="rgba(255,255,255,.35)" stroke-width="1"/>
        </svg>
      </div>
      <strong>NovAtel Agent</strong>
      <p>Ask about logs, message formats, or upload a receiver log file to begin analysis.</p>
      <div class="chips">
        <div class="chip" onclick="sendChip(this)">What logs show receiver status?</div>
        <div class="chip" onclick="sendChip(this)">Explain BESTPOS message fields</div>
        <div class="chip" onclick="sendChip(this)">Common Positioning Logs</div>
      </div>
    </div>
  </div>

  <!-- Upload progress -->
  <div id="progress-row">
    <div class="progress-head">
      <span class="name" id="prog-name">uploading...</span>
      <span id="prog-pct">0%</span>
    </div>
    <div class="bar"><div class="fill" id="prog-fill"></div></div>
    <div id="prog-status"></div>
  </div>

  <!-- Footer input bar -->
  <footer>
    <input type="file" id="file-input" accept=".log,.txt,.asc,.ascii,.dat,.bin,.json,.csv"/>
    <button id="attach" title="Attach log file">
      <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"
           stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg">
        <path d="M10.5 4.5L5 10a2 2 0 0 0 2.8 2.8l6.2-6.2a3.5 3.5 0 0 0-5-5L2.8 8a5 5 0 1 0 7 7l4.7-4.7"/>
      </svg>
    </button>
    <textarea id="input" rows="1" placeholder="Ask about NovAtel logs, message formats, GNSS..." maxlength="2000"></textarea>
    <button id="send" title="Send (Enter)">
      <svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg"><path d="M1 8l13-6-5 6 5 6z"/></svg>
    </button>
  </footer>
</div>

<script>
  const messagesEl = document.getElementById('messages');
  const inputEl    = document.getElementById('input');
  const sendBtn    = document.getElementById('send');
  const statusLabel= document.getElementById('status-label');
  const statusDot  = document.getElementById('status-dot');
  const welcome    = document.getElementById('welcome');
  const attachBtn  = document.getElementById('attach');
  const fileInput  = document.getElementById('file-input');
  const progRow    = document.getElementById('progress-row');
  const progName   = document.getElementById('prog-name');
  const progPct    = document.getElementById('prog-pct');
  const progFill   = document.getElementById('prog-fill');
  const progStatus = document.getElementById('prog-status');

  let clientId = null, sessionId = null, busy = false;

  const SMALL_FILE_LIMIT = 5 * 1024 * 1024;
  const MAX_FILE         = 350 * 1024 * 1024;

  inputEl.addEventListener('input', () => {
    inputEl.style.height = 'auto';
    inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
  });
  inputEl.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });
  sendBtn.addEventListener('click', send);
  attachBtn.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', handleFile);

  function sendChip(el) { inputEl.value = el.textContent; send(); }
  function now() { return new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}); }
  function fmtMB(b) { return (b / (1024 * 1024)).toFixed(1) + ' MB'; }

  function simpleMarkdown(text) {
    // Convert markdown to HTML for better rendering
    // Bold: **text** -> <strong>text</strong>
    text = text.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
    // Bullet points: - text
    text = text.replace(/^- (.+)$/gm, '&nbsp;&nbsp;&bull; $1');
    // Preserve line breaks
    text = text.replace(/\\n/g, '<br>');
    return text;
  }

  function appendMsg(role, text, opts) {
    if (welcome) welcome.style.display = 'none';
    const wrap   = document.createElement('div'); wrap.className = 'msg ' + role;
    const av     = document.createElement('div'); av.className = 'avatar ' + role;
    av.textContent = role === 'user' ? 'YOU' : 'AI';
    const col    = document.createElement('div');
    const bubble = document.createElement('div'); bubble.className = 'bubble';
    
    // For agent messages, render markdown; for user messages, keep plain text
    if (role === 'agent' && !opts?.html) {
      bubble.innerHTML = simpleMarkdown(text);
    } else if (opts && opts.html) {
      bubble.innerHTML = text;
    } else {
      bubble.textContent = text;
    }
    
    const ts = document.createElement('div'); ts.className = 'ts'; ts.textContent = now();
    col.appendChild(bubble); col.appendChild(ts);
    wrap.appendChild(av); wrap.appendChild(col);
    messagesEl.appendChild(wrap);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return bubble;
  }

  function showTyping(statusText) {
    const w = document.createElement('div');
    w.className = 'msg agent typing'; w.id = 'typing';
    const statusMsg = statusText || 'Thinking...';
    w.innerHTML = '<div class="avatar agent">AI</div><div><div class="bubble"><div class="status-text" id="status-text">' + statusMsg + '</div><div class="dot"></div><div class="dot"></div><div class="dot"></div></div></div>';
    messagesEl.appendChild(w);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
  function removeTyping() { document.getElementById('typing')?.remove(); }
  
  function updateTypingStatus(text) {
    const statusEl = document.getElementById('status-text');
    if (statusEl) statusEl.textContent = text;
  }

  function showSessionBadge(sid) {
    const existing = document.getElementById('sess-badge');
    if (existing) existing.remove();
    const d = document.createElement('div');
    d.className = 'session-badge'; d.id = 'sess-badge';
    d.innerHTML = '<span>session &middot; ' + sid + '</span>';
    messagesEl.insertBefore(d, messagesEl.firstChild);
  }

  function setBusy(on, label) {
    busy = on; sendBtn.disabled = on; attachBtn.disabled = on;
    if (on) {
      statusLabel.textContent = label || 'THINKING';
      statusDot.style.background = '#f0b840';
      statusDot.style.boxShadow  = '0 0 6px rgba(240,184,64,.8)';
    } else {
      statusLabel.textContent = 'READY';
      statusDot.style.background = '#4a9fd4';
      statusDot.style.boxShadow  = '0 0 6px rgba(74,159,212,.8)';
    }
  }

  function showProgress(name, pct, status) {
    progRow.classList.add('active');
    progName.textContent    = name;
    progPct.textContent     = pct + '%';
    progFill.style.width    = pct + '%';
    progStatus.textContent  = status || '';
  }
  function hideProgress() {
    progRow.classList.remove('active');
    progFill.style.width = '0%';
  }

  async function handleFile(e) {
    const f = e.target.files[0]; fileInput.value = '';
    if (!f) return;
    if (f.size > MAX_FILE) {
      appendMsg('agent', 'File is ' + fmtMB(f.size) + '. Max allowed is ' + fmtMB(MAX_FILE) + '.');
      return;
    }
    if (welcome) welcome.style.display = 'none';
    appendMsg('user', '<span class="file-chip">&#128206; ' + f.name + ' &middot; ' + fmtMB(f.size) + '</span>', {html: true});
    setBusy(true, 'UPLOADING');
    try {
      if (f.size <= SMALL_FILE_LIMIT) { await uploadSmall(f); }
      else { await uploadLarge(f); }
    } catch(err) {
      hideProgress();
      appendMsg('agent', 'Upload failed: ' + (err.message || err));
      setBusy(false);
    }
  }

  async function uploadSmall(f) {
    showProgress(f.name, 10, 'Encoding...');
    const b64 = await fileToBase64(f);
    showProgress(f.name, 60, 'Sending to agent...');
    const res = await fetch('/upload-small', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({filename: f.name, file_b64: b64, client_id: clientId}),
    });
    if (!res.ok) {
      const e = await res.json().catch(() => ({detail: res.statusText}));
      throw new Error(e.detail || 'upload failed');
    }
    const data = await res.json();
    showProgress(f.name, 100, 'Done');
    finishUpload(data, f.name);
  }

  function fileToBase64(f) {
    return new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload  = () => resolve(r.result.split(',')[1]);
      r.onerror = () => reject(new Error('read error'));
      r.readAsDataURL(f);
    });
  }

  async function uploadLarge(f) {
    const uploadId = crypto.randomUUID ? crypto.randomUUID() : String(Math.random());
    showProgress(f.name, 0, 'Uploading to S3...');
    let pollActive = true;
    (async function poll() {
      while (pollActive) {
        try {
          const r = await fetch('/upload-progress?id=' + uploadId);
          if (r.ok) {
            const p = await r.json();
            if (p.total > 0) {
              const pct = Math.min(99, Math.round(p.done / p.total * 100));
              showProgress(f.name, pct, p.status || 'Uploading to S3...');
            }
            if (p.status === 'done' || p.status === 'error') break;
          }
        } catch(_) {}
        await new Promise(r => setTimeout(r, 500));
      }
    })();
    const fd = new FormData();
    fd.append('file', f);
    fd.append('upload_id', uploadId);
    if (clientId) fd.append('client_id', clientId);
    const res = await fetch('/upload-large', {method: 'POST', body: fd});
    pollActive = false;
    if (!res.ok) {
      const e = await res.json().catch(() => ({detail: res.statusText}));
      throw new Error(e.detail || 'upload failed');
    }
    const data = await res.json();
    showProgress(f.name, 100, 'Done');
    finishUpload(data, f.name);
  }

  function finishUpload(data, filename) {
    clientId = data.client_id; sessionId = data.session_id;
    showSessionBadge(sessionId);
    appendMsg('agent', data.reply || ('Processed ' + filename + '. You can now ask questions about this log.'));
    setTimeout(hideProgress, 800);
    setBusy(false);
    inputEl.focus();
  }

  async function send() {
    const text = inputEl.value.trim(); if (!text || busy) return;
    setBusy(true, 'THINKING');
    inputEl.value = ''; inputEl.style.height = 'auto';
    appendMsg('user', text); showTyping('Thinking...');
    
    // Start polling for status
    let statusPoll = setInterval(async () => {
      if (sessionId) {
        try {
          const res = await fetch('/status?session=' + sessionId);
          if (res.ok) {
            const data = await res.json();
            if (data.status) updateTypingStatus(data.status);
          }
        } catch(e) {}
      }
    }, 500);
    
    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: text, client_id: clientId}),
      });
      clearInterval(statusPoll);
      removeTyping();
      if (!res.ok) {
        const err = await res.json().catch(() => ({detail: res.statusText}));
        appendMsg('agent', 'Error: ' + (err.detail || 'Unknown'));
      } else {
        const data = await res.json();
        clientId = data.client_id; sessionId = data.session_id;
        showSessionBadge(sessionId);
        appendMsg('agent', data.reply);
      }
    } catch(e) {
      clearInterval(statusPoll);
      removeTyping();
      appendMsg('agent', 'Network error -- is ui_server.py running?');
    }
    setBusy(false);
    inputEl.focus();
  }
</script>
</body>
</html>"""


# ── Multipart form parser (stdlib) ────────────────────────────────────
from email.parser import BytesParser
from email.policy import default as email_default_policy

def parse_multipart(body: bytes, content_type: str) -> dict:
    """Return {field_name: (filename_or_None, bytes_or_str)}"""
    header = f"Content-Type: {content_type}\r\n\r\n".encode("utf-8")
    msg    = BytesParser(policy=email_default_policy).parsebytes(header + body)
    out    = {}
    for part in msg.iter_parts():
        disp = part.get("Content-Disposition", "")
        if "name=" not in disp:
            continue
        params = {}
        for kv in disp.split(";"):
            kv = kv.strip()
            if "=" in kv:
                k, v = kv.split("=", 1)
                params[k.strip().lower()] = v.strip().strip('"')
        name     = params.get("name")
        filename = params.get("filename")
        payload  = part.get_payload(decode=True)
        if filename is None and isinstance(payload, (bytes, bytearray)):
            try:
                payload = payload.decode("utf-8")
            except UnicodeDecodeError:
                pass
        out[name] = (filename, payload)
    return out


# ── Upload helpers ────────────────────────────────────────────────────
def _session_for_client(client_id: str, new_for_file: bool = False) -> str:
    if new_for_file or client_id not in _sessions:
        _sessions[client_id] = "session-" + uuid.uuid4().hex[:10]
    return _sessions[client_id]


def _s3_upload_with_progress(file_bytes: bytes, filename: str, upload_id: str) -> str:
    key   = f"logs/{filename}"
    total = len(file_bytes)

    with _uploads_lock:
        _uploads[upload_id] = {"done": 0, "total": total, "status": "uploading"}

    cfg = TransferConfig(
        multipart_threshold = 8 * 1024 * 1024,
        multipart_chunksize = 8 * 1024 * 1024,
        max_concurrency     = 4,
        use_threads         = True,
    )

    def cb(n):
        with _uploads_lock:
            if upload_id in _uploads:
                _uploads[upload_id]["done"] += n

    get_s3_client().upload_fileobj(
        io.BytesIO(file_bytes),
        S3_BUCKET,
        key,
        Config=cfg,
        Callback=cb,
    )

    with _uploads_lock:
        if upload_id in _uploads:
            _uploads[upload_id]["status"] = "processing"
    return key


# ── HTTP handler ──────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass

    def _json(self, code: int, obj: dict):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode('utf-8'))
            return

        if self.path == "/debug":
            import main as m
            debug_info = {
                "log_store_sessions": list(m._log_store.keys()),
                "ui_sessions": dict(_sessions),
                "log_store_details": {
                    sid: {
                        "filename": entry.get("filename"),
                        "records": len(entry.get("df", [])),
                    }
                    for sid, entry in m._log_store.items()
                }
            }
            self._json(200, debug_info)
            return

        if self.path.startswith("/upload-progress"):
            upload_id = ""
            if "?" in self.path:
                q = self.path.split("?", 1)[1]
                for kv in q.split("&"):
                    if kv.startswith("id="):
                        upload_id = kv[3:]
                        break
            with _uploads_lock:
                p = _uploads.get(upload_id)
                payload = dict(p) if p else {"done": 0, "total": 0, "status": "unknown"}
            self._json(200, payload)
            return

        if self.path.startswith("/status"):
            session_id = ""
            if "?" in self.path:
                q = self.path.split("?", 1)[1]
                for kv in q.split("&"):
                    if kv.startswith("session="):
                        session_id = kv[8:]
                        break
            status = agent_get_status(session_id) if session_id else ""
            self._json(200, {"status": status})
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path == "/chat":
            return self._handle_chat()
        if self.path == "/upload-small":
            return self._handle_upload_small()
        if self.path == "/upload-large":
            return self._handle_upload_large()
        self.send_response(404)
        self.end_headers()

    def _handle_chat(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)
        print(f"\n{'='*60}\n[UI_SERVER] Received chat request\n{'='*60}")
        try:
            req        = json.loads(body)
            message    = req.get("message", "")
            client_id  = req.get("client_id") or str(uuid.uuid4())
            session_id = _session_for_client(client_id)

            print(f"[UI_SERVER] message={message!r} session_id={session_id!r}")
            result = asyncio.run(agent_invoke({"prompt": message, "session_id": session_id}))
            answer = result.get("result", str(result))
            print(f"[UI_SERVER] answer length={len(answer)}")

            self._json(200, {"reply": answer, "client_id": client_id, "session_id": session_id})
        except Exception as e:
            print(f"[UI_SERVER] ERROR: {e}")
            self._json(500, {"detail": str(e)})

    def _handle_upload_small(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)
        try:
            req        = json.loads(body)
            file_b64   = req.get("file_b64", "")
            filename   = req.get("filename", "log.txt")
            client_id  = req.get("client_id") or str(uuid.uuid4())

            session_id = _session_for_client(client_id, new_for_file=True)

            result = asyncio.run(agent_invoke({
                "file":       file_b64,
                "filename":   filename,
                "session_id": session_id,
            }))
            reply = result.get("result", str(result))
            self._json(200, {"reply": reply, "client_id": client_id, "session_id": session_id})
        except Exception as e:
            self._json(500, {"detail": str(e)})

    def _handle_upload_large(self):
        import main as _m
        print(f"[DEBUG] main.S3_BUCKET = {_m.S3_BUCKET!r}")
        length       = int(self.headers.get("Content-Length", 0))
        content_type = self.headers.get("Content-Type", "")
        if length > MAX_UPLOAD:
            self._json(413, {"detail": f"File exceeds {MAX_UPLOAD // (1024*1024)} MB limit"})
            return

        try:
            body   = self.rfile.read(length)
            fields = parse_multipart(body, content_type)

            upload_id  = fields.get("upload_id", (None, ""))[1]
            client_id  = fields.get("client_id", (None, None))[1] or str(uuid.uuid4())
            file_tuple = fields.get("file")
            if not file_tuple or not file_tuple[0]:
                self._json(400, {"detail": "no file uploaded"})
                return
            filename   = file_tuple[0]
            file_bytes = file_tuple[1]
            if isinstance(file_bytes, str):
                file_bytes = file_bytes.encode("utf-8")

            s3_key = _s3_upload_with_progress(file_bytes, filename, upload_id)

            session_id = _session_for_client(client_id, new_for_file=True)
            result     = asyncio.run(agent_invoke({
                "s3_key":     s3_key,
                "filename":   filename,
                "session_id": session_id,
            }))
            reply = result.get("result", str(result))

            with _uploads_lock:
                if upload_id in _uploads:
                    _uploads[upload_id]["status"] = "done"
                    _uploads[upload_id]["done"]   = _uploads[upload_id]["total"]

            self._json(200, {"reply": reply, "client_id": client_id, "session_id": session_id})

        except Exception as e:
            with _uploads_lock:
                upload_id = locals().get("upload_id", "")
                if upload_id in _uploads:
                    _uploads[upload_id]["status"] = "error"
            self._json(500, {"detail": str(e)})


# ── Threaded server so progress polls work during large upload ────────
from http.server import ThreadingHTTPServer

if __name__ == "__main__":
    port   = 8080
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print("NovAtel Agent UI running at http://localhost:" + str(port))
    print("Upload files with the paperclip icon. S3 bucket: " + S3_BUCKET)
    print("Press Ctrl+C to stop.")
    server.serve_forever()
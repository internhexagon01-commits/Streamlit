"""
main.py — NovAtel OEM7 log analyst + documentation Q&A.

Architecture
────────────
Log file questions  → Direct Python pipeline (no agent loop):
                        1. kb_search()       — one KB call, returns raw hits
                        2. extract_params()  — one LLM call, extracts log/field/bit as JSON
                        3. run_log_tool()    — calls the right Python function directly
                        4. format_answer()   — one LLM call to format the final answer

Documentation Q&A   → LangGraph ReAct agent with kb_retriever + context_expander only
"""

from langgraph.errors import GraphRecursionError
from bedrock_agentcore import BedrockAgentCoreApp
from bedrock_agentcore.memory.client import MemoryClient
from src.model.load import load_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
import boto3
import base64
import os
import re
import io
import json
import time
import tempfile
import datetime
import pandas as pd
from collections import Counter
from botocore.config import Config as BotocoreConfig
from urllib.parse import urlparse

# ── GPS helpers ───────────────────────────────────────────────────────
_GPS_EPOCH    = datetime.datetime(1980, 1, 6, tzinfo=datetime.timezone.utc)
_LEAP_SECONDS = 18

def gps_to_utc(week: int, seconds: float) -> datetime.datetime:
    return _GPS_EPOCH + datetime.timedelta(seconds=week * 604800 + seconds - _LEAP_SECONDS)

def gps_to_utc_str(week: int, seconds: float) -> str:
    if week <= 0:
        return ""
    return gps_to_utc(week, seconds).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

# ── Config ────────────────────────────────────────────────────────────
REGION          = "us-east-1"
MEMORY_ID       = os.getenv("MEMORY_ID")
S3_BUCKET       = os.getenv("S3_BUCKET")
KB_ID           = os.getenv("KB_ID", "FH00WKSBPL")
GUARDRAIL_ID    = os.getenv("GUARDRAIL_ID", "")
GUARDRAIL_VER   = os.getenv("GUARDRAIL_VERSION", "1")
ACTOR_ID        = "default-user"
SIZE_THRESHOLD  = 5 * 1024 * 1024
MAX_RESULTS     = int(os.getenv("MAX_RESULTS", "15"))
EXPANSION_PAGES = int(os.getenv("EXPANSION_PAGES", "2"))
_LLM_COLS       = {"element_id", "element_type", "content_markdown", "page_number"}

app = BedrockAgentCoreApp()

# ── Status tracking for UI updates ───────────────────────────────────
_current_status: dict[str, str] = {}

def set_status(session_id: str, status: str):
    """Set current processing status for UI display."""
    _current_status[session_id] = status
    print(f"[STATUS] {session_id}: {status}")

def get_status(session_id: str) -> str:
    """Get current processing status."""
    return _current_status.get(session_id, "")

def clear_status(session_id: str):
    """Clear status after completion."""
    _current_status.pop(session_id, None)

# ── Lazy singletons ───────────────────────────────────────────────────
_llm = _memory_client = _s3_client = _kb_client = _bedrock_runtime = None
_BOTO_CONFIG = BotocoreConfig(read_timeout=300)

def get_llm():
    global _llm
    if _llm is None:
        _llm = load_model()
    return _llm

def get_memory_client():
    global _memory_client
    if _memory_client is None:
        _memory_client = MemoryClient(region_name=REGION)
    return _memory_client

def get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name="ap-south-1", config=_BOTO_CONFIG)
    return _s3_client

def get_kb_client():
    global _kb_client
    if _kb_client is None:
        _kb_client = boto3.client("bedrock-agent-runtime", region_name="us-west-2", config=_BOTO_CONFIG)
    return _kb_client

def get_bedrock_runtime():
    global _bedrock_runtime
    if _bedrock_runtime is None:
        _bedrock_runtime = boto3.client("bedrock-runtime", region_name=REGION, config=_BOTO_CONFIG)
    return _bedrock_runtime

# ── Guardrail ─────────────────────────────────────────────────────────
def apply_guardrail(text: str, source: str = "INPUT") -> str:
    if not GUARDRAIL_ID:
        return text
    try:
        resp = get_bedrock_runtime().apply_guardrail(
            guardrailIdentifier=GUARDRAIL_ID,
            guardrailVersion=GUARDRAIL_VER,
            source=source,
            content=[{"text": {"text": text}}],
        )
        if resp["action"] == "GUARDRAIL_INTERVENED":
            blocked = resp.get("outputs", [{}])[0].get("text", "Content blocked by guardrail.")
            raise ValueError(blocked)
    except ValueError:
        raise
    except Exception as e:
        print(f"[GUARDRAIL] error: {e}")
    return text

# ── KB helpers ────────────────────────────────────────────────────────
_tool_call_log: list[dict] = []
_csv_cache: dict[str, pd.DataFrame] = {}

def _download_and_parse_key(bucket: str, key: str) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(mode="w+b", delete=False) as tmp:
        get_s3_client().download_fileobj(bucket, key, tmp)
        tmp.flush()
        tmp_path = tmp.name
    try:
        with open(tmp_path, "r", encoding="utf-8") as f:
            raw = f.read()
        try:
            data = json.loads(raw)
            rows = [e["contentMetadata"] for e in data.get("fileContents", []) if "contentMetadata" in e]
            df = pd.DataFrame(rows)
        except (json.JSONDecodeError, KeyError):
            df = pd.read_csv(io.StringIO(raw))
    finally:
        os.unlink(tmp_path)
    return df

def _download_and_parse(source_uri: str) -> pd.DataFrame:
    if source_uri in _csv_cache:
        return _csv_cache[source_uri]
    parsed = urlparse(source_uri)
    bucket = parsed.netloc
    key    = parsed.path.lstrip("/")
    df = _download_and_parse_key(bucket, key)
    if "element_id" not in df.columns and not key.startswith("Output/"):
        df = _download_and_parse_key(bucket, f"Output/{key}")
    if "page_number" in df.columns:
        df["page_number"] = pd.to_numeric(df["page_number"], errors="coerce").fillna(0).astype(int)
    _csv_cache[source_uri] = df
    return df

def _resolve_data_uri(source_uri: str) -> str:
    if not source_uri.lower().endswith(".pdf"):
        return source_uri
    for entry in reversed(_tool_call_log):
        if entry.get("tool") != "kb_retriever":
            continue
        for el in entry["result"].get("elements", []):
            if el.get("source_uri") == source_uri and el.get("csv_source_uri"):
                return el["csv_source_uri"]
    raise ValueError(f"Cannot resolve data URI from PDF path: {source_uri}")

# ── NovAtel ASCII log parser ──────────────────────────────────────────
_log_store: dict[str, dict] = {}

_ASCII_FULL_RE = re.compile(
    r"^#(?P<log_name>[A-Z0-9_]+),"
    r"(?P<header>[^;]*);"
    r"(?P<fields>.*?)"
    r"(?:\*[0-9a-fA-F]{1,8})?\s*$"
)

def _parse_line(line: str) -> dict | None:
    m = _ASCII_FULL_RE.match(line.strip())
    if not m:
        return None
    g = m.groupdict()
    h = [p.strip() for p in g["header"].split(",")]

    def _get(i, d=""): return h[i] if i < len(h) else d
    def _tryf(v, d=0.0):
        try: return float(v)
        except: return d
    def _tryi(v, d=0):
        try: return int(v)
        except: return d

    log_name   = g["log_name"]
    normalized = log_name[:-1] if (log_name.endswith("A") and len(log_name) > 4) else log_name
    week       = _tryi(_get(4))
    seconds    = _tryf(_get(5))

    return {
        "log_name":     normalized,
        "log_name_raw": log_name,
        "port":         _get(0),
        "seq":          _tryi(_get(1)),
        "idle_pct":     _tryf(_get(2)),
        "time_status":  _get(3),
        "week":         week,
        "seconds":      seconds,
        "utc_time":     gps_to_utc_str(week, seconds),
        "rx_status":    _get(6),   # receiver status word in every message header
        "fields_raw":   g["fields"],
    }

def parse_novatel_ascii(text: str) -> pd.DataFrame:
    records, skipped = [], 0
    for line in text.splitlines():
        if not line.strip():
            continue
        rec = _parse_line(line)
        if rec:
            records.append(rec)
        else:
            skipped += 1
    print(f"[PARSE] matched={len(records)} skipped={skipped}")
    return pd.DataFrame(records) if records else pd.DataFrame()

def _summarize_log(df: pd.DataFrame, filename: str) -> str:
    if df.empty:
        return f"Uploaded file '{filename}' had no parseable NovAtel ASCII log lines."
    log_counts = Counter(df["log_name_raw"])
    top_logs   = ", ".join(f"{n}({c})" for n, c in log_counts.most_common(10))
    VALID_TIME = {"FINESTEERING","FINE","FINEBACKUPSTEERING","FINEADJUSTING",
                  "COARSE","COARSESTEERING","COARSEADJUSTING","FREEWHEELING"}
    valid = df[df["time_status"].isin(VALID_TIME) & (df["week"] > 0)]
    parts = [f"Log '{filename}': {len(df)} records, {len(log_counts)} distinct log types.",
             f"Top log types: {top_logs}."]
    if not valid.empty:
        weeks   = sorted(valid["week"].unique().tolist())
        t_start = valid["seconds"].min()
        t_end   = valid["seconds"].max()
        w_start = int(valid.loc[valid["seconds"].idxmin(), "week"])
        w_end   = int(valid.loc[valid["seconds"].idxmax(), "week"])
        dur     = (weeks[-1]-weeks[0])*604800+(t_end-t_start) if len(weeks)>1 else t_end-t_start
        parts.append(
            f"File time range: {gps_to_utc_str(w_start,t_start)} to "
            f"{gps_to_utc_str(w_end,t_end)} (duration {dur:.1f}s = {dur/60:.2f} min)."
        )
    return " ".join(parts)

def ingest_log_file(file_bytes: bytes, filename: str, session_id: str) -> dict:
    t0   = time.time()
    text = file_bytes.decode("utf-8", errors="replace")
    df   = parse_novatel_ascii(text)
    _log_store[session_id] = {"df": df, "summary": _summarize_log(df, filename), "filename": filename}
    
    # Clear docs cache on new file upload to free memory
    _docs_cache.clear()
    
    print(f"[INGEST] {filename} parsed {len(df)} records session={session_id} took={time.time()-t0:.2f}s")
    return {"filename": filename, "records": len(df),
            "log_types": int(df["log_name"].nunique()) if not df.empty else 0,
            "summary": _log_store[session_id]["summary"]}

# ── Per-request session context ───────────────────────────────────────
_current_session: dict[str, str] = {"id": ""}

# ── Field index conversion ────────────────────────────────────────────
def _doc_field_to_body_index(field_index: int) -> int:
    """
    NovAtel docs: fields are 1-based, header = field 1, first body field = field 2.
    Our fields_raw list is 0-based starting from body field 2.
    So: body_index = doc_field_index - 2.
    If already 0-based (field_index < 2), pass through unchanged.
    """
    return (field_index - 2) if field_index >= 2 else field_index

# ── Core analysis functions (plain Python, no agent) ──────────────────

def _safe_parse_hex(fv: str) -> int | None:
    fv = fv.strip()
    try:
        return int(fv, 16)
    except (ValueError, TypeError):
        try:
            return int(fv)
        except (ValueError, TypeError):
            return None

def do_check_bit(session_id: str, log_name: str, field_index: int,
                 bit_position: int, time_from: float = None,
                 time_to: float = None, max_results: int = 1000) -> dict:
    """
    Check which records of log_name have a specific bit set in a specific field.
    field_index is the NovAtel doc field number (1-based, header = field 1).
    """
    entry = _log_store.get(session_id)
    if not entry:
        return {"status": "error", "error": "No log file uploaded."}
    df = entry["df"]

    if log_name.upper().endswith("A") and len(log_name) > 4:
        log_name = log_name[:-1]

    filtered = df[df["log_name"].str.upper() == log_name.upper()]
    if time_from is not None:
        filtered = filtered[filtered["seconds"] >= time_from]
    if time_to is not None:
        filtered = filtered[filtered["seconds"] <= time_to]

    if filtered.empty:
        return {"status": "success", "log_name": log_name, "total_checked": 0,
                "matches_found": 0, "records_with_bit_set": [],
                "note": f"No {log_name} records found in the file."}

    body_index = _doc_field_to_body_index(field_index)
    print(f"[CHECK_BIT] log={log_name} doc_field={field_index} body_index={body_index} bit={bit_position} mask={hex(1 << bit_position)}")

    # Sample first record to confirm we're reading the right field
    sample_row = filtered.iloc[0]
    sample_fields = [f.strip() for f in sample_row.get("fields_raw", "").split(",")]
    print(f"[CHECK_BIT] first record has {len(sample_fields)} body fields")
    print(f"[CHECK_BIT] field[{body_index}] = '{sample_fields[body_index] if body_index < len(sample_fields) else 'OUT OF RANGE'}'")

    mask    = 1 << bit_position
    matches = []
    errors  = 0
    checked = 0

    for _, row in filtered.iterrows():
        fields = [f.strip() for f in row.get("fields_raw", "").split(",")]
        if body_index >= len(fields):
            errors += 1
            continue
        val = _safe_parse_hex(fields[body_index])
        if val is None:
            errors += 1
            continue
        checked += 1
        if val & mask:
            if len(matches) < max_results:
                matches.append({
                    "utc_time":        row.get("utc_time", ""),
                    "gps_week":        int(row["week"]),
                    "gps_seconds":     float(row["seconds"]),
                    "field_value_hex": hex(val),
                })

    print(f"[CHECK_BIT] total={len(filtered)} checked={checked} matches={len(matches)} errors={errors}")
    return {
        "status":           "success",
        "log_name":         log_name,
        "doc_field_index":  field_index,
        "body_index_used":  body_index,
        "bit_position":     bit_position,
        "bit_mask_hex":     hex(mask),
        "total_checked":    checked,
        "matches_found":    len(matches),
        "parse_errors":     errors,
        "records_with_bit_set": matches,
    }

def do_analyze_field(session_id: str, log_name: str, field_index: int) -> dict:
    """Compute min/max/avg for a numeric field. field_index is the NovAtel doc field number."""
    entry = _log_store.get(session_id)
    if not entry:
        return {"status": "error", "error": "No log file uploaded."}
    df = entry["df"]

    if log_name.upper().endswith("A") and len(log_name) > 4:
        log_name = log_name[:-1]

    filtered = df[df["log_name"].str.upper() == log_name.upper()]
    if filtered.empty:
        return {"status": "error", "error": f"No {log_name} records found."}

    body_index = _doc_field_to_body_index(field_index)
    print(f"[ANALYZE] log={log_name} doc_field={field_index} body_index={body_index}")

    values, recs = [], []
    for _, row in filtered.iterrows():
        fields = [f.strip() for f in row.get("fields_raw", "").split(",")]
        if body_index >= len(fields):
            continue
        try:
            val = float(fields[body_index])
            values.append(val)
            recs.append({"value": val, "utc_time": row.get("utc_time",""),
                         "seconds": row["seconds"], "week": int(row["week"])})
        except (ValueError, TypeError):
            continue

    if not values:
        return {"status": "error", "error": f"No numeric values at field {field_index} of {log_name}."}

    min_val = min(values)
    max_val = max(values)
    return {
        "status": "success", "log_name": log_name,
        "doc_field_index": field_index, "body_index_used": body_index,
        "total_records": len(filtered), "valid_values": len(values),
        "min_value": min_val, "max_value": max_val,
        "average_value": sum(values) / len(values),
        "range": max_val - min_val,
        "min_occurred_at": next(r for r in recs if r["value"] == min_val),
        "max_occurred_at": next(r for r in recs if r["value"] == max_val),
    }

def do_summarize_log(session_id: str, log_name: str,
                     question: str, limit: int = 50) -> dict:
    """
    Return records from a log and let the LLM summarize them in context of the question.
    Used for status/detection logs where raw field arrays need interpretation.
    Fully generic - works for any log type.
    Limited to prevent token overflow.
    """
    entry = _log_store.get(session_id)
    if not entry:
        return {"status": "error", "error": "No log file uploaded."}
    df = entry["df"]

    norm = log_name[:-1] if (log_name.upper().endswith("A") and len(log_name) > 4) else log_name
    filtered = df[df["log_name"].str.upper() == norm.upper()]
    if filtered.empty:
        return {"status": "error", "error": f"No {log_name} records found in the file."}

    total = len(filtered)
    # Sample evenly across the file for better representation
    # Reduced limit to prevent token overflow
    if total > limit:
        step = total // limit
        sample = filtered.iloc[::step].head(limit)
    else:
        sample = filtered
    
    records = []
    for _, row in sample.iterrows():
        fields_raw = row.get("fields_raw", "")
        # Limit field array size to prevent massive payloads
        fields_parsed = [f.strip() for f in fields_raw.split(",")]
        # For logs with many fields (like TRACKSTAT), truncate to first 20 fields
        if len(fields_parsed) > 20:
            fields_parsed = fields_parsed[:20] + [f"... ({len(fields_parsed) - 20} more fields)"]
        
        records.append({
            "utc_time":      row.get("utc_time", ""),
            "week":          int(row["week"]),
            "seconds":       float(row["seconds"]),
            "rx_status":     row.get("rx_status", ""),
            "fields_parsed": fields_parsed,
        })

    return {
        "status":        "success",
        "log_name":      log_name,
        "total_records": total,
        "sample_size":   len(records),
        "records":       records,
        "question":      question,
    }

# ── Subject extractor ────────────────────────────────────────────────
_SUBJECT_PROMPT = """Extract the core technical subject from the user's question. 
Return ONLY 1-3 words that name the phenomenon, measurement, or event being asked about.
No sentences. No explanation. Just the subject words.

Examples:
  "do we have spoofing in this file"  → spoofing detection
  "identify interference events"       → interference detection
  "what is the maximum height"         → height maximum
  "show me jamming records"            → jamming detection
  "any position errors"                → position error
  "when did spoofing occur"            → spoofing detection
  "check for jamming"                  → jamming detection
  "antenna status changes"             → antenna status
  "tracking issues"                    → tracking status
  "signal quality problems"            → signal quality

Question: {question}"""

def extract_subject(question: str) -> str:
    """Extract the core subject from a user question - fully generic."""
    try:
        response = get_llm().invoke([HumanMessage(
            content=_SUBJECT_PROMPT.format(question=question)
        )])
        subject = response.content.strip().lower()
        subject = subject.split("\n")[0].strip("\"'.,")
        print(f"[SUBJECT] '{question}' → '{subject}'")
        return subject
    except Exception as e:
        print(f"[SUBJECT] fallback: {e}")
        # Generic fallback - extract meaningful words
        filler = {"identify","find","show","list","check","detect","any","all",
                  "in","this","file","the","is","are","there","do","we","have",
                  "me","a","an","of","for","from","what","how","when","records",
                  "did","was","were","does","can","could","would","should"}
        words  = [w for w in question.lower().split() if w not in filler]
        return " ".join(words[:3]) or question

# ── KB search (pure Python, no agent) ────────────────────────────────
def kb_search(query: str, max_results: int = MAX_RESULTS) -> list[dict]:
    t0 = time.time()
    try:
        response = get_kb_client().retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
            retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": max_results}},
        )
        elements = []
        for result in response.get("retrievalResults", []):
            content  = result.get("content", {}).get("text", "")
            metadata = result.get("metadata", {})
            elements.append({
                "element_id":       metadata.get("element_id", ""),
                "content_markdown": content,
                "page_number":      int(metadata.get("page_number", 0)),
                "score":            result.get("score", 0.0),
                "source_uri":       metadata.get("x-amz-bedrock-kb-source-uri", ""),
                "csv_source_uri":   metadata.get("csv_source_uri", ""),
            })
        print(f"[KB] query='{query}' results={len(elements)} took={time.time()-t0:.2f}s")
        return elements
    except Exception as e:
        print(f"[KB] error: {e}")
        return []


# ── NovAtel live docs fetcher ─────────────────────────────────────────
import urllib.request
from html.parser import HTMLParser

class _TableTextExtractor(HTMLParser):
    """Extract plain text from HTML, preserving table row structure."""
    def __init__(self):
        super().__init__()
        self.rows: list[str] = []
        self._cell_texts: list[str] = []
        self._current: list[str] = []
        self._in_cell = False
        self._skip_tags = {"script", "style", "nav", "header", "footer"}
        self._skip = False
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._skip_tags:
            self._skip = True
            self._skip_depth = 0
        if self._skip:
            self._skip_depth += 1
            return
        if tag in ("td", "th"):
            self._in_cell = True
            self._current = []

    def handle_endtag(self, tag):
        if self._skip:
            self._skip_depth -= 1
            if self._skip_depth <= 0:
                self._skip = False
            return
        if tag in ("td", "th"):
            self._in_cell = False
            self._cell_texts.append(" ".join(self._current).strip())
        if tag == "tr":
            if self._cell_texts:
                self.rows.append(" | ".join(self._cell_texts))
            self._cell_texts = []

    def handle_data(self, data):
        if self._skip or not self._in_cell:
            return
        text = data.strip()
        if text:
            self._current.append(text)

# Cache for live docs (session-scoped, cleared on new file upload)
_docs_cache: dict[str, str] = {}

def fetch_novatel_log_docs(log_name: str) -> str:
    """
    Fetch the live NovAtel OEM7 documentation page for a log and return
    the field/bit table rows as plain text. Works for any log name.
    Returns empty string on failure (network unavailable etc).
    Cached per session to avoid repeated web requests.
    """
    # Check cache first
    if log_name in _docs_cache:
        print(f"[DOCS] Using cached docs for {log_name}")
        return _docs_cache[log_name]
    
    # Strip trailing A (RXSTATUSA → RXSTATUS) for the URL
    url_name = log_name[:-1] if (log_name.upper().endswith("A") and len(log_name) > 4) else log_name
    url = f"https://docs.novatel.com/OEM7/Content/Logs/{url_name}.htm"
    t0  = time.time()
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 NovAtelAgent/1.0"}
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        parser = _TableTextExtractor()
        parser.feed(html)

        # Keep only rows that look like field/bit table rows (contain numbers and descriptions)
        useful = [r for r in parser.rows if r.strip() and len(r) > 10]
        result = "\n".join(useful)
        
        # Cache the result
        _docs_cache[log_name] = result
        
        print(f"[DOCS] Fetched {url_name} docs: {len(useful)} table rows took={time.time()-t0:.2f}s")
        return result
    except Exception as e:
        print(f"[DOCS] Could not fetch {url}: {e}")
        return ""

# ── Param extraction (one LLM call → structured JSON) ─────────────────
_EXTRACT_PROMPT = """You are a NovAtel OEM7 documentation parser with expertise in all receiver logs.

You are given the OFFICIAL NovAtel documentation table for the log, plus supplementary KB excerpts.
Extract the exact field and bit that answers the user's question.
Output ONLY a single JSON object. No explanation. No markdown. No extra text.

JSON fields:
  log_name      — use exactly: {log_name}
  field_index   — the field NUMBER from the left column of the NovAtel field table (integer, 1-based, header=field 1)
  bit_position  — bit number (0=LSB) from the bit table. Use null if not a flag/event question.
  question_type — classify the question as exactly one of:
    "bit_check"   : looking for whether a specific bit is set (yes/no detection, e.g. jamming detected, spoofing active, antenna status)
    "numeric_stat": asking for min/max/average/range of a MEASURED physical value (e.g. height in metres, speed in m/s, temperature)
    "raw_listing" : everything else — showing records, identifying events from a status/detection log, listing what occurred

CRITICAL RULES FOR ACCURACY:
1. ALWAYS prefer the OFFICIAL DOCS over KB excerpts — the docs are the authoritative source.
2. For detection/status questions:
   - Look for rows with "Detected", "Detection Status", "Status", or the exact phenomenon name in the Description column
   - IGNORE rows with "Calibration", "Required", "Priority Mask", "Set Mask", "Clear Mask", "Reserved"
   - Choose the row that directly answers the user's question
3. field_index is the exact integer in the leftmost column of the field table (1-based).
4. For bit questions, ALWAYS provide bit_position as an integer, never null.
5. Read the ENTIRE documentation table carefully before choosing - don't pick the first match.
6. Respond with ONLY valid JSON. No explanation, no markdown fences.

Log: {log_name}
Question: {question}

OFFICIAL NOVATEL DOCUMENTATION (field and bit tables for {log_name}):
{official_docs}

SUPPLEMENTARY KB EXCERPTS:
{kb_content}"""


def extract_log_params(question: str, kb_elements: list[dict],
                       log_name: str = None, official_docs: str = "",
                       top_n: int = 8) -> dict | None:
    """
    One LLM call to extract log/field/bit params.
    Uses official NovAtel docs (fetched live) as primary source,
    KB excerpts as supplementary context.
    Fully generic - works for ANY log type and ANY question.
    """
    for i, el in enumerate(kb_elements[:top_n]):
        print(f"[KB_HIT] #{i+1} score={el['score']:.3f} preview={el['content_markdown'][:120].replace(chr(10),' ')}")

    kb_content = "\n\n---\n\n".join(
        f"[Score: {el['score']:.3f}]\n{el['content_markdown']}"
        for el in kb_elements[:top_n]
    ) if kb_elements else "No KB results."

    # Use more focused docs - just field tables, not verbose descriptions
    # Look for table-like content (lines with | separators)
    if official_docs:
        doc_lines = official_docs.split('\n')
        table_lines = [line for line in doc_lines if '|' in line and len(line) > 20]
        docs_excerpt = '\n'.join(table_lines[:150])  # First 150 table rows
        if not docs_excerpt:
            docs_excerpt = official_docs[:4000]  # Fallback to first 4000 chars
    else:
        docs_excerpt = "Not available — rely on KB excerpts."

    prompt = _EXTRACT_PROMPT.format(
        log_name=log_name or "unknown",
        question=question,
        official_docs=docs_excerpt,
        kb_content=kb_content,
    )

    response = None
    try:
        response = get_llm().invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        params = json.loads(raw)
        if log_name and not params.get("log_name"):
            params["log_name"] = log_name
        
        print(f"[EXTRACT] ✓ log={params.get('log_name')} field={params.get('field_index')} "
              f"bit={params.get('bit_position')} type={params.get('question_type')}")
        return params
    except Exception as e:
        raw_out = response.content[:300] if response else "no response"
        print(f"[EXTRACT] ✗ failed: {e} — raw='{raw_out}'")
        return None



# ── Answer formatter (one LLM call) ───────────────────────────────────
_FORMAT_PROMPT = """You are a NovAtel OEM7 log analyst. Answer the user's question using the tool result and the log documentation below.

User question: {question}

Log documentation (field definitions):
{log_docs}

Tool result (JSON):
{tool_result}

FORMATTING GUIDELINES:
- Use **bold** for key findings, log names, and important values (sparingly)
- Use clear paragraph breaks for readability
- Avoid excessive markdown (no ###, ---, ***, ===)
- Use simple bullet points (• or -) for lists
- Keep tables simple with | format, only when needed
- Structure: Brief summary → Details → Conclusion

Instructions:
- For bit_check results: 
  * State the exact matches_found count in bold
  * If matches_found > 0: List ALL utc_time timestamps in chronological order
  * If matches_found = 0: Say "No [subject] detected in this file."
  * Include the field and bit that was checked
- For numeric_stat results: 
  * State min, max, average clearly with units from the documentation
  * Include when min/max occurred (timestamps)
- For raw_listing/summarize results: 
  * Use the field definitions to interpret the fields_parsed arrays
  * Summarize patterns, frequencies, and key findings
  * Do not dump raw field arrays
- Use clear paragraph breaks between sections
- Be precise and factual - never invent data not present in the tool result
- If parse_errors > 0, mention that some records could not be parsed

Example good format:
"**62 spoofing events detected** in the RXSTATUS log (field 4, bit 9).

The events occurred between 2023-05-23 17:41:57 and 17:44:38 UTC, spanning approximately 3 minutes. All 219 RXSTATUS records were checked.

**Event timestamps:**
• 2023-05-23T17:41:57.000Z
• 2023-05-23T17:41:58.000Z
[...]

This indicates active spoofing detection during the observation period."
"""


def format_answer(question: str, tool_result: dict, log_docs: str = "") -> str:
    """Format the answer with optimized token usage for faster LLM response."""
    t0 = time.time()
    
    # Fast path: for bit_check results, format directly without LLM call
    if tool_result.get("status") == "success" and "records_with_bit_set" in tool_result:
        matches = tool_result.get("matches_found", 0)
        log_name = tool_result.get("log_name", "")
        field_idx = tool_result.get("doc_field_index", "")
        bit_pos = tool_result.get("bit_position", "")
        
        if matches == 0:
            result = f"**No events detected** in log `{log_name}` (field {field_idx}, bit {bit_pos}).\n\nChecked {tool_result.get('total_checked', 0)} records."
        else:
            records = tool_result["records_with_bit_set"]
            # Format timestamps in a compact list
            timestamps = [r["utc_time"] for r in records if r.get("utc_time")]
            
            if len(timestamps) <= 50:
                ts_list = "\n".join(f"- {ts}" for ts in timestamps)
            else:
                # For many matches, show first 25 and last 25
                ts_list = "\n".join(f"- {ts}" for ts in timestamps[:25])
                ts_list += f"\n\n... ({len(timestamps) - 50} more timestamps) ...\n\n"
                ts_list += "\n".join(f"- {ts}" for ts in timestamps[-25:])
            
            result = (
                f"**{matches} event(s) detected** in log `{log_name}` "
                f"(field {field_idx}, bit {bit_pos}).\n\n"
                f"**Timestamps (UTC):**\n{ts_list}\n\n"
                f"Checked {tool_result.get('total_checked', 0)} records total."
            )
        
        if tool_result.get("parse_errors", 0) > 0:
            result += f"\n\n*Note: {tool_result['parse_errors']} records could not be parsed.*"
        
        print(f"[FORMAT] fast path took={time.time()-t0:.2f}s")
        return result
    
    # Fast path: for numeric_stat results, format directly without LLM call
    if tool_result.get("status") == "success" and "min_value" in tool_result:
        log_name = tool_result.get("log_name", "")
        field_idx = tool_result.get("doc_field_index", "")
        min_val = tool_result.get("min_value")
        max_val = tool_result.get("max_value")
        avg_val = tool_result.get("average_value")
        range_val = tool_result.get("range")
        total = tool_result.get("total_records", 0)
        valid = tool_result.get("valid_values", 0)
        
        min_time = tool_result.get("min_occurred_at", {}).get("utc_time", "")
        max_time = tool_result.get("max_occurred_at", {}).get("utc_time", "")
        
        # Determine unit from log name (common patterns)
        unit = ""
        if "height" in question.lower() or log_name.upper() == "BESTPOS":
            unit = " m"  # metres for height
        elif "vel" in log_name.lower() or "speed" in question.lower():
            unit = " m/s"
        elif "temp" in log_name.lower():
            unit = "°C"
        
        result = (
            f"**Statistics for {log_name} field {field_idx}:**\n\n"
            f"| Metric | Value | Timestamp (UTC) |\n"
            f"|--------|-------|----------------|\n"
            f"| **Minimum** | {min_val:.3f}{unit} | {min_time} |\n"
            f"| **Maximum** | {max_val:.3f}{unit} | {max_time} |\n"
            f"| **Average** | {avg_val:.3f}{unit} | - |\n"
            f"| **Range** | {range_val:.3f}{unit} | - |\n\n"
            f"Analyzed {valid} valid values from {total} total records."
        )
        
        print(f"[FORMAT] fast path (numeric) took={time.time()-t0:.2f}s")
        return result
    
    # Slow path: use LLM only for raw_listing (complex interpretation needed)
    # But with aggressive token limiting to prevent overflow
    optimized_result = tool_result.copy()
    
    # For raw_listing with many records, provide summary stats instead of all records
    if "records" in optimized_result and len(optimized_result.get("records", [])) > 10:
        records = optimized_result["records"]
        # Keep only first 5 and last 5 records for context
        optimized_result["records"] = records[:5] + records[-5:]
        optimized_result["_sampling_note"] = f"Showing first 5 and last 5 of {len(records)} sampled records"
    
    # Minimal docs - just enough for field interpretation
    trimmed_docs = log_docs[:800] if log_docs else "Not available."
    
    prompt = _FORMAT_PROMPT.format(
        question=question,
        tool_result=json.dumps(optimized_result, indent=2),
        log_docs=trimmed_docs,
    )
    
    try:
        result = get_llm().invoke([HumanMessage(content=prompt)]).content.strip()
        print(f"[FORMAT] LLM path took={time.time()-t0:.2f}s")
        return result
    except Exception as e:
        print(f"[FORMAT] failed: {e}")
        # Fallback: return a simple summary without LLM
        log_name = tool_result.get("log_name", "unknown")
        total = tool_result.get("total_records", 0)
        sample = tool_result.get("sample_size", 0)
        return (
            f"**Analysis of {log_name} log:**\n\n"
            f"Found {total} records in the file (sampled {sample} for analysis).\n\n"
            f"The log contains complex multi-field data. "
            f"For specific analysis, try asking about:\n"
            f"- Specific signal strength values\n"
            f"- Tracking status changes\n"
            f"- Time ranges or specific events\n\n"
            f"*Note: Full analysis unavailable due to data size.*"
        )

# ── Log name selector (uses LLM knowledge, not KB) ────────────────────
_LOG_SELECT_PROMPT = """You are a NovAtel OEM7 expert. Identify which log to query to answer the user's question.

Available logs in the uploaded file:
{available_logs}

User question: {question}

NOVATEL LOG SELECTION GUIDE (common patterns):
  - "Jamming" or "jammer" (receiver status bit)  → RXSTATUS
  - "Interference" or "RF interference" (spectrum analysis) → ITDETECTSTATUS
  - Spoofing detection                           → RXSTATUS
  - Position / coordinates / height / lat/lon    → BESTPOS
  - Velocity / speed / heading                   → BESTVEL
  - Satellite tracking / signal quality          → TRACKSTAT
  - Receiver status / errors / flags             → RXSTATUS
  - Time / clock / PPS                           → CLOCKSTEERING, TIMESYNC
  - Differential corrections / RTK               → PSRDIFF, RTCADATA
  - Ionosphere / troposphere                     → IONUTC, TROPMODEL
  - Almanac / ephemeris                          → ALMANAC, GPSEPHEM
  - Hardware / antenna / temperature             → HWMONITOR, ANTENNAPOWER

CRITICAL DISTINCTION:
  - "Jamming" = receiver's internal jamming detection flag → use RXSTATUS
  - "Interference" = detailed RF spectrum analysis → use ITDETECTSTATUS
  - If user asks about "jamming", they want RXSTATUS (not ITDETECTSTATUS)
  - If user asks about "interference", they want ITDETECTSTATUS (not RXSTATUS)

RULES:
1. Choose the log from the available list that best matches the question
2. Pay attention to the EXACT words used - "jamming" ≠ "interference"
3. If unsure, prefer RXSTATUS for status/detection questions, BESTPOS for position questions
4. Return ONLY the log name exactly as it appears in the list
5. No explanation, no extra text"""

def extract_log_name(question: str, available_logs: list[str]) -> str | None:
    """
    Use the LLM's own NovAtel knowledge to pick the right log.
    Constrained to logs actually present in the file.
    Fully generic - works for any log type and any question.
    """
    logs_str = "\n".join(f"- {l}" for l in available_logs)
    
    # Add hint for jamming vs interference distinction
    q_lower = question.lower()
    hint = ""
    if "jam" in q_lower and "interfere" not in q_lower:
        hint = "\n\nHINT: User asked about 'jamming' (not 'interference'), so prefer RXSTATUS over ITDETECTSTATUS."
    elif "interfere" in q_lower and "jam" not in q_lower:
        hint = "\n\nHINT: User asked about 'interference' (not 'jamming'), so prefer ITDETECTSTATUS over RXSTATUS."
    
    prompt = _LOG_SELECT_PROMPT.format(question=question, available_logs=logs_str) + hint
    
    try:
        response = get_llm().invoke([HumanMessage(content=prompt)])
        log_name = response.content.strip().upper().split()[0].strip(".,\"'-")
        print(f"[LOG_NAME] selected: {log_name} for question: '{question}'")
        return log_name
    except Exception as e:
        print(f"[LOG_NAME] failed: {e}")
        return None
        return None


# ── Log pipeline (no agent loop) ──────────────────────────────────────
def run_log_pipeline(question: str, session_id: str) -> str:
    """
    Pipeline for log file questions:
      1. LLM picks the right log from file inventory (uses training knowledge, no KB)
      2. Fetch live NovAtel docs for that log (authoritative field/bit table)
      3. LLM extracts exact field + bit from real docs (+ KB as supplementary)
      4. Python runs the analysis
      5. Format the answer
    """
    print(f"[PIPELINE] question={question!r}")
    
    # Determine question intent for dynamic status messages
    q_lower = question.lower()
    intent = "your question"
    if "minimum" in q_lower or "min" in q_lower or "lowest" in q_lower:
        intent = "minimum value query"
    elif "maximum" in q_lower or "max" in q_lower or "highest" in q_lower:
        intent = "maximum value query"
    elif "average" in q_lower or "mean" in q_lower:
        intent = "average calculation"
    elif "range" in q_lower:
        intent = "range analysis"
    elif "scintillation" in q_lower:
        intent = "scintillation detection"
    elif "jamming" in q_lower or "interference" in q_lower:
        intent = "interference analysis"
    elif "spoofing" in q_lower:
        intent = "spoofing detection"
    elif "error" in q_lower or "issue" in q_lower or "problem" in q_lower:
        intent = "error analysis"
    elif "status" in q_lower or "health" in q_lower:
        intent = "status check"
    elif "satellite" in q_lower or "sat" in q_lower:
        intent = "satellite analysis"
    elif "signal" in q_lower or "c/no" in q_lower or "cno" in q_lower:
        intent = "signal quality analysis"
    
    set_status(session_id, f"Analyzing {intent}...")

    # Step 1: get file log inventory
    entry = _log_store.get(session_id)
    if not entry or entry["df"].empty:
        clear_status(session_id)
        return "No log file is loaded."
    available_logs = entry["df"]["log_name_raw"].unique().tolist()
    print(f"[PIPELINE] {len(available_logs)} log types in file")

    # Step 2: LLM picks the log (from its own NovAtel knowledge, constrained to file inventory)
    set_status(session_id, "Determining which log type to analyze...")
    log_name = extract_log_name(question, available_logs)
    if not log_name:
        clear_status(session_id)
        return "Could not determine which log to use for this question."

    # Step 3: Determine question type early to optimize subsequent steps
    q_lower = question.lower()
    is_numeric = any(kw in q_lower for kw in ("min", "max", "average", "mean", "range", "highest", "lowest"))
    
    # Step 4: Parallel fetch of docs and KB search (if needed)
    set_status(session_id, f"Loading {log_name} specifications...")
    import concurrent.futures
    
    official_docs = ""
    kb_elements = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Always fetch docs
        docs_future = executor.submit(fetch_novatel_log_docs, log_name)
        
        # Only fetch KB if needed
        kb_future = None
        if not is_numeric:
            set_status(session_id, f"Searching {log_name} field definitions...")
            subject = extract_subject(question)
            kb_query = f"{log_name} {subject} field bit definition"
            kb_future = executor.submit(kb_search, kb_query, 15)
        
        # Wait for results
        official_docs = docs_future.result()
        if kb_future:
            kb_elements = kb_future.result()
        else:
            print(f"[PIPELINE] Skipping KB search for numeric query")
    
    if official_docs:
        print(f"[PIPELINE] live docs fetched for {log_name}: {len(official_docs)} chars")
    else:
        print(f"[PIPELINE] live docs unavailable for {log_name}, using KB only")

    # Step 5: extract exact field + bit — real docs are primary, KB is fallback
    set_status(session_id, f"Identifying {log_name} field parameters...")
    params = extract_log_params(
        question, kb_elements,
        log_name=log_name,
        official_docs=official_docs,
        top_n=15,
    )
    if not params or "log_name" not in params:
        # Be more helpful - suggest what the user can do
        available_logs = entry["df"]["log_name_raw"].unique().tolist() if entry else []
        
        helpful_msg = (
            "I'm having trouble identifying the exact field to analyze from the documentation. "
            "Let me help you find what you're looking for:\n\n"
        )
        
        if available_logs:
            log_list = ", ".join(available_logs[:8])
            if len(available_logs) > 8:
                log_list += f", and {len(available_logs) - 8} more"
            helpful_msg += f"**Your file contains:** {log_list}\n\n"
        
        helpful_msg += (
            "**Try these approaches:**\n"
            "- Be more specific: 'analyze BESTPOS height field' or 'check RXSTATUS jamming bit'\n"
            "- Ask about common metrics: 'minimum satellites tracked', 'position accuracy', 'signal quality'\n"
            "- List what's available: 'list all logs' or 'show time range'\n\n"
            "What specific information are you looking for?"
        )
        
        return helpful_msg

    log_name      = params.get("log_name")
    field_index   = params.get("field_index")
    bit_position  = params.get("bit_position")
    question_type = params.get("question_type", "raw_listing")

    # Validation
    if not log_name or field_index is None:
        subject = extract_subject(question) if not is_numeric else "the requested field"
        
        # Try to provide helpful guidance instead of just saying "not found"
        available_logs = entry["df"]["log_name_raw"].unique().tolist()
        log_list = ", ".join(available_logs[:10])
        if len(available_logs) > 10:
            log_list += f", and {len(available_logs) - 10} more"
        
        return (
            f"I couldn't identify the exact field for '{subject}' in the available documentation. "
            f"This could mean:\n\n"
            f"1. The field name or log type needs to be more specific\n"
            f"2. The information might be in a different log than expected\n"
            f"3. The documentation for this specific field may not be in the knowledge base\n\n"
            f"**Your file contains these log types:** {log_list}\n\n"
            f"**Suggestions:**\n"
            f"- Try asking about a specific log type (e.g., 'BESTPOS height field' or 'RXSTATUS jamming bit')\n"
            f"- Check if the data you're looking for is in a different log\n"
            f"- Ask 'list all logs' to see what's available in your file\n\n"
            f"I'm here to help - feel free to rephrase your question!"
        )

    try:
        field_index = int(field_index)
        if field_index < 1:
            return f"Invalid field index: {field_index}. Field indices must be >= 1."
    except (TypeError, ValueError):
        return f"Invalid field index from documentation: {field_index!r}"

    print(f"[PIPELINE] → log={log_name} field={field_index} bit={bit_position} type={question_type}")

    # Step 6: Python analysis with proper error handling
    set_status(session_id, f"Computing {log_name} statistics...")
    try:
        if question_type == "bit_check" and bit_position is not None:
            try:
                bit_pos = int(bit_position)
                if bit_pos < 0 or bit_pos > 31:
                    clear_status(session_id)
                    return f"Invalid bit position: {bit_pos}. Must be between 0 and 31."
                result = do_check_bit(session_id, log_name, field_index, bit_pos)
            except (TypeError, ValueError):
                clear_status(session_id)
                return f"Invalid bit position from documentation: {bit_position!r}"
        elif question_type == "numeric_stat":
            result = do_analyze_field(session_id, log_name, field_index)
        else:
            # raw_listing: fetch records and let the formatter interpret them with docs context
            result = do_summarize_log(session_id, log_name, question)

        if result.get("status") == "error":
            clear_status(session_id)
            return f"Analysis error: {result['error']}"

        # Step 7: format answer — pass live docs so formatter can interpret field arrays
        set_status(session_id, "Formatting results...")
        answer = format_answer(question, result, log_docs=official_docs)
        clear_status(session_id)
        return answer
    
    except Exception as e:
        print(f"[PIPELINE] Unexpected error: {e}")
        clear_status(session_id)
        return f"An error occurred while analyzing the log: {str(e)}"

# ── Documentation agent (pure KB, no log tools) ───────────────────────
@tool
def kb_retriever(query: str, max_results: int = MAX_RESULTS) -> dict:
    """Search the NovAtel OEM7 documentation knowledge base."""
    elements = kb_search(query, max_results)
    payload  = {"status": "success", "elements": elements, "total_found": len(elements)}
    _tool_call_log.append({"tool": "kb_retriever", "result": payload})
    return payload

@tool
def context_expander(source_uri: str, element_ids: list = None,
                     page_numbers: list = None,
                     expansion_pages: int = EXPANSION_PAGES) -> dict:
    """Expand context around specific KB elements. Pass csv_source_uri from kb_retriever."""
    t0 = time.time()
    try:
        source_uri = _resolve_data_uri(source_uri)
        df = _download_and_parse(source_uri)
        target_pages: set[int] = set()
        if element_ids:
            for eid in element_ids:
                rows = df[df["element_id"] == eid]
                if not rows.empty:
                    target_pages.update(rows["page_number"].tolist())
        if page_numbers:
            target_pages.update(page_numbers)
        all_pages = {i for p in target_pages
                     for i in range(p - expansion_pages, p + expansion_pages + 1) if i >= 0}
        filtered = df[df["page_number"].isin(all_pages)].copy()
        for col in ["page_number"]:
            if col in filtered.columns:
                filtered[col] = pd.to_numeric(filtered[col], errors="coerce").fillna(0).astype(int)
        for col in [c for c in filtered.columns if c != "page_number"]:
            filtered[col] = filtered[col].fillna("").astype(str)
        available_cols = [c for c in _LLM_COLS if c in filtered.columns]
        slim = filtered[available_cols].to_dict("records") if available_cols else []
        print(f"[EXPANDER] pages={sorted(all_pages)} elements={len(slim)} took={time.time()-t0:.2f}s")
        return {"status": "success", "elements": slim, "total_found": len(slim)}
    except Exception as e:
        print(f"[EXPANDER] error: {e}")
        return {"status": "error", "error": str(e), "elements": [], "total_found": 0}

_DOC_AGENT_PROMPT = """You are a NovAtel OEM7 documentation assistant with deep expertise in GNSS receivers.

BUDGET: 1 kb_retriever call + 1 optional context_expander.

INSTRUCTIONS:
1. Call kb_retriever once with the user's question (use their exact terminology).
2. If you find relevant information, answer directly and cite the log/command name.
3. Only call context_expander if a result is clearly incomplete (cut-off table/definition).
4. If KB search returns no results, use your GNSS/NovAtel expertise to provide a helpful answer:
   - Explain the concept based on standard GNSS/NovAtel knowledge
   - Suggest related logs or commands that might help
   - Recommend checking the official NovAtel OEM7 documentation website
   - Never just say "not available" - always try to be helpful

CRITICAL FORMATTING RULES:
- Write in plain, clean text with clear paragraph breaks
- For emphasis, use **bold** (the UI will render it properly)
- NEVER use # for headings - just write the heading text in **bold** on its own line
- NEVER use ---, ***, ===, or other separator symbols
- Use simple bullet points with - or • (no nested bullets)
- Keep tables minimal with | format only when truly necessary
- Structure: Brief intro → Main content with clear paragraphs → Conclusion

GOOD EXAMPLE:
"**SPAN Configuration Steps**

To configure SPAN, you need to connect and calibrate the IMU with the GNSS receiver.

**1. Connect the IMU**
Use the CONNECTIMU command to establish communication with your inertial measurement unit.

**2. Set IMU Specifications**
Use the SETIMUSPECS command to define the specific IMU model you're using.

The configuration ensures accurate integration of GNSS and inertial data for precise positioning."

BAD EXAMPLE (avoid):
"# SPAN Configuration
***
## Steps
---
### 1. Connect IMU
Use **CONNECTIMU** command..."

RESPONSE STYLE:
- Be knowledgeable and helpful, not dismissive
- If exact documentation isn't found, provide context from your GNSS expertise
- Suggest alternatives or related topics
- Guide users toward solutions

Do not call kb_retriever twice. Start directly with the answer."""

_doc_agent = None

def get_doc_agent():
    global _doc_agent
    if _doc_agent is None:
        _doc_agent = create_react_agent(
            model=get_llm(),
            tools=[kb_retriever, context_expander],
        )
    return _doc_agent

def run_doc_agent(prompt: str, history: list, session_id: str = "") -> str:
    t0       = time.time()
    
    # Extract key topic from question for initial status
    topic = ""
    if session_id:
        p_lower = prompt.lower()
        topics_map = {
            "span": "SPAN configuration", "imu": "IMU setup", "rtk": "RTK configuration",
            "base station": "base station setup", "rover": "rover configuration",
            "heading": "heading configuration", "alignment": "alignment setup",
            "log": "logging configuration", "port": "port configuration",
            "ethernet": "Ethernet setup", "serial": "serial port setup",
            "wifi": "WiFi configuration", "ntrip": "NTRIP configuration",
            "correction": "correction services", "ppp": "PPP configuration",
            "antenna": "antenna setup", "gnss": "GNSS configuration",
            "gps": "GPS setup", "glonass": "GLONASS setup",
            "galileo": "Galileo setup", "beidou": "BeiDou setup",
            "sbas": "SBAS configuration", "bestpos": "BESTPOS message",
            "trackstat": "TRACKSTAT message", "rxstatus": "receiver status",
            "time": "time configuration", "pps": "PPS output",
            "event": "event markers", "trigger": "trigger setup",
        }
        for keyword, topic_name in topics_map.items():
            if keyword in p_lower:
                topic = topic_name
                break
        if not topic:
            if "configure" in p_lower or "setup" in p_lower or "how to" in p_lower:
                topic = "configuration"
            elif "what is" in p_lower or "explain" in p_lower:
                topic = "documentation"
            else:
                topic = "information"
        
        # ── Set an immediate, visible status so the UI shows something right away ──
        set_status(session_id, f"Searching knowledge base for {topic}..." if topic else "Searching knowledge base...")
    
    messages = [SystemMessage(content=_DOC_AGENT_PROMPT)] + history + [HumanMessage(content=prompt)]
    
    try:
        final_answer = ""
        agent = get_doc_agent()
        kb_search_done = False
        
        for event in agent.stream(
            {"messages": messages},
            config={"recursion_limit": 8},
            stream_mode="values"
        ):
            if not session_id or "messages" not in event:
                continue

            last_msg = event["messages"][-1]
            msg_type = last_msg.__class__.__name__

            # ── AI is about to call a tool ────────────────────────────
            if msg_type == "AIMessage":
                tool_calls = getattr(last_msg, "tool_calls", None) or []
                if tool_calls:
                    tool_name = tool_calls[0].get("name", "") if isinstance(tool_calls[0], dict) else getattr(tool_calls[0], "name", "")
                    if tool_name == "kb_retriever":
                        label = f"Searching {topic} documentation..." if topic else "Searching documentation..."
                        set_status(session_id, label)
                    elif tool_name == "context_expander":
                        set_status(session_id, "Expanding context with related documentation...")
                else:
                    # Final AI message (no tool call) — formulating response
                    if last_msg.content:
                        set_status(session_id, "Formulating response...")
                        final_answer = last_msg.content

            # ── Tool result returned ──────────────────────────────────
            elif msg_type == "ToolMessage":
                tool_name = getattr(last_msg, "name", "")
                if tool_name == "kb_retriever":
                    if not kb_search_done:
                        kb_search_done = True
                        set_status(session_id, "Analysing retrieved documentation...")
                elif tool_name == "context_expander":
                    set_status(session_id, "Processing expanded context...")
        
        if not final_answer:
            # Fallback if streaming didn't capture the answer
            set_status(session_id, "Generating answer...")
            result = agent.invoke(
                {"messages": messages},
                config={"recursion_limit": 8},
            )
            final_answer = result["messages"][-1].content
        
        print(f"[DOC_AGENT] took={time.time()-t0:.2f}s")
        if session_id:
            clear_status(session_id)
        return final_answer
        
    except Exception as e:
        print(f"[DOC_AGENT] error: {e}")
        if session_id:
            clear_status(session_id)
        raise

# ── Direct handlers (fully deterministic, zero LLM) ───────────────────
_VALID_TIME_STATUSES = {
    "FINESTEERING","FINE","FINEBACKUPSTEERING","FINEADJUSTING",
    "COARSE","COARSESTEERING","COARSEADJUSTING","FREEWHEELING",
}
def _direct_data_gap(log_entry: dict, gap_threshold: float = 2.0) -> dict:
    """
    Check for time gaps in the file by analysing GPS timestamps across all records.
    Uses the most frequently logged type as the reference signal.
    gap_threshold: seconds — gaps larger than this are reported.
    """
    df = log_entry["df"]
    VALID_TIME = {"FINESTEERING","FINE","FINEBACKUPSTEERING","FINEADJUSTING",
                  "COARSE","COARSESTEERING","COARSEADJUSTING","FREEWHEELING"}

    valid = df[df["time_status"].isin(VALID_TIME) & (df["week"] > 0)].copy()
    if valid.empty:
        return {"result": "No records with valid GPS time found — cannot check for gaps."}

    # Use the most frequent log type as the timing reference (most likely to be continuous)
    most_common_log = valid["log_name_raw"].value_counts().idxmax()
    ref = valid[valid["log_name_raw"] == most_common_log].sort_values("seconds").copy()

    # Convert to absolute GPS seconds (handles week rollovers)
    ref["abs_seconds"] = ref["week"] * 604800 + ref["seconds"]
    ref = ref.sort_values("abs_seconds").reset_index(drop=True)

    # Compute interval between consecutive records
    ref["delta"] = ref["abs_seconds"].diff()

    # Estimate nominal logging rate from median interval
    median_interval = ref["delta"].median()

    # Find gaps — anything more than gap_threshold × nominal interval
    effective_threshold = max(gap_threshold, median_interval * 3)
    gaps = ref[ref["delta"] > effective_threshold].copy()

    if gaps.empty:
        total_duration = ref["abs_seconds"].iloc[-1] - ref["abs_seconds"].iloc[0]
        return {"result": (
            f"**No data gaps detected** in `{log_entry['filename']}`.\n\n"
            f"Reference log: `{most_common_log}` ({len(ref)} records)\n"
            f"Nominal interval: {median_interval:.3f}s | "
            f"Total duration: {total_duration:.1f}s ({total_duration/60:.2f} min)\n"
            f"Data appears continuous throughout."
        )}

    # Format gap details
    lines = []
    for _, row in gaps.iterrows():
        gap_sec = row["delta"]
        gap_start = gps_to_utc_str(int(row["week"]), float(row["seconds"]) - gap_sec)
        gap_end   = gps_to_utc_str(int(row["week"]), float(row["seconds"]))
        lines.append(f"| {gap_start} | {gap_end} | {gap_sec:.2f}s |")

    table = "| Gap Start (UTC) | Gap End (UTC) | Duration |\n|---|---|---|\n" + "\n".join(lines)
    total_duration = ref["abs_seconds"].iloc[-1] - ref["abs_seconds"].iloc[0]
    total_gap = gaps["delta"].sum()

    return {"result": (
        f"**{len(gaps)} data gap(s) detected** in `{log_entry['filename']}`.\n\n"
        f"Reference log: `{most_common_log}` ({len(ref)} records) | "
        f"Nominal interval: {median_interval:.3f}s\n"
        f"Total file duration: {total_duration:.1f}s | "
        f"Total missing time: {total_gap:.1f}s\n\n"
        f"{table}"
    )}

_LIST_TRIGGERS  = ("list all log","what log","all log","log type","how many log",
                   "available log","logs in","logs are","logs present","logs there","what data")
_TIME_TRIGGERS  = ("start time","end time","duration","time range","start and end",
                   "file time","gps time","utc time","how long","begin","finish")
_EVENT_KEYWORDS = ("when","occur","detect","spoof","jam","interfere","bit","status",
                   "error","flag","max","min","average","height","position","velocity",
                   "signal","drop","strength","quality","tracking","lock","loss")

def _direct_list_logs(log_entry: dict) -> dict:
    df         = log_entry["df"]
    log_counts = df.groupby("log_name_raw").size().sort_values(ascending=False)
    lines      = [f"| {n} | {c} |" for n, c in log_counts.items()]
    table      = "| Log Type | Count |\n|---|---|\n" + "\n".join(lines)
    return {"result": f"**{len(log_counts)} log types** in `{log_entry['filename']}` "
                      f"({len(df)} total records):\n\n{table}"}

def _direct_time_range(log_entry: dict) -> dict:
    df    = log_entry["df"]
    valid = df[df["time_status"].isin(_VALID_TIME_STATUSES) & (df["week"] > 0)]
    if valid.empty:
        return {"result": "No records with valid GPS time found in this file."}
    w_s   = int(valid.loc[valid["seconds"].idxmin(), "week"])
    w_e   = int(valid.loc[valid["seconds"].idxmax(), "week"])
    s_s   = float(valid["seconds"].min())
    s_e   = float(valid["seconds"].max())
    weeks = sorted(valid["week"].unique().tolist())
    dur   = (weeks[-1]-weeks[0])*604800+(s_e-s_s) if len(weeks)>1 else s_e-s_s
    return {"result": (
        f"**File time range for `{log_entry['filename']}`:**\n\n"
        f"| | GPS | UTC |\n|---|---|---|\n"
        f"| Start | Week {w_s}, {s_s:.3f}s | {gps_to_utc_str(w_s, s_s)} |\n"
        f"| End   | Week {w_e}, {s_e:.3f}s | {gps_to_utc_str(w_e, s_e)} |\n\n"
        f"**Duration:** {dur:.3f}s ({dur/60:.2f} min)"
    )}

# ── S3 helper ─────────────────────────────────────────────────────────
def upload_to_s3(content: bytes, filename: str) -> str:
    key = f"logs/{filename}"
    get_s3_client().put_object(Bucket=S3_BUCKET, Key=key, Body=content)
    return key


# ── Binary pre-processor ──────────────────────────────────────────────
_BINARY_EXTENSIONS = ('.gps', '.bin', '.raw', '.nov', '.novb')

def is_binary_log(filename: str, file_bytes: bytes) -> bool:
    if any(filename.lower().endswith(ext) for ext in _BINARY_EXTENSIONS):
        return True
    if len(file_bytes) >= 3 and file_bytes[:3] == b'\xaa\x44\x12':
        return True
    return False

def convert_binary_to_ascii(file_bytes: bytes, filename: str) -> tuple[bytes, str]:
    import novatel_edie as edie

    ascii_lines = []
    skipped     = 0

    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    fp = None
    try:
        fp = edie.FileParser(tmp_path)

        while True:
            try:
                result = fp.convert(edie.ENCODE_FORMAT.ASCII)
                if isinstance(result, edie.MessageData):
                    line = result.message.decode('utf-8', errors='replace').strip()
                    if line:
                        ascii_lines.append(line)
                else:
                    skipped += 1
            except edie.StreamEmptyException:
                break
            except Exception as e:
                skipped += 1
                continue

    finally:
        del fp  # ✅ release EDIE's file handle first (Windows fix)
        try:
            os.remove(tmp_path)  # now safe to delete
        except Exception as e:
            print(f"[PREPROCESS] Warning: could not delete temp file: {e}")

    ascii_content = '\n'.join(ascii_lines)
    new_filename  = os.path.splitext(filename)[0] + '_converted.ascii'
    print(f"[PREPROCESS] '{filename}' → '{new_filename}' "
          f"({len(ascii_lines)} messages, {skipped} skipped)")
    return ascii_content.encode('utf-8'), new_filename

def preprocess_file(file_bytes: bytes, filename: str) -> tuple[bytes, str]:
    if is_binary_log(filename, file_bytes):
        print(f"[PREPROCESS] Binary detected: '{filename}', converting via EDIE...")
        return convert_binary_to_ascii(file_bytes, filename)
    print(f"[PREPROCESS] ASCII file: '{filename}', no conversion needed.")
    return file_bytes, filename


# ── Main entrypoint ───────────────────────────────────────────────────
@app.entrypoint
async def invoke(payload):
    if isinstance(payload, dict):
        prompt     = payload.get("prompt", "")
        file_b64   = payload.get("file", None)
        s3_key_in  = payload.get("s3_key", None)
        filename   = payload.get("filename", "log.txt")
        session_id = payload.get("session_id", "default-session")
    else:
        prompt = str(payload); file_b64 = s3_key_in = None
        filename = "log.txt"; session_id = "default-session"

    # ── File ingest ───────────────────────────────────────────────────
    if file_b64:
        try:
            file_bytes = base64.b64decode(file_b64)
            file_bytes, filename = preprocess_file(file_bytes, filename)
            if S3_BUCKET and len(file_bytes) > SIZE_THRESHOLD:
                upload_to_s3(file_bytes, filename)
            info = ingest_log_file(file_bytes, filename, session_id)
            return {
                "result": f"Parsed '{info['filename']}': {info['records']} records across "
                          f"{info['log_types']} log types. Ask me anything about this file.",
                "summary": info["summary"],
            }
        except Exception as e:
            return {"result": f"Error parsing log file: {e}"}

    elif s3_key_in:
        try:
            obj  = get_s3_client().get_object(Bucket=S3_BUCKET, Key=s3_key_in)
            info = ingest_log_file(obj["Body"].read(), filename, session_id)
            return {
                "result": f"Parsed '{info['filename']}': {info['records']} records across "
                          f"{info['log_types']} log types. Ask me anything about this file.",
                "summary": info["summary"],
            }
        except Exception as e:
            return {"result": f"Error reading from S3: {e}"}

    # ── Q&A path ─────────────────────────────────────────────────────
    print(f"[QA] prompt={prompt!r} session_id={session_id!r}")
    try:
        apply_guardrail(prompt, source="INPUT")
    except ValueError as e:
        return {"result": str(e)}

    _current_session["id"] = session_id
    _tool_call_log.clear()

    # Check if log file is loaded
    log_entry = _log_store.get(session_id)

    # Load memory history
    history = []
    if MEMORY_ID:
        try:
            events = get_memory_client().list_events(
                memory_id=MEMORY_ID, actor_id=ACTOR_ID,
                session_id=session_id, max_results=10,
            )
            for event in events:
                for item in event.get("payload", []):
                    conv = item.get("conversational", {})
                    role = conv.get("role", "")
                    text = conv.get("content", {}).get("text", "")
                    if role == "USER":
                        history.append(HumanMessage(content=text))
                    elif role == "ASSISTANT":
                        history.append(AIMessage(content=text))
        except Exception as e:
            print(f"Memory retrieve error: {e}")

    history = history[-6:]
    history = [
        msg.__class__(content=msg.content[:800] + "…[truncated]")
        if isinstance(msg.content, str) and len(msg.content) > 800 else msg
        for msg in history
    ]

    # ── Routing ───────────────────────────────────────────────────────
    # log_entry already defined above
    p = prompt.lower()

    # Check if question is clearly non-GNSS/non-technical even with log file loaded
    _NON_GNSS_INDICATORS = (
        "war", "conflict", "battle", "military", "army", "weapon", "soldier",
        "politics", "election", "president", "government", "congress", "parliament",
        "news", "current events", "today", "yesterday", "breaking",
        "weather", "temperature", "rain", "snow", "forecast", "climate",
        "stock", "market", "trading", "investment", "finance", "economy",
        "recipe", "cooking", "food", "restaurant", "meal", "dish",
        "movie", "film", "actor", "actress", "cinema", "netflix",
        "music", "song", "album", "artist", "concert", "band",
        "sports", "football", "basketball", "soccer", "game", "match", "player",
        "medical", "disease", "symptom", "doctor", "hospital", "medicine",
        "travel", "vacation", "hotel", "flight", "tourism", "destination",
        "history", "ancient", "medieval", "century", "historical",
        "programming", "python", "javascript", "code", "software development"
    )
    
    is_non_gnss = any(indicator in p for indicator in _NON_GNSS_INDICATORS)
    
    # If clearly non-GNSS question, route to doc agent (which will use its general knowledge)
    if is_non_gnss and log_entry:
        print("[ROUTE] → doc agent (non-GNSS question, ignoring log file)")
        try:
            output = run_doc_agent(prompt, [], session_id)
        except GraphRecursionError:
            return {"result": "Hit the search budget. Please try rephrasing."}
        except Exception as e:
            return {"result": f"Error: {e}"}
        
        # Save to memory and return
        if MEMORY_ID:
            try:
                get_memory_client().create_event(
                    memory_id=MEMORY_ID, actor_id=ACTOR_ID, session_id=session_id,
                    messages=[(prompt, "USER"), (output, "ASSISTANT")],
                )
            except Exception as e:
                print(f"Memory save error: {e}")
        return {"result": output}

    # 0. Hardcoded scintillation question handler - routes to log pipeline with optimized prompt
    if log_entry and ("scintillation" in p and ("c/no" in p or "cno" in p or "carrier" in p)):
        print("[ROUTE] → scintillation analysis via log pipeline")
        # Rewrite the question to ensure it routes correctly through the log pipeline
        optimized_prompt = "analyze TRACKSTAT field 11 C/No statistics and identify if there is scintillation based on the signal variations"
        try:
            output = run_log_pipeline(optimized_prompt, session_id)
            
            # Enhance the output with scintillation-specific interpretation
            if "Statistics for TRACKSTAT field 11" in output or "TRACKSTAT" in output:
                # Extract statistics from the output to add scintillation assessment
                import re
                min_match = re.search(r'Minimum.*?(\d+\.?\d*)', output)
                max_match = re.search(r'Maximum.*?(\d+\.?\d*)', output)
                range_match = re.search(r'Range.*?(\d+\.?\d*)', output)
                
                if min_match and max_match and range_match:
                    min_val = float(min_match.group(1))
                    max_val = float(max_match.group(1))
                    range_val = float(range_match.group(1))
                    
                    # Add scintillation assessment
                    assessment = "\n\n**Scintillation Assessment:**\n"
                    
                    if range_val > 40 or min_val < 10:
                        assessment += "✅ **Yes, strong indicators of ionospheric scintillation are present.**\n\n"
                        assessment += "**Evidence:**\n"
                        
                        if range_val > 40:
                            assessment += f"- **Large signal variation:** C/No range of {range_val:.1f} dB-Hz indicates rapid amplitude fluctuations characteristic of scintillation\n"
                        
                        if min_val < 10:
                            assessment += f"- **Deep signal fades:** Minimum C/No of {min_val:.1f} dB-Hz shows severe signal degradation\n"
                        
                        if min_val == 0:
                            assessment += "- **Complete signal loss:** C/No drops to 0 dB-Hz indicate periods of total signal disruption\n"
                        
                        if range_val > 50:
                            assessment += "- **Extreme fluctuations:** Range exceeding 50 dB-Hz suggests severe scintillation conditions\n"
                        
                        assessment += "\n**Typical causes:** Ionospheric irregularities (especially in equatorial/auroral regions), solar activity, or geomagnetic storms.\n"
                        assessment += "**Impact:** May cause positioning errors, loss of lock, and degraded navigation accuracy."
                    
                    elif range_val > 20 or min_val < 25:
                        assessment += "⚠️ **Moderate scintillation indicators detected.**\n\n"
                        assessment += f"The C/No shows moderate variations (range: {range_val:.1f} dB-Hz, min: {min_val:.1f} dB-Hz) that suggest some ionospheric disturbance, though not severe."
                    
                    else:
                        assessment += "✅ **No significant scintillation detected.**\n\n"
                        assessment += f"The C/No values are stable (range: {range_val:.1f} dB-Hz, min: {min_val:.1f} dB-Hz), indicating normal signal conditions."
                    
                    output = output + assessment
            
            return {"result": output}
            
        except Exception as e:
            print(f"[SCINTILLATION] error: {e}")
            # Fall through to normal routing
            pass

    # 0a. Hardcoded RXSTATUSEVENT summary handler
    if log_entry and ("rxstatus" in p and ("event" in p or "message" in p) and ("summar" in p or "all" in p or "list" in p)):
        print("[ROUTE] → RXSTATUSEVENT summary")
        try:
            output = run_log_pipeline("summarize all RXSTATUSEVENT records and identify what events occurred", session_id)
            return {"result": output}
        except Exception as e:
            print(f"[RXSTATUSEVENT] error: {e}")
            # Fall through to normal routing
            pass

    # 0b. Hardcoded minimum satellites tracked handler
    if log_entry and ("minimum" in p or "min" in p or "lowest" in p) and ("satellite" in p or "sat" in p) and ("track" in p):
        print("[ROUTE] → minimum satellites tracked")
        try:
            # TRACKSTAT field 2 is typically the number of satellites being tracked
            # But we need to check BESTPOS field for number of satellites used in solution
            # Try BESTPOS first (field 7 = # of satellites in solution)
            result = do_analyze_field(session_id, "BESTPOS", 7)
            
            if result.get("status") == "error":
                # Fallback: try TRACKSTAT if BESTPOS not available
                result = do_analyze_field(session_id, "TRACKSTAT", 2)
            
            if result.get("status") == "error":
                return {"result": f"Could not analyze satellite count: {result['error']}"}
            
            min_val = result.get("min_value", 0)
            max_val = result.get("max_value", 0)
            avg_val = result.get("average_value", 0)
            log_name = result.get("log_name", "")
            field_idx = result.get("doc_field_index", "")
            total = result.get("total_records", 0)
            valid = result.get("valid_values", 0)
            
            min_time = result.get("min_occurred_at", {}).get("utc_time", "")
            max_time = result.get("max_occurred_at", {}).get("utc_time", "")
            
            output = (
                f"**Satellite Tracking Statistics ({log_name} field {field_idx}):**\n\n"
                f"| Metric | Value | Timestamp (UTC) |\n"
                f"|--------|-------|----------------|\n"
                f"| **Minimum** | {int(min_val)} satellites | {min_time} |\n"
                f"| **Maximum** | {int(max_val)} satellites | {max_time} |\n"
                f"| **Average** | {avg_val:.1f} satellites | - |\n\n"
                f"Analyzed {valid} valid values from {total} total records.\n\n"
            )
            
            # Add assessment
            if min_val < 4:
                output += "⚠️ **Warning:** Minimum satellite count is below 4, which is insufficient for 3D positioning. This indicates periods of poor satellite visibility or signal loss.\n"
            elif min_val < 6:
                output += "⚠️ **Caution:** Minimum satellite count is low (4-5 satellites). While sufficient for basic positioning, accuracy may be degraded during these periods.\n"
            else:
                output += "✅ **Good:** Minimum satellite count is adequate for reliable positioning throughout the observation period.\n"
            
            if max_val - min_val > 8:
                output += f"\n**Note:** Large variation in satellite count ({int(max_val - min_val)} satellites) suggests changing sky visibility conditions or signal obstructions."
            
            return {"result": output}
            
        except Exception as e:
            print(f"[MIN_SATELLITES] error: {e}")
            # Fall through to normal routing
            pass

    # 1. Ultra-fast path for simple conversational questions (no KB needed)
    _SIMPLE_CONVERSATIONAL = (
        "hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "yes", "no",
        "good", "great", "nice", "cool", "awesome", "perfect", "got it",
        "i see", "understood", "makes sense", "you are", "you're", "your",
        "bad", "good job", "well done", "terrible", "wrong", "right", "correct",
        "stupid", "smart", "dumb", "clever", "useless", "helpful", "unhelpful"
    )
    
    # Check if it's a very short conversational message or feedback about the agent
    is_about_agent = any(phrase in p for phrase in ["you are", "you're", "your performance", "your answer"])
    is_short_conversational = len(prompt.strip()) < 50 and any(p.startswith(phrase) or p == phrase for phrase in _SIMPLE_CONVERSATIONAL)
    
    if is_short_conversational or is_about_agent:
        print("[ROUTE] → fast conversational response")
        responses = {
            "hi": "Hello! I'm ready to help you analyze NovAtel OEM7 receiver logs. Upload a log file or ask me about NovAtel documentation.",
            "hello": "Hi! I can help you analyze GNSS receiver logs and answer questions about NovAtel OEM7 documentation.",
            "hey": "Hey! Ready to analyze your receiver logs. What would you like to know?",
            "thanks": "You're welcome! Let me know if you need anything else.",
            "thank you": "Happy to help! Feel free to ask more questions.",
            "ok": "Great! What would you like to analyze next?",
            "okay": "Sounds good! Anything else I can help with?",
            "yes": "Understood. What would you like to do?",
            "good": "Glad to help! What's next?",
            "great": "Excellent! Let me know if you need more analysis.",
            "nice": "Thanks! Anything else you'd like to check?",
            "cool": "Great! What else can I analyze for you?",
            "perfect": "Wonderful! Let me know if you need anything else.",
            "got it": "Perfect! Feel free to ask more questions.",
        }
        
        # Handle feedback about the agent
        if is_about_agent:
            if any(word in p for word in ["bad", "wrong", "terrible", "stupid", "useless", "unhelpful"]):
                return {"result": "I apologize if my response wasn't helpful. Could you please clarify what you're looking for? I'm here to help with NovAtel log analysis and GNSS questions."}
            elif any(word in p for word in ["good", "great", "helpful", "smart", "correct", "right"]):
                return {"result": "Thank you! I'm glad I could help. Let me know if you need anything else."}
            else:
                return {"result": "I'm a NovAtel log analysis assistant. How can I help you with your GNSS data?"}
        
        for key, response in responses.items():
            if key in p:
                return {"result": response}
        return {"result": "I'm here to help! What would you like to know?"}

    # 1. Check if this is a general question (not requiring log analysis)
    _GENERAL_QUESTIONS = (
        "what is", "what are", "how does", "how do", "explain", "define",
        "tell me about", "describe", "should i", "is this trustable",
        "is it trustable", "can i trust", "is the data", "trustworthy",
        "reliable", "accuracy", "how accurate", "quality of", "good data",
        "bad data", "valid data", "correct data", "is this good", "is this bad",
        "should i use", "can i use", "is it safe", "safe to use"
    )
    
    is_general = any(phrase in p for phrase in _GENERAL_QUESTIONS)
    
    # If it's a general question without specific log/field reference, use doc agent
    # But if question contains analysis keywords (guess, detect, check, any, find), prefer log pipeline
    _ANALYSIS_KEYWORDS = ("guess", "detect", "check", "any", "find", "identify", 
                          "show", "list", "analyze", "analyse", "scintillation",
                          "jamming", "spoofing", "interference", "error", "issue")
    has_analysis_intent = any(kw in p for kw in _ANALYSIS_KEYWORDS)
    has_specific_reference = any(kw in p for kw in ["in this file", "in the file", "in my file", 
                                                      "of this file", "of the file", "of my file",
                                                      "this log", "the log", "my log", "field", "bit"])
    
    if is_general and not has_specific_reference and not has_analysis_intent and log_entry:
        print("[ROUTE] → doc agent (general question, ignoring file context)")
        try:
            output = run_doc_agent(prompt, history, session_id)
        except GraphRecursionError:
            return {"result": "Hit the search budget. Please try rephrasing."}
        except Exception as e:
            return {"result": f"Error: {e}"}
    # 2. Fully deterministic direct handlers (no LLM)
    elif log_entry:
        is_event = any(kw in p for kw in _EVENT_KEYWORDS)

        # Data gap / continuity check
        _GAP_TRIGGERS = ("gap","missing","continuous","every second","data loss",
                         "time missing","dropped","interval","recording","recorded at",
                         "any second","each second","per second","all second")
        if any(kw in p for kw in _GAP_TRIGGERS):
            print("[DIRECT] → data gap analysis")
            set_status(session_id, "Checking for time gaps in log data...")
            result = _direct_data_gap(log_entry)
            clear_status(session_id)
            return result

        if not is_event and any(kw in p for kw in _LIST_TRIGGERS):
            print("[DIRECT] → list logs")
            set_status(session_id, "Retrieving log inventory from file...")
            result = _direct_list_logs(log_entry)
            clear_status(session_id)
            return result
        if not is_event and any(kw in p for kw in _TIME_TRIGGERS):
            print("[DIRECT] → time range")
            set_status(session_id, "Computing file time range from GPS timestamps...")
            result = _direct_time_range(log_entry)
            clear_status(session_id)
            return result

    # 3. Main routing logic
    try:
        if log_entry:
            # Log pipeline — deterministic, no agent loop
            print("[ROUTE] → log pipeline")
            output = run_log_pipeline(prompt, session_id)
        else:
            # Documentation agent — no file loaded
            print("[ROUTE] → doc agent")
            output = run_doc_agent(prompt, history, session_id)

    except GraphRecursionError:
        return {"result": "Hit the search budget. Please try rephrasing."}
    except Exception as e:
        return {"result": f"Error: {e}"}

    try:
        apply_guardrail(output, source="OUTPUT")
    except ValueError as e:
        return {"result": str(e)}

    if MEMORY_ID:
        try:
            get_memory_client().create_event(
                memory_id=MEMORY_ID, actor_id=ACTOR_ID, session_id=session_id,
                messages=[(prompt, "USER"), (output, "ASSISTANT")],
            )
        except Exception as e:
            print(f"Memory save error: {e}")

    return {"result": output}


if __name__ == "__main__":
    app.run()
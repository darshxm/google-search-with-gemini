"""
gemini_grounding.core
Reusable module for issuing grounded Gemini model queries and extracting / resolving
citations produced via Google Search grounding.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Iterable

import requests
from tqdm import tqdm

# Gemini imports
from google import genai
from google.genai import types

__all__ = [
    'load_api_key', 'get_client', 'add_citations', 'extract_markdown_links',
    'resolve_redirect', 'resolve_all_redirects', 'generate_with_grounding',
    'query_with_grounded_citations', 'run_prompt_and_resolve',
    'CitationResult', 'ResolvedRedirect'
]

# -----------------------------------------------------------------------------
# API KEY LOADING
# -----------------------------------------------------------------------------

def load_api_key(path: str = "api.json") -> Optional[str]:
    """Return API key string or None.

    Order of precedence:
      1. Environment variable 'GEMINI_API_KEY'
      2. JSON file at `path` containing {"gemini_key": "..."}
    """
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key:
        return env_key.strip()
    p = Path(path)
    if p.exists():
        try:
            with p.open('r') as f:
                return json.load(f).get("gemini_key")
        except Exception:
            return None
    return None


_client: Optional[genai.Client] = None


def get_client(force_reload: bool = False) -> genai.Client:
    """Return a cached Gemini client instance, creating it lazily."""
    global _client
    if force_reload or _client is None:
        api_key = load_api_key()
        if not api_key:
            raise RuntimeError(
                "API key not found. Set GEMINI_API_KEY or provide api.json with gemini_key."
            )
        _client = genai.Client(api_key=api_key)
    return _client

# -----------------------------------------------------------------------------
# CITATION RENDERING
# -----------------------------------------------------------------------------

def add_citations(response) -> str:
    """Insert inline citation links into model response text in-place."""
    text = response.text
    md = getattr(response.candidates[0], 'grounding_metadata', None)
    if not md:
        return text
    supports = getattr(md, 'grounding_supports', [])
    chunks = getattr(md, 'grounding_chunks', [])
    sorted_supports = sorted(
        supports, key=lambda s: s.segment.end_index, reverse=True
    )
    for support in sorted_supports:
        end_index = support.segment.end_index
        indices = getattr(support, 'grounding_chunk_indices', None)
        if indices:
            links = []
            for i in indices:
                if i < len(chunks) and getattr(chunks[i], 'web', None):
                    uri = chunks[i].web.uri
                    links.append(f"[{i + 1}]({uri})")
            if links:
                citation_string = ", ".join(links)
                text = text[:end_index] + citation_string + text[end_index:]
    return text

# -----------------------------------------------------------------------------
# LINK EXTRACTION
# -----------------------------------------------------------------------------

LINK_PATTERN = re.compile(r"\[(\d+)]\((https?://[^)\s]+)\)")

def extract_markdown_links(markdown_text: str) -> List[Tuple[int, str]]:
    """Parse inline citation-style markdown links of the form [n](url)."""
    return [(int(idx), url) for idx, url in LINK_PATTERN.findall(markdown_text)]

# -----------------------------------------------------------------------------
# REDIRECT RESOLUTION
# -----------------------------------------------------------------------------

def resolve_redirect(url: str, session: Optional[requests.Session] = None, timeout: int = 15) -> Dict[str, Any]:
    """Resolve a URL to its final destination following redirects & simple meta refresh."""
    sess = session or requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) PreciousRedirectResolver/1.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    def _meta_refresh_find(html_text: str) -> Optional[str]:
        m = re.search(
            r'<meta[^>]*http-equiv=["\']?refresh["\']?[^>]*content=["\']?\d+;\s*url=([^"\'>]+)',
            html_text,
            re.IGNORECASE,
        )
        return m.group(1) if m else None

    attempt_method = 'HEAD'
    history_urls: List[str] = []
    status_code = None
    final_url = None
    error = None
    try:
        resp = sess.head(url, allow_redirects=True, timeout=timeout, headers=headers)
        status_code = resp.status_code
        history_urls = [r.url for r in resp.history] + [resp.url]
        final_url = resp.url
        if status_code in {403, 405} or (status_code == 200 and url == final_url and not resp.history):
            attempt_method = 'GET'
            resp = sess.get(url, allow_redirects=True, timeout=timeout, headers=headers)
            status_code = resp.status_code
            history_urls = [r.url for r in resp.history] + [resp.url]
            final_url = resp.url
            if 'text/html' in resp.headers.get('Content-Type', ''):
                extra = _meta_refresh_find(resp.text)
                if extra and extra != final_url:
                    resp2 = sess.get(extra, allow_redirects=True, timeout=timeout, headers=headers)
                    history_urls.append(resp2.url)
                    final_url = resp2.url
    except Exception as e:
        error = str(e)

    return {
        "input_url": url,
        "final_url": final_url,
        "status_code": status_code,
        "method_used": attempt_method,
        "redirect_chain": history_urls,
        "error": error,
    }

def resolve_all_redirects(urls: Iterable[str], delay_seconds: float = 0.2) -> Dict[str, Dict[str, Any]]:
    """Resolve many URLs, returning a mapping {input_url: resolution_dict}."""
    sess = requests.Session()
    resolved: Dict[str, Dict[str, Any]] = {}
    for u in tqdm(list(urls), desc="Resolving citation redirects"):
        resolved[u] = resolve_redirect(u, sess)
        if delay_seconds:
            time.sleep(delay_seconds)  # polite delay
    return resolved

# -----------------------------------------------------------------------------
# CORE GENERATION LOGIC
# -----------------------------------------------------------------------------

def generate_with_grounding(prompt: str, model: str = "gemini-2.5-flash"):
    """Call Gemini model with Google Search grounding enabled; returns raw SDK response."""
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(tools=[grounding_tool])
    client = get_client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    return response

@dataclass
class ResolvedRedirect:
    input_url: str
    final_url: Optional[str]
    status_code: Optional[int]
    method_used: Optional[str]
    redirect_chain: List[str]
    error: Optional[str]

@dataclass
class CitationResult:
    citation_index: int  # numeric label inside []
    redirect_url: str    # the Gemini grounding redirect URL
    final_url: Optional[str] = None
    status_code: Optional[int] = None
    redirect_chain: Optional[List[str]] = None
    error: Optional[str] = None


def query_with_grounded_citations(
    prompt: str,
    model: str = "gemini-2.5-flash",
    follow_redirects: bool = True,
    include_raw_resolution: bool = False,
) -> Dict[str, Any]:
    """Produce structured grounded model answer with optional redirect resolution."""
    response = generate_with_grounding(prompt, model=model)
    raw_text = response.text
    enriched_text = add_citations(response)
    links = extract_markdown_links(enriched_text)  # list[(idx, url)] in order encountered

    # Build initial citation results preserving order (even if same redirect repeats)
    citation_objects: List[CitationResult] = [
        CitationResult(citation_index=idx, redirect_url=url) for idx, url in links
    ]

    raw_resolved_map: Dict[str, Dict[str, Any]] = {}
    if follow_redirects and citation_objects:
        unique_redirects = sorted({c.redirect_url for c in citation_objects})
        raw_resolved_map = resolve_all_redirects(unique_redirects)
        # Attach resolution info back to each citation occurrence
        for c in citation_objects:
            data = raw_resolved_map.get(c.redirect_url, {})
            c.final_url = data.get("final_url")
            c.status_code = data.get("status_code")
            c.redirect_chain = data.get("redirect_chain")
            c.error = data.get("error")

    result: Dict[str, Any] = {
        "prompt": prompt,
        "model": model,
        "raw_text": raw_text,
        "enriched_text": enriched_text,
        "citations": [asdict(c) for c in citation_objects],
        "timestamp": datetime.utcnow().isoformat() + 'Z'
    }
    if include_raw_resolution and follow_redirects:
        result["raw_resolved_map"] = raw_resolved_map
    return result


def run_prompt_and_resolve(
    prompt: str,
    model: str,
    output_json: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Convenience wrapper combining query + redirect resolution + optional JSON dump."""
    result = query_with_grounded_citations(prompt=prompt, model=model, follow_redirects=True, include_raw_resolution=True)
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open('w') as f:
            json.dump(result, f, indent=2)
    if verbose:
        print("\n=== RAW MODEL TEXT ===\n", result["raw_text"])
        print("\n=== ENRICHED WITH CITATIONS ===\n", result["enriched_text"])
        print("\n=== RESOLVED REDIRECTS (input -> final) ===")
        seen = set()
        for c in result["citations"]:
            r = c["redirect_url"]
            if r in seen:
                continue
            seen.add(r)
            print(f"{r} -> {c.get('final_url')}")
    return result

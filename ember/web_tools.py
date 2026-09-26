"""Web search and fetch, backed by the TinyFish APIs.

Importing this module registers `web_search` and `web_fetch` into the global
TOOLS registry in ember.core -- see the @tool decorator there.

Raw REST over urllib rather than the `tinyfish` SDK, so the package keeps its
single third-party dependency. Docs: https://docs.tinyfish.ai
Auth: export TINYFISH_API_KEY. It is read at call time and never logged.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

from ember import config as app_config
from ember.core import ToolContext, ToolResult, log, tool

SEARCH_URL = app_config.TINYFISH_SEARCH_URL
FETCH_URL = app_config.TINYFISH_FETCH_URL
TIMEOUT = app_config.TINYFISH_TIMEOUT
MAX_URLS = app_config.TINYFISH_MAX_URLS
RESULTS_CAP = app_config.WEB_SEARCH_RESULTS_CAP
SNIPPET_CAP = app_config.WEB_SNIPPET_CAP
FETCH_CHAR_CAP = app_config.WEB_FETCH_CHAR_CAP
EMPTY_RETRY_DELAY = 1.0   # pause before the one silent retry on an empty result set

_MISSING_KEY = (
    "TINYFISH_API_KEY is not set. Get a key at https://agent.tinyfish.ai/api-keys "
    "and export TINYFISH_API_KEY, or add it to .env."
)


def _api_key() -> Optional[str]:
    key = os.environ.get("TINYFISH_API_KEY", "").strip()
    return key or None


def _request(url: str, key: str, payload: Optional[dict[str, Any]]) -> tuple[Any, Optional[str]]:
    """Return (parsed_json, error_message). Never raises."""
    data = None
    headers = {"X-API-Key": key, "Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
        except Exception:  # noqa: BLE001
            pass
        if exc.code in (401, 403):
            return None, f"TinyFish rejected the API key (HTTP {exc.code}). {detail}"
        if exc.code == 429:
            return None, f"TinyFish rate limit hit (HTTP 429). Slow down and retry. {detail}"
        return None, f"TinyFish HTTP {exc.code}: {detail}"
    except urllib.error.URLError as exc:
        return None, f"Cannot reach TinyFish ({url}): {exc.reason}"
    except TimeoutError:
        return None, f"TinyFish request timed out after {TIMEOUT:.0f}s"

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        return None, f"TinyFish returned non-JSON: {exc}"

    # A 200 can still carry an error payload. Without this check it would
    # flatten into an empty result set and read as "nothing found", which is
    # a different claim entirely. Note `errors` (plural) is NOT checked here:
    # the Fetch API uses it for legitimate per-URL failures.
    if isinstance(parsed, dict) and parsed.get("error"):
        err = parsed["error"]
        if isinstance(err, dict):
            err = err.get("message") or json.dumps(err)
        return None, f"error payload in an HTTP 200 response: {err}"

    return parsed, None


# --- web_search --------------------------------------------------------------


@tool(
    name="web_search",
    description=(
        "Search the live web and return ranked results (title, URL, snippet). "
        "Use this to DISCOVER sources; snippets are not evidence -- call "
        "web_fetch on a result before citing it. Leave domain_type unset "
        "unless you specifically need press coverage or academic papers."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search keywords."},
            "max_results": {
                "type": "integer",
                "description": f"1-{RESULTS_CAP}, default 5.",
            },
            "domain_type": {
                "type": "string",
                "enum": ["web", "news", "research_paper"],
                "description": (
                    "Corpus to search. 'web' (default) covers general sites, "
                    "forums and aggregators, and is right for most queries. "
                    "'news' returns press articles with publisher and date. "
                    "'research_paper' returns academic work with authors, "
                    "venue, year and citation count."
                ),
            },
            "recency_minutes": {
                "type": "integer",
                "description": "Only results newer than this many minutes old.",
            },
        },
        "required": ["query"],
    },
    requires_confirmation=False,
)
def web_search_tool(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    query = str(args.get("query") or "").strip()
    if not query:
        return ToolResult("web_search requires a non-empty 'query'.", is_error=True)

    key = _api_key()
    if not key:
        return ToolResult(_MISSING_KEY, is_error=True)

    n = max(1, min(int(args.get("max_results") or 5), RESULTS_CAP))
    params: dict[str, Any] = {"query": query}

    domain_type = str(args.get("domain_type") or "").strip()
    if domain_type in ("web", "news", "research_paper"):
        params["domain_type"] = domain_type

    recency = args.get("recency_minutes")
    if recency is not None:
        try:
            params["recency_minutes"] = max(1, int(recency))
        except (TypeError, ValueError):
            return ToolResult("recency_minutes must be an integer", is_error=True)

    url = f"{SEARCH_URL}?{urllib.parse.urlencode(params)}"
    payload, err = _request(url, key, None)
    results = [] if err else ((payload or {}).get("results") or [])

    # One silent retry. The API intermittently answers 200 with an empty list
    # for queries that do have results, and search is free and fast. This is
    # infrastructure flakiness, so it is handled here rather than spent as an
    # agent step -- same reasoning as LLMClient._with_retry and the SQLite
    # busy-retry loop, neither of which the model is told about either.
    if not results:
        log.info("web_search: empty/failed first attempt, retrying once")
        time.sleep(EMPTY_RETRY_DELAY)
        payload, err = _request(url, key, None)
        results = [] if err else ((payload or {}).get("results") or [])

    if err:
        log.warning("web_search failed: %s", err)
        return ToolResult(f"web_search: {err}", is_error=True)

    if not results:
        # Empty twice. Now it is a real signal rather than a blip.
        return ToolResult(
            f"web_search found nothing for {query!r} (searched twice). "
            "Try different or broader search terms.",
            is_error=True,
        )

    lines: list[str] = []
    for i, r in enumerate(results[:n], start=1):
        title = str(r.get("title") or "").strip()
        link = str(r.get("url") or "").strip()
        site = str(r.get("site_name") or "").strip()
        snippet = " ".join(str(r.get("snippet") or "").split())
        if len(snippet) > SNIPPET_CAP:
            snippet = snippet[:SNIPPET_CAP] + "..."
        head = f"[{i}] {title}" + (f"  ({site})" if site else "")
        lines.append(f"{head}\n    {link}\n    {snippet}")

    total = (payload or {}).get("total_results")
    header = f"[web_search {query!r}] showing {len(lines)}"
    if isinstance(total, int):
        header += f" of ~{total}"
    return ToolResult(header + "\n" + "\n".join(lines))


# --- web_fetch ---------------------------------------------------------------


def _as_url_list(args: dict[str, Any]) -> list[str]:
    """Accept either 'urls': [...] or a single 'url': '...'."""
    raw = args.get("urls")
    if raw is None:
        single = args.get("url")
        raw = [single] if single else []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [str(u).strip() for u in raw if str(u).strip()]


@tool(
    name="web_fetch",
    description=(
        "Read web pages from the live internet. Give it one or more URLs and it "
        "returns each page's full text as clean Markdown. Use it to read a page "
        "before relying on it or citing it. This is real web access, not a "
        "simulation: call it whenever you need what a page actually says."
    ),
    parameters={
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": f"1-{MAX_URLS} absolute http(s) URLs.",
            },
            "url": {"type": "string", "description": "Convenience: a single URL."},
            "max_chars": {
                "type": "integer",
                "description": f"Per-document truncation cap, default {FETCH_CHAR_CAP}.",
            },
        },
    },
    requires_confirmation=False,
)
def web_fetch_tool(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    urls = _as_url_list(args)
    if not urls:
        return ToolResult("web_fetch requires 'urls' (array) or 'url' (string).", is_error=True)

    bad = [u for u in urls if not u.startswith(("http://", "https://"))]
    if bad:
        return ToolResult(f"web_fetch: not absolute http(s) URLs: {bad}", is_error=True)

    if len(urls) > MAX_URLS:
        return ToolResult(
            f"web_fetch accepts at most {MAX_URLS} URLs per call, got {len(urls)}.",
            is_error=True,
        )

    key = _api_key()
    if not key:
        return ToolResult(_MISSING_KEY, is_error=True)

    cap = max(500, min(int(args.get("max_chars") or FETCH_CHAR_CAP), FETCH_CHAR_CAP))

    payload, err = _request(FETCH_URL, key, {"urls": urls, "format": "markdown"})
    if err:
        log.warning("web_fetch failed: %s", err)
        return ToolResult(f"web_fetch: {err}", is_error=True)

    payload = payload or {}
    results = payload.get("results") or []
    errors = payload.get("errors") or []

    blocks: list[str] = []
    for r in results:
        final_url = str(r.get("final_url") or r.get("url") or "").strip()
        title = str(r.get("title") or "").strip()
        text = r.get("text")
        if not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False) if text is not None else ""
        truncated = len(text) > cap
        text = text[:cap]
        if truncated:
            text += f"\n\n[...truncated at {cap} chars; ask for a narrower page...]"
        head = f"--- {title or '(untitled)'} --- {final_url}"
        blocks.append(f"{head}\n{text}")

    for e in errors:
        blocks.append(
            f"--- FAILED --- {e.get('url', '')}\n{e.get('error', 'unknown error')}"
        )

    if not blocks:
        return ToolResult("web_fetch: no content returned.", is_error=True)

    header = f"[web_fetch] {len(results)} fetched, {len(errors)} failed"
    # Per-URL failures come back with HTTP 200, so flag the all-failed case.
    return ToolResult(header + "\n\n" + "\n\n".join(blocks), is_error=not results)


__all__ = ["web_search_tool", "web_fetch_tool"]

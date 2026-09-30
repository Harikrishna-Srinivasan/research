"""Web search and page reading tools — free, no API keys needed."""

from __future__ import annotations

import os
import re
from typing import Optional


def web_search(query: str, max_results: int = 8) -> str:
    """Search the web using DuckDuckGo (free, no API key)."""
    try:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"**{r['title']}**\n{r['href']}\n{r['body']}\n")

        if results:
            return f"Search results for '{query}':\n\n" + "\n---\n".join(results)
        return f"No results found for '{query}'"
    except ImportError:
        return "ERROR: duckduckgo-search not installed. Run: pip install duckduckgo-search"
    except Exception as e:
        return f"ERROR searching: {e}"


def tavily_web_search(query: str, max_results: int = 8) -> str:
    """Search the web using Tavily (requires TAVILY_API_KEY env var)."""
    try:
        from tavily import TavilyClient

        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            return "ERROR: TAVILY_API_KEY environment variable is not set."

        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=max_results)

        results = []
        for r in response.get("results", []):
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "")
            results.append(f"**{title}**\n{url}\n{content}\n")

        if results:
            return f"Search results for '{query}':\n\n" + "\n---\n".join(results)
        return f"No results found for '{query}'"
    except ImportError:
        return "ERROR: tavily-python not installed. Run: pip install tavily-python"
    except Exception as e:
        return f"ERROR searching: {e}"


def read_webpage(url: str, max_chars: int = 6000) -> str:
    """Fetch a web page and extract its text content as markdown."""
    try:
        import requests
        from bs4 import BeautifulSoup
        from markdownify import markdownify as md

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; DeepAgent/1.0; Research Bot)"
        }
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove script/style/nav
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()

        text = md(str(soup), heading_style="ATX", strip=["img"])
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n... [truncated]"

        return f"Content from {url}:\n\n{text}"
    except ImportError:
        return "ERROR: Install dependencies: pip install requests beautifulsoup4 markdownify"
    except Exception as e:
        return f"ERROR fetching {url}: {e}"


def read_github_repo(owner: str, repo: str, path: str = "") -> str:
    """Fetch file listing or file content from a GitHub repository via API."""
    try:
        import requests

        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        resp = requests.get(api_url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, list):
            entries = [f"{'📁' if d['type'] == 'dir' else '📄'} {d['name']}" for d in data[:50]]
            return f"GitHub: {owner}/{repo}/{path}\n" + "\n".join(entries)
        elif isinstance(data, dict) and "content" in data:
            import base64
            content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
            return f"File: {owner}/{repo}/{path}\n\n{content[:6000]}"
        else:
            return f"Unexpected response from GitHub API for {api_url}"
    except Exception as e:
        return f"ERROR reading GitHub repo: {e}"


# Search provider tools keyed by provider name used in config.tools.web_search_provider
SEARCH_PROVIDER_TOOLS: dict[str, dict] = {
    "duckduckgo": {
        "name": "web_search",
        "description": "Search the web using DuckDuckGo. Returns titles, URLs, and snippets.",
        "parameters": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "description": "Max results", "default": 8},
        },
        "function": web_search,
    },
    "tavily": {
        "name": "tavily_web_search",
        "description": "Search the web using Tavily. Returns titles, URLs, and snippets. Requires TAVILY_API_KEY.",
        "parameters": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "description": "Max results", "default": 8},
        },
        "function": tavily_web_search,
    },
}

# Non-search web tools (always registered)
WEB_TOOLS = [
    {
        "name": "read_webpage",
        "description": "Fetch and extract text content from a URL as markdown.",
        "parameters": {
            "url": {"type": "string", "description": "URL to read"},
        },
        "function": read_webpage,
    },
    {
        "name": "read_github_repo",
        "description": "Browse a GitHub repository — list files or read file content.",
        "parameters": {
            "owner": {"type": "string", "description": "GitHub username/org"},
            "repo": {"type": "string", "description": "Repository name"},
            "path": {"type": "string", "description": "Path within repo (empty for root)", "default": ""},
        },
        "function": read_github_repo,
    },
]

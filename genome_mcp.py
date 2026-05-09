#!/usr/bin/env python3
"""
Industrial Genome MCP Server

Exposes Genome signal intelligence as MCP tools so any AI assistant
(Claude, Cursor, OpenBrain, etc.) can query live signals, convictions,
screener results, and macro regime directly.

Tools:
  get_signal         — Full investment signal for a ticker
  get_screener       — Top conviction signals across all tracked tickers
  get_ecl            — Live macro regime + ECL module scores
  get_horizons       — 3M/6M/12M divergence for a ticker
  get_supply_chain   — Upstream supply chain risk for a ticker
  get_pfsl           — Pre-financial signal layer (EDGAR) for a ticker
  get_regime         — Current market regime summary
  get_diligence      — Full industrial operational diligence brief for a company (6 sections)
  ask_genome         — Natural language Q&A over Genome data (Claude-powered)

Usage:
  python mcp/genome_mcp.py

Config (env vars):
  GENOME_API_URL   — defaults to http://localhost:8000
  ANTHROPIC_API_KEY — required for ask_genome tool
"""

from __future__ import annotations

import json
import os
import sys
import asyncio
from typing import Any

import httpx

# ── Config ─────────────────────────────────────────────────────────────────────

GENOME_API = os.environ.get("GENOME_API_URL", "https://genome-api-production.up.railway.app")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Commercial key (gk_live_*) used by external buyers; falls back to internal key
GENOME_COMMERCIAL_KEY = os.environ.get("GENOME_COMMERCIAL_KEY", "")
GENOME_INTERNAL_KEY = os.environ.get("GENOME_API_KEY", "")

def _auth_headers() -> dict:
    """Return Authorization header — commercial key takes precedence over internal."""
    key = GENOME_COMMERCIAL_KEY or GENOME_INTERNAL_KEY
    if key:
        if GENOME_COMMERCIAL_KEY:
            return {"Authorization": f"Bearer {key}"}
        return {"X-API-Key": key}
    return {}

_CLIENT = httpx.AsyncClient(base_url=GENOME_API, timeout=60.0)


# ── API helpers ────────────────────────────────────────────────────────────────

async def _get(path: str, params: dict | None = None) -> dict:
    try:
        resp = await _CLIENT.get(path, params=params, headers=_auth_headers())
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return {"error": str(e)}


async def _post(path: str, body: dict) -> dict:
    try:
        resp = await _CLIENT.post(path, json=body, headers=_auth_headers())
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return {"error": str(e)}


# ── Tool implementations ────────────────────────────────────────────────────────

async def tool_get_signal(ticker: str, industry: str | None = None) -> str:
    """Full investment signal for a ticker."""
    params = {}
    if industry:
        params["industry"] = industry
    data = await _get(f"/v1/signal/{ticker.upper()}", params or None)
    if "error" in data:
        return f"Error fetching signal for {ticker}: {data['error']}"

    sig = data.get("signal", "UNKNOWN")
    conv = data.get("conviction", 0.0)
    arch = data.get("archetype", "unknown")
    traj = data.get("trajectory", "unknown")
    ecl_adj = data.get("ecl_adjustment", 0.0)
    pfsl_adj = data.get("pfsl_adjustment", 0.0)
    regime = data.get("regime_multiplier", 1.0)
    sc_risk = data.get("supply_chain_risk", 0.0)

    lines = [
        f"**{ticker.upper()} — {sig}**",
        f"Conviction: {conv:+.3f}",
        f"Archetype: {arch} / {traj}",
        f"ECL adjustment: {ecl_adj:+.3f}  |  PFSL adjustment: {pfsl_adj:+.3f}",
        f"Regime multiplier: {regime:.2f}  |  Supply chain risk: {sc_risk:+.3f}",
    ]

    if data.get("archetype_transition"):
        t = data["archetype_transition"]
        lines.append(f"⚠ Archetype transition: {t.get('from')} → {t.get('to')}")

    if data.get("options"):
        opt = data["options"]
        lines.append(f"Options: {opt.get('strategy', 'N/A')} ({opt.get('expiry', '')})")

    if data.get("notes"):
        lines.append("\nSignal notes:")
        for n in data["notes"][:5]:
            lines.append(f"  · {n}")

    return "\n".join(lines)


async def tool_get_screener(
    min_conviction: float = 0.25,
    signal: str | None = None,
    archetype: str | None = None,
    limit: int = 20,
) -> str:
    """Top conviction signals across all tracked tickers."""
    params: dict = {"min_conviction": min_conviction, "limit": limit}
    if signal:
        params["signal"] = signal
    if archetype:
        params["archetype"] = archetype

    data = await _get("/v1/screener", params)
    if "error" in data:
        return f"Error: {data['error']}"

    rows = data.get("results", [])
    if not rows:
        return "No results matching criteria."

    lines = [f"**Screener — {len(rows)} results** (min conviction {min_conviction:+.2f})\n"]
    for r in rows:
        sig = r.get("signal", "?")
        tick = r.get("ticker", "?")
        conv = r.get("conviction", 0.0)
        arch = r.get("archetype", "?")
        traj = r.get("trajectory", "?")
        lines.append(f"  {tick:6s}  {sig:14s}  {conv:+.3f}  [{arch}/{traj}]")

    return "\n".join(lines)


async def tool_get_ecl(industry: str | None = None) -> str:
    """Live macro regime + ECL module scores."""
    params = {}
    if industry:
        params["industry"] = industry
    data = await _get("/v1/ecl", params or None)
    if "error" in data:
        return f"Error: {data['error']}"

    regime = data.get("regime", "unknown")
    roc = data.get("roc_multiplier", 1.0)
    modules = data.get("modules", {})

    lines = [
        f"**ECL Regime: {regime.upper()}**  (ROC multiplier: {roc:.2f})",
        "",
        "Module scores (positive = stress/headwind, negative = tailwind):",
    ]
    for name, score in modules.items():
        if isinstance(score, (int, float)):
            bar = "▓" * int(abs(score) * 10) + "░" * (10 - int(abs(score) * 10))
            sign = "+" if score >= 0 else ""
            lines.append(f"  {name:8s}  {sign}{score:.3f}  [{bar}]")

    if data.get("notes"):
        lines.append("\nNotes:")
        for n in data["notes"][:4]:
            lines.append(f"  · {n}")

    return "\n".join(lines)


async def tool_get_horizons(ticker: str) -> str:
    """3M/6M/12M multi-horizon signals for a ticker."""
    data = await _get(f"/v1/signal/{ticker.upper()}/horizons")
    if "error" in data:
        return f"Error: {data['error']}"

    lines = [f"**{ticker.upper()} — Multi-Horizon Signals**\n"]
    for horizon in ["3m", "6m", "12m"]:
        h = data.get(horizon, {})
        if h:
            sig = h.get("signal", "?")
            conv = h.get("conviction", 0.0)
            pattern = h.get("divergence_pattern", "")
            lines.append(f"  {horizon.upper()}:  {sig:14s}  {conv:+.3f}  {pattern}")

    div = data.get("divergence")
    if div:
        lines.append(f"\nDivergence: {div}")

    return "\n".join(lines)


async def tool_get_supply_chain(ticker: str) -> str:
    """Upstream supply chain risk for a ticker's industry."""
    data = await _get(f"/v1/supply-chain/{ticker.upper()}")
    if "error" in data:
        return f"Error: {data['error']}"

    risk = data.get("supply_chain_adjustment", 0.0)
    industry = data.get("industry", "unknown")
    upstream = data.get("upstream_industries", [])
    reasons = data.get("reasons", [])

    lines = [
        f"**{ticker.upper()} Supply Chain Risk** ({industry})",
        f"Aggregate adjustment: {risk:+.3f}",
    ]
    if upstream:
        lines.append(f"Upstream dependencies: {', '.join(upstream)}")
    if reasons:
        lines.append("\nSignal details:")
        for r in reasons[:6]:
            lines.append(f"  · {r}")

    return "\n".join(lines)


async def tool_get_pfsl(ticker: str, industry: str | None = None) -> str:
    """Pre-financial signal layer (EDGAR) for a ticker."""
    params = {}
    if industry:
        params["industry"] = industry
    data = await _get(f"/v1/pfsl/{ticker.upper()}", params or None)
    if "error" in data:
        return f"Error: {data['error']}"

    agg = data.get("aggregate_score", 0.0)
    signals = data.get("signals", {})

    lines = [
        f"**{ticker.upper()} PFSL** (aggregate: {agg:+.3f})",
        "",
        "Signal families:",
    ]
    for family, info in signals.items():
        if isinstance(info, dict) and "score" in info:
            score = info["score"]
            notes = info.get("notes", [])
            note_str = notes[0] if notes else ""
            lines.append(f"  {family:22s}  {score:+.3f}  {note_str}")

    return "\n".join(lines)


async def tool_get_regime() -> str:
    """Current market regime summary across all ECL modules."""
    data = await _get("/v1/ecl")
    if "error" in data:
        return f"Error: {data['error']}"

    regime = data.get("regime", "unknown")
    roc = data.get("roc_multiplier", 1.0)
    macro = data.get("macro_summary", {})

    lines = [
        f"**Current Regime: {regime.upper()}**",
        f"ROC multiplier: {roc:.2f}x",
    ]

    if macro:
        for k, v in macro.items():
            lines.append(f"  {k}: {v}")

    lines += [
        "",
        f"Interpretation:",
        f"  ROC {roc:.2f} → BUY convictions scaled {'down' if roc < 1.0 else 'up'} by {abs(1.0 - roc) * 100:.0f}%",
    ]

    return "\n".join(lines)


async def tool_get_diligence(ticker: str, format: str = "full") -> str:
    """
    Full industrial operational diligence brief for a company (6 sections).
    Uses /v1/report when a commercial key is set, /v1/diligence for internal use.
    format: 'full' | 'json_compact' (compact strips verbose sections for token efficiency)
    """
    if GENOME_COMMERCIAL_KEY:
        data = await _post(f"/v1/report/{ticker.upper()}", {"format": format})
    else:
        data = await _post(f"/v1/diligence/{ticker.upper()}", {})
    if "error" in data:
        return f"Error fetching diligence brief for {ticker}: {data['error']}"
    if format == "json_compact":
        # Return only the most agent-useful fields
        return json.dumps({
            "ticker": data.get("ticker"),
            "company": data.get("company_name"),
            "conviction": data.get("headline", {}).get("conviction"),
            "flags": data.get("section_4_cross_engine_flags", []) or data.get("flags", []),
            "questions": data.get("section_6_diligence_priorities", []) or data.get("diligence_questions", []),
            "comparables": [c.get("ticker") for c in (data.get("section_5_comparables") or data.get("comparables") or [])],
        }, indent=2)
    return json.dumps(data, indent=2)


async def tool_ask_genome(question: str, context_tickers: list[str] | None = None) -> str:
    """
    Natural language Q&A over Genome signal data.
    Fetches relevant data then uses Claude to synthesize an answer.
    """
    if not ANTHROPIC_API_KEY:
        return "ask_genome requires ANTHROPIC_API_KEY to be set."

    # Gather context
    context_parts: list[str] = []

    # Always pull regime
    ecl_text = await tool_get_ecl()
    context_parts.append(f"=== CURRENT REGIME ===\n{ecl_text}")

    # Pull signals for mentioned tickers
    tickers = context_tickers or []
    # Extract ticker-like words from question (2-5 uppercase chars)
    import re
    found = re.findall(r'\b([A-Z]{2,5})\b', question)
    tickers = list(dict.fromkeys(tickers + found))[:5]

    for ticker in tickers:
        sig_text = await tool_get_signal(ticker)
        pfsl_text = await tool_get_pfsl(ticker)
        context_parts.append(f"=== {ticker} SIGNAL ===\n{sig_text}")
        context_parts.append(f"=== {ticker} PFSL ===\n{pfsl_text}")

    # Screener for broad questions
    if not tickers or any(w in question.lower() for w in ("best", "top", "screener", "which", "list")):
        screen_text = await tool_get_screener(min_conviction=0.25, limit=10)
        context_parts.append(f"=== SCREENER (top signals) ===\n{screen_text}")

    context = "\n\n".join(context_parts)

    # Claude call
    import httpx as _httpx
    payload = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 1024,
        "system": (
            "You are an industrial equity analyst with access to the Industrial Genome signal platform. "
            "Answer concisely using the data provided. Lead with the key insight. "
            "When giving conviction calls, always cite the archetype and trajectory. "
            "Use signal terminology: STRONG_BUY, BUY, WATCH, NEUTRAL, SELL, STRONG_SELL."
        ),
        "messages": [
            {
                "role": "user",
                "content": f"Context data:\n\n{context}\n\n---\n\nQuestion: {question}",
            }
        ],
    }

    async with _httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            json=payload,
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        if not resp.is_success:
            return f"Claude API error: {resp.status_code}"
        result = resp.json()
        return result["content"][0]["text"]


# ── MCP protocol (stdio) ────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "get_signal",
        "description": (
            "Get the full investment signal for an industrial ticker: conviction score, "
            "archetype, trajectory, ECL/PFSL adjustments, options structure, and regime impact."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker (e.g. HON, CAT, GE)"},
                "industry": {"type": "string", "description": "Optional industry hint (e.g. hvac, auto, aerospace_defense)"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_screener",
        "description": (
            "Screen all tracked industrial tickers by conviction, signal type, or archetype. "
            "Returns ranked list of top signals."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "min_conviction": {"type": "number", "description": "Minimum conviction threshold (default 0.25)"},
                "signal": {"type": "string", "description": "Filter by signal: STRONG_BUY, BUY, WATCH, NEUTRAL, SELL, STRONG_SELL"},
                "archetype": {"type": "string", "description": "Filter by archetype: constrained, oscillatory, fragile, resilient, saturated, decoupled"},
                "limit": {"type": "integer", "description": "Max results (default 20)"},
            },
        },
    },
    {
        "name": "get_ecl",
        "description": (
            "Get the current External Constraint Layer (ECL) macro regime and all 7 module scores: "
            "CPI (commodity), EPI (energy), LPI (labor), DCSI (demand/channel), DLPS (decision latency), "
            "ROC (regime multiplier), DCS (data confidence)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "industry": {"type": "string", "description": "Optional industry for sector-specific ECL weights"},
            },
        },
    },
    {
        "name": "get_horizons",
        "description": "Get 3M, 6M, and 12M multi-horizon investment signals for a ticker with divergence patterns.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_supply_chain",
        "description": "Get upstream supply chain risk for a ticker — which upstream industries are stressed and how much that drags the conviction score.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_pfsl",
        "description": (
            "Get Pre-Financial Signal Layer (PFSL) scores from SEC EDGAR financials: "
            "operational stress, demand signals, capex activity, narrative shift, labor/talent, "
            "and Reddit sentiment blend."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker"},
                "industry": {"type": "string", "description": "Optional industry hint"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_regime",
        "description": "Get a concise summary of the current macro market regime and its impact on conviction scaling.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_diligence",
        "description": (
            "Get a full industrial operational diligence brief for a company. "
            "Returns 6-section structured analysis covering operational genome, leadership, "
            "supply chain, financial signals, risk flags, and diligence priority questions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker (e.g. HON, CAT, ROK)"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "ask_genome",
        "description": (
            "Ask a natural language question and get an AI-synthesized answer grounded in live Genome data. "
            "Examples: 'Which industrials are best positioned right now?', "
            "'What is HON's signal and why?', 'Is the macro regime favorable for HVAC names?'"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Your question in plain English"},
                "context_tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of tickers to include as context",
                },
            },
            "required": ["question"],
        },
    },
]


async def handle_call(name: str, arguments: dict) -> str:
    if name == "get_signal":
        return await tool_get_signal(arguments["ticker"], arguments.get("industry"))
    elif name == "get_screener":
        return await tool_get_screener(
            min_conviction=arguments.get("min_conviction", 0.25),
            signal=arguments.get("signal"),
            archetype=arguments.get("archetype"),
            limit=arguments.get("limit", 20),
        )
    elif name == "get_ecl":
        return await tool_get_ecl(arguments.get("industry"))
    elif name == "get_horizons":
        return await tool_get_horizons(arguments["ticker"])
    elif name == "get_supply_chain":
        return await tool_get_supply_chain(arguments["ticker"])
    elif name == "get_pfsl":
        return await tool_get_pfsl(arguments["ticker"], arguments.get("industry"))
    elif name == "get_regime":
        return await tool_get_regime()
    elif name == "get_diligence":
        return await tool_get_diligence(arguments["ticker"])
    elif name == "ask_genome":
        return await tool_ask_genome(
            arguments["question"],
            arguments.get("context_tickers"),
        )
    else:
        return f"Unknown tool: {name}"


async def run_stdio():
    """MCP stdio transport — reads JSON-RPC from stdin, writes to stdout."""
    while True:
        line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg_id = msg.get("id")
        method = msg.get("method", "")

        async def send(result=None, error=None):
            resp: dict = {"jsonrpc": "2.0", "id": msg_id}
            if error:
                resp["error"] = error
            else:
                resp["result"] = result
            print(json.dumps(resp), flush=True)

        if method == "initialize":
            await send({
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "genome", "version": "1.0.0"},
            })

        elif method == "tools/list":
            await send({"tools": TOOLS})

        elif method == "tools/call":
            params = msg.get("params", {})
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})
            try:
                result_text = await handle_call(tool_name, tool_args)
                await send({
                    "content": [{"type": "text", "text": result_text}],
                    "isError": False,
                })
            except Exception as e:
                await send({
                    "content": [{"type": "text", "text": f"Tool error: {e}"}],
                    "isError": True,
                })

        elif method == "notifications/initialized":
            pass  # ack, no response needed

        else:
            if msg_id is not None:
                await send(error={"code": -32601, "message": f"Method not found: {method}"})


if __name__ == "__main__":
    asyncio.run(run_stdio())

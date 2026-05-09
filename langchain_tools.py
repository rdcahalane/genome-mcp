"""
Industrial Genome — LangChain Tool Wrappers

Exposes Genome signal intelligence as LangChain BaseTool subclasses so any
LangChain agent can query live industrial signals, screener results, macro
regime, and diligence reports.

Usage:
    from langchain_tools import get_genome_tools
    from langchain.agents import initialize_agent, AgentType

    tools = get_genome_tools()
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
    agent.run("Which industrials have the strongest conviction right now?")

Config (environment variables):
    GENOME_API_KEY    — Internal key (X-API-Key header). Falls back to built-in.
    GENOME_API_URL    — API base URL. Defaults to https://genome-api-production.up.railway.app
    GENOME_COMMERCIAL_KEY — Commercial key (gk_live_*). Takes precedence if set.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Type

import httpx
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


# ── Config ─────────────────────────────────────────────────────────────────────

_BASE_URL = os.environ.get(
    "GENOME_API_URL", "https://genome-api-production.up.railway.app"
)
_COMMERCIAL_KEY = os.environ.get("GENOME_COMMERCIAL_KEY", "")
_INTERNAL_KEY = os.environ.get("GENOME_API_KEY", "")


def _auth_headers() -> dict:
    """Return auth header — commercial key takes precedence over internal."""
    key = _COMMERCIAL_KEY or _INTERNAL_KEY
    if not key:
        return {}
    if _COMMERCIAL_KEY:
        return {"Authorization": f"Bearer {key}"}
    return {"X-API-Key": key}


def _get(path: str, params: dict | None = None) -> dict:
    try:
        with httpx.Client(base_url=_BASE_URL, timeout=60.0) as client:
            resp = client.get(path, params=params, headers=_auth_headers())
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return {"error": str(e)}


def _post(path: str, body: dict) -> dict:
    try:
        with httpx.Client(base_url=_BASE_URL, timeout=60.0) as client:
            resp = client.post(path, json=body, headers=_auth_headers())
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return {"error": str(e)}


# ── Input schemas ───────────────────────────────────────────────────────────────

class GetSignalInput(BaseModel):
    ticker: str = Field(..., description="Stock ticker, e.g. HON, CAT, GE, ROK")
    industry: Optional[str] = Field(
        None,
        description="Optional industry hint: hvac, auto, aerospace_defense, utilities, "
                    "energy_midstream, financials, tech, reit",
    )


class ScreenerInput(BaseModel):
    limit: int = Field(20, description="Max results (1–100)")
    min_conviction: float = Field(
        0.25,
        description="Minimum conviction threshold (-1.0 to +1.0). "
                    "0.25=WATCH, 0.50=BUY, 0.72=STRONG_BUY",
    )
    signal: Optional[str] = Field(
        None,
        description="Filter by signal: STRONG_BUY, BUY, WATCH, NEUTRAL, SELL, "
                    "STRONG_SELL, EARLY_WARNING",
    )
    archetype: Optional[str] = Field(
        None,
        description="Filter by archetype: constrained, oscillatory, fragile, "
                    "resilient, saturated, decoupled",
    )


class MacroRegimeInput(BaseModel):
    industry: Optional[str] = Field(
        None,
        description="Optional industry for sector-specific ECL weights. "
                    "Omit for general industrial regime.",
    )


class DiligenceInput(BaseModel):
    ticker: str = Field(..., description="Stock ticker, e.g. HON, CAT, ROK")
    format: str = Field(
        "json_compact",
        description="'full' for complete report, 'json_compact' for conviction + "
                    "flags + questions + comparables (token-efficient)",
    )


class HorizonsInput(BaseModel):
    ticker: str = Field(..., description="Stock ticker")


class SupplyChainInput(BaseModel):
    ticker: str = Field(..., description="Stock ticker")


class PFSLInput(BaseModel):
    ticker: str = Field(..., description="Stock ticker")
    industry: Optional[str] = Field(None, description="Optional industry hint")


# ── Tool implementations ────────────────────────────────────────────────────────

class GetIndustrialSignalTool(BaseTool):
    """
    Get the full Genome investment signal for an industrial ticker.

    Returns conviction score, archetype (constrained/oscillatory/fragile/
    resilient/saturated/decoupled), trajectory (improving/stable/deteriorating),
    ECL and PFSL adjustments, regime multiplier, supply chain risk drag,
    options structure recommendation, and signal notes.

    Signal thresholds:
      STRONG_BUY  > +0.72   BUY  > +0.50   WATCH  > +0.25
      NEUTRAL  -0.25 to +0.25
      SELL < -0.45   STRONG_SELL < -0.65

    Use this before any investment discussion about an industrial, utility,
    financial, or tech company.
    """

    name: str = "get_industrial_signal"
    description: str = (
        "Get the full Genome investment signal for an industrial ticker. "
        "Returns conviction score, archetype, trajectory, ECL/PFSL adjustments, "
        "regime multiplier, supply chain risk, and options structure. "
        "Input: ticker (required), industry (optional)."
    )
    args_schema: Type[BaseModel] = GetSignalInput

    def _run(self, ticker: str, industry: Optional[str] = None) -> str:
        params = {}
        if industry:
            params["industry"] = industry
        data = _get(f"/v1/signal/{ticker.upper()}", params or None)
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
            f"{ticker.upper()} — {sig}",
            f"Conviction: {conv:+.3f}",
            f"Archetype: {arch} / {traj}",
            f"ECL adjustment: {ecl_adj:+.3f}  |  PFSL adjustment: {pfsl_adj:+.3f}",
            f"Regime multiplier: {regime:.2f}  |  Supply chain risk: {sc_risk:+.3f}",
        ]

        if data.get("archetype_transition"):
            t = data["archetype_transition"]
            lines.append(f"Archetype transition: {t.get('from')} → {t.get('to')}")

        if data.get("options"):
            opt = data["options"]
            lines.append(f"Options: {opt.get('strategy', 'N/A')} ({opt.get('expiry', '')})")

        if data.get("notes"):
            lines.append("Signal notes:")
            for n in data["notes"][:5]:
                lines.append(f"  - {n}")

        return "\n".join(lines)

    async def _arun(self, ticker: str, industry: Optional[str] = None) -> str:
        raise NotImplementedError("Use sync _run or wrap with asyncio.to_thread")


class ScreenIndustrialSignalsTool(BaseTool):
    """
    Screen all tracked industrial tickers by conviction, signal type, or archetype.

    Returns a ranked list of matching tickers. Use this when you need to find
    the best-positioned industrials, identify high-conviction opportunities, or
    scan for specific archetypes (e.g. all constrained companies improving).

    Examples:
      - "Which industrials have the strongest buy signals?" → min_conviction=0.50
      - "Show me fragile companies" → archetype="fragile"
      - "Any strong sells I should know about?" → signal="STRONG_SELL"
    """

    name: str = "screen_industrial_signals"
    description: str = (
        "Screen all tracked industrial tickers by conviction, signal type, or archetype. "
        "Returns ranked list of top signals. "
        "Inputs: limit (default 20), min_conviction (default 0.25), "
        "signal (optional: STRONG_BUY/BUY/WATCH/NEUTRAL/SELL/STRONG_SELL), "
        "archetype (optional: constrained/oscillatory/fragile/resilient/saturated/decoupled)."
    )
    args_schema: Type[BaseModel] = ScreenerInput

    def _run(
        self,
        limit: int = 20,
        min_conviction: float = 0.25,
        signal: Optional[str] = None,
        archetype: Optional[str] = None,
    ) -> str:
        params: dict = {"min_conviction": min_conviction, "limit": limit}
        if signal:
            params["signal"] = signal
        if archetype:
            params["archetype"] = archetype

        data = _get("/v1/screener", params)
        if "error" in data:
            return f"Error: {data['error']}"

        rows = data.get("results", [])
        if not rows:
            return "No results matching criteria."

        lines = [f"Screener — {len(rows)} results (min conviction {min_conviction:+.2f})\n"]
        for r in rows:
            tick = r.get("ticker", "?")
            sig = r.get("signal", "?")
            conv = r.get("conviction", 0.0)
            arch = r.get("archetype", "?")
            traj = r.get("trajectory", "?")
            lines.append(f"  {tick:6s}  {sig:14s}  {conv:+.3f}  [{arch}/{traj}]")

        return "\n".join(lines)

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError("Use sync _run or wrap with asyncio.to_thread")


class GetMacroRegimeTool(BaseTool):
    """
    Get the current External Constraint Layer (ECL) macro regime and module scores.

    Returns all 7 ECL modules:
      CPI  — Commodity cost pressure (FRED: PPIACO, PPICRM + industry-specific)
      EPI  — Energy cost pressure (WTI crude, nat gas, diesel)
      LPI  — Labor cost pressure (Employment Cost Index)
      DCSI — Demand/channel stress (manufacturing orders, consumer sentiment)
      DLPS — Decision latency proxy (vs. industry median)
      ROC  — Regime multiplier (0.6–1.3×, scales all conviction scores)
      DCS  — Data confidence

    Always call this first when advising on sector positioning. The ROC multiplier
    directly scales every BUY and SELL signal in the platform.
    """

    name: str = "get_macro_regime"
    description: str = (
        "Get the current ECL macro regime and all 7 module scores: CPI (commodity), "
        "EPI (energy), LPI (labor), DCSI (demand/channel), DLPS (decision latency), "
        "ROC (regime multiplier that scales all conviction scores), DCS (data confidence). "
        "Input: industry (optional, for sector-specific ECL weights)."
    )
    args_schema: Type[BaseModel] = MacroRegimeInput

    def _run(self, industry: Optional[str] = None) -> str:
        params = {}
        if industry:
            params["industry"] = industry
        data = _get("/v1/ecl", params or None)
        if "error" in data:
            return f"Error: {data['error']}"

        regime = data.get("regime", "unknown")
        roc = data.get("roc_multiplier", 1.0)
        modules = data.get("modules", {})

        lines = [
            f"ECL Regime: {regime.upper()}  (ROC multiplier: {roc:.2f}x)",
            "",
            "Module scores (positive = headwind/stress, negative = tailwind):",
        ]
        for name, score in modules.items():
            if isinstance(score, (int, float)):
                sign = "+" if score >= 0 else ""
                lines.append(f"  {name:8s}  {sign}{score:.3f}")

        if data.get("notes"):
            lines.append("\nNotes:")
            for n in data["notes"][:4]:
                lines.append(f"  - {n}")

        return "\n".join(lines)

    async def _arun(self, industry: Optional[str] = None) -> str:
        raise NotImplementedError("Use sync _run or wrap with asyncio.to_thread")


class GetDiligenceReportTool(BaseTool):
    """
    Get a full industrial operational diligence brief for a company.

    Returns a 6-section structured analysis:
      1. Operational genome fingerprint (system ID dimensions)
      2. Leadership and organizational signals
      3. Supply chain risk and upstream dependencies
      4. Pre-financial signal layer from SEC EDGAR (inventory, capex, margins)
      5. Comparable companies by archetype similarity
      6. Priority diligence questions for deal teams

    Use this for M&A diligence, PE platform assessments, or any deep-dive
    company research. The 'json_compact' format returns only the most
    agent-useful fields: conviction, risk flags, diligence questions, comparables.
    """

    name: str = "get_diligence_report"
    description: str = (
        "Get a full industrial operational diligence brief for a company. "
        "6-section analysis: genome fingerprint, leadership signals, supply chain, "
        "EDGAR pre-financial signals, comparable companies, and diligence questions. "
        "Inputs: ticker (required), format ('full' or 'json_compact', default json_compact)."
    )
    args_schema: Type[BaseModel] = DiligenceInput

    def _run(self, ticker: str, format: str = "json_compact") -> str:
        if _COMMERCIAL_KEY:
            data = _post(f"/v1/report/{ticker.upper()}", {"format": format})
        else:
            data = _post(f"/v1/diligence/{ticker.upper()}", {})

        if "error" in data:
            return f"Error fetching diligence for {ticker}: {data['error']}"

        if format == "json_compact":
            compact = {
                "ticker": data.get("ticker"),
                "company": data.get("company_name"),
                "conviction": data.get("headline", {}).get("conviction"),
                "flags": (
                    data.get("section_4_cross_engine_flags", [])
                    or data.get("flags", [])
                ),
                "questions": (
                    data.get("section_6_diligence_priorities", [])
                    or data.get("diligence_questions", [])
                ),
                "comparables": [
                    c.get("ticker")
                    for c in (
                        data.get("section_5_comparables")
                        or data.get("comparables")
                        or []
                    )
                ],
            }
            return json.dumps(compact, indent=2)

        return json.dumps(data, indent=2)

    async def _arun(self, ticker: str, format: str = "json_compact") -> str:
        raise NotImplementedError("Use sync _run or wrap with asyncio.to_thread")


class GetMultiHorizonSignalsTool(BaseTool):
    """
    Get 3M, 6M, and 12M multi-horizon investment signals for a ticker.

    Returns signal, conviction, and divergence pattern for each horizon.
    Useful for detecting timeframe divergences — e.g. a 3M SELL with a 12M BUY
    suggests a near-term flush before a longer-term recovery.

    Known divergences:
      oscillatory:stable at 3M → SHORT (−4.0% mean excess return)
      resilient:* at 12M → strongest long (+77.1% XS vs XLI)
    """

    name: str = "get_multi_horizon_signals"
    description: str = (
        "Get 3M, 6M, and 12M multi-horizon investment signals for a ticker. "
        "Detects timeframe divergences (e.g. near-term SELL + long-term BUY). "
        "Input: ticker (required)."
    )
    args_schema: Type[BaseModel] = HorizonsInput

    def _run(self, ticker: str) -> str:
        data = _get(f"/v1/signal/{ticker.upper()}/horizons")
        if "error" in data:
            return f"Error: {data['error']}"

        lines = [f"{ticker.upper()} — Multi-Horizon Signals\n"]
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

    async def _arun(self, ticker: str) -> str:
        raise NotImplementedError("Use sync _run or wrap with asyncio.to_thread")


class GetSupplyChainRiskTool(BaseTool):
    """
    Get upstream supply chain risk for a ticker.

    Returns aggregate supply chain conviction adjustment, which upstream
    industries are stressed, and the specific signal details driving the risk.
    A negative adjustment (e.g. −0.12) means upstream stress is dragging
    down the ticker's conviction score.

    Uses 19 upstream signal types including ISM, inventory, freight, steel
    capacity utilization, and port volume.
    """

    name: str = "get_supply_chain_risk"
    description: str = (
        "Get upstream supply chain risk for a ticker. Returns aggregate conviction "
        "adjustment, stressed upstream industries, and signal details. "
        "Negative adjustment = upstream stress dragging down conviction. "
        "Input: ticker (required)."
    )
    args_schema: Type[BaseModel] = SupplyChainInput

    def _run(self, ticker: str) -> str:
        data = _get(f"/v1/supply-chain/{ticker.upper()}")
        if "error" in data:
            return f"Error: {data['error']}"

        risk = data.get("supply_chain_adjustment", 0.0)
        industry = data.get("industry", "unknown")
        upstream = data.get("upstream_industries", [])
        reasons = data.get("reasons", [])

        lines = [
            f"{ticker.upper()} Supply Chain Risk ({industry})",
            f"Aggregate adjustment: {risk:+.3f}",
        ]
        if upstream:
            lines.append(f"Upstream dependencies: {', '.join(upstream)}")
        if reasons:
            lines.append("Signal details:")
            for r in reasons[:6]:
                lines.append(f"  - {r}")

        return "\n".join(lines)

    async def _arun(self, ticker: str) -> str:
        raise NotImplementedError("Use sync _run or wrap with asyncio.to_thread")


class GetPFSLTool(BaseTool):
    """
    Get Pre-Financial Signal Layer (PFSL) scores from SEC EDGAR for a ticker.

    PFSL extracts leading signals from quarterly financials before they show up
    in reported earnings. Signal families:
      Operational Stress   — inventory build, gross margin compression
      Demand Signals       — revenue YoY velocity, 3-quarter momentum
      Capex/Physical       — capex intensity trend (expansion vs retrenchment)
      Narrative Shift      — operating income trend proxy for earnings tone

    A negative aggregate PFSL score is a forward-looking headwind. A positive
    score amplifies BUY conviction.
    """

    name: str = "get_pfsl_signals"
    description: str = (
        "Get Pre-Financial Signal Layer (PFSL) scores from SEC EDGAR for a ticker. "
        "Extracts leading signals before earnings: operational stress, demand velocity, "
        "capex activity, and narrative shift. "
        "Inputs: ticker (required), industry (optional)."
    )
    args_schema: Type[BaseModel] = PFSLInput

    def _run(self, ticker: str, industry: Optional[str] = None) -> str:
        params = {}
        if industry:
            params["industry"] = industry
        data = _get(f"/v1/pfsl/{ticker.upper()}", params or None)
        if "error" in data:
            return f"Error: {data['error']}"

        agg = data.get("aggregate_score", 0.0)
        signals = data.get("signals", {})

        lines = [
            f"{ticker.upper()} PFSL (aggregate: {agg:+.3f})",
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

    async def _arun(self, ticker: str, industry: Optional[str] = None) -> str:
        raise NotImplementedError("Use sync _run or wrap with asyncio.to_thread")


# ── Public API ──────────────────────────────────────────────────────────────────

def get_genome_tools() -> list[BaseTool]:
    """
    Return the full list of Genome LangChain tools.

    Use with any LangChain agent:

        from langchain_tools import get_genome_tools
        from langchain.agents import initialize_agent, AgentType

        tools = get_genome_tools()
        agent = initialize_agent(
            tools, llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        agent.run("Which industrials have the strongest conviction right now?")

    Required env vars:
        GENOME_API_KEY     — or GENOME_COMMERCIAL_KEY (gk_live_* format)
        GENOME_API_URL     — defaults to https://genome-api-production.up.railway.app
    """
    return [
        GetIndustrialSignalTool(),
        ScreenIndustrialSignalsTool(),
        GetMacroRegimeTool(),
        GetDiligenceReportTool(),
        GetMultiHorizonSignalsTool(),
        GetSupplyChainRiskTool(),
        GetPFSLTool(),
    ]


def get_genome_tools_core() -> list[BaseTool]:
    """
    Return only the 4 primary tools — signal, screener, regime, diligence.
    Use when you want a minimal toolset to reduce agent decision overhead.
    """
    return [
        GetIndustrialSignalTool(),
        ScreenIndustrialSignalsTool(),
        GetMacroRegimeTool(),
        GetDiligenceReportTool(),
    ]

# Industrial Genome — Agent Integration Guide

Industrial Genome exposes live signal intelligence for industrial companies as AI-native tools. Any agent — Claude, GPT-4, LangChain, custom — can query conviction scores, macro regime, screener results, and diligence reports in real time.

**API base:** `https://genome-api-production.up.railway.app`  
**Auth:** `X-API-Key: <your-key>` (internal) or `Authorization: Bearer gk_live_<your-key>` (commercial)  
**Keys:** [genome.axiomsystems.io/api-keys](https://genome.axiomsystems.io/api-keys)

---

## Quickstart: Claude Desktop (MCP)

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "genome": {
      "command": "python3",
      "args": ["/path/to/genome/mcp/genome_mcp.py"],
      "env": {
        "GENOME_API_KEY": "your-key-here"
      }
    }
  }
}
```

Restart Claude Desktop. You now have 9 tools: `get_signal`, `get_screener`, `get_ecl`, `get_horizons`, `get_supply_chain`, `get_pfsl`, `get_regime`, `get_diligence`, `ask_genome`.

**Test it:** Ask Claude — *"What is HON's current signal?"*

---

## Quickstart: OpenAI Function Calling

Load the spec from `mcp/openai_functions.json` and pass it as `tools` to any Chat Completions call:

```python
import json, os
from openai import OpenAI
import httpx

client = OpenAI()
GENOME_KEY = os.environ["GENOME_API_KEY"]
GENOME_BASE = "https://genome-api-production.up.railway.app"

with open("mcp/openai_functions.json") as f:
    tools = json.load(f)

messages = [
    {"role": "user", "content": "What industrials have the strongest buy signals right now?"}
]

response = client.chat.completions.create(
    model="gpt-4o",
    tools=tools,
    messages=messages,
)

# Dispatch tool calls
for tool_call in response.choices[0].message.tool_calls or []:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    if name == "screen_industrial_signals":
        result = httpx.get(
            f"{GENOME_BASE}/v1/screener",
            params=args,
            headers={"X-API-Key": GENOME_KEY},
        ).json()

    elif name == "get_industrial_signal":
        ticker = args.pop("ticker")
        result = httpx.get(
            f"{GENOME_BASE}/v1/signal/{ticker}",
            params=args,
            headers={"X-API-Key": GENOME_KEY},
        ).json()

    elif name == "get_macro_regime":
        result = httpx.get(
            f"{GENOME_BASE}/v1/ecl",
            params=args,
            headers={"X-API-Key": GENOME_KEY},
        ).json()

    elif name == "get_diligence_report":
        ticker = args.pop("ticker")
        result = httpx.post(
            f"{GENOME_BASE}/v1/report/{ticker}",
            json=args,
            headers={"Authorization": f"Bearer {GENOME_KEY}"},
        ).json()

    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)})

# Final response with tool results injected
final = client.chat.completions.create(model="gpt-4o", tools=tools, messages=messages)
print(final.choices[0].message.content)
```

---

## Quickstart: LangChain

```python
pip install langchain langchain-openai httpx pydantic
```

```python
import os
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

# Genome tools are in mcp/langchain_tools.py — copy or add to PYTHONPATH
from langchain_tools import get_genome_tools

os.environ["GENOME_API_KEY"] = "your-key-here"

llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = get_genome_tools()

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

agent.run("Which industrials are best positioned right now given the macro regime?")
```

**Minimal toolset** (4 tools instead of 7, lower decision overhead):

```python
from langchain_tools import get_genome_tools_core
tools = get_genome_tools_core()
```

**With LangChain Expression Language (LCEL):**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_tools import get_genome_tools

tools = get_genome_tools()
llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an industrial equity analyst using the Genome signal platform."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
executor.invoke({"input": "Analyze ROK's current signal and supply chain risk."})
```

---

## Direct REST (Any Agent Framework)

Every endpoint accepts `X-API-Key` in the header or `Authorization: Bearer gk_live_*` for commercial keys.

```bash
# Single ticker signal
curl -H "X-API-Key: $GENOME_API_KEY" \
  https://genome-api-production.up.railway.app/v1/signal/HON

# Screener — top 10 BUY signals
curl -H "X-API-Key: $GENOME_API_KEY" \
  "https://genome-api-production.up.railway.app/v1/screener?signal=BUY&limit=10"

# Macro regime
curl -H "X-API-Key: $GENOME_API_KEY" \
  https://genome-api-production.up.railway.app/v1/ecl

# Multi-horizon (3M/6M/12M)
curl -H "X-API-Key: $GENOME_API_KEY" \
  https://genome-api-production.up.railway.app/v1/signal/CAT/horizons

# Batch signals (up to 500 tickers)
curl -X POST -H "X-API-Key: $GENOME_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"tickers":["HON","CAT","GE","ROK","EMR"]}' \
  https://genome-api-production.up.railway.app/v1/signals/batch

# Diligence brief (commercial key required)
curl -X POST -H "Authorization: Bearer gk_live_your-key" \
  -H "Content-Type: application/json" \
  -d '{"format":"json_compact"}' \
  https://genome-api-production.up.railway.app/v1/report/ROK
```

---

## Agent Prompts That Work Well With Genome

Genome tools are most useful when the agent combines regime context with individual signals. These prompts produce high-quality analysis:

**Sector scan:**
> "Pull the current macro regime, then screen for all industrials with conviction above 0.50. Summarize the top 5 and explain what's driving each signal."

**Single-company deep dive:**
> "Give me a complete picture of [TICKER]: current signal, multi-horizon outlook (3M/6M/12M), supply chain risk, and PFSL pre-financial signals. What's the options structure recommendation?"

**M&A diligence:**
> "I'm evaluating [TICKER] as an acquisition target. Pull the diligence brief and identify the top 3 operational risk flags and the priority questions I should ask management."

**Macro-aware positioning:**
> "What is the current ECL regime? Given the regime multiplier, which archetype-trajectory combinations are most attractive right now? Give me 5 specific tickers."

**Earnings pre-read:**
> "Before [TICKER]'s earnings call, pull the PFSL signals from EDGAR and the current conviction score. What should I watch for?"

**Comparative analysis:**
> "Compare HON, EMR, and ROK across conviction, archetype, and supply chain risk. Which is best positioned for the next 6 months and why?"

**Volatility/options play:**
> "Identify all oscillatory:stable tickers in the screener. These are candidates for straddle strategies. Show me their current signals and options recommendations."

---

## Signal Reference

| Signal | Conviction | Walk-forward Accuracy |
|--------|-----------|----------------------|
| STRONG_BUY | > +0.72 | 86% win rate (oscillatory:improving) |
| BUY | > +0.50 | 100% WR (constrained, N=4) |
| WATCH | > +0.25 | Positive edge, early-stage |
| NEUTRAL | −0.25 to +0.25 | No directional edge |
| SELL | < −0.45 | 61–67% win rate |
| STRONG_SELL | < −0.65 | 74% win rate |
| EARLY_WARNING | Near neutral + stress conditions | Pre-deterioration signal |

**Archetypes:**

| Archetype | Description | Default Signal |
|-----------|-------------|----------------|
| constrained | Capacity-bound, demand exceeds supply | BUY |
| oscillatory | Inventory/demand cycles; trajectory matters most | Varies by trajectory |
| resilient | Durable competitive position, pricing power | BUY / STRONG_BUY |
| fragile | Demand-dependent, limited buffer; improving = TRAP | SELL / STRONG_SELL |
| saturated | Market saturation, growth ceiling | NEUTRAL / SELL |
| decoupled | Disconnected from macro cycle | WATCH |

**ECL Regime Multiplier (ROC):**
- > 1.0x — BUY signals amplified (favorable macro)
- = 1.0x — Neutral regime
- < 1.0x — BUY signals damped, SELL signals amplified (stressed macro)
- Typical range: 0.6× (stress) to 1.3× (expansion)

---

## Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/signal/{ticker}` | GET | Full signal: conviction, archetype, trajectory, adjustments |
| `/v1/signal/{ticker}/horizons` | GET | 3M/6M/12M multi-horizon signals |
| `/v1/signals/batch` | POST | Batch signals for up to 500 tickers |
| `/v1/screener` | GET | Screener: filter by conviction, signal, archetype |
| `/v1/ecl` | GET | Macro regime + 7 ECL module scores |
| `/v1/pfsl/{ticker}` | GET | Pre-financial signals from SEC EDGAR |
| `/v1/supply-chain/{ticker}` | GET | Upstream supply chain risk |
| `/v1/report/{ticker}` | POST | Full diligence brief (commercial key) |
| `/v1/diligence/{ticker}` | POST | Full diligence brief (internal key) |
| `/v1/options/{ticker}` | GET | Options structure recommendation |
| `/v1/anomalies` | GET | Active signal contradictions |
| `/v1/hypotheses` | GET | Signal hypothesis tracking |
| `/health` | GET | Health check |

Full API reference: `https://genome-api-production.up.railway.app/docs`

---

## Pricing and Keys

| Tier | Price | Calls/mo | Features |
|------|-------|----------|----------|
| Developer | Free | 500 | Signal + screener + ECL |
| Pro | $99/mo | 10,000 | All endpoints + diligence |
| Institutional | $499/mo | Unlimited | All endpoints + batch + priority support |

Get your key: [genome.axiomsystems.io/api-keys](https://genome.axiomsystems.io/api-keys)

Commercial keys use `Authorization: Bearer gk_live_<key>` format and unlock `/v1/report/{ticker}`.

---

## Files in This Directory

| File | Purpose |
|------|---------|
| `genome_mcp.py` | MCP server for Claude Desktop (stdio transport) |
| `openai_functions.json` | OpenAI function calling spec (4 primary tools) |
| `langchain_tools.py` | LangChain BaseTool wrappers (7 tools) |
| `AGENT_INTEGRATION.md` | This file |

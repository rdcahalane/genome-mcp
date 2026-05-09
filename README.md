<!-- mcp-name: io.github.rdcahalane/genome-industrial-intelligence -->
# Genome Industrial Intelligence — MCP Server

Industrial system intelligence for AI agents. Conviction scores, archetypes, and operational diligence for 10,000+ industrial companies from public signals. Built for PE firms, M&A advisors, and AI agents doing industrial research.

**→ Get an API key:** [genomestudio.vercel.app/dealmakers](https://genomestudio.vercel.app/dealmakers)  
**→ Available on Smithery:** [smithery.ai/servers/rdcahalane/genome-industrial-intelligence](https://smithery.ai/servers/rdcahalane/genome-industrial-intelligence)

## Tools

| Tool | Description |
|------|-------------|
| `get_signal` | Full investment signal for a ticker (conviction, archetype, ECL/PFSL, options) |
| `get_screener` | Top conviction signals across all tracked tickers |
| `get_ecl` | Live macro regime + 7 ECL module scores |
| `get_horizons` | 3M/6M/12M multi-horizon signals with divergence |
| `get_supply_chain` | Upstream supply chain risk and conviction drag |
| `get_pfsl` | Pre-financial signal layer from SEC EDGAR |
| `get_regime` | Current macro regime summary |
| `ask_genome` | Natural language Q&A grounded in live Genome data |

## Setup

### Option 1: Smithery (recommended — no local install)

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "genome": {
      "command": "npx",
      "args": ["-y", "@smithery/cli@latest", "run", "rdcahalane/genome-industrial-intelligence", "--key", "<your-smithery-key>"],
      "env": {
        "GENOME_COMMERCIAL_KEY": "gk_live_<your-key>"
      }
    }
  }
}
```

### Option 2: Direct (requires Python 3.11+)

```bash
pip install httpx mcp
```

Then add to Claude Desktop config:

```json
{
  "mcpServers": {
    "genome": {
      "command": "python3",
      "args": ["-m", "genome_mcp"],
      "env": {
        "GENOME_COMMERCIAL_KEY": "gk_live_<your-key>",
        "GENOME_API_URL": "https://genome-api-production.up.railway.app"
      }
    }
  }
}
```

### OpenAI Function Calling

See [`openai_functions.json`](./openai_functions.json) for the complete tool spec.

### LangChain

See [`langchain_tools.py`](./langchain_tools.py) for BaseTool implementations.

```python
from langchain_tools import get_genome_tools
tools = get_genome_tools()
```

## Example agent queries

- "Which industrials have the strongest buy signals right now?"
- "What's ROK's conviction and why?"
- "Is the current macro regime favorable for HVAC names?"
- "Give me a diligence brief on Emerson Electric"
- "Which PE-owned industrials are showing fragile archetypes?"
- "Compare HON and EMR on supply chain risk"
- "What's the CPI pipeline pressure on diversified industrials?"

## Pricing

Purchase at [genomestudio.vercel.app/dealmakers](https://genomestudio.vercel.app/dealmakers)

| Plan | Price | Lookups |
|------|-------|---------|
| Pay-as-you-go | $49 one-time | 10 lookups |
| Pro | $199/month | 200 lookups/month |
| Enterprise | Custom | Custom SLA |

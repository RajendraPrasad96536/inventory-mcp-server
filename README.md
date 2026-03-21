# Maharashtra Medicine Purchase — FastMCP Server

AI-powered MCP server that lets Claude (or any MCP client) analyse
Maharashtra wholesale medicine purchase data through 10 focused tools.

---

## Project Structure

```
medicine_mcp_server/
├── server.py                         # MCP server (all tools)
├── data/
│   └── maharashtra_wholesale_medicine_purchase.csv
├── pyproject.toml
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install fastmcp pandas google-generativeai
```

### 2. Set your Gemini API key (needed only for `natural_language_query`)

```bash
export GEMINI_API_KEY=AIza...
# or: export GOOGLE_API_KEY=AIza...
```

Get a free key at https://aistudio.google.com/app/apikey

### 3. Run locally (stdio transport — Claude Desktop / mcp-remote)

```bash
python server.py
```

### 4. Run as HTTP server (SSE transport — Azure App Service / any HTTP host)

```python
# In server.py, change the last line to:
mcp.run(transport="sse", host="0.0.0.0", port=8000)
```

Or pass via CLI:

```bash
fastmcp run server.py --transport sse --host 0.0.0.0 --port 8000
```

---

## Claude Desktop Config (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "medicine": {
      "command": "python",
      "args": ["/path/to/medicine_mcp_server/server.py"],
      "env": {
        "GEMINI_API_KEY": "AIza..."
      }
    }
  }
}
```

---

## Azure App Service Deployment

1. Push the project to your Azure App Service.
2. Set `ANTHROPIC_API_KEY` as an Application Setting.
3. Set startup command:
   ```
   fastmcp run server.py --transport sse --host 0.0.0.0 --port 8000
   ```
4. In Claude Desktop / `mcp-remote`, point to:
   ```
   https://<your-app>.azurewebsites.net/sse
   ```

---

## Available Tools

| # | Tool | Purpose |
|---|------|---------|
| 1 | `search_medicines` | Search by product / manufacturer / supplier / buyer |
| 2 | `get_invoice_details` | Full line-items for one or more invoices |
| 3 | `filter_by_schedule` | Filter by drug schedule (H, H1, X, G, OTC) |
| 4 | `get_expiry_alerts` | Medicines expiring within N days |
| 5 | `analyse_supplier` | Spend & invoice summary for a supplier |
| 6 | `analyse_buyer` | Purchase history & schedule mix for a buyer |
| 7 | `top_products_by_spend` | Ranked products by taxable amount / quantity |
| 8 | `gst_summary` | CGST / SGST / IGST breakdown by invoice/supplier/buyer |
| 9 | `cold_chain_and_narcotic_items` | Cold-chain & Schedule X items |
| 10 | `natural_language_query` | Free-form NL question answered by Claude |

---

## Example Queries (Natural Language Tool)

- "Which supplier sold the most Schedule H drugs?"
- "What is the total GST paid by Ganesh Medical Store?"
- "List all Cipla products purchased in April 2024."
- "Which medicines expire before December 2025?"
- "Show top 5 products by total spend."
- "Which invoices had the highest discount percentage?"
- "What is the average MRP of Schedule X drugs?"

---

## Extending to a Larger Dataset

The CSV path is set in `server.py`:

```python
CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "maharashtra_wholesale_medicine_purchase.csv")
```

Replace the CSV with a larger file using the same column schema and restart
the server. All tools will automatically work on the new data. For datasets
> 100k rows consider loading into Azure Cognitive Search and replacing the
`_load_df()` function with search-index queries.

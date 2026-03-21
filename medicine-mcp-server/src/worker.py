"""
FastMCP Server: Maharashtra Wholesale Medicine Purchase Analyser
================================================================
Cloudflare Workers deployment — Python Worker via Pyodide

Key differences from the Azure/local version:
  - CSV is loaded from Cloudflare R2 (no local filesystem)
  - Transport is Streamable HTTP (not SSE)
  - Entry point is a Durable Object wrapping the ASGI app
  - Secrets (GEMINI_API_KEY) come from Cloudflare Secrets via env

Tools exposed:
  1. search_medicines          – fuzzy search by product/manufacturer/supplier/buyer
  2. get_invoice_details       – full details of one or more invoices
  3. filter_by_schedule        – filter by drug schedule (H, H1, X, G, OTC)
  4. get_expiry_alerts         – medicines expiring within N days
  5. analyse_supplier          – purchase summary for a given supplier
  6. analyse_buyer             – purchase summary for a given buyer
  7. top_products_by_spend     – ranked products by taxable amount
  8. gst_summary               – GST breakdown by invoice / supplier / buyer
  9. cold_chain_and_narcotic_items – list cold-chain and narcotic items
 10. natural_language_query    – free-form NL query answered by Gemini
"""

import io
import json
import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from fastmcp import FastMCP

# ── Cloudflare-specific imports ────────────────────────────────────────────────
from js import Response                              # noqa: F401  (CF Workers JS bridge)
import asgi                                          # noqa: F401  (CF ASGI adapter)
from cloudflare.workers import DurableObject         # noqa: F401

# ── In-memory DataFrame cache (lives for the lifetime of the Worker isolate) ──
_DF_CACHE: Optional[pd.DataFrame] = None


def _parse_df(csv_text: str) -> pd.DataFrame:
    """Parse CSV text → cleaned DataFrame."""
    df = pd.read_csv(io.StringIO(csv_text))
    df["Invoice_Date"] = pd.to_datetime(df["Invoice_Date"], errors="coerce")
    df["Expiry_Date"]  = pd.to_datetime(df["Expiry_Date"],  format="%b-%y", errors="coerce")
    df["Mfg_Date"]     = pd.to_datetime(df["Mfg_Date"],     format="%b-%y", errors="coerce")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("")
    return df


async def _get_df(env) -> pd.DataFrame:
    """
    Fetch CSV from R2 on first request, then serve from in-memory cache.
    env.MEDICINE_DATA is the R2 bucket binding declared in wrangler.toml.
    """
    global _DF_CACHE
    if _DF_CACHE is not None:
        return _DF_CACHE

    r2_object = await env.MEDICINE_DATA.get(
        "maharashtra_wholesale_medicine_purchase.csv"
    )
    if r2_object is None:
        raise RuntimeError(
            "CSV not found in R2 bucket 'medicine-mcp-data'. "
            "Run: npx wrangler r2 object put medicine-mcp-data/"
            "maharashtra_wholesale_medicine_purchase.csv "
            "--file ./data/maharashtra_wholesale_medicine_purchase.csv"
        )

    csv_text = await r2_object.text()
    _DF_CACHE = _parse_df(csv_text)
    return _DF_CACHE


# ── Helper ─────────────────────────────────────────────────────────────────────
def _to_records(df: pd.DataFrame, max_rows: int = 100) -> list[dict]:
    """Convert DataFrame slice to JSON-serialisable list of dicts."""
    df = df.head(max_rows).copy()
    for col in df.select_dtypes(
        include=["datetime64[ns]", "datetime64[ns, UTC]"]
    ).columns:
        df[col] = df[col].dt.strftime("%Y-%m-%d").where(df[col].notna(), "")
    return df.where(pd.notnull(df), None).to_dict(orient="records")


# ── FastMCP server ─────────────────────────────────────────────────────────────
mcp = FastMCP(
    name="MaharashtraMedicinePurchase",
    instructions=(
        "Analyse Maharashtra wholesale medicine purchase data. "
        "Use the available tools to search products, filter by schedule, "
        "check expiry, analyse suppliers/buyers, and run GST breakdowns."
    ),
)


# ── Tool 1: Search medicines ───────────────────────────────────────────────────
@mcp.tool(
    description=(
        "Search purchase records by product name, manufacturer, supplier, or buyer. "
        "Returns matching rows sorted by Invoice_Date descending."
    )
)
async def search_medicines(
    query: str,
    search_in: str = "all",      # all | product | manufacturer | supplier | buyer
    max_results: int = 20,
) -> dict:
    """
    Args:
        query:       The search string (case-insensitive substring match).
        search_in:   Which column(s) to search. Options: all | product | manufacturer | supplier | buyer
        max_results: Maximum rows to return (default 20).
    """
    ctx = mcp.get_context()
    df  = await _get_df(ctx.request_context.request.state.env)
    df  = df.copy()
    q   = query.lower()

    col_map = {
        "product":      ["Product_Name"],
        "manufacturer": ["Manufacturer_Name"],
        "supplier":     ["Supplier_Name"],
        "buyer":        ["Buyer_Name"],
        "all":          ["Product_Name", "Manufacturer_Name", "Supplier_Name", "Buyer_Name"],
    }
    cols = col_map.get(search_in.lower(), col_map["all"])

    mask = pd.Series([False] * len(df), index=df.index)
    for col in cols:
        mask |= df[col].str.lower().str.contains(q, na=False)

    result  = df[mask].sort_values("Invoice_Date", ascending=False)
    records = _to_records(result, max_results)

    return {
        "query": query,
        "search_in": search_in,
        "total_matches": int(mask.sum()),
        "returned": len(records),
        "results": records,
    }


# ── Tool 2: Get invoice details ────────────────────────────────────────────────
@mcp.tool(
    description=(
        "Retrieve all line-items for one or more invoice numbers. "
        "Pass a single invoice like 'INV/2024-25/00341' or a comma-separated list."
    )
)
async def get_invoice_details(invoice_numbers: str) -> dict:
    """
    Args:
        invoice_numbers: Comma-separated invoice numbers, e.g. 'INV/2024-25/00341,INV/2024-25/00342'
    """
    ctx  = mcp.get_context()
    DF   = await _get_df(ctx.request_context.request.state.env)
    nums = [n.strip() for n in invoice_numbers.split(",")]
    df   = DF[DF["Invoice_No"].isin(nums)].copy()

    if df.empty:
        return {"error": f"No records found for invoice(s): {invoice_numbers}"}

    summary = (
        df.groupby("Invoice_No")
        .agg(
            Supplier=("Supplier_Name", "first"),
            Buyer=("Buyer_Name", "first"),
            Invoice_Date=("Invoice_Date", "first"),
            Line_Items=("Product_Name", "count"),
            Total_Taxable=("Taxable_Amount", "sum"),
            Total_GST=("CGST_Amt", "sum"),
            Total_Invoice_Amt=("Total_Invoice_Amt", "sum"),
        )
        .reset_index()
    )

    return {
        "invoices": _to_records(summary),
        "line_items": _to_records(df),
    }


# ── Tool 3: Filter by drug schedule ───────────────────────────────────────────
@mcp.tool(
    description=(
        "Filter purchase records by drug schedule. "
        "Valid schedules: Schedule H, Schedule H1, Schedule X, Schedule G, OTC."
    )
)
async def filter_by_schedule(
    schedule: str,
    supplier_name: Optional[str] = None,
    max_results: int = 50,
) -> dict:
    """
    Args:
        schedule:      Drug schedule name, e.g. 'Schedule H1' or 'OTC'.
        supplier_name: Optional – further filter by supplier (substring match).
        max_results:   Maximum rows to return.
    """
    ctx = mcp.get_context()
    DF  = await _get_df(ctx.request_context.request.state.env)
    df  = DF[DF["Drug_Schedule"].str.lower() == schedule.lower()].copy()

    if supplier_name:
        df = df[df["Supplier_Name"].str.lower().str.contains(
            supplier_name.lower(), na=False
        )]

    records = _to_records(df.sort_values("Invoice_Date", ascending=False), max_results)

    return {
        "schedule": schedule,
        "total_records": len(df),
        "returned": len(records),
        "results": records,
    }


# ── Tool 4: Expiry alerts ──────────────────────────────────────────────────────
@mcp.tool(
    description=(
        "List medicines expiring within the next N days (default 180). "
        "Useful for near-expiry stock monitoring."
    )
)
async def get_expiry_alerts(days_ahead: int = 180) -> dict:
    """
    Args:
        days_ahead: Number of days from today to check for expiry (default 180).
    """
    ctx    = mcp.get_context()
    DF     = await _get_df(ctx.request_context.request.state.env)
    today  = datetime.today()
    cutoff = today + timedelta(days=days_ahead)
    df     = DF.dropna(subset=["Expiry_Date"]).copy()

    expiring = df[(df["Expiry_Date"] >= today) & (df["Expiry_Date"] <= cutoff)]
    expired  = df[df["Expiry_Date"] < today]

    def _fmt(frame: pd.DataFrame) -> list[dict]:
        cols = ["Product_Name", "Manufacturer_Name", "Batch_No", "Expiry_Date",
                "Qty_Packs", "Buyer_Name", "Invoice_No"]
        return _to_records(frame[cols].sort_values("Expiry_Date"))

    return {
        "check_date":             today.strftime("%Y-%m-%d"),
        "cutoff_date":            cutoff.strftime("%Y-%m-%d"),
        "expiring_soon_count":    len(expiring),
        "already_expired_count":  len(expired),
        "expiring_soon":          _fmt(expiring),
        "already_expired":        _fmt(expired),
    }


# ── Tool 5: Supplier analysis ──────────────────────────────────────────────────
@mcp.tool(
    description=(
        "Summarise all purchases from a given supplier: "
        "total spend, top products, invoice list, and GST totals."
    )
)
async def analyse_supplier(supplier_name: str) -> dict:
    """
    Args:
        supplier_name: Full or partial supplier name (case-insensitive).
    """
    ctx = mcp.get_context()
    DF  = await _get_df(ctx.request_context.request.state.env)
    df  = DF[DF["Supplier_Name"].str.lower().str.contains(
        supplier_name.lower(), na=False
    )].copy()

    if df.empty:
        return {"error": f"No records found for supplier containing '{supplier_name}'"}

    top_products = (
        df.groupby("Product_Name")["Taxable_Amount"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
        .rename(columns={"Taxable_Amount": "Total_Taxable_Amount"})
    )

    invoice_summary = (
        df.groupby("Invoice_No")
        .agg(
            Date=("Invoice_Date", "first"),
            Buyer=("Buyer_Name", "first"),
            Items=("Product_Name", "count"),
            Total_Amt=("Total_Invoice_Amt", "sum"),
        )
        .reset_index()
    )

    return {
        "supplier_name":          df["Supplier_Name"].iloc[0],
        "supplier_gst":           df["Supplier_GST_No"].iloc[0],
        "total_invoices":         invoice_summary.shape[0],
        "total_line_items":       len(df),
        "total_taxable_amount":   round(float(df["Taxable_Amount"].sum()), 2),
        "total_gst_collected":    round(float(
            (df["CGST_Amt"] + df["SGST_Amt"] + df["IGST_Amt"]).sum()
        ), 2),
        "total_invoice_amount":   round(float(df["Total_Invoice_Amt"].sum()), 2),
        "top_products_by_spend":  _to_records(top_products),
        "invoices":               _to_records(invoice_summary),
    }


# ── Tool 6: Buyer analysis ─────────────────────────────────────────────────────
@mcp.tool(
    description=(
        "Summarise all purchases by a given buyer: "
        "total spend, top products purchased, and invoice history."
    )
)
async def analyse_buyer(buyer_name: str) -> dict:
    """
    Args:
        buyer_name: Full or partial buyer name (case-insensitive).
    """
    ctx = mcp.get_context()
    DF  = await _get_df(ctx.request_context.request.state.env)
    df  = DF[DF["Buyer_Name"].str.lower().str.contains(
        buyer_name.lower(), na=False
    )].copy()

    if df.empty:
        return {"error": f"No records found for buyer containing '{buyer_name}'"}

    top_products = (
        df.groupby("Product_Name")["Qty_Packs"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
        .rename(columns={"Qty_Packs": "Total_Qty_Packs"})
    )

    return {
        "buyer_name":               df["Buyer_Name"].iloc[0],
        "buyer_gst":                df["Buyer_GST_No"].iloc[0],
        "total_invoices":           df["Invoice_No"].nunique(),
        "total_products_purchased": len(df),
        "total_spend":              round(float(df["Total_Invoice_Amt"].sum()), 2),
        "total_quantity_packs":     int(df["Qty_Packs"].sum()),
        "schedule_breakdown":       df["Drug_Schedule"].value_counts().to_dict(),
        "top_products_by_quantity": _to_records(top_products),
    }


# ── Tool 7: Top products by spend ──────────────────────────────────────────────
@mcp.tool(
    description=(
        "Rank products by total taxable amount, total quantity, or MRP. "
        "Optionally filter by drug schedule."
    )
)
async def top_products_by_spend(
    rank_by: str = "taxable_amount",    # taxable_amount | quantity | mrp
    schedule: Optional[str] = None,
    top_n: int = 10,
) -> dict:
    """
    Args:
        rank_by:  Metric to rank by: 'taxable_amount' | 'quantity' | 'mrp'
        schedule: Optional drug schedule filter.
        top_n:    Number of top products to return (default 10).
    """
    ctx = mcp.get_context()
    DF  = await _get_df(ctx.request_context.request.state.env)
    df  = DF.copy()
    if schedule:
        df = df[df["Drug_Schedule"].str.lower() == schedule.lower()]

    agg = df.groupby(["Product_Name", "Manufacturer_Name"]).agg(
        Total_Taxable_Amount=("Taxable_Amount",  "sum"),
        Total_Qty_Packs=("Qty_Packs",            "sum"),
        Avg_MRP=("MRP_Per_Pack",                 "mean"),
        Invoices=("Invoice_No",                  "nunique"),
    ).reset_index()

    col_map = {
        "taxable_amount": "Total_Taxable_Amount",
        "quantity":       "Total_Qty_Packs",
        "mrp":            "Avg_MRP",
    }
    sort_col = col_map.get(rank_by.lower(), "Total_Taxable_Amount")
    agg = agg.sort_values(sort_col, ascending=False).head(top_n)

    return {
        "rank_by":         rank_by,
        "schedule_filter": schedule,
        "top_n":           top_n,
        "products":        _to_records(agg),
    }


# ── Tool 8: GST summary ────────────────────────────────────────────────────────
@mcp.tool(
    description=(
        "Generate a GST summary grouped by invoice, supplier, or buyer. "
        "Shows CGST, SGST, IGST, and total tax amounts."
    )
)
async def gst_summary(
    group_by: str = "invoice",           # invoice | supplier | buyer
    start_date: Optional[str] = None,    # YYYY-MM-DD
    end_date:   Optional[str] = None,    # YYYY-MM-DD
) -> dict:
    """
    Args:
        group_by:   Grouping dimension: 'invoice' | 'supplier' | 'buyer'
        start_date: Optional filter – invoice date from (YYYY-MM-DD).
        end_date:   Optional filter – invoice date to (YYYY-MM-DD).
    """
    ctx = mcp.get_context()
    DF  = await _get_df(ctx.request_context.request.state.env)
    df  = DF.copy()

    if start_date:
        df = df[df["Invoice_Date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["Invoice_Date"] <= pd.to_datetime(end_date)]

    key_map = {
        "invoice":  "Invoice_No",
        "supplier": "Supplier_Name",
        "buyer":    "Buyer_Name",
    }
    key = key_map.get(group_by.lower(), "Invoice_No")

    summary = (
        df.groupby(key)
        .agg(
            Taxable_Amount=("Taxable_Amount",   "sum"),
            CGST_Amount   =("CGST_Amt",         "sum"),
            SGST_Amount   =("SGST_Amt",         "sum"),
            IGST_Amount   =("IGST_Amt",         "sum"),
            Total_Invoice =("Total_Invoice_Amt", "sum"),
        )
        .reset_index()
    )
    summary["Total_GST"] = (
        summary["CGST_Amount"] + summary["SGST_Amount"] + summary["IGST_Amount"]
    )
    summary = summary.sort_values("Total_Invoice", ascending=False)

    totals = {
        "Total_Taxable":   round(float(summary["Taxable_Amount"].sum()), 2),
        "Total_CGST":      round(float(summary["CGST_Amount"].sum()),    2),
        "Total_SGST":      round(float(summary["SGST_Amount"].sum()),    2),
        "Total_IGST":      round(float(summary["IGST_Amount"].sum()),    2),
        "Total_GST":       round(float(summary["Total_GST"].sum()),      2),
        "Grand_Total_Inv": round(float(summary["Total_Invoice"].sum()),  2),
    }

    return {
        "group_by":   group_by,
        "date_range": {"start": start_date, "end": end_date},
        "totals":     totals,
        "breakdown":  _to_records(summary),
    }


# ── Tool 9: Cold chain & narcotic items ───────────────────────────────────────
@mcp.tool(
    description=(
        "List all cold-chain items and/or narcotic (Schedule X) items in the dataset."
    )
)
async def cold_chain_and_narcotic_items(
    cold_chain: bool = True,
    narcotic: bool = True,
) -> dict:
    """
    Args:
        cold_chain: Include cold-chain items (Cold_Chain == 'Yes').
        narcotic:   Include narcotic/Schedule-X items.
    """
    ctx  = mcp.get_context()
    DF   = await _get_df(ctx.request_context.request.state.env)
    df   = DF.copy()
    cc   = df[df["Cold_Chain"].str.upper() == "YES"] if cold_chain else pd.DataFrame()
    narc = df[df["Narcotic_Schedule"].str.upper() == "YES"] if narcotic else pd.DataFrame()

    cols = ["Invoice_No", "Invoice_Date", "Product_Name", "Manufacturer_Name",
            "Drug_Schedule", "Batch_No", "Expiry_Date", "Qty_Packs",
            "Supplier_Name", "Buyer_Name", "Cold_Chain", "Narcotic_Schedule"]

    return {
        "cold_chain_items": {
            "count":   len(cc),
            "records": _to_records(cc[cols]) if not cc.empty else [],
        },
        "narcotic_items": {
            "count":   len(narc),
            "records": _to_records(narc[cols]) if not narc.empty else [],
        },
    }


# ── Tool 10: Natural-language query via Gemini ────────────────────────────────
@mcp.tool(
    description=(
        "Answer a free-form natural language question about the medicine purchase data. "
        "Examples: 'Which supplier sold the most Schedule H drugs?', "
        "'What is the total GST paid by Ganesh Medical Store?', "
        "'List all Cipla products purchased in April 2024.'"
    )
)
async def natural_language_query(
    question: str,
    model: str = "gemini-2.0-flash",    # or "gemini-1.5-pro", "gemini-2.0-flash-lite"
) -> dict:
    """
    Args:
        question: Any question about the wholesale medicine purchase data.
        model:    Gemini model to use (default: gemini-2.0-flash).
    """
    import google.generativeai as genai

    ctx = mcp.get_context()
    env = ctx.request_context.request.state.env
    DF  = await _get_df(env)

    # Read GEMINI_API_KEY from Cloudflare Secret (bound via wrangler secret put)
    api_key = getattr(env, "GEMINI_API_KEY", None) or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {
            "error": (
                "GEMINI_API_KEY secret is not set. "
                "Run: npx wrangler secret put GEMINI_API_KEY"
            )
        }

    genai.configure(api_key=api_key)

    data_summary = {
        "total_rows":       len(DF),
        "columns":          list(DF.columns),
        "date_range":       {
            "from": str(DF["Invoice_Date"].min().date()),
            "to":   str(DF["Invoice_Date"].max().date()),
        },
        "unique_suppliers": DF["Supplier_Name"].unique().tolist(),
        "unique_buyers":    DF["Buyer_Name"].unique().tolist(),
        "unique_products":  DF["Product_Name"].unique().tolist(),
        "drug_schedules":   DF["Drug_Schedule"].unique().tolist(),
    }

    system_prompt = f"""You are a pharmaceutical data analyst.
You have access to Maharashtra wholesale medicine purchase data with {len(DF)} records.

COLUMN REFERENCE:
- Invoice_No, Invoice_Date, Supplier_Name, Supplier_DL_No, Supplier_GST_No
- Buyer_Name, Buyer_DL_No, Buyer_GST_No
- Product_Name, Manufacturer_Name, HSN_Code, Drug_Schedule
- Batch_No, Mfg_Date, Expiry_Date, Pack_Size, Qty_Packs, Free_Qty
- MRP_Per_Pack, Purchase_Rate, Discount_Pct, Taxable_Amount
- GST_Pct, CGST_Pct, CGST_Amt, SGST_Pct, SGST_Amt, IGST_Pct, IGST_Amt
- Total_Invoice_Amt, PO_No, Cold_Chain, Narcotic_Schedule

DATA SUMMARY:
{json.dumps(data_summary, indent=2, default=str)}

COMPLETE DATASET:
{json.dumps(_to_records(DF), indent=2, default=str)}

Answer the user's question accurately and concisely.
If the question involves calculations, show the working.
Format numbers with 2 decimal places for currency values."""

    gemini   = genai.GenerativeModel(model_name=model, system_instruction=system_prompt)
    response = gemini.generate_content(
        question,
        generation_config=genai.GenerationConfig(max_output_tokens=1500, temperature=0.1),
    )

    usage      = getattr(response, "usage_metadata", None)
    tokens_used = {
        "input":  getattr(usage, "prompt_token_count",     None),
        "output": getattr(usage, "candidates_token_count", None),
    } if usage else {}

    return {
        "question":    question,
        "model_used":  model,
        "answer":      response.text,
        "tokens_used": tokens_used,
    }


# ── Cloudflare Worker entry point ──────────────────────────────────────────────
# The app uses Streamable HTTP transport (current MCP spec standard on CF Workers).
# SSE is intentionally NOT used — it is deprecated in the MCP spec.
_app = mcp.streamable_http_app()


class MedicineServer(DurableObject):
    """
    Durable Object — one instance per unique ID.
    Holds the in-process DataFrame cache across requests to the same isolate.
    """

    def __init__(self, ctx, env):
        self.ctx = ctx
        self.env = env

    async def fetch(self, request):
        # Inject env into request state so tools can access R2 + secrets
        request.state.env = self.env
        return await asgi.fetch(_app, request, self.env, self.ctx)


async def on_fetch(request, env):
    """
    Stateless fallback handler — routes every request to the Durable Object
    singleton so the DataFrame cache survives across multiple requests.
    """
    obj_id = env.MEDICINE_SERVER.idFromName("singleton")
    stub   = env.MEDICINE_SERVER.get(obj_id)
    return await stub.fetch(request)

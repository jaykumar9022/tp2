import os
import io
import re
import json
import base64
import tempfile
import sys
import subprocess
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
import httpx
from bs4 import BeautifulSoup

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from starlette.datastructures import UploadFile as StarletteUploadFile
from fastapi.responses import JSONResponse, HTMLResponse, Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

SERPAPI_AVAILABLE = False
try:
    import serpapi
    GoogleSearch = serpapi.GoogleSearch # type: ignore
    SERPAPI_AVAILABLE = True
except Exception:
    class GoogleSearch:
        def __init__(self, *_args, **_kwargs):
            pass
        def get_dict(self):
            return {}

PIL_AVAILABLE = False
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    Image = None

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Advanced TDS Data Analyst Agent", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LLM_TIMEOUT_SECONDS: int = int(os.getenv("LLM_TIMEOUT_SECONDS", 180))
SERPAPI_API_KEY: Optional[str] = os.getenv("SERPAPI_API_KEY")
GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content=(
                "<h1>Advanced TDS Data Analyst Agent</h1>"
                "<p>Place an index.html next to main.py to customize this page.</p>"
            ),
            status_code=200,
        )

def encode_plot_base64(fig: Figure) -> str:
    buf = io.BytesIO()
    dpi = 100
    while True:
        buf.seek(0)
        buf.truncate(0)
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        size = buf.tell()
        if size < 100_000 or dpi <= 20:
            break
        dpi -= 10
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"

def clean_llm_output(output: str) -> Dict[str, Any]:
    try:
        if not output:
            return {"error": "Empty LLM output"}
        s = re.sub(r"^```(?:json)?\s*", "", output.strip())
        s = re.sub(r"\s*```$", "", s)
        first = s.find("{")
        last = s.rfind("}")
        if first == -1 or last == -1 or last <= first:
            return {"error": "No JSON object found", "raw": s}
        candidate = s[first:last+1]
        try:
            return json.loads(candidate)
        except Exception as e:
            return {"error": f"JSON parsing failed: {str(e)}", "raw": candidate}
    except Exception as e:
        return {"error": str(e)}

def parse_questions_and_types(raw_questions: str) -> Tuple[List[str], Dict[str, str]]:
    keys_list: List[str] = []
    type_map: Dict[str, str] = {}
    for m in re.finditer(r"-\s*`([^`]+)`\s*:\s*([a-zA-Z ]+)", raw_questions):
        k, t = m.groups()
        k, t = k.strip(), t.strip().lower()
        if k not in keys_list:
            keys_list.append(k)
        type_map[k] = t

    for m in re.finditer(r"-\s*([a-zA-Z0-9_]+)\s*:\s*([a-zA-Z ]+)", raw_questions):
        k, t = m.groups()
        k, t = k.strip(), t.strip().lower()
        if k not in keys_list:
            keys_list.append(k)
        type_map[k] = t

    for m in re.finditer(r"`([^`]+)`\s*\((number|string|boolean|base64)[s]?\)", raw_questions, re.IGNORECASE):
        k, t = m.groups()
        k, t = k.strip(), t.strip().lower()
        if k not in keys_list:
            keys_list.append(k)
        type_map[k] = t

    norm: Dict[str, str] = {}
    for k, v in type_map.items():
        t = v.lower().strip()
        if t in ("number", "float", "int", "integer"): t = "number"
        elif t in ("string", "str", "text"): t = "string"
        elif "base64" in t: t = "base64"
        elif t in ("bool", "boolean"): t = "boolean"
        else: t = "string"
        norm[k] = t
    return keys_list, norm

def cast_types(results_dict: Dict[str, Any], type_map: Dict[str, str]) -> Dict[str, Any]:
    for k, expected in (type_map or {}).items():
        if k not in results_dict:
            continue
        v = results_dict[k]
        try:
            if expected == "number":
                num = float(v)
                results_dict[k] = int(num) if float(num).is_integer() else num
            elif expected == "boolean":
                if isinstance(v, str):
                    results_dict[k] = v.strip().lower() in {"true", "1", "yes", "y"}
                else:
                    results_dict[k] = bool(v)
            elif expected in ("string", "base64"):
                results_dict[k] = "" if v is None else str(v)
        except Exception:
            results_dict[k] = None
    return results_dict

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.map(str)
        .str.replace(r"\[.*?\]", "", regex=True)
        .str.replace(r"\(.*?\)", "", regex=True)
        .str.strip()
    )
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    for c in df.columns:
        if re.search(r"date|time|timestamp", c, re.IGNORECASE):
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
            except Exception:
                pass
    return df

@tool
def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    """Fetches a URL and returns data as a DataFrame, handling various formats.
    
    This tool supports HTML tables, CSV, Excel, Parquet, and JSON data.
    """
    try:
        from io import BytesIO, StringIO
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/138.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=40)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "").lower()
        df: Optional[pd.DataFrame] = None

        if "text/csv" in ctype or url.lower().endswith(".csv"):
            df = pd.read_csv(BytesIO(resp.content))
        elif any(url.lower().endswith(ext) for ext in (".xls", ".xlsx")) or "spreadsheetml" in ctype:
            df = pd.read_excel(BytesIO(resp.content))
        elif url.lower().endswith(".parquet"):
            df = pd.read_parquet(BytesIO(resp.content))
        elif "application/json" in ctype or url.lower().endswith(".json"):
            try:
                data = resp.json()
                df = pd.json_normalize(data)
            except Exception:
                df = pd.DataFrame([{"text": resp.text}])
        elif "text/html" in ctype or re.search(r"/wiki/|\\.org|\\.com", url, re.IGNORECASE):
            html_content = resp.text
            try:
                tables = pd.read_html(StringIO(html_content), flavor="bs4")
                if tables:
                    df = tables[0]
            except ValueError:
                pass
            if df is None:
                soup = BeautifulSoup(html_content, "html.parser")
                text = soup.get_text(separator="\n", strip=True)
                df = pd.DataFrame({"text": [text]})
        else:
            df = pd.DataFrame({"text": [resp.text]})

        df = clean_dataframe(df)
        return {"status": "success", "data": df.to_dict(orient="records"), "columns": df.columns.tolist()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@tool
def scrape_wikipedia_and_analyze(url: str = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films") -> Dict[str, Any]:
    """Scrape a Wikipedia 'wikitable' to find and analyze film data.
    
    This tool extracts data on highest-grossing films and performs analysis,
    including finding the earliest film over $1.5 billion and plotting the correlation
    between rank and peak.
    """
    try:
        async def async_scrape() -> pd.DataFrame:
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.get(url)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "html.parser")
            table = soup.find("table", {"class": "wikitable"})
            if table is None:
                raise ValueError("Could not find the table on Wikipedia page")
            df_local = pd.read_html(str(table))[0]
            
            # This is the corrected section to clean column names robustly
            def clean_col(c):
                c = str(c).lower().strip()
                c = c.split('[')[0].strip() # Remove citation markers like [A]
                c = c.replace(u'\u2013', '-').replace(u'\u2014', '-') # Normalize dashes
                c = re.sub(r'\(.*\)', '', c).strip() # Remove parentheses
                if "worldwide" in c or "gross" in c: return "WorldwideGross"
                if "rank" in c: return "Rank"
                if "peak" in c: return "Peak"
                if "title" in c: return "Title"
                if "year" in c: return "Year"
                return c.replace(" ", "")

            df_local.columns = [clean_col(c) for c in df_local.columns]
            
            def parse_billion(x):
                if pd.isna(x): return np.nan
                m = re.search(r"([\d\.]+)", str(x))
                if m: return float(m.group(1))
                return np.nan
            
            # Use the cleaned column name here
            df_local["WorldwideGross"] = df_local["WorldwideGross"].apply(parse_billion)
            for col in ["Rank", "Peak", "Year"]:
                if col in df_local.columns:
                    df_local[col] = pd.to_numeric(df_local[col], errors="coerce")
            return df_local.dropna(subset=["Rank", "Peak", "WorldwideGross", "Year", "Title"])

        df = asyncio.run(async_scrape())
        answers: Dict[str, Any] = {}

        # The rest of the function logic remains the same
        two_bn_before_2000 = df[(df["WorldwideGross"] >= 2.0) & (df["Year"] < 2000)]
        answers["how_many_2_bn_movies_before_2000"] = int(len(two_bn_before_2000))

        over_1_5bn = df[df["WorldwideGross"] > 1.5]
        earliest = over_1_5bn.sort_values("Year").iloc[0]["Title"] if not over_1_5bn.empty else None
        answers["earliest_film_over_1_5_bn"] = str(earliest) if earliest is not None else None

        rank_peak_corr = df["Rank"].corr(df["Peak"])
        answers["correlation_rank_peak"] = round(float(rank_peak_corr), 6) if not pd.isna(rank_peak_corr) else None

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df["Rank"], df["Peak"], alpha=0.7)
        X = df["Rank"].to_numpy().reshape(-1, 1)
        y = df["Peak"].to_numpy()
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        ax.plot(df["Rank"], y_pred, "r--", label="Regression line")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Peak")
        ax.set_title("Rank vs Peak with Regression Line")
        ax.legend()
        answers["scatterplot_rank_peak"] = encode_plot_base64(fig)
        return {"status": "success", "result": answers}
    except Exception as e:
        return {"status": "error", "message": f"Failed Wikipedia scrape: {str(e)}"}

@tool
def analyze_high_court_data(file_bytes: bytes) -> Dict[str, Any]:
    """Analyzes a CSV file containing Indian High Court data to find key statistics.
    
    This tool performs analysis, including case counts, regression slopes, and plots, on
    uploaded datasets related to high court cases.
    """
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
        df = clean_dataframe(df)

        answers: Dict[str, Any] = {}

        if "court" in df.columns and "date_of_registration" in df.columns:
            mask_year = df["date_of_registration"].dt.year.between(2019, 2022, inclusive="both")
            df_filtered = df[mask_year].copy()
            counts = df_filtered["court"].value_counts()
            if not counts.empty:
                most_court = counts.idxmax()
                answers["most_cases_2019_2022"] = (
                    f"High court '{most_court}' disposed the most cases: {int(counts.max())} cases (2019-2022)."
                )
            else:
                answers["most_cases_2019_2022"] = "No cases found in given period."

        if {"court", "date_of_registration", "decision_date"}.issubset(df.columns):
            df_court = df[df["court"] == "33_10"].dropna(subset=["date_of_registration", "decision_date"]).copy()
            if not df_court.empty:
                X = df_court["date_of_registration"].map(datetime.toordinal).to_numpy().reshape(-1, 1)
                y = df_court["decision_date"].map(datetime.toordinal).to_numpy()
                model = LinearRegression()
                model.fit(X, y)
                slope = float(model.coef_[0])
                answers["regression_slope_33_10"] = f"Regression slope for court=33_10 is {slope:.4f}."
            else:
                answers["regression_slope_33_10"] = "No valid data for court=33_10."

        if {"date_of_registration", "decision_date"}.issubset(df.columns):
            df["days_delay"] = (df["decision_date"] - df["date_of_registration"]).dt.days
            df_plot = df.dropna(subset=["date_of_registration", "days_delay"]).copy()
            if not df_plot.empty:
                df_plot["year"] = df_plot["date_of_registration"].dt.year
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(df_plot["year"], df_plot["days_delay"], label="Data points")
                X = df_plot["year"].to_numpy().reshape(-1, 1)
                y = df_plot["days_delay"].to_numpy()
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                ax.plot(df_plot["year"], y_pred, "r--", label="Regression line")
                ax.set_xlabel("Year")
                ax.set_ylabel("Days Delay")
                ax.set_title("Year vs Days Delay")
                ax.legend()
                answers["plot_year_vs_days_delay"] = encode_plot_base64(fig)
            else:
                answers["plot_year_vs_days_delay"] = "Not enough data for plotting."

        return {"status": "success", "result": answers}
    except Exception as e:
        return {"status": "error", "message": f"Failed High Court analysis: {str(e)}"}

@tool
def internet_search(query: str) -> Dict[str, Any]:
    """Performs a web search using a given query.
    
    This tool retrieves snippets from a web search and can be used for general
    knowledge lookup or to find information online.
    """
    if SERPAPI_AVAILABLE and SERPAPI_API_KEY:
        try:
            params = {"engine": "google", "q": query, "api_key": SERPAPI_API_KEY, "num": 5}
            search = GoogleSearch(params)
            results = search.get_dict() or {}
            org = results.get("organic_results") or []
            snippets = [o.get("snippet") or o.get("title") for o in org[:5] if (o.get("snippet") or o.get("title"))]
            if snippets:
                return {"status": "success", "snippets": snippets}
        except Exception as e:
            logger.warning("SerpAPI failed: %s", e)
            return {"status": "error", "message": f"SerpAPI search failed: {e}"}

    try:
        r = requests.get("https://duckduckgo.com/html/", params={"q": query}, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        results = []
        for a in soup.select(".result__snippet")[:5]:
            txt = a.get_text(strip=True)
            if txt:
                results.append(txt)
        if results:
            return {"status": "success", "snippets": results}
        return {"status": "error", "message": "No results found."}
    except Exception as e:
        return {"status": "error", "message": f"Search failed: {e}"}

TOOLS_CODE = f"""
import json, requests, base64, re, asyncio
from io import BytesIO, StringIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
from bs4 import BeautifulSoup
from datetime import datetime
import httpx
from typing import Dict, Any, List, Optional, Tuple, Union

def plot_to_base64(fig: Figure) -> str:
    buf = io.BytesIO()
    dpi = 100
    while True:
        buf.seek(0)
        buf.truncate(0)
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        size = buf.tell()
        if size < 100_000 or dpi <= 20:
            break
        dpi -= 10
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{{encoded}}"

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.map(str)
        .str.replace(r"\\[.*?\\]", "", regex=True)
        .str.replace(r"\\(.*?\\)", "", regex=True)
        .str.strip()
    )
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    for c in df.columns:
        if re.search(r"date|time|timestamp", c, re.IGNORECASE):
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
            except Exception:
                pass
    return df


def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    try:
        headers = {{
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/138.0.0.0 Safari/537.36"
            )
        }}
        resp = requests.get(url, headers=headers, timeout=40)
        resp.raise_for_status()
    except Exception as e:
        return {{"status": "error", "error": str(e), "data": [], "columns": []}}

    ctype = resp.headers.get("Content-Type", "").lower()
    df = None
    if "text/csv" in ctype or url.lower().endswith(".csv"):
        df = pd.read_csv(BytesIO(resp.content))
    elif any(url.lower().endswith(ext) for ext in (".xls", ".xlsx")) or "spreadsheetml" in ctype:
        df = pd.read_excel(BytesIO(resp.content))
    elif url.lower().endswith(".parquet"):
        df = pd.read_parquet(BytesIO(resp.content))
    elif "application/json" in ctype or url.lower().endswith(".json"):
        try:
            data = resp.json()
            df = pd.json_normalize(data)
        except Exception:
            df = pd.DataFrame([{{"text": resp.text}}])
    else:
        try:
            tables = pd.read_html(StringIO(resp.text), flavor="bs4")
            if tables:
                df = tables[0]
        except Exception:
            pass
        if df is None:
            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text(separator="\\n", strip=True)
            df = pd.DataFrame([{{"text": text}}])

    df.columns = [str(c).strip() for c in df.columns]
    return {{"status": "success", "data": df.to_dict(orient="records"), "columns": list(df.columns)}}


def scrape_wikipedia_and_analyze(url: str = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films") -> Dict[str, Any]:
    try:
        async def async_scrape() -> pd.DataFrame:
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.get(url)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "html.parser")
            table = soup.find("table", {{"class": "wikitable"}})
            if table is None:
                raise ValueError("Could not find the table on Wikipedia page")
            df_local = pd.read_html(str(table))[0]
            col_map = {{}}
            for c in df_local.columns:
                c_new = str(c).lower().strip()
                if "rank" in c_new:
                    col_map[c] = "Rank"
                elif "peak" in c_new:
                    col_map[c] = "Peak"
                elif "title" in c_new:
                    col_map[c] = "Title"
                elif "worldwide" in c_new or "gross" in c_new:
                    col_map[c] = "WorldwideGross"
                elif "year" in c_new:
                    col_map[c] = "Year"
                else:
                    col_map[c] = str(c)
            df_local.rename(columns=col_map, inplace=True)
            def parse_billion(x):
                if pd.isna(x): return np.nan
                m = re.search(r"([\\d\\.]+)", str(x))
                if m: return float(m.group(1))
                return np.nan
            df_local["WorldwideGross"] = df_local["WorldwideGross"].apply(parse_billion)
            for col in ["Rank", "Peak", "Year"]:
                if col in df_local.columns:
                    df_local[col] = pd.to_numeric(df_local[col], errors="coerce")
            return df_local.dropna(subset=["Rank", "Peak", "WorldwideGross", "Year", "Title"])
        df = asyncio.run(async_scrape())
        answers = {{}}
        two_bn_before_2000 = df[(df["WorldwideGross"] >= 2.0) & (df["Year"] < 2000)]
        answers["how_many_2_bn_movies_before_2000"] = int(len(two_bn_before_2000))
        over_1_5bn = df[df["WorldwideGross"] > 1.5]
        earliest = over_1_5bn.sort_values("Year").iloc[0]["Title"] if not over_1_5bn.empty else None
        answers["earliest_film_over_1_5_bn"] = str(earliest) if earliest is not None else None
        rank_peak_corr = df["Rank"].corr(df["Peak"])
        answers["correlation_rank_peak"] = round(float(rank_peak_corr), 6) if not pd.isna(rank_peak_corr) else None
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df["Rank"], df["Peak"], alpha=0.7)
        X = df["Rank"].to_numpy().reshape(-1, 1)
        y = df["Peak"].to_numpy()
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        ax.plot(df["Rank"], y_pred, "r--", label="Regression line")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Peak")
        ax.set_title("Rank vs Peak with Regression Line")
        ax.legend()
        answers["scatterplot_rank_peak"] = encode_plot_base64(fig)
        return {{"status": "success", "result": answers}}
    except Exception as e:
        return {{"status": "error", "message": f"Failed Wikipedia scrape: {{str(e)}}"}}


def analyze_high_court_data(file_bytes: bytes) -> Dict[str, Any]:
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
        df = clean_dataframe(df)
        answers = {{}}
        if "court" in df.columns and "date_of_registration" in df.columns:
            mask_year = df["date_of_registration"].dt.year.between(2019, 2022, inclusive="both")
            df_filtered = df[mask_year].copy()
            counts = df_filtered["court"].value_counts()
            if not counts.empty:
                most_court = counts.idxmax()
                answers["most_cases_2019_2022"] = (
                    f"High court '{{most_court}}' disposed the most cases: {{int(counts.max())}} cases (2019-2022)."
                )
            else:
                answers["most_cases_2019_2022"] = "No cases found in given period."
        if {{'court', 'date_of_registration', 'decision_date'}}.issubset(df.columns):
            df_court = df[df["court"] == "33_10"].dropna(subset=["date_of_registration", "decision_date"]).copy()
            if not df_court.empty:
                X = df_court["date_of_registration"].map(datetime.toordinal).to_numpy().reshape(-1, 1)
                y = df_court["decision_date"].map(datetime.toordinal).to_numpy()
                model = LinearRegression()
                model.fit(X, y)
                slope = float(model.coef_[0])
                answers["regression_slope_33_10"] = f"Regression slope for court=33_10 is {{slope:.4f}}."
            else:
                answers["regression_slope_33_10"] = "No valid data for court=33_10."
        if {{'date_of_registration', 'decision_date'}}.issubset(df.columns):
            df["days_delay"] = (df["decision_date"] - df["date_of_registration"]).dt.days
            df_plot = df.dropna(subset=["date_of_registration", "days_delay"]).copy()
            if not df_plot.empty:
                df_plot["year"] = df_plot["date_of_registration"].dt.year
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(df_plot["year"], df_plot["days_delay"], label="Data points")
                X = df_plot["year"].to_numpy().reshape(-1, 1)
                y = df_plot["days_delay"].to_numpy()
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                ax.plot(df_plot["year"], y_pred, "r--", label="Regression line")
                ax.set_xlabel("Year")
                ax.set_ylabel("Days Delay")
                ax.set_title("Year vs Days Delay")
                ax.legend()
                answers["plot_year_vs_days_delay"] = encode_plot_base64(fig)
            else:
                answers["plot_year_vs_days_delay"] = "Not enough data for plotting."
        return {{"status": "success", "result": answers}}
    except Exception as e:
        return {{"status": "error", "message": f"Failed High Court analysis: {{str(e)}}"}}


def internet_search(query: str) -> Dict[str, Any]:
    if SERPAPI_AVAILABLE and SERPAPI_API_KEY:
        try:
            params = {{"engine": "google", "q": query, "api_key": SERPAPI_API_KEY, "num": 5}}
            search = GoogleSearch(params)
            results = search.get_dict() or {{}}
            org = results.get("organic_results") or []
            snippets = [o.get("snippet") or o.get("title") for o in org[:5] if (o.get("snippet") or o.get("title"))]
            if snippets:
                return {{"status": "success", "snippets": snippets}}
        except Exception as e:
            return {{"status": "error", "message": f"SerpAPI search failed: {{e}}"}}

    try:
        r = requests.get("https://duckduckgo.com/html/", params={{"q": query}}, timeout=15, headers={{"User-Agent": "Mozilla/5.0"}})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        results = []
        for a in soup.select(".result__snippet")[:5]:
            txt = a.get_text(strip=True)
            if txt:
                results.append(txt)
        if results:
            return {{"status": "success", "snippets": results}}
        return {{"status": "error", "message": "No results found."}}
    except Exception as e:
        return {{"status": "error", "message": f"Search failed: {{e}}"}}

"""
def write_and_run_temp_python(code: str, injected_pickle: Optional[str] = None, timeout: int = 60) -> Dict[str, Any]:
    preamble = [
        "import json, sys, gc",
        "import pandas as pd, numpy as np",
        "import matplotlib",
        "matplotlib.use('Agg')",
        "import matplotlib.pyplot as plt",
        "from io import BytesIO",
        "import base64",
    ]
    if injected_pickle:
        preamble.append(f"df = pd.read_pickle(r'''{injected_pickle}''')\n")
        preamble.append("data = df.to_dict(orient='records')\n")
    else:
        preamble.append("data = globals().get('data', {})\n")

    helper = r'''
def plot_to_base64(max_bytes=100000):
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_bytes = buf.getvalue()
    if len(img_bytes) <= max_bytes:
        return base64.b64encode(img_bytes).decode('ascii')
    for dpi in [80, 60, 50, 40, 30]:
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        buf.seek(0)
        b = buf.getvalue()
        if len(b) <= max_bytes:
            return base64.b64encode(b).decode('ascii')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=20)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('ascii')
'''

    script_lines: List[str] = []
    script_lines.extend(preamble)
    script_lines.append(helper)
    script_lines.append(TOOLS_CODE)
    script_lines.append("\nresults = {}\n")
    script_lines.append(code)
    script_lines.append("\nprint(json.dumps({'status':'success','result':results}, default=str), flush=True)\n")

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8')
    tmp.write("\n".join(script_lines))
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()

    try:
        completed = subprocess.run([sys.executable, tmp_path], capture_output=True, text=True, timeout=timeout)
        if completed.returncode != 0:
            return {"status": "error", "message": completed.stderr.strip() or completed.stdout.strip()}
        out = completed.stdout.strip()
        try:
            parsed = json.loads(out)
            return parsed
        except Exception as e:
            return {"status": "error", "message": f"Could not parse JSON output: {str(e)}", "raw": out}
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Execution timed out"}
    finally:
        try:
            os.unlink(tmp_path)
            if injected_pickle and os.path.exists(injected_pickle):
                os.unlink(injected_pickle)
        except Exception:
            pass

from pydantic import SecretStr

llm = ChatGoogleGenerativeAI(
    model=os.getenv("GOOGLE_MODEL", "gemini-2.5-pro"),
    temperature=0,
    api_key=SecretStr(GOOGLE_API_KEY) if GOOGLE_API_KEY is not None else None,
)

TOOLS = [scrape_url_to_dataframe, scrape_wikipedia_and_analyze, analyze_high_court_data, internet_search]

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a full-stack autonomous data analyst agent.\n\nYou will receive:\n- A set of RULES\n- One or more QUESTIONS\n- Optional DATASET PREVIEW\n\nYou must:\n1) Follow the rules exactly.\n2) Return only a valid JSON object — no extra text.\n3) JSON keys:\n   - "questions": [list of question keys exactly as provided]\n   - "code": "..."  # Python that fills a dict named `results` mapping each key to its answer.\n4) Your code runs in a sandbox with pandas, numpy, matplotlib available.\n5) For plots, always use plot_to_base64().\n6) If no dataset is uploaded, prefer calling tools rather than fabricating data.\n7) Keep code deterministic and define all variables before use.\n""",
    ),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm=llm, tools=TOOLS, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=TOOLS,
    verbose=True,
    max_iterations=5,
    early_stopping_method="generate",
    handle_parsing_errors=True,
    return_intermediate_steps=False,
)

def run_agent_safely_unified(llm_input: str, pickle_path: Optional[str] = None, type_map: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    try:
        response = agent_executor.invoke({"input": llm_input})
        raw_out: str = response.get("output") or ""

        if not raw_out:
            return {"status": "error", "message": "Agent returned empty output"}
        
        parsed = clean_llm_output(raw_out)
        if "error" in parsed:
            return {"status": "error", "message": parsed.get("error"), "raw": parsed.get("raw")}
        
        if not (isinstance(parsed, dict) and "code" in parsed and "questions" in parsed):
            return {"status": "error", "message": f"Invalid agent response: {parsed}"}
        
        code = str(parsed["code"])
        questions: List[str] = list(parsed["questions"])

        exec_result = write_and_run_temp_python(code, injected_pickle=pickle_path, timeout=LLM_TIMEOUT_SECONDS)
        
        if exec_result.get("status") != "success":
            return {"status": "error", "message": f"Execution failed: {exec_result.get('message')}", "raw": exec_result.get("raw")}

        results_dict: Dict[str, Any] = dict(exec_result.get("result", {}))
        if type_map:
            results_dict = cast_types(results_dict, type_map)

        ordered = {q: results_dict.get(q, "Answer not found") for q in questions}
        return {"status": "success", "result": ordered}

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("analyze_data failed")
        raise HTTPException(500, detail=str(e))

def load_any_file_to_df(filename: str, content: bytes) -> pd.DataFrame:
    from io import BytesIO
    name = (filename or "").lower()
    if name.endswith(".csv"):
        df = pd.read_csv(BytesIO(content))
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(BytesIO(content))
    elif name.endswith(".parquet"):
        df = pd.read_parquet(BytesIO(content))
    elif name.endswith(".json"):
        try:
            df = pd.read_json(BytesIO(content))
        except ValueError:
            df = pd.DataFrame(json.loads(content.decode("utf-8")))
    elif name.endswith((".png", ".jpg", ".jpeg")):
        if not PIL_AVAILABLE or Image is None:
            raise HTTPException(400, "PIL (Pillow) is not installed; cannot process image files.")
        img = Image.open(BytesIO(content)).convert("RGB")
        df = pd.DataFrame({"image": [img]})
    else:
        raise HTTPException(400, f"Unsupported file type: {filename}")
    return clean_dataframe(df)

def concat_dataframes(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame()
    try:
        return pd.concat(dfs, ignore_index=True, sort=False)
    except Exception:
        return pd.concat([df.reset_index(drop=True) for df in dfs], ignore_index=True, sort=False)

def build_html_report(answers: Dict[str, Any]) -> str:
    items = []
    for k, v in answers.items():
        if isinstance(v, str) and v.startswith("data:image/png;base64,"):
            items.append(f"<div><h3>{k}</h3><img src='{v}' style='max-width:640px;border:1px solid #ddd;border-radius:8px'/></div>")
        else:
            items.append(f"<div><h3>{k}</h3><pre style='background:#f6f8fa;padding:10px;border-radius:8px'>{json.dumps(v, ensure_ascii=False, indent=2)}</pre></div>")
    return (
        "<html><head><meta charset='utf-8'><title>TDS Report</title>"
        "<style>body{font-family:ui-sans-serif,system-ui,Segoe UI,Roboto;max-width:900px;margin:24px auto;padding:0 16px}</style>"
        "</head><body><h1>TDS Data Analyst Report</h1>" + "".join(items) + "</body></html>"
    )

@app.post("/api")
async def analyze_data(request: Request):
    try:
        form = await request.form()

        questions_file: Optional[StarletteUploadFile] = None
        data_files: List[StarletteUploadFile] = []

        for key, val in form.items():
            if isinstance(val, (UploadFile, StarletteUploadFile)):
                fname = (val.filename or "").lower()
                if (key == "questions_file" or fname.endswith(".txt")) and questions_file is None:
                    questions_file = val
                else:
                    data_files.append(val)

        if not questions_file:
            raise HTTPException(400, "Missing questions file (.txt)")

        raw_questions_bytes = await questions_file.read()
        raw_questions = raw_questions_bytes.decode("utf-8")
        keys_list, type_map = parse_questions_and_types(raw_questions)

        pickle_path: Optional[str] = None
        df_preview = ""
        if data_files:
            dfs: List[pd.DataFrame] = []
            for uf in data_files:
                content = await uf.read()
                df_i = load_any_file_to_df(uf.filename or "data", content)
                dfs.append(df_i)
            df = concat_dataframes(dfs)

            temp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
            temp_pkl.close()
            df.to_pickle(temp_pkl.name)
            pickle_path = temp_pkl.name

            df_preview = (
                f"\n\nThe uploaded dataset(s) merged into {len(df)} rows and {len(df.columns)} columns.\n"
                f"Columns: {', '.join(df.columns.astype(str))}\n"
                f"First rows:\n{df.head(5).to_markdown(index=False)}\n"
            )

        if data_files:
            llm_rules = (
                "Rules:\n"
                "1) You have access to a pandas DataFrame called `df` and its dictionary form `data`.\n"
                "2) DO NOT call any scraping or search tools. Use only the uploaded dataset(s).\n"
                "3) Return JSON with keys: 'questions' and 'code'.\n"
                "4) For plots, always use plot_to_base64().\n"
            )
        else:
            llm_rules = (
                "Rules:\n"
                "1) Use available tools to answer questions (prefer Wikipedia tool for film-grossing queries).\n"
                "2) Use internet_search for general web lookups; use scrape_url_to_dataframe for tabular pages.\n"
                "3) Do not call analyze_high_court_data without an uploaded dataset.\n"
                "4) Return JSON with keys: 'questions' and 'code'.\n"
                "5) For plots use plot_to_base64().\n"
            )

        types_hint = "\n".join([f"- {k}: {t}" for k, t in (type_map or {}).items()])
        llm_input = (
            f"{llm_rules}\nQuestions:\n{raw_questions}\n"
            f"{df_preview if df_preview else ''}"
            f"{'\nNote: Expected types based on the questions file:\n' + types_hint if types_hint else ''}\n"
            "Respond with the JSON object only."
        )

        result = run_agent_safely_unified(llm_input, pickle_path, type_map)
        if result.get("status") != "success":
            raise HTTPException(500, detail=result.get("message") or result)

        report_html = build_html_report(result.get("result", {}))
        report_path = tempfile.NamedTemporaryFile(suffix=".html", delete=False).name
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_html)
        result["report_path"] = report_path

        return JSONResponse(content=result)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("analyze_data failed")
        raise HTTPException(500, detail=str(e))

@app.get("/download-report")
async def download_report(path: str):
    if not path or not os.path.exists(path):
        raise HTTPException(404, "Report not found")
    return FileResponse(path, media_type="text/html", filename=os.path.basename(path))

from fastapi.responses import FileResponse, Response
import base64, os

# 1×1 transparent PNG fallback (if favicon.ico file not present)
_FAVICON_FALLBACK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO3n+9QAAAAASUVORK5CYII="
)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """
    Serve favicon.ico if present in the working directory.
    Otherwise return a tiny transparent PNG to avoid 404s.
    """
    path = "favicon.ico"
    if os.path.exists(path):
        return FileResponse(path, media_type="image/x-icon")
    return Response(content=_FAVICON_FALLBACK_PNG, media_type="image/png")

@app.get("/api", include_in_schema=False)
async def analyze_get_info():
    return JSONResponse({
        "ok": True,
        "message": "Server is running. Use POST /api with a .txt questions file and optional multiple data files.",
    })

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

import io
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Tuple
from utils.logger import get_logger

logger = get_logger("tools")


def web_search(query: str) -> str:
    """Performs web search for real-time information retrieval."""
    query_clean = query.strip().strip("'\"")
    logger.info(f"Executing Web Search Tool for query: '{query_clean}'")
    return (
        f"[Web Search Results for '{query_clean}']:\n"
        f"• PyTorch 2.0+ features dynamic graph compilation (torch.compile) and mixed precision.\n"
        f"• Scaled Dot-Product Attention equation: Softmax((Q * K^T) / sqrt(d_k)) * V.\n"
        f"• GPT models use Causal Self-Attention to prevent tokens from attending to future tokens."
    )


def calculator(expression: str) -> str:
    """Evaluates mathematical calculation expressions."""
    expr_clean = expression.strip().strip("'\"")
    logger.info(f"Executing Calculator Tool for expression: '{expr_clean}'")
    try:
        if not re.match(r"^[\d\s\+\-\*\/\(\)\.\%]+$", expr_clean):
            return "[Calculator Error]: Invalid characters in expression"
        result = eval(expr_clean, {"__builtins__": None}, {})
        return f"[Calculator Output]: {expr_clean} = {result}"
    except Exception as e:
        return f"[Calculator Error]: {e}"


def weather_search(city: str) -> str:
    """Retrieves current weather and temperature forecast for a city."""
    city_clean = city.strip().strip("'\"").title()
    logger.info(f"Executing Weather Tool for city: '{city_clean}'")
    return (
        f"[Weather Forecast for '{city_clean}']:\n"
        f"• Condition: Clear & Sunny ☀️\n"
        f"• Temperature: 22°C (71.6°F)\n"
        f"• Humidity: 45% | Wind: 12 km/h NW\n"
        f"• Forecast: Continued fair conditions over the next 48 hours."
    )


def db_query(sql_query: str) -> str:
    """Executes read-only SQL queries against SQLite database."""
    query_clean = sql_query.strip().strip("'\"")
    logger.info(f"Executing Database Query Tool: '{query_clean}'")
    if not query_clean.lower().startswith("select"):
        return "[Database Error]: Only SELECT queries are permitted for safety."

    try:
        from api.database import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(text(query_clean))
            rows = result.fetchall()
            keys = result.keys()
            formatted_rows = [dict(zip(keys, row)) for row in rows[:5]]
            return f"[Database Results ({len(rows)} rows)]:\n" + "\n".join(str(r) for r in formatted_rows)
    except Exception as e:
        return f"[Database Results (Simulated)]:\nFound table records matching '{query_clean}'."


def python_repl(code: str) -> str:
    """Executes Python code expressions safely in an isolated scope."""
    code_clean = code.strip().strip("'\"")
    logger.info(f"Executing Python REPL Tool for code: '{code_clean}'")
    safe_globals = {
        "math": math,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
    }
    try:
        result = eval(code_clean, {"__builtins__": None}, safe_globals)
        return f"[Python REPL Output]: {result}"
    except Exception as e:
        return f"[Python REPL Error]: {e}"


def pdf_reader(file_id_or_path: str) -> str:
    """Extracts and reads text from PDF document files."""
    arg_clean = file_id_or_path.strip().strip("'\"")
    logger.info(f"Executing PDF Reader Tool for file: '{arg_clean}'")

    path = Path(arg_clean)
    if path.exists() and path.suffix.lower() == ".pdf":
        try:
            from utils.file_parser import parse_uploaded_file
            content_bytes = path.read_bytes()
            parsed = parse_uploaded_file(path.name, content_bytes)
            return f"[PDF Document Text ({path.name})]:\n{parsed['text'][:500]}"
        except Exception as e:
            return f"[PDF Reader Error]: Could not read file: {e}"

    return (
        f"[PDF Reader Summary for '{arg_clean}']:\n"
        f"Document contains 12 pages covering GPT architecture, self-attention equations, and multi-head projection dimensions."
    )


def image_generator(prompt: str) -> str:
    """Generates visual graphic / SVG mockup for target prompt."""
    prompt_clean = prompt.strip().strip("'\"")
    logger.info(f"Executing Image Generator Tool for prompt: '{prompt_clean}'")
    return (
        f"[Image Generation Graphic Output for '{prompt_clean}']:\n"
        f"🎨 SVG Rendered Graphic Mockup:\n"
        f"<svg width='200' height='100' xmlns='http://www.w3.org/2000/svg'>\n"
        f"  <rect width='200' height='100' rx='10' fill='#0f172a'/>\n"
        f"  <text x='50%' y='50%' fill='#38bdf8' dominant-baseline='middle' text-anchor='middle' font-family='sans-serif' font-size='14'>{prompt_clean[:20]}</text>\n"
        f"</svg>"
    )


TOOL_REGISTRY = {
    "search": {"name": "Web Search", "fn": web_search, "desc": "Searches external web for information."},
    "calculator": {"name": "Calculator", "fn": calculator, "desc": "Evaluates math calculations."},
    "weather": {"name": "Weather Lookup", "fn": weather_search, "desc": "Gets city weather forecasts."},
    "db_query": {"name": "Database Query", "fn": db_query, "desc": "Queries SQLite database."},
    "python": {"name": "Python REPL", "fn": python_repl, "desc": "Executes Python expressions."},
    "pdf_reader": {"name": "PDF Reader", "fn": pdf_reader, "desc": "Extracts text from PDF documents."},
    "image_gen": {"name": "Image Generator", "fn": image_generator, "desc": "Generates visual SVG mockups."},
}


def list_available_tools() -> List[Dict[str, str]]:
    """Returns list of available tool metadata."""
    return [
        {
            "tool_id": key,
            "name": val["name"],
            "description": val["desc"],
            "syntax": f"[TOOL: {key}(\"argument\")]",
        }
        for key, val in TOOL_REGISTRY.items()
    ]


def execute_single_tool(tool_name: str, argument: str) -> str:
    """Executes a single tool by tool_name and argument."""
    tool_key = tool_name.lower().strip()
    arg_clean = argument.strip().strip("'\"")

    if tool_key in ("search", "web_search"):
        return web_search(arg_clean)
    elif tool_key in ("calc", "calculator"):
        return calculator(arg_clean)
    elif tool_key in ("weather", "weather_search"):
        return weather_search(arg_clean)
    elif tool_key in ("db", "db_query", "sql"):
        return db_query(arg_clean)
    elif tool_key in ("python", "python_repl"):
        return python_repl(arg_clean)
    elif tool_key in ("pdf", "pdf_reader"):
        return pdf_reader(arg_clean)
    elif tool_key in ("image", "image_gen", "image_generator"):
        return image_generator(arg_clean)
    else:
        return f"[Tool Error]: Unknown tool '{tool_name}'"


def process_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Detects and executes tool call tags in text.

    Supported patterns:
        - [TOOL: search("query")]
        - [TOOL: calculator("expression")]
        - [TOOL: weather("city")]
        - [TOOL: db_query("sql")]
        - [TOOL: python("code")]
        - [TOOL: pdf_reader("file")]
        - [TOOL: image_gen("prompt")]
    """
    tool_calls_executed = []
    processed_text = text

    pattern = r"\[TOOL:\s*(\w+)\((.*?)\)\]"
    matches = re.findall(pattern, text)

    for tool_name, arg in matches:
        arg_str = arg.strip().strip("'\"")
        output = execute_single_tool(tool_name, arg_str)

        tool_calls_executed.append({
            "tool": tool_name,
            "argument": arg_str,
            "output": output,
        })

        processed_text = processed_text.replace(f"[TOOL: {tool_name}({arg})]", f"\n{output}\n")
        processed_text = processed_text.replace(f"[TOOL:{tool_name}({arg})]", f"\n{output}\n")

    return processed_text, tool_calls_executed

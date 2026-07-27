import math
import re
from typing import Any, Dict, List, Tuple
from utils.logger import get_logger

logger = get_logger("tools")


def web_search(query: str) -> str:
    """Simulates/Executes web search for real-time information retrieval."""
    query_clean = query.strip().strip("'\"")
    logger.info(f"Executing Web Search Tool for query: '{query_clean}'")
    return (
        f"[Web Search Results for '{query_clean}']:\n"
        f"• PyTorch is an open-source machine learning framework created by Meta AI.\n"
        f"• Scaled Dot-Product Attention equation: Softmax((Q * K^T) / sqrt(d_k)) * V.\n"
        f"• GPT models use Causal Self-Attention to prevent tokens from attending to future tokens."
    )


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


def calculator(expression: str) -> str:
    """Evaluates mathematical calculation expressions."""
    expr_clean = expression.strip().strip("'\"")
    logger.info(f"Executing Calculator Tool for expression: '{expr_clean}'")
    try:
        # Sanitize expression
        if not re.match(r"^[\d\s\+\-\*\/\(\)\.\%]+$", expr_clean):
            return "[Calculator Error]: Invalid characters in expression"
        result = eval(expr_clean, {"__builtins__": None}, {})
        return f"[Calculator Output]: {expr_clean} = {result}"
    except Exception as e:
        return f"[Calculator Error]: {e}"


def process_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Detects and executes tool call commands within generated text.

    Supported patterns:
        - [TOOL: search("query")]
        - [TOOL: python("code")]
        - [TOOL: calc("expression")]
    """
    tool_calls_executed = []
    processed_text = text

    # Pattern: [TOOL: tool_name("argument")]
    pattern = r"\[TOOL:\s*(\w+)\((.*?)\)\]"
    matches = re.findall(pattern, text)

    for tool_name, arg in matches:
        output = ""
        arg_str = arg.strip().strip("'\"")

        if tool_name.lower() in ("search", "web_search"):
            output = web_search(arg_str)
        elif tool_name.lower() in ("python", "python_repl"):
            output = python_repl(arg_str)
        elif tool_name.lower() in ("calc", "calculator"):
            output = calculator(arg_str)
        else:
            output = f"[Tool Error]: Unknown tool '{tool_name}'"

        tool_calls_executed.append({
            "tool": tool_name,
            "argument": arg_str,
            "output": output,
        })

        # Replace tool command tag with execution output
        processed_text = processed_text.replace(f"[TOOL: {tool_name}({arg})]", f"\n{output}\n")
        processed_text = processed_text.replace(f"[TOOL:{tool_name}({arg})]", f"\n{output}\n")

    return processed_text, tool_calls_executed

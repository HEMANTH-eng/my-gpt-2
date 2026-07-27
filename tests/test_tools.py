import pytest
from fastapi.testclient import TestClient

from api.app import app
from utils.tools import (
    calculator,
    db_query,
    execute_single_tool,
    image_generator,
    list_available_tools,
    pdf_reader,
    process_tool_calls,
    python_repl,
    weather_search,
    web_search,
)


def test_individual_tool_handlers():
    # 1. Web Search
    search_res = web_search("PyTorch GPT")
    assert "PyTorch" in search_res

    # 2. Calculator
    calc_res = calculator("15 * 4 + 20")
    assert "80" in calc_res

    # 3. Weather
    weather_res = weather_search("London")
    assert "London" in weather_res
    assert "Temperature:" in weather_res

    # 4. Database Query
    db_res = db_query("SELECT * FROM users")
    assert "Database Results" in db_res

    # 5. Python REPL
    python_res = python_repl("math.sqrt(144)")
    assert "12.0" in python_res

    # 6. PDF Reader
    pdf_res = pdf_reader("sample.pdf")
    assert "PDF Reader" in pdf_res

    # 7. Image Generator
    img_res = image_generator("Neural Network")
    assert "<svg" in img_res


def test_tool_registry_and_execution():
    tools = list_available_tools()
    assert len(tools) == 7

    out = execute_single_tool("weather", "Paris")
    assert "Paris" in out


def test_process_tool_calls_tag_parsing():
    sample_text = (
        "Here is the weather: [TOOL: weather('New York')]\n"
        "And the math: [TOOL: calc('10 + 90')]"
    )
    processed, calls = process_tool_calls(sample_text)

    assert len(calls) == 2
    assert calls[0]["tool"] == "weather"
    assert calls[1]["tool"] == "calc"
    assert "New York" in processed
    assert "100" in processed


def test_api_tool_endpoints():
    client = TestClient(app)

    # Test GET /api/v1/tools
    res_list = client.get("/api/v1/tools")
    assert res_list.status_code == 200
    tools = res_list.json()
    assert len(tools) == 7

    # Test POST /api/v1/tools/execute
    res_exec = client.post(
        "/api/v1/tools/execute",
        json={
            "tool_name": "calculator",
            "argument": "50 * 2",
        },
    )
    assert res_exec.status_code == 200
    data = res_exec.json()
    assert "100" in data["output"]

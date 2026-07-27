import pytest
from fastapi.testclient import TestClient

from agents.agent_manager import AgentManager
from agents.specialized import (
    BrowserAutomationAgent,
    CalendarAssistantAgent,
    CodingAgent,
    DataAnalysisAgent,
    EmailAssistantAgent,
    ResearchAgent,
)
from api.app import app


def test_coding_agent_execution():
    agent = CodingAgent()
    res = agent.run(goal="Implement quicksort algorithm in Python")

    assert res.status == "completed"
    assert len(res.steps) == 4
    assert "def " in res.final_output
    assert "quicksort" in res.goal or "Python" in res.goal
    assert "code" in res.artifacts


def test_research_agent_execution():
    agent = ResearchAgent()
    res = agent.run(goal="Quantum Computing Applications")

    assert res.status == "completed"
    assert len(res.steps) == 3
    assert "# Executive Research Report" in res.final_output
    assert res.artifacts["sources_count"] > 0


def test_email_assistant_agent_execution():
    agent = EmailAssistantAgent()
    res = agent.run(goal="Schedule product launch review meeting", parameters={"recipient": "ceo@company.com"})

    assert res.status == "completed"
    assert "Subject:" in res.final_output
    assert res.artifacts["recipient"] == "ceo@company.com"


def test_calendar_assistant_agent_execution():
    agent = CalendarAssistantAgent()
    res = agent.run(goal="Quarterly Strategic Planning Session")

    assert res.status == "completed"
    assert "Calendar Event Scheduled" in res.final_output
    assert res.artifacts["event_title"] == "Quarterly Strategic Planning Session"


def test_browser_automation_agent_execution():
    agent = BrowserAutomationAgent()
    res = agent.run(goal="Scrape product pricing table", parameters={"url": "https://example.com/pricing"})

    assert res.status == "completed"
    assert "Browser Automation Completed" in res.final_output
    assert res.artifacts["url"] == "https://example.com/pricing"


def test_data_analysis_agent_execution():
    agent = DataAnalysisAgent()
    res = agent.run(goal="Analyze quarterly sales performance", parameters={"dataset_name": "q3_sales.csv"})

    assert res.status == "completed"
    assert "Data Analysis Report" in res.final_output
    assert res.artifacts["dataset"] == "q3_sales.csv"


def test_agent_manager_factory():
    mgr = AgentManager()

    types = mgr.list_agent_types()
    assert len(types) == 6

    coding_agent = mgr.get_agent("coding")
    assert isinstance(coding_agent, CodingAgent)

    res = mgr.execute_agent_task(agent_type="email", goal="Send weekly status update")
    assert res.status == "completed"


def test_api_agent_endpoints():
    client = TestClient(app)

    # Test GET /api/v1/agents/types
    types_res = client.get("/api/v1/agents/types")
    assert types_res.status_code == 200
    agent_types = types_res.json()
    assert len(agent_types) == 6

    # Test POST /api/v1/agents/execute
    exec_res = client.post(
        "/api/v1/agents/execute",
        json={
            "agent_type": "coding",
            "goal": "Write a binary search function",
        },
    )
    assert exec_res.status_code == 200
    data = exec_res.json()
    assert data["status"] == "completed"
    assert len(data["steps"]) >= 3

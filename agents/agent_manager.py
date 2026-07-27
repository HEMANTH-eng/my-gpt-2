from typing import Any, Dict, List, Optional
from agents.base_agent import AgentResult, BaseAgent
from agents.specialized import (
    BrowserAutomationAgent,
    CalendarAssistantAgent,
    CodingAgent,
    DataAnalysisAgent,
    EmailAssistantAgent,
    ResearchAgent,
)
from utils.logger import get_logger

logger = get_logger("agent_manager")


class AgentManager:
    """Agent Factory and Registry for routing task goals to specialized AI Agents."""

    def __init__(self, model: Any = None, tokenizer: Any = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.agents: Dict[str, BaseAgent] = {}
        self._register_default_agents()

    def _register_default_agents(self) -> None:
        agents_list = [
            CodingAgent(model=self.model, tokenizer=self.tokenizer),
            ResearchAgent(model=self.model, tokenizer=self.tokenizer),
            EmailAssistantAgent(model=self.model, tokenizer=self.tokenizer),
            CalendarAssistantAgent(model=self.model, tokenizer=self.tokenizer),
            BrowserAutomationAgent(model=self.model, tokenizer=self.tokenizer),
            DataAnalysisAgent(model=self.model, tokenizer=self.tokenizer),
        ]
        for agent in agents_list:
            self.agents[agent.agent_type] = agent
        logger.info(f"AgentManager initialized with {len(self.agents)} specialized AI Agents.")

    def get_agent(self, agent_type: str) -> BaseAgent:
        """Retrieves specialized agent instance or defaults to ResearchAgent."""
        return self.agents.get(agent_type.lower(), self.agents["research"])

    def list_agent_types(self) -> List[Dict[str, str]]:
        """Returns list of all available agent types and descriptions."""
        return [
            {
                "agent_type": agent.agent_type,
                "name": agent.name,
                "description": agent.description,
            }
            for agent in self.agents.values()
        ]


    def execute_agent_task(
        self,
        agent_type: str,
        goal: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """Routes and executes task goal via target specialized agent."""
        agent = self.get_agent(agent_type)
        return agent.run(goal=goal, parameters=parameters)

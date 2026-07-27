from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time


from utils.logger import get_logger

logger = get_logger("base_agent")


@dataclass
class AgentStep:
    """Represents a single step in an Agent's ReAct execution loop."""

    step_index: int
    thought: str
    action: str
    observation: str
    timestamp: str = field(default_factory=lambda: time.strftime("%H:%M:%S"))


@dataclass
class AgentResult:
    """Final output result of an Agent task execution."""

    task_id: str
    agent_type: str
    goal: str
    status: str  # 'completed', 'failed'
    steps: List[AgentStep]
    final_output: str
    artifacts: Optional[Dict[str, Any]] = None


class BaseAgent:
    """Base class for Autonomous ReAct AI Agents."""

    def __init__(self, agent_type: str, name: str, description: str, model: Any = None, tokenizer: Any = None) -> None:
        self.agent_type = agent_type
        self.name = name
        self.description = description
        self.model = model
        self.tokenizer = tokenizer

    def run(self, goal: str, parameters: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Executes the agent task goal using the ReAct loop."""
        task_id = f"agent_task_{int(time.time())}"
        logger.info(f"[{self.name}] Initiating task goal: '{goal}' (Task ID: {task_id})")

        steps: List[AgentStep] = []
        parameters = parameters or {}

        try:
            final_output, artifacts = self._execute_react_loop(goal, parameters, steps)
            logger.info(f"[{self.name}] Task completed successfully.")
            return AgentResult(
                task_id=task_id,
                agent_type=self.agent_type,
                goal=goal,
                status="completed",
                steps=steps,
                final_output=final_output,
                artifacts=artifacts,
            )
        except Exception as e:
            logger.error(f"[{self.name}] Task failed with error: {e}")
            return AgentResult(
                task_id=task_id,
                agent_type=self.agent_type,
                goal=goal,
                status="failed",
                steps=steps,
                final_output=f"Error executing agent task: {str(e)}",
            )

    def _execute_react_loop(
        self,
        goal: str,
        parameters: Dict[str, Any],
        steps: List[AgentStep],
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Abstract ReAct loop method implemented by specialized agents."""
        raise NotImplementedError("Specialized agents must implement _execute_react_loop()")

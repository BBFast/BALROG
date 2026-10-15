import copy
import re

from balrog.agents.base import BaseAgent
from balrog.client import LLMClientWrapper


class ReactBaby(BaseAgent):
    """An agent that performs actions using the react approach."""

    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.plan = None

    def act(self, obs, prev_action=None):
        """Generate the next action and update the plan based on the current observation.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            LLMResponse: The response containing the updated plan and selected action.
        """
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)

        plan_text = f"Current thought:\n{self.plan}\n" if self.plan else "You have no thought yet.\n"

        react_instructions = """
Thought 1:  Active rules: baba is you, door is win. door is the goal. I am baba. 
The goal is 2 steps down.
Plan: go down twice.
Action 1: down
Observation 1:
rule `door` 5 step to the left and 1 step up
rule `is`   4 step to the left and 1 step up
rule `win`  3 step to the left and 1 step up
door 1 step down
rule `baba` 5 step to the left and 4 steps down
rule `is`   4 step to the left and 4 steps down
rule `you`  3 step to the left and 4 steps down
Active rules: door is win, baba is you
Thought 2: Everything shifted up by one, that confirms I moved down.
Rules unchanged. Door is now 1 step down. One more move.
Action 2:  down
Observation 2: You win!
NEXT THOUGHT: <your next though what will bring you closer to the goal>
NEXT ACTION: <your next action>
        """.strip()

        messages = self.prompt_builder.get_prompt()
        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + plan_text + "\n" + react_instructions 

        response = self.client.generate(messages)

        # Extract the plan and action from the LLM's response
        plan, action = self._extract_plan_and_action(response.completion)

        # Update the internal plan if it has changed
        if plan != "No changes to the plan.":
            self.plan = plan

        # Save the plan in the response.reasoning field and the action in response.completion
        response = response._replace(reasoning=plan, completion=action)
        return response

    def _extract_plan_and_action(self, response_text):
        """Extract the plan and action from the LLM's response text.

        Args:
            response_text (str): The response text from the LLM.

        Returns:
            tuple: A tuple containing the extracted plan and action.
        """
        plan_match = re.search(r"NEXT THOUGHT:\s*(.*?)(?=\nNEXT ACTION:|\Z)", response_text, re.IGNORECASE | re.DOTALL)
        action_match = re.search(r"NEXT ACTION:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL)

        plan = plan_match.group(1).strip() if plan_match else "No changes to the plan."
        action = action_match.group(1).strip() if action_match else None

        return plan, action

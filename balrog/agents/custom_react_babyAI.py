import re

from balrog.agents.base import BaseAgent


class ReactBaby(BaseAgent):
    """An agent that uses react to solve babyAI."""

    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()

    def act(self, obs, prev_action=None):
        """Generate a thought and the next action based on the history.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            LLMResponse: The response containing the thought and selected action.
        """
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)

        react_instructions = """
To solve the task use the steps: Observation, Thought, and Action.
A Thought reasons about the current situation:  Devide your problem in many small steps to achieve the goal, keeping track of your progress and how to solve unexpected observations.
The Observation is the state of your enviroment and that should be important when formulating your thoughts.

Here is an example for the mission "go to the red ball":

Observation: a wall 4 steps forward, a red ball 1 step left and 2 steps forward
Thought: The read bol is my goal because it is my mission. I need to get to my goal. My goal is 2 steps forward and one to the left. To reach my goal I should use eather go forward or left.
Action: go forward
Observation: a wall 3 steps forward, a red ball 1 step left and 1 step forward
Thought: I used one action as I should. The read bol is my goal because it is my mission. I need to get to my goal. My goal is 1 steps forward and 1 to the left. To reach my goal I should use eather go forward or left.
Action: go forward
Observation: a wall 2 steps forward, a red ball 1 step left
Thought: I used one action as I should. The read bol is my goal because it is my mission. I need to get to my goal. My goal is 1 to the left. To reach my goal I should use left.
Action: turn left
Observation: a red ball 1 step forward
Thought: The red ball is right in front of me, so I have reached it.

Here is an example for navigation:
Thought: There is a wall forward and on my right, therefore I can not go foward or to the right. To move I should turn left.
Observation: a wall 1 step forward; a wall 1 step right
Action: turn left

Action Control:
Thought: There is a wall forward and on my left, therefore I can not go foward or to the left. To move I should turn right.
Observation: a wall 1 step forward; a wall 1 step left
Action: go forward, turn right
Thought: My last action had two components go forward and turn left. I should use only one action. To move I should turn right
Observation: a wall 1 step forward; a wall 1 step left
Action: turn right

Answer in the following format:
Thought: <your reasoning>
Action: <choose your ONE SINGLE action>
        """.strip()

        messages = self.prompt_builder.get_prompt()
        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + react_instructions

        response = self.client.generate(messages)

        thought, action = self._extract_thought_and_action(response.completion)

        # Keep the thought in the trajectory so the context interleaves thoughts, actions and observations
        self.prompt_builder.update_reasoning(f"Thought: {thought}\nAction: {action}")

        # Save the thought in the response.reasoning field and the action in response.completion
        response = response._replace(reasoning=thought, completion=action)
        return response

    def _extract_thought_and_action(self, response_text):
        """Extract the thought and action from the LLM's response text.

        Args:
            response_text (str): The response text from the LLM.

        Returns:
            tuple: A tuple containing the extracted thought and action.
        """
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|\Z)", response_text, re.IGNORECASE | re.DOTALL)
        action_match = re.search(r"Action:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL)

        thought = thought_match.group(1).strip() if thought_match else ""
        action = action_match.group(1).strip() if action_match else None

        return thought, action

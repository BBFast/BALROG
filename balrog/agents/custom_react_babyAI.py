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
To solve the task use the steps:  Thought, Observation, and Action.
A Thought reasons about the current situation:  Reflect how the current observation, your mission appears it? Plan of how to reach your mission and the actions to help you. Devide your problem in many small steps to achieve the mission.
The Observation is the state of your enviroment and that should be important when formulating your thoughts.
A Action is a unique term from the allowed actions of the game.

Rules of the world:
- All positions are relative to you. "X steps left" or "X steps right" means you must first turn that way, then go forward. Only "forward" moves you.
- You can only interact with an object that is exactly "1 step forward", with no left or right part. "pick up" and "toggle" do nothing in any other position.
- If a wall or any object is "1 step forward", you cannot "go forward". Turn left or right instead.
- Before each action, check whether it brings the target closer. Don't repeat an action that didn't change the observation.
- Only talk about objects that appear in the current observation. Never invent objects.

Missions:
- "go to the <object>": you succeed when the object is "1 step forward". Then you are done.
- "open the <color> door": go to the door until it is "1 step forward", then use "toggle".
  - A closed door opens with "toggle".
  - A locked door needs the key of the SAME color:
    1. Move until "a <color> key 1 step forward".
    2. "pick up". The observation will then show "You carry a <color> key".
    3. Move until "a locked <color> door 1 step forward".
    4. "toggle". The door becomes "open".

Here is an example for the mission "go to the red ball":
Thought: There is a wall forward and on my left, therefore I can not go foward or to the left and I should act turn right.
Observation: a wall 1 step forward; a wall 1 step left
Action: turn right
Thought: The read bol is my mission. My mission is 2 steps forward and one to the left. To reach my mission I should act: go forward or left.
Observation: a wall 4 steps forward, a red ball 1 step left and 2 steps forward
Action: go forward
Thought: the wall is closer to me, 4-3=1, that confirms that I moved foward. I should evaluate again how to get to my mission. The red ball is 1 steps forward and 1 to the left and to reach it I should act: go forward or left.
Observation: a wall 3 steps forward, a red ball 1 step left and 1 step forward
Action: go forward
Thought: the wall is closer to me, 3-2=1, that confirms that I moved foward. I should investigate how to get to the red ball. My mission is on the left, therefore I will act turn left.
Observation: a wall 2 steps forward, a red ball 1 step left
Action: turn left
Thought: The red ball is right in front of me, so I have reached it.
Observation: a red ball 1 step forward


Answer in the following format and follow the format rigorously:
Thought: <your reasoning>
Action: <Output exactly ONE action from the action list. "turn left" or "turn right" or "go forward" or "pick up" or "drop" or "toggle".>
""".strip()

        messages = self.prompt_builder.get_prompt()
        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + react_instructions

        response = self.client.generate(messages)

        thought, action = self._extract_thought_and_action(response.completion)

        # add the thought
        self.prompt_builder.update_reasoning(f"Thought: {thought}\nAction: {action}")

        # Save the thought 
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

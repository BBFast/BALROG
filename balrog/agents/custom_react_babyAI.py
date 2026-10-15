import re

from balrog.agents.base import BaseAgent


COMMON_RULES = """
Rules of the world:
- All positions are relative to you. "go forward" moves you; "turn left" and "turn right" rotate you.
- "pick up", "toggle" and "drop" only work on the cell exactly "1 step forward".
- "pick up" picks the object in front of you
- "drop" drops the object that you carry
- "toggle" interacts with the object in front of you
- Never repeat an action that did not change the observation.
- Only talk about objects in the CURRENT observation and the CURRENT mission. Never invent objects.
""".strip()

FEW_SHOTS = {
    "goto": """Example (mission "go to the purple box"; the objects in your game are different):
Observation: a wall 3 steps forward; a purple box 1 step left and 2 steps forward
Thought: Target purple box: 1 left, 2 forward. Forward part is not 0 -> go forward.
Action: go forward
Observation: a wall 2 steps forward; a purple box 1 step left and 1 step forward
Thought: Target purple box: 1 left, 1 forward. Forward part is not 0 -> go forward.
Action: go forward
Observation: a wall 1 step forward; a purple box 1 step left
Thought: Target purple box: 1 left, 0 forward. Turn toward it -> turn left.
Action: turn left
Observation: a purple box 1 step forward
Thought: Target purple box is 1 step forward. Mission complete.""",
    "open": """Example (mission "open the yellow door"; the objects in your game are different):
Observation: a locked yellow door 2 steps forward; a yellow key 1 step right
Thought: The door is locked and I carry nothing -> first get the yellow key. Key: 1 right -> turn right.
Action: turn right
Observation: a yellow key 1 step forward
Thought: Yellow key is 1 step forward -> pick up.
Action: pick up
Observation: You carry a yellow key; a wall 3 steps forward
Thought: I carry the yellow key. The door was on my left -> turn left.
Action: turn left
Observation: You carry a yellow key; a locked yellow door 2 steps forward
Thought: Door: 2 forward, not yet 1 -> go forward.
Action: go forward
Observation: You carry a yellow key; a locked yellow door 1 step forward
Thought: Door is 1 step forward and I carry the key of the same color -> toggle.
Action: toggle
Observation: You carry a yellow key; a open yellow door 1 step forward
Thought: The door is open. Mission complete.""",
    "pickup": """Example (mission "pick up the blue key"; the objects in your game are different):
Observation: a wall 3 steps forward; a blue key 2 steps right and 2 steps forward
Thought: Target blue key: 2 right, 2 forward. Forward part is not 0 -> go forward.
Action: go forward
Observation: a wall 2 steps forward; a blue key 2 steps right and 1 step forward
Thought: Target blue key: 2 right, 1 forward. Forward part is not 0 -> go forward.
Action: go forward
Observation: a wall 1 step forward; a blue key 2 steps right
Thought: Target blue key: 2 right, 0 forward. Turn toward it -> turn right.
Action: turn right
Observation: a wall 1 step left; a blue key 2 steps forward
Thought: Target blue key: 2 forward, not yet 1 -> go forward.
Action: go forward
Observation: a wall 1 step left; a blue key 1 step forward
Thought: Target blue key is 1 step forward -> pick up.
Action: pick up
Observation: You carry a blue key; a wall 1 step left
Thought: I carry the blue key. Mission complete.""",
    "putnext": """Example (mission "put the green ball next to the yellow box"; the objects in your game are different):
Two phases: (1) pick up the FIRST object, (2) carry it and "drop" when the SECOND object is "2 steps forward", "1 step left and 1 step forward" or "1 step right and 1 step forward".
Observation: a green ball 1 step forward; a yellow box 1 step left and 3 steps forward
Thought: Phase 1, I do not carry the green ball. Green ball is 1 step forward -> pick up.
Action: pick up
Observation: You carry a green ball; a yellow box 1 step left and 3 steps forward
Thought: Phase 2, I carry the green ball. Yellow box: 1 left, 3 forward. Not a drop spot -> go forward.
Action: go forward
Observation: You carry a green ball; a yellow box 1 step left and 2 steps forward
Thought: Phase 2. Yellow box: 1 left, 2 forward. Not a drop spot -> go forward.
Action: go forward
Observation: You carry a green ball; a yellow box 1 step left and 1 step forward
Thought: Phase 2. Yellow box: 1 left, 1 forward. This IS a drop spot -> drop.
Action: drop
Observation: a green ball 1 step forward; a yellow box 1 step left and 1 step forward
Thought: The green ball is next to the yellow box. Mission complete.""",
    "seq": """Example (mission "pick up the purple key, then go to the grey ball"; the objects in your game are different):
Do the sub-goals in order. "A, then B" means A first. "A after you B" means B first. Keep carrying what you picked up.
Observation: a grey ball 2 steps left and 1 step forward; a purple key 1 step right
Thought: Sub-goal 1: pick up the purple key. Key: 1 right -> turn right.
Action: turn right
Observation: a wall 3 steps forward; a purple key 1 step forward
Thought: Sub-goal 1. Purple key is 1 step forward -> pick up.
Action: pick up
Observation: You carry a purple key; a wall 3 steps forward
Thought: Sub-goal 1 done. Sub-goal 2: go to the grey ball. It was on my left side -> turn left.
Action: turn left
Observation: You carry a purple key; a grey ball 2 steps left and 1 step forward
Thought: Sub-goal 2. Grey ball: 2 left, 1 forward. Forward part is not 0 -> go forward.
Action: go forward
Observation: You carry a purple key; a grey ball 2 steps left
Thought: Sub-goal 2. Grey ball: 2 left, 0 forward -> turn left.
Action: turn left
Observation: You carry a purple key; a grey ball 2 steps forward
Thought: Sub-goal 2. Grey ball: 2 forward, not yet 1 -> go forward.
Action: go forward
Observation: You carry a purple key; a grey ball 1 step forward
Thought: The grey ball is 1 step forward. Mission complete.""",
}

ANSWER_FORMAT = """
Answer in the following format and follow the format rigorously:
Thought: Target <object from YOUR mission>: <its position>. <rule> -> <action>.
Action: <exactly ONE action from the action list, in lowercase. Never combine actions like "go forward, turn right".>
""".strip()

def _pick_example(mission):
        m = mission.lower()
        if "then" in m or "after you" in m or " and " in m:
            return FEW_SHOTS["seq"]
        if m.startswith("put"):
            return FEW_SHOTS["putnext"]
        if m.startswith("pick up"):
            return FEW_SHOTS["pickup"]
        if m.startswith("open"):
            return FEW_SHOTS["open"]
        return FEW_SHOTS["goto"]

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

        mission = obs.get("mission", "")
        react_instructions = "\n\n".join(
            [
                "To solve the task, reason in a short Thought, then output one Action.",
                COMMON_RULES,
                _pick_example(mission),
                f"Your mission: {mission}",
                ANSWER_FORMAT,
            ]
        )

        print(react_instructions)

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

import functools
from typing import List
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from utilities.helper import HelperUtilities
from tools.tool_empty import placeholder_tool

class ChefStrengthTeam:
    def __init__(self, model):
        self.llm = ChatOpenAI(model=model)
        self.utilities = HelperUtilities()
        self.tools = {
            'placeholder': placeholder_tool,
            # Add any specific tools needed
        }

    def strength_personal_agent(self):
        """Creates an agent that evaluates personal strengths of the applicant."""
        system_prompt_template = (
            """
            You are an agent evaluating the personal strengths of a chef applicant.
            Use the company's instructions for personal strengths:

            Strength Instructions:
            {inst_chef_strength_personal}

            Applicant Summary:
            {summary}

            Use the arguments from strength_arguer_agent to refine your persona strengths.
            {strength_arguer['personal']}
            
            **Your Task:**
            - Identify personal strengths based on the summary.
            - Store them in the `strength_personal` dictionary.

            **Format:**
            {{
                "Strength 1": "Description",
                "Strength 2": "Description",
                ...
            }}

            **Do not include any extra text; output only the JSON object.**
            """
        )

        personal_strength_agent = self.utilities.create_agent(
            self.llm,
            [self.tools['placeholder']],
            system_prompt_template
        )
        return functools.partial(
            self.utilities.agent_node,
            agent=personal_strength_agent,
            name="personal_strength_agent"
        )

    def strength_experience_agent(self):
        """Creates an agent that evaluates experience strengths of the applicant."""
        # TODO 
        # 1. Include chat history between strength_experience and strength_arguer
        system_prompt_template = (
            """
            You are an agent evaluating the experience strengths of a chef applicant.
            Use the company's instructions for experience strengths:

            Strength Instructions:
            {inst_chef_strength_experience}

            Applicant Summary:
            {summary}

            Use the arguments from strength_arguer_agent to refine your persona strengths.
            {strength_arguer['experience']}

            **Your Task:**
            - Identify experience strengths based on the summary.
            - Store them in the `strength_experience` dictionary.

            **Format:**
            {{
                "Experience 1": "Description",
                "Experience 2": "Description",
                ...
            }}

            **Do not include any extra text; output only the JSON object.**
            """
        )

        experience_strength_agent = self.utilities.create_agent(
            self.llm,
            [self.tools['placeholder']],
            system_prompt_template
        )
        return functools.partial(
            self.utilities.agent_node,
            agent=experience_strength_agent,
            name="experience_strength_agent"
        )

    def strength_arguer_agent(self):
        """Creates a conversable agent that argues against the selected strengths."""
        # TODO 
        # 1. Include chat history between strength_personal and strength_arguer
        # 2. Include chat history between strength_experience and strength_arguer
        system_prompt_template = (
            """
            You are an agent responsible for critically evaluating the selected strengths.

            Applicant Summary:
            {summary}

            Personal Strengths:
            {strength_personal}

            Experience Strengths:
            {strength_experience}

            **Your Task:**
            - Argue against the strengths provided, pointing out any potential weaknesses or concerns.
            - Engage in a conversation with the other agents to test the validity of the strengths.
            - If you accept the strengths from an agent after discussion, respond with 'OKAY' + the name of the agent.
            Example of reply: 'OKAY personal_strength_agent' or 'OKAY experience_strength_agent'.

            Store your arguments in either the 'strength_arguer' dictionary.
            Be sure to be expressive and provide a thorough evaluation.
            {{
                "personal": "Argument",
                "experience": "Argument"
            }}
            **Note:**
            - Be constructive and aim for a thorough evaluation.
            """
        )

        strength_arguer_agent = self.utilities.create_agent(
            self.llm,
            [self.tools['placeholder']],
            system_prompt_template
        )
        return functools.partial(
            self.utilities.agent_node,
            agent=strength_arguer_agent,
            name="strength_arguer_agent"
        )

    def supervisor_agent(self, members: List[str]):
        """Creates a supervisor agent that oversees the conversation."""
        system_prompt_template = (
            """
            You are the supervisor agent managing the discussion between agents.

            **Agents Involved:**
            - personal_strength_agent
            - experience_strength_agent
            - strength_arguer_agent

            **Your Role:**
            - Monitor the conversation between agents.
            - Determine when the strength_arguer_agent has accepted the strengths (i.e., responds with 'OKAY').
            - When consensus is reached, signal the relevant agent to produce the final strengths dictionary.
            - End the conversation when appropriate.

            **Conversation History:**
            {chat_history}

            **Instructions:**
            - Ensure all agents have contributed adequately before concluding.
            - If strength_arguer_agent accepts the strengths, by responding with 'OKAY' + agent name, don't prompt that agent again.
            - When the strength_arguer_agent have accepted the strengths, from both agents end the conversation by saying 'END'.
            """
        )

        supervisor_agent = self.utilities.create_team_supervisor(
            self.llm,
            system_prompt_template,
            members
        )
        return supervisor_agent
    
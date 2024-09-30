import functools
from typing import List
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from utilities.helper import HelperUtilities
from tools.tool_empty import placeholder_tool

class ChefWeaknessTeam:
    def __init__(self, model):
        self.llm = ChatOpenAI(model=model)
        self.utilities = HelperUtilities()
        self.tools = {
            'placeholder': placeholder_tool,
            # Add any specific tools needed
        }

    def weakness_personal_agent(self):
        """Creates an agent that evaluates personal weaknesses of the applicant."""
        system_prompt_template = (
            """
            You are an agent evaluating the personal weaknesses of a chef applicant.
            Use the company's instructions for personal weaknesses:

            Weakness Instructions:
            {inst_chef_weakness_personal}

            Applicant Summary:
            {summary}

            **Your Task:**
            - Identify personal weaknesses based on the summary.
            - Store them in the `weakness_personal` dictionary.

            **Format:**
            {{
                "Weakness 1": "Description",
                "Weakness 2": "Description",
                ...
            }}

            **Do not include any extra text; output only the JSON object.**
            """
        )

        personal_weakness_agent = self.utilities.create_agent(
            self.llm,
            [self.tools['placeholder']],
            system_prompt_template
        )
        return functools.partial(
            self.utilities.agent_node,
            agent=personal_weakness_agent,
            name="personal_weakness_agent"
        )

    def weakness_experience_agent(self):
        """Creates an agent that evaluates experience weaknesses of the applicant."""
        system_prompt_template = (
            """
            You are an agent evaluating the experience weaknesses of a chef applicant.
            Use the company's instructions for experience weaknesses:

            Weakness Instructions:
            {inst_chef_weakness_experience}

            Applicant Summary:
            {summary}

            **Your Task:**
            - Identify experience weaknesses based on the summary.
            - Store them in the `weakness_experience` dictionary.

            **Format:**
            {{
                "Weakness 1": "Description",
                "Weakness 2": "Description",
                ...
            }}

            **Do not include any extra text; output only the JSON object.**
            """
        )

        experience_weakness_agent = self.utilities.create_agent(
            self.llm,
            [self.tools['placeholder']],
            system_prompt_template
        )
        return functools.partial(
            self.utilities.agent_node,
            agent=experience_weakness_agent,
            name="experience_weakness_agent"
        )

    def weakness_arguer_agent(self):
        """Creates a conversable agent that argues against the identified weaknesses."""
        system_prompt_template = (
            """
            You are an agent responsible for critically evaluating the selected weaknesses.

            Applicant Summary:
            {summary}

            Personal Weaknesses:
            {weakness_personal}

            Experience Weaknesses:
            {weakness_experience}

            **Your Task:**
            - Argue against the weaknesses provided, pointing out any strengths or mitigating factors.
            - Engage in a conversation with the other agents to test the validity of the weaknesses.
            - If you accept the weaknesses from an agent after discussion, respond with 'OKAY' + the name of the agent.
            Example of reply: 'OKAY personal_weakness_agent' or 'OKAY experience_weakness_agent'.

            **Note:**
            - Be constructive and aim for a thorough evaluation.
            """
        )

        weakness_arguer_agent = self.utilities.create_agent(
            self.llm,
            [self.tools['placeholder']],
            system_prompt_template
        )
        return functools.partial(
            self.utilities.agent_node,
            agent=weakness_arguer_agent,
            name="weakness_arguer_agent"
        )

    def supervisor_agent(self, members: List[str]):
        """Creates a supervisor agent that oversees the conversation."""
        system_prompt_template = (
            """
            You are the supervisor agent managing the discussion between agents.

            **Agents Involved:**
            - personal_weakness_agent
            - experience_weakness_agent
            - weakness_arguer_agent

            **Your Role:**
            - Monitor the conversation between agents.
            - Determine when the weakness_arguer_agent has accepted the weaknesses (i.e., responds with 'OKAY').
            - When consensus is reached, signal the relevant agent to produce the final weaknesses dictionary.
            - End the conversation when appropriate.

            **Conversation History:**
            {chat_history}

            **Instructions:**
            - Ensure all agents have contributed adequately before concluding.
            - If weakness_arguer_agent accepts the weaknesses, by responding with 'OKAY' + agent name, don't prompt that agent again.
            - When the weakness_arguer_agent has accepted the weaknesses, from both agents, end the conversation by saying 'END'.
            """
        )

        supervisor_agent = self.utilities.create_team_supervisor(
            self.llm,
            system_prompt_template,
            members
        )
        return supervisor_agent

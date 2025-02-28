import datetime
from dotenv import load_dotenv

load_dotenv()
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from schemas import AnswerQuestion, ReviseAnswer

llm = ChatOpenAI(model="gpt-4-turbo-preview")
parser = JsonOutputToolsParser(return_id=True)
# take response from LLM, search for function calling invocation, and parse and transform into answer question object
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

# Use MessageGraph, state of graph nodes that changes upon every node is a list of messages
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        # system prompt
        (
            "system",
            """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
        ),
        # reuse prompt by reviser node to keep criticizing, MessagesPlaceholder contains all history info what was criticized
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
# to populate some already known placeholders, when invoking template, plug in here the date
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction = "Provide a detailed ~250 words answer."
)
first_responder = first_responder_prompt_template | llm.bind_tools(
    # tool choice force LLM to ground the response to the Pydantic object we want to receive
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

# this instruction will plug in actor_prompt_template's 1.{first instruction}.
revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

# function calling to supply tools and tool_choice to enforce schema of pydantic object revise answer class and make LLM arrive at answer of that kind of object
revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")

if __name__ == "__main__":
    human_message = HumanMessage(
        content="Write about DeepSeek MoE and GRPO, list its impact and applications to future AI research."
    )
    chain = (
        first_responder_prompt_template
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | parser_pydantic
    )

    res = chain.invoke(input={"messages": [human_message]})
    print(res)
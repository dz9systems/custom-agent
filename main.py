from dotenv import load_dotenv
import os
import openai

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.agents import Tool, AgentExecutor, BaseSingleActionAgent
from langchain.llms import OpenAI

# Importing the serpapi i.e, SerpAPIWrapper class
from langchain.utilities import SerpAPIWrapper

# Importing necessary types from the typing module
from typing import List, Tuple, Any, Union

# Importing specific classes from the langchain.schema module
from langchain.schema import AgentAction, AgentFinish


# Load environment variables from a .env file
load_dotenv()
# Set API key for the OpenAI module
openai.openai_api_key = os.getenv("OPENAI_API_KEY")
# Set API key for the serpAPI module
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

# Create instances of OpenAI and ChatOpenAI
llm = OpenAI()
chat_model = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))


# Create a function that will be used to run the OpenAI model
search = SerpAPIWrapper(serpapi_api_key =serpapi_api_key)
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
        return_direct=True,
    )
]

# Defining a class named FakeAgent that inherits from BaseSingleActionAgent
class FakeAgent(BaseSingleActionAgent):
    """Fake Custom Agent."""

    # Defining a property named input_keys that returns a list with one element, "input"
    @property
    def input_keys(self):
        return ["input"]

    # Defining a method named plan that takes intermediate_steps, a list of tuples, and additional keyword arguments
    # It returns an AgentAction object with specific properties
    def plan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decide what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        # Returning an AgentAction object with tool set to "Search",
        # tool_input set to the value associated with the "input" key in kwargs, and an empty log
        return AgentAction(tool="Search", tool_input=kwargs["input"], log="")

    # Defining an asynchronous version of the plan method, named aplan
    async def aplan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decide what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        # Returning an AgentAction object with tool set to "Search",
        # tool_input set to the value associated with the "input" key in kwargs, and an empty log
        return AgentAction(tool="Search", tool_input=kwargs["input"], log="")

        # Creating an instance of the FakeAgent class
agent = FakeAgent()

# Initializing an AgentExecutor with the FakeAgent instance, a set of tools, and verbose mode set to True
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)

# Running the agent executor with the input "How many people live in Canada as of 2023?"
agent_executor.run("Who mike douglas?")





from typing import List, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END


@tool
def get_current_time(city: str) -> str:
    """Return the current time in the given city (no timezone accuracy, just demo)."""
    from datetime import datetime

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"Current time in {city}: {now}"


# @tool
# def execute_code(code: str) -> str:
#     """Executes the given Python code."""
#     result = exec(code)
#     return str(result)


class ChatState(TypedDict):
    messages: List[BaseMessage]


class Agent:
    """Agent with tool calling capabilities."""

    def __init__(self, base_url: str, model: str):
        """
        Initialize the agent with LLM and tools.

        Args:
            base_url: Base URL for the LLM API endpoint
            model: Model identifier
        """
        self.base_url = base_url
        self.model = model
        self.state = None
        self.tools = [get_current_time]

        # Initialize LLM with tool binding
        self.llm = ChatOpenAI(
            model=self.model,
            base_url=self.base_url,
            api_key="", # dummy, required by the client
        ).bind_tools(self.tools)

        # Build the graph
        self.graph = self._build_graph()

    def _call_tools(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        Execute tool calls from AIMessage responses.

        Args:
            messages: List of messages to process

        Returns:
            List of messages with tool results appended
        """
        for message in messages:
            if isinstance(message, AIMessage) and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call["name"]
                    tool_input = tool_call["args"]
                    for tool in self.tools:
                        if tool.name == tool_name:
                            tool_output = tool.invoke(tool_input)
                            messages.append(
                                AIMessage(
                                    content=f"Tool '{tool_name}' output: {tool_output}"
                                )
                            )
        return messages

    def _chat_node(self, state: ChatState) -> ChatState:
        """
        Main chat node that invokes LLM and executes tools.

        Args:
            state: Current chat state

        Returns:
            Updated chat state with new messages
        """
        messages = state["messages"]
        response = self.llm.invoke(messages)
        response = self._call_tools([response])
        return {"messages": messages + response}

    def _build_graph(self):
        """
        Build and compile the LangGraph workflow.

        Returns:
            Compiled StateGraph
        """
        builder = StateGraph(ChatState)
        builder.add_node("chat", self._chat_node)
        builder.set_entry_point("chat")
        builder.add_edge("chat", END)
        return builder.compile()

    def _invoke(self, message: str) -> ChatState:
        """
        Invoke the agent graph with the given state.

        Args:
            message: User message to process

        Returns:
            Final chat state after graph execution
        """
        if self.state is None:
            self.state = {"messages": []}

        # Append new user message to existing conversation history
        self.state["messages"].append(HumanMessage(content=message))
        state = self.graph.invoke(self.state)

        # Update state with the result from graph execution
        self.state = state
        return state
    
    def __call__(self, message: str) -> str:
        self._invoke(message)
        return self.state["messages"][-1].content


class Qwen3_4B_2507(Agent):
    """Agent using Qwen 3 4B model with tool calling capabilities."""

    def __init__(self, base_url: str ):
        """
        Initialize the Qwen agent with LLM and tools.

        Args:
            base_url: Base URL for the LLM API endpoint
        """
        super().__init__(base_url=base_url, model="qwen/qwen3-4b-2507")
        # Rebuild LLM with updated tools
        self.llm = ChatOpenAI(
            model=self.model,
            base_url=self.base_url,
            api_key="", # dummy, required by the client
        ).bind_tools(self.tools)


def from_name(model: str, base_url: str) -> Agent:
    """
    Factory function to create an agent based on model name.

    Args:
        model: Model identifier
        base_url: Base URL for the LLM API endpoint

    Returns:
        Appropriate Agent instance for the model
    """
    if model == "qwen/qwen3-4b-2507":
        return Qwen3_4B_2507(base_url=base_url)
    else:
        return Agent(model=model, base_url=base_url)

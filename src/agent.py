import os
from typing import List

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage
from langchain.tools import tool, BaseTool
from callbacks import AgentCallbackHandler

from tools import calculate_shipping_cost, inventory_lookup
from rag import create_rag_chain


load_dotenv()
GOOGLE_MODEL_ID = os.getenv("GOOGLE_MODEL_ID")


def find_tool_by_name(tools: List[BaseTool], tool_name: str) -> BaseTool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")


@tool("get_shipping_cost", description="Returns the cost of shipping in USD based on distance in km and weight in kg")
def get_shipping_cost(distance_km: float, weight_kg: float)->float:
    """
    Returns the cost of shipping in USD based on distance in km and weight in kg
    """
    cost = calculate_shipping_cost(distance_km, weight_kg)
    return cost


@tool("get_inventory_lookup", description="Returns a string of info that details whether a request can be fulfilled given an item name and number of items required")
def get_inventory_lookup(item: str, num_required: int)->str:
    """
    Returns a string of info that details whether a request can be fulfilled given an item name and number of items required
    """
    lookup_res = inventory_lookup(item, num_required)
    return lookup_res
    
    
@tool("get_rag_info", description="Search the medical logistics knowledge docs to answer questions and to seek item names")
def get_rag_info(question: str)->str:
    """
    Search the medical logistics knowledge docs to answer questions and to seek item names
    """
    return rag_chain.invoke(question)
    

tools = [get_shipping_cost, get_inventory_lookup, get_rag_info]

llm = ChatGoogleGenerativeAI(
    model=GOOGLE_MODEL_ID,
    callbacks=[AgentCallbackHandler()]
)

llm_with_tools = llm.bind_tools(tools)
rag_chain = create_rag_chain(llm_with_tools)

if __name__ == "__main__":
    user_input = input("Enter input: ")
    messages: List[BaseMessage] = [HumanMessage(content=user_input)]
    
    while True:
        ai_message = llm_with_tools.invoke(messages)

        tool_calls = getattr(ai_message, "tool_calls", None) or []
        if len(tool_calls) > 0:
            messages.append(ai_message)
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id")

                tool_to_use = find_tool_by_name(tools, tool_name)
                observation = tool_to_use.invoke(tool_args)
                print(f"observation={observation}")

                messages.append(
                    ToolMessage(content=str(observation), tool_call_id=tool_call_id)
                )
            continue

        print(ai_message.content)
        break
        
    
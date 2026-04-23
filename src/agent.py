from typing import List

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage
from langchain.tools import tool, BaseTool
from callbacks import AgentCallbackHandler
# from src.rag import load_vector_db
from tools import calculate_shipping_cost, inventory_lookup, keep_inventory, parse_model_output

load_dotenv()



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

@tool("get_inventory_lookup", description="Returns a string of info regarding the item's stock based upon the input string 'Item, Number Requested'")
def get_inventory_lookup(item_str: str)->str:
    """
    Returns a string of info regarding the item's stock based upon the input string 'Item, Number Requested'
    """
    lookup_res = inventory_lookup(item_str)
    return lookup_res
    
if __name__ == "__main__":
    tools = [get_shipping_cost, get_inventory_lookup]
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        callbacks=[AgentCallbackHandler()]
    )
    
    llm_with_tools = llm.bind_tools(tools)
    
    user_input = input("Enter input: ")
    messages: List[BaseMessage] = [HumanMessage(content=user_input)]
    
    while True:
        ai_message = llm_with_tools.invoke(messages)

        # If the model decides to call tools, execute them and return results
        tool_calls = getattr(ai_message, "tool_calls", None) or []
        if len(tool_calls) > 0:
            messages.append(ai_message)
            for tool_call in tool_calls:
                # tool_call is typically a dict with keys: id, type, name, args
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id")

                tool_to_use = find_tool_by_name(tools, tool_name)
                observation = tool_to_use.invoke(tool_args)
                print(f"observation={observation}")

                messages.append(
                    ToolMessage(content=str(observation), tool_call_id=tool_call_id)
                )
            # Continue loop to allow the model to use the observations
            continue

        # No tool calls -> final answer
        print(ai_message.content)
        break
        
    
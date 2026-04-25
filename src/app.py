import streamlit as st
from langchain_core.messages import HumanMessage, ToolMessage
from agent import llm_with_tools, tools, find_tool_by_name

st.title("Medical Logistics Agent")


def run_agent(user_input: str):
    messages = [HumanMessage(content=user_input)]

    while True:
        ai_message = llm_with_tools.invoke(messages)

        tool_calls = getattr(ai_message, "tool_calls", None) or []
        if tool_calls:
            messages.append(ai_message)

            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id")

                tool = find_tool_by_name(tools, tool_name)
                observation = tool.invoke(tool_args)

                messages.append(
                    ToolMessage(content=str(observation), tool_call_id=tool_call_id)
                )
            continue

        return ai_message.content


if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Enter your request:")

if st.button("Submit"):
    if user_input:
        response = run_agent(user_input)

        st.session_state.history.append({
            "user": user_input,
            "response": response
        })

for chat in st.session_state.history:
    st.write("User:", chat["user"])
    st.write("Agent:", chat["response"])
import streamlit as st
from langchain_core.messages import HumanMessage, ToolMessage
from agent import tools, find_tool_by_name

st.title("Medical Logistics Agent")

@st.cache_resource
def load_agent():
    from agent import llm_with_tools, rag_chain
    return llm_with_tools, rag_chain


llm_with_tools, rag_chain = load_agent()


def run_agent(user_input: str):
    st.session_state.messages.append(HumanMessage(content=user_input))

    while True:
        ai_message = llm_with_tools.invoke(st.session_state.messages)

        tool_calls = getattr(ai_message, "tool_calls", None) or []
        if tool_calls:
            st.session_state.messages.append(ai_message) # type: ignore

            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id")

                tool_to_use = find_tool_by_name(tools, tool_name)
                observation = tool_to_use.invoke(tool_args)

                st.session_state.messages.append(
                    ToolMessage(content=str(observation), tool_call_id=tool_call_id) # type: ignore
                )
            continue

        return ai_message.content


if "history" not in st.session_state:
    st.session_state.history = []
    
if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("Enter your request:")

if st.button("Submit"):
    if user_input.strip():
        response = run_agent(user_input)

        st.session_state.history.append({
            "user": user_input,
            "response": response
        })

for chat in st.session_state.history:
    st.write("**:blue[User]:**", chat["user"])
    st.write("**:green[Agent]:**", chat["response"][0]["text"])
import streamlit as st

from agent import get_agent
from memory import load_memory, save_memory
from rag import create_vector_db
from tools import parse_model_output


st.set_page_config(page_title="Medical Logistics Agent", page_icon="+", layout="centered")
st.title("Medical Logistics Agent")
st.caption("RAG + tools for medical logistics support")


@st.cache_resource
def get_cached_agent():
    return get_agent()


if "history" not in st.session_state:
    st.session_state.history = load_memory()


col1, col2 = st.columns(2)
with col1:
    if st.button("Build / Refresh RAG DB"):
        try:
            create_vector_db()
            st.success("RAG database is ready")
        except Exception as e:
            st.error(str(e))

with col2:
    if st.button("Clear Chat History"):
        st.session_state.history = []
        save_memory([])
        st.success("Chat history cleared")


user_input = st.text_input("Ask a question")

if st.button("Submit"):
    if user_input:
        try:
            agent = get_cached_agent()
            raw_response = agent.run(user_input)
            parsed_response = parse_model_output(raw_response)

            st.session_state.history.append(
                {
                    "user": user_input,
                    "response": raw_response,
                }
            )
            save_memory(st.session_state.history)

            st.subheader("Latest Response")
            st.write(raw_response)
            if parsed_response is not None:
                st.subheader("Parsed JSON")
                st.json(parsed_response)

        except Exception as e:
            st.error(str(e))


st.subheader("Chat History")
for chat in st.session_state.history:
    st.write("User:", chat["user"])
    st.write("Agent:", chat["response"])
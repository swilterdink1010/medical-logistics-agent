from memory import load_memory, save_memory

if "history" not in st.session_state:
    st.session_state.history = load_memory()

if st.button("Submit"):
    if user_input:
        response = agent.run(user_input)

        st.session_state.history.append({
            "user": user_input,
            "response": response
        })

        save_memory(st.session_state.history)

for chat in st.session_state.history:
    st.write("User:", chat["user"])
    st.write("Agent:", chat["response"])
    
import streamlit as st

from models import from_name


if "base_url" not in st.session_state:
    st.session_state["base_url"] = "http://localhost:1234/v1"

if "model" not in st.session_state:
    st.session_state["model"] = "qwen/qwen3-4b-2507"

if "agent" not in st.session_state:
    st.session_state["agent"] = from_name(
        st.session_state["model"], st.session_state["base_url"]
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Databot")

# Display all previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept new input
if prompt := st.chat_input("How can I assist you?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    response = st.session_state["agent"](prompt)

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

"""
Nano-Cog 0.1 - A laptop-scale language agent with high reasoning-per-FLOP efficiency
Streamlit UI
"""

import os
import json
import time
import streamlit as st
from src.core.model import NanoCogModel
from src.tools.dispatcher import ToolDispatcher
from src.core.retrieval import RetrievalSystem
from src.utils.config import load_config

# Set page configuration
st.set_page_config(
    page_title="Nano-Cog 0.1",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .user-bubble {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1E88E5;
    }
    .assistant-bubble {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
        border-left: 4px solid #4CAF50;
    }
    .tool-call {
        background-color: #FFF8E1;
        color: #FF8F00;
        padding: 0.5rem;
        border-radius: 0.5rem;
        font-family: monospace;
        margin: 0.5rem 0;
        border-left: 3px solid #FF8F00;
    }
    .tool-result {
        background-color: #F1F8E9;
        color: #558B2F;
        padding: 0.5rem;
        border-radius: 0.5rem;
        font-family: monospace;
        margin: 0.5rem 0;
        border-left: 3px solid #558B2F;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(model_path=None, config_path=None):
    """
    Load model and cache it with Streamlit

    Args:
        model_path (str, optional): Path to model checkpoint
        config_path (str, optional): Path to config file

    Returns:
        tuple: (model, tool_dispatcher, retrieval_system, config)
    """
    # Load configuration
    config = load_config(config_path)

    # Initialize model
    model = NanoCogModel(config_path)

    if model_path and os.path.exists(model_path):
        model.load(model_path)

    # Initialize tool dispatcher
    tool_dispatcher = ToolDispatcher(config_path)

    # Initialize retrieval system
    retrieval = RetrievalSystem(config_path)

    return model, tool_dispatcher, retrieval, config


def format_response(response):
    """
    Format response with syntax highlighting for tool calls

    Args:
        response (str): Response text

    Returns:
        str: Formatted HTML
    """
    # Format tool calls and results
    formatted = response

    # Highlight calculator tool calls
    formatted = formatted.replace("<<calc>>", "<span class='tool-call'><<calc>>")
    formatted = formatted.replace("<<calc>>", "<<calc>></span>")

    # Highlight Python tool calls
    formatted = formatted.replace("<<py>>", "<span class='tool-call'><<py>>")
    formatted = formatted.replace("<<py>>", "<<py>></span>")

    # Highlight bash tool calls
    formatted = formatted.replace("<<bash>>", "<span class='tool-call'><<bash>>")
    formatted = formatted.replace("<<bash>>", "<<bash>></span>")

    # Highlight tool results
    formatted = formatted.replace(
        "calc result:", "<span class='tool-result'>calc result:"
    )
    formatted = formatted.replace(
        "python result:", "<span class='tool-result'>python result:"
    )
    formatted = formatted.replace(
        "bash result:", "<span class='tool-result'>bash result:"
    )

    # Close tool result spans (imperfect but good enough for demo)
    formatted = formatted.replace("\n", "</span>\n")

    return formatted


def main():
    """Main Streamlit app"""
    # Display header
    st.markdown(
        "<div class='main-header'>🧠 Nano-Cog 0.1</div>", unsafe_allow_html=True
    )
    st.markdown(
        "<div class='subtitle'>A laptop-scale language agent with high reasoning-per-FLOP efficiency</div>",
        unsafe_allow_html=True,
    )

    # Sidebar configuration
    st.sidebar.title("Configuration")

    model_path = st.sidebar.text_input("Model path (optional)", "")
    config_path = st.sidebar.text_input("Config path (optional)", "")

    # Advanced settings expander
    with st.sidebar.expander("Advanced Settings"):
        temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.1)
        max_length = st.slider("Max response length", 100, 4096, 2048, 100)
        top_p = st.slider("Top-p sampling", 0.1, 1.0, 0.9, 0.1)

    # Load model
    with st.spinner("Loading model..."):
        model, tool_dispatcher, retrieval, config = load_model(model_path, config_path)

    # Display model info
    st.sidebar.info(f"Model: {config['model']['backbone']['name']}")

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(
                f"<div class='user-bubble'>{message['content']}</div>",
                unsafe_allow_html=True,
            )
        else:
            formatted_content = format_response(message["content"])
            st.markdown(
                f"<div class='assistant-bubble'>{formatted_content}</div>",
                unsafe_allow_html=True,
            )

    # Chat input
    user_input = st.chat_input("Ask Nano-Cog something...")

    if user_input:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user message
        st.markdown(
            f"<div class='user-bubble'>{user_input}</div>", unsafe_allow_html=True
        )

        # Compose prompt
        if len(st.session_state.messages) > 1:
            # Build context from history
            context = ""
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    context += f"User: {msg['content']}\n"
                else:
                    context += f"Assistant: {msg['content']}\n"

            # Append new input
            full_prompt = f"{context}User: {user_input}\nAssistant:"
        else:
            # If first message, use retrieval
            full_prompt = retrieval.compose_prompt(user_input)

        # Generate response
        with st.spinner("Nano-Cog is thinking..."):
            response = model.generate(
                full_prompt, temperature=temperature, max_length=max_length, top_p=top_p
            )

            # Process tool calls
            processed_response = tool_dispatcher.process_text(response)

        # Add assistant message to history
        st.session_state.messages.append(
            {"role": "assistant", "content": processed_response}
        )

        # Display assistant message
        formatted_response = format_response(processed_response)
        st.markdown(
            f"<div class='assistant-bubble'>{formatted_response}</div>",
            unsafe_allow_html=True,
        )

    # Add sidebar controls
    st.sidebar.title("Actions")
    if st.sidebar.button("Clear conversation"):
        st.session_state.messages = []
        st.experimental_rerun()

    # Save conversation button
    if st.sidebar.button("Save conversation"):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"conversation-{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(st.session_state.messages, f, indent=2)

        st.sidebar.success(f"Conversation saved to {filename}")

    # About section
    st.sidebar.title("About")
    st.sidebar.info(
        "Nano-Cog 0.1 combines state-space sequence modeling (Mamba), "
        "retrieval, tool use, and reinforcement-tuned chain-of-thought (CoT) "
        "for high reasoning-per-FLOP efficiency."
    )


if __name__ == "__main__":
    main()

import ollama
import streamlit as st

st.title("TRJ Chatbot")

# Initialize history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Initialize model and model parameters
if "model" not in st.session_state:
    st.session_state["model"] = ""
if "temperature" not in st.session_state:
    st.session_state["temperature"] = 0.7  # Default temperature
if "top_p" not in st.session_state:
    st.session_state["top_p"] = 0.9      # Default top_p for randomness control

# Sidebar for model and randomness controls
with st.sidebar:
    st.header("Model Configuration")

    # Corrected model listing
    try:
        models = [model['model'] for model in ollama.list().get('models', [])]
        if not models:
            st.warning("No Ollama models found. Please ensure Ollama is running and models are downloaded.")
            st.stop() # Stop execution if no models are available
    except Exception as e:
        st.error(f"Error connecting to Ollama: {e}. Please ensure Ollama is running.")
        st.stop() # Stop execution if there's an error connecting

    # Show dropdown for model selection
    st.session_state["model"] = st.selectbox("Choose your model", models, key="model_selector")

    st.subheader("Randomness Control")
    # Temperature slider
    st.session_state["temperature"] = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=st.session_state["temperature"],
        step=0.01,
        help="Controls the 'creativity' of the model. Higher values mean more random outputs.",
        key="temp_slider"
    )

    # Top P slider
    st.session_state["top_p"] = st.slider(
        "Top P",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state["top_p"],
        step=0.01,
        help="Controls the diversity of the output. Lower values mean less diverse but more focused outputs.",
        key="top_p_slider"
    )


def model_res_generator():
    stream = ollama.chat(
        model=st.session_state["model"],
        messages=st.session_state["messages"],
        options={
            "temperature": st.session_state["temperature"],
            "top_p": st.session_state["top_p"],
        },
        stream=True,
    )
    for chunk in stream:
        yield chunk["message"]["content"]

# Display chat messages from history on app rerun
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask a question..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Ensure a model is selected before attempting to generate a response
        if st.session_state["model"]:
            message = st.write_stream(model_res_generator())
            st.session_state["messages"].append({"role": "assistant", "content": message})
        else:
            st.warning("Please select a model from the sidebar to start chatting.")
import streamlit as st
import sys
import os

# Adjust path to import backend correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from middleware.backend import stream

st.set_page_config(
    page_title="AI Agent with Middleware",
    layout="centered"
)


st.title("AI Agent with Middleware")
st.markdown("A search-enabled assistant demonstrating PII redaction and summarization middleware.")

# Setup history state by thread ID
if "history_by_thread" not in st.session_state:
    st.session_state.history_by_thread = {}

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Configuration")
    thread_id = st.text_input("Active Thread ID", value="tax_session_1")
    
    st.markdown("---")
    if st.button("Clear Thread History"):
        st.session_state.history_by_thread[thread_id] = []
        st.rerun()

# Retrieve or create history for this thread
if thread_id not in st.session_state.history_by_thread:
    st.session_state.history_by_thread[thread_id] = []

messages = st.session_state.history_by_thread[thread_id]

# Render chat history
for msg in messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "tool_call":
        with st.status(f"🛠️ Tool Call: {msg['name']}", state="complete"):
            st.write("Parameters:")
            st.json(msg["args"])
    elif msg["role"] == "tool_response":
        with st.status("📥 Tool Response Received", state="complete"):
            st.code(msg["content"], language="json")
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# User prompt input
if prompt := st.chat_input("Ask about tax brackets, calculations, etc..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Store user message
    messages.append({"role": "user", "content": prompt})
    st.session_state.history_by_thread[thread_id] = messages
    
    # Generate response
    with st.chat_message("assistant"):
        text_placeholder = st.empty()
        full_response = ""
        
        for chunk in stream(prompt, thread_id):
            # Handle tools node updates specifically for status block formatting
            if "tools" in chunk:
                tool_msg = chunk["tools"]["messages"][0]
                with st.status("📥 Tool Response Received", state="complete"):
                    st.code(tool_msg.content, language="json")
                
                messages.append({
                    "role": "tool_response",
                    "content": tool_msg.content
                })
                st.session_state.history_by_thread[thread_id] = messages
                continue

            # Handle other node updates (e.g. model or middleware)
            for node_name, node_update in chunk.items():
                if isinstance(node_update, dict) and "messages" in node_update:
                    for msg in node_update["messages"]:
                        # If it is a ToolMessage (e.g. blocked tool message injected by middleware)
                        if msg.__class__.__name__ == "ToolMessage" or getattr(msg, "type", None) == "tool":
                            with st.status(f"📥 Injected Tool Response ({msg.name or 'Tool'})", state="complete"):
                                st.code(msg.content, language="json")
                            messages.append({
                                "role": "tool_response",
                                "content": msg.content
                            })
                            st.session_state.history_by_thread[thread_id] = messages
                        
                        # If it is an AIMessage
                        elif msg.__class__.__name__ == "AIMessage" or getattr(msg, "type", None) == "ai":
                            # Render tool calls
                            if getattr(msg, "tool_calls", None):
                                for tc in msg.tool_calls:
                                    with st.status(f"🛠️ Tool Call: {tc['name']}", state="running") as status:
                                        st.write("Parameters:")
                                        st.json(tc["args"])
                                        status.update(label=f"🛠️ Tool Call: {tc['name']}", state="complete")
                                    
                                    messages.append({
                                        "role": "tool_call",
                                        "name": tc["name"],
                                        "args": tc["args"]
                                    })
                                    st.session_state.history_by_thread[thread_id] = messages
                            
                            # Render stream content
                            if getattr(msg, "content", None):
                                text_content = ""
                                if isinstance(msg.content, str):
                                    text_content = msg.content
                                elif isinstance(msg.content, list):
                                    for part in msg.content:
                                        if isinstance(part, dict) and part.get("type") == "text":
                                            text_content += part.get("text", "")
                                        elif isinstance(part, str):
                                            text_content += part
                                
                                if text_content:
                                    full_response += text_content
                                    text_placeholder.markdown(full_response)

        # Save assistant text content
        if full_response:
            messages.append({"role": "assistant", "content": full_response})
            st.session_state.history_by_thread[thread_id] = messages

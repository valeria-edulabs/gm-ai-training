import pprint
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

import dotenv
dotenv.load_dotenv()

model = init_chat_model("google_genai:gemini-3.5-flash-lite")


# Use with chat models
messages = [
    SystemMessage("You are a helpful assistant."),
    HumanMessage("Hello, how are you?"),
    AIMessage("I'm a helpful assistant. How can I help you?"),
    HumanMessage("What exactly can you do?")
]

# Alternative 1: Using (role, content) tuples
# messages = [
#     ("system", "You are a helpful assistant."),
#     ("human", "Hello, how are you?"),
#     ("ai", "I'm a helpful assistant. How can I help you?"),
#     ("human", "What exactly can you do?"),
# ]

# Alternative 2: Using dictionaries (OpenAI / standard schema)
# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Hello, how are you?"},
#     {"role": "assistant", "content": "I'm a helpful assistant. How can I help you?"},
#     {"role": "user", "content": "What exactly can you do?"},
# ]

response = model.invoke(messages)  # Returns AIMessage
pprint.pprint(response.model_dump())
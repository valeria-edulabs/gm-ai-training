from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage



# Load environment variables from .env file
load_dotenv()

# create model
gemini_model = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest")
open_ai_model = ChatOpenAI(model="gpt-4o-mini")
ollama_model = ChatOllama(model="gemma3:1b", temperature=0)

# invoke
# response = gemini_model.invoke("Tell me a joke, but good one")
# print(response)

# stream
# for chunk in gemini_model.stream("Tell me 10 jokes, but good one"):
#     print(chunk.text, end="|", flush=True)

conversation = [
    {"role": "system", "content": "You are a helpful assistant that translates English to French."},
    {"role": "user", "content": "Translate: I love programming."},
    {"role": "assistant", "content": "J'adore la programmation."},
    {"role": "user", "content": "Translate: I love building applications."}
]

response = ollama_model.invoke(conversation)
print(response)  # AIMessage("J'adore créer des applications.")

parser = StrOutputParser()
print(parser.invoke(response))
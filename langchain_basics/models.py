# pip install -U "langchain[openai]"
# pip install -U "langchain[anthropic]"
# pip install -U "langchain[google-genai]"
# pip install -U langchain-ollama

import dotenv
dotenv.load_dotenv()

import pprint

from langchain.chat_models import init_chat_model
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_anthropic import ChatAnthropic



# Long text context (~50,000 tokens, well over Gemini's 32,768 token caching threshold)
article_section = (
    "In artificial intelligence and natural language processing, large language models (LLMs) "
    "are deep neural networks trained on vast text corpora. Context caching allows systems to "
    "store and reuse representations of large documents, books, or codebases across multiple requests. "
    "This significantly lowers latency and reduces cost for repetitive retrieval and reasoning tasks. "
)

long_prompt = (
    f"Here is a comprehensive research document:\n\n{article_section * 600}\n\n"
    "Based on the text above, summarize the key benefits of context caching in 2 sentences."
)

# no caching
short_prompt = "Why do parrots talk?"
short_prompt_he = "היי"


# model = init_chat_model("ollama:gemma4:12b")
# model = init_chat_model("ollama:gemma2:2b")
model = init_chat_model("google_genai:gemini-3.5-flash-lite")
# llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest")

response = model.invoke(short_prompt)
# 1. To print the full object hierarchy as a structured dict:
pprint.pprint(response.model_dump())

# 2. To print the actual formatted text cleanly (without raw dict blocks):
# print(response.text)

# 3. Or print key structured sections:
# print(f"\n[Usage]: {response.usage_metadata}")
# print(f"[Finish Reason]: {response.response_metadata.get('finish_reason')}")
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="openai/gpt-4.1-mini",
    api_key="YOUR_API_KEY",
    base_url="https://openrouter.ai/api/v1"
)

response = llm.invoke("What is AI?")

print(response.content)

import requests


API_URL = "http://127.0.0.1:8000/chat"


queries = [
    "What is Agentic AI?",

    "What are the core components of Agentic AI?",

    "What are the use cases of Agentic AI?",

    "How is Agentic AI different from traditional generative AI chatbots?",

    "What are the challenges and limitations of Agentic AI?",

    "What is the capital of France?",
]


for query in queries:

    print()
    print("=" * 70)
    print("QUESTION:", query)
    print("=" * 70)

    response = requests.post(
        API_URL,
        json={"query": query},
    )

    print(response.json())
import ollama


MODEL_NAME = "qwen2.5:7b"


def ask_llm(prompt: str) -> str:
    """
    Send a prompt to the local Ollama model
    and return the model's response.
    """

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


if __name__ == "__main__":

    print("Testing Local LLM Tool")
    print("=" * 60)

    prompt = """
    You are a data analyst.

    Explain in 2-3 sentences why understanding a database
    schema is important before generating SQL.
    """

    answer = ask_llm(prompt)

    print("\nPrompt:")
    print(prompt)

    print("\nModel Response:")
    print(answer)

    print("\nLLM Tool test completed successfully.")
import os
from groq import Groq


def call_grok(prompt: str) -> str:
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")

    client = Groq(api_key=api_key)

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=2048,
        )

        return chat_completion.choices[0].message.content

    except Exception as e:
        raise RuntimeError(f"Groq API error: {e}")
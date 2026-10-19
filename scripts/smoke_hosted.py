"""One-off check (spec §9): does a hosted provider return token usage in its stream?

Usage: GROQ_API_KEY=... uv run python scripts/smoke_hosted.py \
         [base_url] [model] [api_key_env]"""

import json
import os
import sys

import httpx


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "https://api.groq.com/openai/v1"
    model = sys.argv[2] if len(sys.argv) > 2 else "openai/gpt-oss-20b"
    key = os.environ[sys.argv[3] if len(sys.argv) > 3 else "GROQ_API_KEY"]
    body = {
        "model": model,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Say hello."}],
    }
    usage = None
    headers = {"Authorization": f"Bearer {key}"}
    url = f"{base}/chat/completions"
    with httpx.stream("POST", url, json=body, headers=headers, timeout=30) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            data = line[5:].strip() if line.startswith("data:") else ""
            if not data or data == "[DONE]":
                continue
            chunk = json.loads(data)
            usage = chunk.get("usage") or (chunk.get("x_groq") or {}).get("usage") or usage
    print(f"{model}: usage in stream = {usage}")
    return 0 if usage else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
First-run setup script.
Asks which LLM provider to use, collects the API key (or Ollama model),
validates it, and writes .env.
"""
import subprocess
import sys
from pathlib import Path

ENV_FILE = Path(".env")

PROVIDERS = {
    "1": "anthropic",
    "2": "openai",
    "3": "ollama",
}

OLLAMA_MODELS = ["llama3.2", "qwen2.5", "mistral", "gemma2"]


def print_header():
    print("\n" + "=" * 55)
    print("  AgentScrape — First-Run Setup")
    print("=" * 55)


def pick_provider() -> str:
    print("\nWhich LLM provider do you want to use?\n")
    print("  1) Anthropic  (Claude Haiku / Sonnet)  — API key required")
    print("  2) OpenAI     (GPT-4o / GPT-4o-mini)   — API key required")
    print("  3) Ollama     (local, free)             — no key needed\n")
    while True:
        choice = input("Enter 1, 2, or 3: ").strip()
        if choice in PROVIDERS:
            return PROVIDERS[choice]
        print("  Invalid choice. Enter 1, 2, or 3.")


def get_api_key(provider: str) -> str:
    labels = {
        "anthropic": "Anthropic API key (sk-ant-...)",
        "openai": "OpenAI API key (sk-...)",
    }
    while True:
        key = input(f"\nEnter your {labels[provider]}: ").strip()
        if not key:
            print("  Key cannot be empty.")
            continue
        if validate_key(provider, key):
            return key
        retry = input("  Try a different key? (y/n): ").strip().lower()
        if retry != "y":
            sys.exit(1)


def validate_key(provider: str, key: str) -> bool:
    print("  Validating key...", end=" ", flush=True)
    try:
        if provider == "anthropic":
            import anthropic

            client = anthropic.Anthropic(api_key=key)
            client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=5,
                messages=[{"role": "user", "content": "hi"}],
            )
        elif provider == "openai":
            import openai

            client = openai.OpenAI(api_key=key)
            client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=5,
                messages=[{"role": "user", "content": "hi"}],
            )
        print("valid.")
        return True
    except Exception as e:
        print(f"failed.\n  Error: {e}")
        return False


def setup_ollama() -> str:
    print("\nChecking if Ollama is installed...")
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True, check=False)
        print(f"  Found: {result.stdout.strip()}")
    except FileNotFoundError:
        print("  Ollama not found. Install it from https://ollama.com and re-run setup.")
        sys.exit(1)

    print("\nAvailable recommended models:")
    for i, model in enumerate(OLLAMA_MODELS, 1):
        print(f"  {i}) {model}")

    while True:
        choice = input("\nEnter model number (or type a custom model name): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(OLLAMA_MODELS):
            model = OLLAMA_MODELS[int(choice) - 1]
        elif choice:
            model = choice
        else:
            continue

        print(f"\n  Pulling {model} (this may take a few minutes on first run)...")
        result = subprocess.run(["ollama", "pull", model], check=False)
        if result.returncode == 0:
            print(f"  Model {model} ready.")
            return model
        print(f"  Failed to pull {model}. Try another model.")


def write_env(provider: str, key: str | None = None, ollama_model: str | None = None):
    lines = [f"LLM_PROVIDER={provider}\n"]
    if provider == "anthropic" and key:
        lines.append(f"ANTHROPIC_API_KEY={key}\n")
    elif provider == "openai" and key:
        lines.append(f"OPENAI_API_KEY={key}\n")
    elif provider == "ollama" and ollama_model:
        lines.append(f"OLLAMA_MODEL={ollama_model}\n")

    ENV_FILE.write_text("".join(lines))
    print(f"\n  Written to {ENV_FILE.resolve()}")


def main():
    print_header()

    if ENV_FILE.exists():
        overwrite = input("\n.env already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite != "y":
            print("Setup cancelled.")
            sys.exit(0)

    provider = pick_provider()

    if provider == "ollama":
        model = setup_ollama()
        write_env(provider, ollama_model=model)
    else:
        key = get_api_key(provider)
        write_env(provider, key=key)

    print("\n" + "=" * 55)
    print("  Setup complete. Run the scraper with:")
    print("  python3 main.py")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()

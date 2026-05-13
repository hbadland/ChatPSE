"""
Provider-agnostic LLM client.
Supports Google Gemini, Anthropic Claude, and OpenAI GPT.

Usage:
    from agents.llm import chat, LLMClient

    # One-shot
    reply = chat("explain VLE", model="gemini-2.5-flash")
    reply = chat("explain VLE", model="claude-sonnet-4-6")
    reply = chat("explain VLE", model="gpt-4o")

    # Multi-turn
    client = LLMClient(system="You are a thermodynamics expert.",
                       model="claude-opus-4-7")
    reply = client.send("What model for ethanol/water?")

Required environment variables (only for providers you use):
    GOOGLE_API_KEY
    ANTHROPIC_API_KEY
    OPENAI_API_KEY
"""
from __future__ import annotations
import os
import time

# ── Retry / rate-limit handling ───────────────────────────────────────────────

_RATE_LIMIT_MARKERS = (
    "rate limit", "429", "quota", "resource_exhausted",
    "too many requests", "overloaded", "503", "service unavailable",
)

def _with_retry(fn, max_retries: int = 6, base_delay: float = 15.0):
    """Call fn(), retrying with exponential backoff on rate-limit errors.

    base_delay=15s because Gemini's RESOURCE_EXHAUSTED is a per-minute RPM
    limit — short delays don't recover.  6 retries gives up to ~7 minutes of
    wait before failing, which covers sustained bursts without hanging forever.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            err = str(e).lower()
            is_transient = any(m in err for m in _RATE_LIMIT_MARKERS)
            if is_transient and attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), 120.0)  # cap at 2 min
                print(f"  [llm] rate limit hit — waiting {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            raise
    raise RuntimeError("Max retries exceeded")  # unreachable but satisfies linters


# ── Provider routing ───────────────────────────────────────────────────────────

_GOOGLE    = {"gemini"}
_ANTHROPIC = {"claude"}
_OPENAI    = {"gpt", "o1", "o3"}

DEFAULT_MODEL = "gemini-2.5-flash"


def _provider(model: str) -> str:
    m = model.lower()
    if any(m.startswith(p) for p in _GOOGLE):
        return "google"
    if any(m.startswith(p) for p in _ANTHROPIC):
        return "anthropic"
    if any(m.startswith(p) for p in _OPENAI):
        return "openai"
    raise ValueError(
        f"Cannot infer provider from model name '{model}'. "
        "Prefix must be gemini/claude/gpt/o1/o3.")


# ── Call counter (for benchmarking) ───────────────────────────────────────────

_call_count: int = 0

def reset_call_count() -> None:
    global _call_count
    _call_count = 0

def get_call_count() -> int:
    return _call_count


# ── One-shot ───────────────────────────────────────────────────────────────────

def chat(prompt: str, system: str = "", model: str = DEFAULT_MODEL,
         max_tokens: int = 8192, temperature: float | None = None) -> str:
    """Send a single prompt and return the text response.

    temperature=0 gives deterministic outputs (recommended for generation agents).
    Leave as None to use the provider default (usually ~1.0).
    """
    global _call_count
    _call_count += 1
    provider = _provider(model)
    if provider == "google":
        return _google_chat(prompt, system, model, max_tokens, temperature)
    if provider == "anthropic":
        return _anthropic_chat(prompt, system, model, max_tokens, temperature)
    return _openai_chat(prompt, system, model, max_tokens, temperature)


def _google_chat(prompt: str, system: str, model: str, max_tokens: int,
                 temperature: float | None) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=_key("GOOGLE_API_KEY"))
    kwargs = dict(system_instruction=system) if system else {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    config = types.GenerateContentConfig(max_output_tokens=max_tokens, **kwargs)
    return _with_retry(
        lambda: client.models.generate_content(
            model=model, contents=prompt, config=config).text)


def _anthropic_chat(prompt: str, system: str, model: str,
                    max_tokens: int, temperature: float | None) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=_key("ANTHROPIC_API_KEY"))
    kwargs: dict = {"system": system} if system else {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    return _with_retry(lambda: client.messages.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    ).content[0].text)


def _openai_chat(prompt: str, system: str, model: str, max_tokens: int,
                 temperature: float | None) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=_key("OPENAI_API_KEY"))
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    kwargs: dict = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    return _with_retry(lambda: client.chat.completions.create(
        model=model, max_tokens=max_tokens,
        messages=messages, **kwargs).choices[0].message.content)


# ── Multi-turn ─────────────────────────────────────────────────────────────────

class LLMClient:
    """Stateful multi-turn chat session, provider-agnostic."""

    def __init__(self, system: str = "", model: str = DEFAULT_MODEL,
                 max_tokens: int = 8192, temperature: float | None = None):
        self._system      = system
        self._model       = model
        self._max_tokens  = max_tokens
        self._temperature = temperature
        self._provider    = _provider(model)
        self._history: list[dict] = []

    def send(self, message: str) -> str:
        self._history.append({"role": "user", "content": message})
        if self._provider == "google":
            reply = self._google_send()
        elif self._provider == "anthropic":
            reply = self._anthropic_send()
        else:
            reply = self._openai_send()
        self._history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self) -> None:
        self._history = []

    def _google_send(self) -> str:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=_key("GOOGLE_API_KEY"))
        kwargs = dict(system_instruction=self._system) if self._system else {}
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        config = types.GenerateContentConfig(
            max_output_tokens=self._max_tokens, **kwargs)
        contents = [
            types.Content(
                role="user" if m["role"] == "user" else "model",
                parts=[types.Part(text=m["content"])]
            )
            for m in self._history
        ]
        return _with_retry(lambda: client.models.generate_content(
            model=self._model, contents=contents, config=config).text)

    def _anthropic_send(self) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=_key("ANTHROPIC_API_KEY"))
        kwargs: dict = {"system": self._system} if self._system else {}
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        return _with_retry(lambda: client.messages.create(
            model=self._model, max_tokens=self._max_tokens,
            messages=self._history, **kwargs,
        ).content[0].text)

    def _openai_send(self) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=_key("OPENAI_API_KEY"))
        messages = []
        if self._system:
            messages.append({"role": "system", "content": self._system})
        messages.extend(self._history)
        kwargs: dict = {}
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        return _with_retry(lambda: client.chat.completions.create(
            model=self._model, max_tokens=self._max_tokens,
            messages=messages, **kwargs).choices[0].message.content)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _key(env_var: str) -> str:
    key = os.environ.get(env_var)
    if not key:
        raise EnvironmentError(f"{env_var} environment variable is not set.")
    return key

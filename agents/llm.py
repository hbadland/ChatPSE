"""
Provider-agnostic LLM client.
Supports Google Gemini, Anthropic Claude, OpenAI GPT, Groq, and Ollama (OpenAI-compatible).

Usage:
    from agents.llm import chat, LLMClient

    # One-shot
    reply = chat("explain VLE", model="gemini-2.5-flash")
    reply = chat("explain VLE", model="claude-sonnet-4-6")
    reply = chat("explain VLE", model="gpt-4o")
    reply = chat("explain VLE", model="llama-3.3-70b-versatile")   # Groq
    reply = chat("explain VLE", model="qwen3:14b")                 # Ollama

    # Multi-turn
    client = LLMClient(system="You are a thermodynamics expert.",
                       model="claude-opus-4-7")
    reply = client.send("What model for ethanol/water?")

Required environment variables (only for providers you use):
    GOOGLE_API_KEY
    ANTHROPIC_API_KEY
    OPENAI_API_KEY
    GROQ_API_KEY     (for Groq models — llama-3.x, qwen-qwq, gemma2-, mixtral-8)
    OLLAMA_API_KEY   (optional — Ollama accepts any value; defaults to "ollama")
    OLLAMA_BASE_URL  (optional — defaults to http://localhost:11434/v1)
"""
from __future__ import annotations
import os
import re
import time

# ── Retry / rate-limit handling ───────────────────────────────────────────────

_RATE_LIMIT_MARKERS = (
    "rate limit", "429", "quota", "resource_exhausted",
    "too many requests", "overloaded", "503", "service unavailable",
)

def _with_retry(fn, max_retries: int = 6, base_delay: float = 15.0):
    """Call fn(), retrying with backoff on rate-limit errors AND empty responses.

    base_delay=15s because Gemini's RESOURCE_EXHAUSTED is a per-minute RPM
    limit — short delays don't recover.  6 retries gives up to ~7 minutes of
    wait before failing, which covers sustained bursts without hanging forever.

    An empty/whitespace-only response is also treated as transient: Ollama
    intermittently returns "" on an otherwise-healthy call, and a plain retry
    usually recovers it.  These use a brief fixed backoff (not the long
    rate-limit schedule).  On exhaustion the empty result is returned unchanged
    so the caller's existing empty-response handling (parse → ValueError) still
    applies — callers degrade the case gracefully rather than retry forever here.
    """
    for attempt in range(max_retries):
        try:
            result = fn()
        except Exception as e:
            err = str(e).lower()
            is_transient = any(m in err for m in _RATE_LIMIT_MARKERS)
            if is_transient and attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), 120.0)
                print(f"  [llm] rate limit hit — waiting {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            raise

        if result is None or (isinstance(result, str) and not result.strip()):
            if attempt < max_retries - 1:
                delay = min(2.0 * (2 ** attempt), 10.0)
                print(f"  [llm] empty response — retrying in {delay:.0f}s "
                      f"(attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            return result  # exhausted — let the caller handle the empty response
        return result
    raise RuntimeError("Max retries exceeded")


# ── Thinking-token stripping ──────────────────────────────────────────────────

def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by Qwen3 thinking mode."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


# ── Provider routing ───────────────────────────────────────────────────────────

_GOOGLE    = {"gemini"}
_ANTHROPIC = {"claude"}
_OPENAI    = {"gpt", "o1", "o3"}
# Groq model names use hyphen-version format (llama-3.x, qwen-qwq, gemma2-, mixtral-8).
# Must be checked BEFORE _OLLAMA since Ollama uses the same base names without hyphens
# (e.g. llama3:8b vs llama-3.3-70b-versatile).
_GROQ   = {"llama-3", "qwen-qwq", "gemma2-", "mixtral-8", "deepseek-r1-distill"}
_OLLAMA = {"qwen", "llama", "mistral", "phi", "deepseek", "gemma"}

DEFAULT_MODEL = "gemini-2.5-flash"

_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
_GROQ_BASE_URL   = "https://api.groq.com/openai/v1"


def _provider(model: str) -> str:
    m = model.lower()
    if any(m.startswith(p) for p in _GOOGLE):
        return "google"
    if any(m.startswith(p) for p in _ANTHROPIC):
        return "anthropic"
    if any(m.startswith(p) for p in _OPENAI):
        return "openai"
    if any(m.startswith(p) for p in _GROQ):
        return "groq"
    if any(m.startswith(p) for p in _OLLAMA):
        return "ollama"
    raise ValueError(
        f"Cannot infer provider from model name '{model}'. "
        "Prefix must be gemini/claude/gpt/o1/o3/"
        "llama-3/qwen-qwq/gemma2-/mixtral-8 (Groq) or "
        "qwen/llama/mistral/phi/deepseek/gemma (Ollama).")


# ── Call counter (for benchmarking) ───────────────────────────────────────────

_call_count: int = 0

def reset_call_count() -> None:
    global _call_count
    _call_count = 0

def get_call_count() -> int:
    return _call_count


# ── Retry temperature schedule ─────────────────────────────────────────────────

def retry_temperature(attempt: int) -> float:
    """
    Temperature schedule for agent retry loops.

    attempt=0 is the first try — temperature=0 gives deterministic output
    (most reliable for structured JSON generation).
    attempt=1+ raises to 0.3 to break wrong-but-internally-consistent outputs
    that a deterministic model will reproduce identically on each retry.

    Capped at 0.3: higher values destabilise small-model (Qwen3:14b) JSON
    generation without meaningfully improving semantic correctness.
    """
    return 0.0 if attempt == 0 else 0.3


# ── One-shot ───────────────────────────────────────────────────────────────────

def chat(prompt: str, system: str = "", model: str = DEFAULT_MODEL,
         max_tokens: int = 8192, temperature: float | None = None,
         thinking: bool = False) -> str:
    """Send a single prompt and return the text response.

    temperature=0 gives deterministic outputs (recommended for generation agents).
    Leave as None to use the provider default (usually ~1.0).
    thinking=True prepends /think token for Ollama models that support it.
    """
    global _call_count
    _call_count += 1
    provider = _provider(model)
    if provider == "google":
        return _google_chat(prompt, system, model, max_tokens, temperature)
    if provider == "anthropic":
        return _anthropic_chat(prompt, system, model, max_tokens, temperature)
    if provider == "groq":
        return _groq_chat(prompt, system, model, max_tokens, temperature)
    if provider == "ollama":
        return _ollama_chat(prompt, system, model, max_tokens, temperature, thinking)
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


def _groq_chat(prompt: str, system: str, model: str, max_tokens: int,
               temperature: float | None) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=_key("GROQ_API_KEY"), base_url=_GROQ_BASE_URL)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    kwargs: dict = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    raw = _with_retry(lambda: client.chat.completions.create(
        model=model, max_tokens=max_tokens,
        messages=messages, **kwargs).choices[0].message.content)
    return _strip_thinking(raw)


_OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "300"))


def _ollama_chat(prompt: str, system: str, model: str, max_tokens: int,
                 temperature: float | None, thinking: bool = False) -> str:
    from openai import OpenAI
    api_key = os.environ.get("OLLAMA_API_KEY", "ollama")
    client = OpenAI(api_key=api_key, base_url=_OLLAMA_BASE_URL, timeout=_OLLAMA_TIMEOUT)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    think_token = "/think" if thinking else "/no_think"
    messages.append({"role": "user", "content": f"{think_token}\n{prompt}"})
    kwargs: dict = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    raw = _with_retry(lambda: client.chat.completions.create(
        model=model, max_tokens=max_tokens,
        messages=messages,
        extra_body={"options": {"num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "16384"))}},
        **kwargs).choices[0].message.content,
        max_retries=3, base_delay=1.0)
    return _strip_thinking(raw)


# ── Multi-turn ─────────────────────────────────────────────────────────────────

class LLMClient:
    """Stateful multi-turn chat session, provider-agnostic."""

    def __init__(self, system: str = "", model: str = DEFAULT_MODEL,
                 max_tokens: int = 8192, temperature: float | None = None,
                 thinking: bool = False):
        self._system      = system
        self._model       = model
        self._max_tokens  = max_tokens
        self._temperature = temperature
        self._thinking    = thinking
        self._provider    = _provider(model)
        self._history: list[dict] = []

    def send(self, message: str) -> str:
        self._history.append({"role": "user", "content": message})
        if self._provider == "google":
            reply = self._google_send()
        elif self._provider == "anthropic":
            reply = self._anthropic_send()
        elif self._provider == "groq":
            reply = self._groq_send()
        elif self._provider == "ollama":
            reply = self._ollama_send(self._thinking)
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

    def _groq_send(self) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=_key("GROQ_API_KEY"), base_url=_GROQ_BASE_URL)
        messages = []
        if self._system:
            messages.append({"role": "system", "content": self._system})
        messages.extend(self._history)
        kwargs: dict = {}
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        raw = _with_retry(lambda: client.chat.completions.create(
            model=self._model, max_tokens=self._max_tokens,
            messages=messages, **kwargs).choices[0].message.content)
        return _strip_thinking(raw)

    def _ollama_send(self, thinking: bool = False) -> str:
        from openai import OpenAI
        api_key = os.environ.get("OLLAMA_API_KEY", "ollama")
        client = OpenAI(api_key=api_key, base_url=_OLLAMA_BASE_URL,
                        timeout=_OLLAMA_TIMEOUT)
        messages = []
        if self._system:
            messages.append({"role": "system", "content": self._system})
        # Inject /think or /no_think on the first user turn only
        history = list(self._history)
        if history and history[0]["role"] == "user":
            think_token = "/think" if thinking else "/no_think"
            history[0] = dict(history[0],
                              content=f"{think_token}\n{history[0]['content']}")
        messages.extend(history)
        kwargs: dict = {}
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        raw = _with_retry(lambda: client.chat.completions.create(
            model=self._model, max_tokens=self._max_tokens,
            messages=messages,
            extra_body={"options": {"num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "16384"))}},
            **kwargs).choices[0].message.content,
            max_retries=3, base_delay=1.0)
        return _strip_thinking(raw)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _key(env_var: str) -> str:
    key = os.environ.get(env_var)
    if not key:
        raise EnvironmentError(f"{env_var} environment variable is not set.")
    return key

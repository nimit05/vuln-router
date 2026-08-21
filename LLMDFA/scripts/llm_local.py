"""Drop-in replacement for LLMDFA's src/utility/llm.py.

Deviations from the published code, all deliberate and disclosed:

1. Talks to a local OpenAI-compatible endpoint (vLLM) instead of the OpenAI /
   Gemini / Anthropic SDKs. The upstream infer() dispatches on substring
   ("gpt" in name, "gemini" in name, ...); an open-weight model name matches
   none of those and infer() silently returns "".
2. Token counts come from the server's usage field, not tiktoken with the
   gpt-3.5 encoder. The upstream number is a wrong-tokenizer estimate, which
   would corrupt the cost axis this reproduction is being run for.
3. Retries are bounded and failures are counted, not silent. Every call site
   upstream loops `while True` on an unparseable answer.
4. <think>...</think> blocks are stripped, so reasoning models do not have
   their answer swallowed by the parser (the QASecClaw failure mode).

Environment:
  LLMDFA_BASE_URL    default http://127.0.0.1:8000/v1
  LLMDFA_MAX_RETRY   default 5
  LLMDFA_TIMEOUT     default 600 (seconds, per request)
  LLMDFA_MAX_TOKENS  default 2048
"""

import os
import re
import time
from typing import Tuple

from openai import *  # noqa: F401,F403  - upstream module star-exports these
from openai import OpenAI

BASE_URL = os.environ.get("LLMDFA_BASE_URL", "http://127.0.0.1:8000/v1")
MAX_RETRY = int(os.environ.get("LLMDFA_MAX_RETRY", "5"))
TIMEOUT = float(os.environ.get("LLMDFA_TIMEOUT", "600"))
MAX_TOKENS = int(os.environ.get("LLMDFA_MAX_TOKENS", "2048"))

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class LLM:
    """Same constructor and infer() contract as the upstream LLM class."""

    # Class-level so the whole run reports one set of counters.
    total_calls = 0
    empty_responses = 0
    failed_calls = 0

    def __init__(
        self,
        online_model_name: str,
        openai_key: str = "EMPTY",
        temperature: float = 0.0,
        system_role: str = "",
    ) -> None:
        self.online_model_name = online_model_name
        self.openai_key = openai_key or "EMPTY"
        self.temperature = temperature
        self.systemRole = system_role
        self.client = OpenAI(
            api_key=self.openai_key, base_url=BASE_URL, timeout=TIMEOUT
        )

    def infer(self, message: str, is_measure_cost: bool = True) -> Tuple[str, int, int]:
        messages = []
        if self.systemRole:
            messages.append({"role": "system", "content": self.systemRole})
        messages.append({"role": "user", "content": message})

        last_err = None
        for attempt in range(MAX_RETRY):
            try:
                resp = self.client.chat.completions.create(
                    model=self.online_model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=MAX_TOKENS,
                )
                raw = resp.choices[0].message.content or ""
                output = _THINK.sub("", raw).strip()

                usage = getattr(resp, "usage", None)
                in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
                out_tok = getattr(usage, "completion_tokens", 0) if usage else 0

                LLM.total_calls += 1
                if not output:
                    LLM.empty_responses += 1
                    print(
                        f"[LLMDFA] empty response (call {LLM.total_calls}, "
                        f"raw len {len(raw)})",
                        flush=True,
                    )
                return output, in_tok, out_tok
            except Exception as e:  # noqa: BLE001 - upstream also catches broadly
                last_err = e
                time.sleep(min(2 ** attempt, 30))

        LLM.total_calls += 1
        LLM.failed_calls += 1
        print(
            f"[LLMDFA] inference FAILED after {MAX_RETRY} attempts "
            f"({LLM.failed_calls} total): {last_err}",
            flush=True,
        )
        return "", 0, 0

    @staticmethod
    def sanity_report() -> str:
        return (
            f"calls={LLM.total_calls} empty={LLM.empty_responses} "
            f"failed={LLM.failed_calls}"
        )

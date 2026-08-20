import os
import ollama
from tqdm.contrib.concurrent import thread_map

from src.models.llm import LLM
from src.utils.mylogger import MyLogger

_model_name_map = {
    "ollama-qwen-coder": "qwen2.5-coder:latest",
    "ollama-qwen": "qwen2.5:32b",
    "ollama-llama3": "llama3.2:latest",
    "ollama-deepseek-32b": "deepseek-r1:32b",
    "ollama-deepseek-7b": "deepseek-r1:latest",
    # --- IRIS paper Table 1 open-weight models (added for reproduction) ---
    "ollama-llama3-8b": "llama3:latest",
    "ollama-llama3-70b": "llama3:70b",
    "ollama-qwen25-coder-32b": "qwen2.5-coder:32b",
    "ollama-gemma2-27b": "gemma2:27b",
    "ollama-deepseekcoder-7b": "deepseek-coder:6.7b",
    # Stronger open models already resident on this box (no extra disk needed).
    # Used to test whether sparse taint specs are a quantization/capacity effect.
    "ollama-llama33-70b": "llama3.3:latest",
    "ollama-qwen3-coder-30b": "qwen3-coder:30b",
    "ollama-gptoss-20b": "gpt-oss:20b",
    "ollama-gemma3-27b": "gemma3:27b",
}

# default model parameters, add or modify according to your needs
# see https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
_OLLAMA_DEFAULT_OPTIONS = {
    "temperature": 0.0,
    # -1 (unlimited) let a schema-constrained model run away generating a huge
    # "explanation" string: observed a single path stalling for 18+ min with the
    # GPU at 67% and ollama at 99% CPU. Cap it; verdict JSON is short.
    "num_predict": int(os.environ.get("IRIS_NUM_PREDICT", "1024")),
    "stop": None,
    "seed": 0,
    # IRIS batches 30 APIs (or 20 func params) per prompt, which is long. Ollama's
    # default context would silently TRUNCATE these prompts, so the model never sees
    # most of the candidate list -> very few labelled specs. Pin it explicitly.
    # Also keeps the KV cache small: llama3.3:70b defaulted to a 131072 context,
    # which pushed a 95.9GiB model past the 80GB A100 and forced CPU offload
    # (observed: ollama runner at 10135% CPU, GPU idle, 0 completed calls in 31 min).
    "num_ctx": int(os.environ.get("IRIS_NUM_CTX", "16384")),
}



# --- Reproduction aid (added) -------------------------------------------------
# IRIS asks for a JSON *array*; its parser does re.findall(r"\[[\s\S]*\]")[0].
# Heavily-quantized local models emit prose or a bare JSON object instead, so
# every spec is silently dropped. Ollama structured outputs let us constrain
# decoding to the exact array-of-objects shape each stage expects.
_API_LABEL_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "package": {"type": "string"},
            "class": {"type": "string"},
            "method": {"type": "string"},
            "signature": {"type": "string"},
            "sink_args": {"type": "array", "items": {"type": "string"}},
            "type": {"type": "string", "enum": ["source", "sink", "taint-propagator", "none"]},
        },
        # sink_args MUST be required: neusym_vul.py:837 only emits a real sink
        # clause when "sink_args" in api, otherwise the predicate degrades to
        # "1 = 0" (CodeQL false) and NO path can ever be found.
        "required": ["package", "class", "method", "signature", "sink_args", "type"],
    },
}

_FUNC_PARAM_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "package": {"type": "string"},
            "class": {"type": "string"},
            "method": {"type": "string"},
            "signature": {"type": "string"},
            "tainted_input": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["package", "class", "method", "signature", "tainted_input"],
    },
}


# Posthoc contextual filtering expects a JSON OBJECT; its parser does
# json.loads(re.findall(r"\{[\s\S]*\}", s)[0]). Quantized models answered in prose,
# so verdicts were unparseable. Worse, a spurious source_is_false_positive=true
# CASCADES: IRIS then auto-suppresses every other path through that source.
# Property order matters: explanation first, so reasoning precedes the verdict.
_POSTHOC_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "source_is_false_positive": {"type": "boolean"},
        "sink_is_false_positive": {"type": "boolean"},
        "is_vulnerable": {"type": "boolean"},
    },
    "required": ["explanation", "source_is_false_positive",
                 "sink_is_false_positive", "is_vulnerable"],
}


def _schema_for(system_prompt: str):
    """Pick the output schema from the stage's system prompt; None = unconstrained."""
    if "taint sources, sinks" in system_prompt:
        return _API_LABEL_SCHEMA
    if "invoked by downstream libraries" in system_prompt:
        return _FUNC_PARAM_SCHEMA
    if "is_vulnerable" in system_prompt:
        return _POSTHOC_SCHEMA
    return None
# -----------------------------------------------------------------------------

# --- User-editable model registry -------------------------------------------
# Anything in IRIS/models.json is merged in here, so you can add models WITHOUT
# editing Python. Format:  "ollama-yourname": "ollama-tag"
try:
    import json as _json, os as _os
    _reg = _os.path.join(_os.path.dirname(__file__), "..", "..", "models.json")
    if _os.path.exists(_reg):
        for _k, _v in _json.load(open(_reg)).items():
            if not _k.startswith("_"):
                _model_name_map[_k.lower()] = _v
except Exception as _e:
    print("[ollama] could not load models.json:", _e)
# -----------------------------------------------------------------------------

class OllamaModel(LLM):
    def __init__(self, model_name, logger: MyLogger, **kwargs):
        super().__init__(model_name, logger, _model_name_map, **kwargs)
        if host := os.environ.get("OLLAMA_HOST"):
            self.client = ollama.Client(host=host)
        else:
            self.log.error("Please set OLLAMA_HOST environment variable")
        # TODO: https://github.com/ollama/ollama/issues/2415
        # self.logprobs = None
        for k in _OLLAMA_DEFAULT_OPTIONS:
            if k in kwargs:
                _OLLAMA_DEFAULT_OPTIONS[k] = kwargs[k]

    def predict(self, prompt, batch_size=0, no_progress_bar=False):
        if batch_size == 0:
            return self._predict(prompt)
        args = range(0, len(prompt))
        responses = thread_map(
            lambda x: self._predict(prompt[x]),
            args,
            max_workers=batch_size,
            disable=no_progress_bar,
        )
        return responses

    def _predict(self, main_prompt):
        # assuming 0 is system and 1 is user
        system_prompt = main_prompt[0]["content"]
        user_prompt = main_prompt[1]["content"]
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            chat_kwargs = {
                "model": self.model_id,
                "messages": prompt,
                "options": _OLLAMA_DEFAULT_OPTIONS,
            }
            schema = _schema_for(system_prompt)
            if schema is not None:
                chat_kwargs["format"] = schema
            response = self.client.chat(**chat_kwargs)
        except ollama.ResponseError as e:
            print("Ollama Response Error:", e.error)
            return None

        return response.message.content

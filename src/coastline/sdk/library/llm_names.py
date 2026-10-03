"""Resolve an LLM name to the key of its entry in Kavier's model catalog.

WorkloadSpec lowercases llm_model and drops the org prefix, but Kavier's catalog has mixed-case
keys ('Llama-3-8B') and keys that are not the HuggingFace short name ('llama3.2-3b' for
meta-llama/Llama-3.2-3B). Kavier and the ML spec features both look models up through here.
"""

from typing import Any, Optional

try:
    from kavier.sdk.library import LLM_SPEC_LIBRARY as _KAVIER_LLMS
except ImportError:  # Kavier is a core dependency; the predictors report its absence themselves
    _KAVIER_LLMS = {}

# HuggingFace ids (org dropped, lowercased) of catalog models whose catalog key differs. Each
# maps to the model name ADO's sfttrainer uses for that checkpoint (its config/models.yaml), or,
# for the Meta-Llama spellings, to the catalog entry with the same architecture.
LLM_ALIASES: dict[str, str] = {
    "llama-3.2-1b": "llama3.2-1b",
    "llama-3.2-3b": "llama3.2-3b",
    "llama-3.1-8b": "llama3.1-8b",
    "meta-llama-3.1-8b": "llama3.1-8b",
    "llama-3.1-70b": "llama3.1-70b",
    "meta-llama-3.1-70b": "llama3.1-70b",
    "meta-llama-3-8b": "llama3-8b",
    "meta-llama-3-70b": "llama3-70b",
    "llama-2-13b-hf": "Llama-2-13B",
    "llama-2-70b-hf": "llama2-70b",
    "granite-3.0-8b-base": "granite-3-8b",
    "granite-3.1-2b-base": "granite-3.1-2b",
    "granite-3.3-8b-base": "granite-3.3-8b",
}

_BY_LOWERCASE = {key.lower(): key for key in _KAVIER_LLMS}


def kavier_llm_name(model_name: Any) -> Optional[str]:
    """The catalog key for a model name, or None when Kavier has no entry for it.

    Tries the name as given, then without org prefix and case, then the alias table. A name
    that is already a catalog key is returned unchanged.
    """
    name = str(model_name)
    if name in _KAVIER_LLMS:
        return name
    short = name.split("/")[-1].lower()  # the form WorkloadSpec stores
    key = _BY_LOWERCASE.get(short) or LLM_ALIASES.get(short)
    return key if key in _KAVIER_LLMS else None

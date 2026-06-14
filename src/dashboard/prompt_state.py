from typing import Any, Dict, Optional

RETRIEVAL_CITATION_SUFFIX = (
    "If you use retrieved context or reference excerpts in your answer, cite the "
    "corresponding Source label inline; if the excerpts do not support the answer, "
    "be explicit."
)
LOCKED_SYSTEM_PROMPT_MESSAGE = "No system prompt was sent on turn 1."


def normalize_system_prompt(prompt: Optional[str]) -> str:
    """Normalize prompt text for storage and first-turn sending."""
    return str(prompt or "").strip()


def append_managed_retrieval_suffix(prompt: Optional[str]) -> str:
    """Append the managed Retrieval citation suffix exactly once."""
    normalized_prompt = normalize_system_prompt(prompt)
    if RETRIEVAL_CITATION_SUFFIX in normalized_prompt:
        return normalized_prompt
    if not normalized_prompt:
        return RETRIEVAL_CITATION_SUFFIX
    return f"{normalized_prompt}\n\n{RETRIEVAL_CITATION_SUFFIX}"


def extract_first_system_prompt(messages: Any) -> Optional[str]:
    """Return the first persisted system prompt from conversation history."""
    if not isinstance(messages, list):
        return None

    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "system":
            continue
        content = message.get("content")
        if isinstance(content, str):
            normalized = content.strip()
            return normalized or None

    return None


def build_system_prompt_state(
    prompt: Optional[str] = None,
    *,
    started: bool = False,
    locked: bool = False,
    committed_prompt: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    """Build consistent dashboard prompt state for one conversation."""
    normalized_prompt = normalize_system_prompt(prompt)
    default_committed_prompt = (
        normalized_prompt if committed_prompt is None and started else committed_prompt
    )
    return {
        "prompt": normalized_prompt,
        "committed_prompt": normalize_system_prompt(default_committed_prompt),
        "reasoning_effort": str(reasoning_effort or "").strip(),
        "started": bool(started),
        "locked": bool(locked),
    }


def first_turn_system_prompt_to_send(
    prompt: Optional[str], started: bool
) -> Optional[str]:
    """Return the prompt only for the first turn of a conversation."""
    if started:
        return None
    normalized_prompt = normalize_system_prompt(prompt)
    return normalized_prompt or None

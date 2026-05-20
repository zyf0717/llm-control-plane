from src.dashboard.prompt_state import (
    RAG_CITATION_SUFFIX,
    append_managed_rag_suffix,
    extract_first_system_prompt,
    first_turn_system_prompt_to_send,
)


def test_append_managed_rag_suffix_appends_once():
    prompt = append_managed_rag_suffix("Be concise.")

    assert prompt == f"Be concise.\n\n{RAG_CITATION_SUFFIX}"


def test_append_managed_rag_suffix_is_idempotent():
    prompt = append_managed_rag_suffix("Be concise.")
    prompt = append_managed_rag_suffix(prompt)

    assert prompt.count(RAG_CITATION_SUFFIX) == 1


def test_append_managed_rag_suffix_uses_suffix_as_prompt_when_empty():
    assert append_managed_rag_suffix("") == RAG_CITATION_SUFFIX


def test_extract_first_system_prompt_returns_first_system_message():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "system", "content": "System one"},
        {"role": "system", "content": "System two"},
    ]

    assert extract_first_system_prompt(messages) == "System one"


def test_extract_first_system_prompt_returns_none_without_system_message():
    assert extract_first_system_prompt([{"role": "user", "content": "Hello"}]) is None


def test_first_turn_system_prompt_to_send_only_sends_before_start():
    assert first_turn_system_prompt_to_send(" Be concise. ", started=False) == "Be concise."
    assert first_turn_system_prompt_to_send("Be concise.", started=True) is None
    assert first_turn_system_prompt_to_send("", started=False) is None

from src.dashboard.chat_client import build_chat_messages


def test_build_chat_messages_with_user_only():
    messages = build_chat_messages(text="Hello")

    assert messages == [{"role": "user", "content": "Hello"}]


def test_build_chat_messages_includes_turn_local_search_context_after_system_prompt():
    messages = build_chat_messages(
        text="Need sources",
        system_prompt="Be concise.",
        extra_turn_messages=[
            {"role": "system", "content": '{"source":"web_search"}'},
            {"role": "assistant", "content": "ignored but allowed"},
            {"role": "", "content": "skip"},
        ],
    )

    assert messages == [
        {"role": "system", "content": "Be concise."},
        {"role": "system", "content": '{"source":"web_search"}'},
        {"role": "assistant", "content": "ignored but allowed"},
        {"role": "user", "content": "Need sources"},
    ]

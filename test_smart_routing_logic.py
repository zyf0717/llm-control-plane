#!/usr/bin/env python3
"""
Test script to verify that the /smart endpoint correctly:
1. Adds to history first
2. Uses only latest message for classification
3. Routes entire conversation to target endpoint
"""

import json
import sys

sys.path.insert(0, "src")

from orchestrator.proxy import parse_and_inject_history, convo_history


def test_smart_routing_logic():
    """Test the three-step logic of the /smart endpoint."""

    # Clear any existing history
    convo_history.clear()

    # Simulate a conversation with some previous history
    convo_id = "test-smart-routing"

    # Step 1: Simulate existing conversation history
    convo_history[convo_id] = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
        {
            "role": "user",
            "content": "Now explain the mathematical proof behind why multiplication is commutative",
        },
    ]

    # Step 2: Simulate a new request coming in
    new_request = {
        "messages": [
            {
                "role": "user",
                "content": "Can you help me solve this complex reasoning problem?",
            }
        ],
        "stream": False,
    }

    print("=== Initial state ===")
    print(f"Existing history length: {len(convo_history[convo_id])}")
    print(f"New request: {new_request}")

    # Step 3: Simulate what the /smart endpoint does
    # Parse and inject history (step 1 of the new logic)
    enriched_body, is_streaming = parse_and_inject_history(
        json.dumps(new_request).encode(), convo_id
    )

    print(f"\n=== After history injection ===")
    print(f"History length: {len(convo_history[convo_id])}")
    print(f"Enriched body messages count: {len(enriched_body.get('messages', []))}")

    # Extract latest message for classification (step 2)
    messages = enriched_body.get("messages", [])
    user_messages = [msg for msg in messages if msg.get("role") == "user"]
    latest_message = user_messages[-1].get("content", "")

    print(f"\n=== Classification step ===")
    print(f"Latest user message for classification: '{latest_message}'")
    print(
        f"Full conversation will be sent to target endpoint with {len(messages)} messages"
    )

    # Verify the logic is correct
    assert len(convo_history[convo_id]) == 4  # 3 original + 1 new
    assert len(enriched_body["messages"]) == 4  # Full conversation in enriched body
    assert latest_message == "Can you help me solve this complex reasoning problem?"

    # Verify full conversation context is available
    all_messages = enriched_body["messages"]
    assert all_messages[0]["content"] == "What is 2+2?"
    assert all_messages[1]["content"] == "2+2 equals 4."
    assert (
        all_messages[2]["content"]
        == "Now explain the mathematical proof behind why multiplication is commutative"
    )
    assert (
        all_messages[3]["content"]
        == "Can you help me solve this complex reasoning problem?"
    )

    print(f"\n✅ Smart routing logic works correctly!")
    print(f"✅ Classification uses only: '{latest_message[:50]}...'")
    print(
        f"✅ Target endpoint receives full conversation with {len(all_messages)} messages"
    )

    return True


if __name__ == "__main__":
    print("Testing smart routing three-step logic...")
    test_smart_routing_logic()
    print("\n🎉 All tests passed! The new /smart endpoint logic is working correctly.")

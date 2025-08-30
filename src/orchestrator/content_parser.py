"""
Content parsing utilities for LLM responses.

This module handles parsing and formatting of special reasoning tags
and channel tags from LLM responses.
"""

import json
import logging
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)


class ContentParser:
    """Handles parsing and formatting of LLM response content."""

    @staticmethod
    def parse_think_tags(content: str) -> str:
        """Parse <think> tags by converting them to emphasis formatting."""
        content = content.replace("<think>", "<em>")
        content = content.replace("</think>", "</em>\n\n---\n\n")
        return content

    @staticmethod
    def parse_channel_tags(content: str) -> str:
        """Parse <|channel|> tags by converting analysis channels to emphasis formatting."""
        content = content.replace("<|channel|>analysis<|message|>", "<em>")
        content = content.replace(
            "<|end|><|start|>assistant<|channel|>final<|message|>",
            "</em>\n\n---\n\n",
        )
        return content

    @staticmethod
    def parse_content_chunk(content: str) -> str:
        """Parse a content chunk by applying all parsing rules."""
        if not content:
            return content

        # Apply think tag parsing
        content = ContentParser.parse_think_tags(content)

        # Apply channel tag parsing
        content = ContentParser.parse_channel_tags(content)

        return content

    @staticmethod
    def format_reasoning_content(reasoning: str) -> str:
        """Format reasoning content with proper emphasis."""
        if not reasoning:
            return ""
        return f"<em>{reasoning}</em>\n\n---\n\n"


class StreamingContentParser:
    """Handles streaming content parsing with stateful buffering."""

    def __init__(self, output_reasoning: bool = False):
        self.output_reasoning = output_reasoning
        self.reasoning_chunk_found = False
        self.reasoning_chunk_buffer = ""

    def process_streaming_chunk(
        self, obj: Dict[str, Any]
    ) -> Generator[str, None, None]:
        """Process a streaming response chunk and yield formatted content."""
        choices = obj.get("choices", [])
        if not choices:
            return

        # Handle reasoning chunks (GPT-style)
        if self.output_reasoning:
            reasoning_chunk = choices[0].get("delta", {}).get("reasoning", "")
            if reasoning_chunk:
                if not self.reasoning_chunk_found:
                    self.reasoning_chunk_found = True
                    yield "<em>"
                yield reasoning_chunk

        # Handle content chunks
        content_chunk = choices[0].get("delta", {}).get("content", "")
        if content_chunk:
            if self.reasoning_chunk_found:
                yield "</em>\n\n---\n\n"
                self.reasoning_chunk_found = False

            # Apply content parsing
            content_chunk = self._process_streaming_content_chunk(content_chunk)

            # Only output content if buffer is empty
            if not self.reasoning_chunk_buffer:
                yield content_chunk

    def _process_streaming_content_chunk(self, content_chunk: str) -> str:
        """Process a streaming content chunk with stateful channel tag parsing."""
        # Apply think tag parsing immediately
        content_chunk = ContentParser.parse_think_tags(content_chunk)

        # Handle channel tag buffering for streaming
        if self.reasoning_chunk_buffer:
            self.reasoning_chunk_buffer += content_chunk

        # Check for start of analysis channel
        if self.reasoning_chunk_buffer.startswith("<|channel|>analysis<|message|>"):
            self.reasoning_chunk_buffer = ""
            return "<em>"

        # Check for end of analysis channel
        if self.reasoning_chunk_buffer.endswith(
            "<|end|><|start|>assistant<|channel|>final<|message|>"
        ):
            self.reasoning_chunk_buffer = ""
            return "</em>\n\n---\n\n"

        # Start buffer if content chunk is a channel message
        if (
            content_chunk in ["<|channel|>", "<|end|>"]
            and not self.reasoning_chunk_buffer
        ):
            self.reasoning_chunk_buffer += content_chunk
            return ""  # Don't output anything

        # Only return content if buffer is empty
        if not self.reasoning_chunk_buffer:
            return content_chunk

        return ""  # Buffer is not empty, don't output yet


def parse_non_streaming_content(
    response_data: Dict[str, Any],
    output_reasoning: bool = False,
    parse_content: bool = True,
) -> str:
    """Parse non-streaming response content."""
    choices = response_data.get("choices", [])
    if not choices:
        return ""

    if not parse_content:
        # Return raw content
        content = choices[0].get("message", {}).get("content", "")
        return str(content)

    result_parts = []

    # Handle reasoning (GPT-style)
    if output_reasoning:
        reasoning = choices[0].get("message", {}).get("reasoning", "")
        if reasoning:
            result_parts.append(ContentParser.format_reasoning_content(str(reasoning)))

    # Handle main content
    content = choices[0].get("message", {}).get("content", "")

    if content:
        parsed_content = ContentParser.parse_content_chunk(str(content))
        result_parts.append(parsed_content)

    return "".join(result_parts)

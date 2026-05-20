from typing import Any, Dict, List, Optional


class RagPromptBuilder:
    @staticmethod
    def _rag_result_label(result: Dict[str, Any], index: int) -> str:
        """Choose a stable label for a retrieved result."""
        for key in (
            "citation_label",
            "title",
            "id",
            "chunk_id",
            "document_id",
            "source",
        ):
            value = result.get(key)
            if value:
                return str(value).strip()
        return f"doc-{index}"

    @staticmethod
    def _extract_entity_texts(result: Dict[str, Any]) -> List[str]:
        """Extract readable entity labels from a retrieved result."""
        matched_entities = result.get("matched_entities")
        if matched_entities is None:
            return []

        if not isinstance(matched_entities, list):
            text = str(matched_entities).strip()
            return [text] if text else []

        entity_values: List[str] = []
        seen = set()
        for entity in matched_entities:
            if isinstance(entity, str):
                text = entity.strip()
            elif isinstance(entity, dict):
                text = str(
                    entity.get("text")
                    or entity.get("value")
                    or entity.get("name")
                    or entity.get("entity")
                    or ""
                ).strip()
            else:
                text = str(entity).strip()

            if not text or text in seen:
                continue
            seen.add(text)
            entity_values.append(text)

        return entity_values

    @classmethod
    def _build_rag_context(
        cls,
        results: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Build formatted retrieved reference excerpts."""
        context_blocks: List[str] = []

        for index, result in enumerate(results, start=1):
            label = cls._rag_result_label(result, index)
            content = str(result.get("content") or "").strip()
            entity_values = cls._extract_entity_texts(result)

            block_lines = [f"Source: {label}"]
            if entity_values:
                block_lines.append(f"Relevant entities: {', '.join(entity_values)}")
            if content:
                block_lines.append("Excerpt:")
                block_lines.append(content)
            if len(block_lines) == 1:
                continue

            context_blocks.append("\n".join(block_lines))

        if not context_blocks:
            return None
        return "\n\n".join(context_blocks)

    @classmethod
    def build_user_message_content(
        cls,
        user_query: str,
        results: List[Dict[str, Any]],
    ) -> str:
        """Build the latest-user-turn content for a retrieval-grounded request."""
        rag_context = cls._build_rag_context(results)
        if not rag_context:
            return user_query

        return (
            "Retrieved reference excerpts:\n\n"
            f"{rag_context}\n\n"
            "Current user question:\n"
            f"{user_query}"
        )

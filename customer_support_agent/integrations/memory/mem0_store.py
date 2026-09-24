from __future__ import annotations

from typing import Any

from customer_support_agent.core.settings import Settings

try:
    import mem0.configs.prompts
    import mem0.memory.main
    
    SHORT_ADDITIVE_EXTRACTION_PROMPT = """# ROLE
You are a Memory Extractor. Your task is to extract memories (facts, preferences, plans, recommendations, or key information) from the new conversation messages.

# OUTPUT FORMAT
Return a JSON object with a single "memory" key containing a list of objects.
Example:
{
  "memory": [
    {"id": "0", "text": "Customer is on the free plan with a 48-hour SLA.", "attributed_to": "user"},
    {"id": "1", "text": "ATM standard savings account withdrawal limit is INR 25,000 per day.", "attributed_to": "assistant"}
  ]
}

# RULES
1. id: Sequential integer as a string starting at "0".
2. text: A contextually rich, self-contained factual statement (10-50 words). Avoid pronouns (use "Customer" or "Support Agent").
3. attributed_to: "user" for facts about the user; "assistant" for recommendations/guidelines/plans.
4. linked_memory_ids: (optional) List of existing memory IDs that are updated/complemented.
5. If nothing worth extracting, return {"memory": []}.
6. Output JSON only. Do not add any extra text or wrappers.
"""
    mem0.configs.prompts.ADDITIVE_EXTRACTION_PROMPT = SHORT_ADDITIVE_EXTRACTION_PROMPT
    mem0.memory.main.ADDITIVE_EXTRACTION_PROMPT = SHORT_ADDITIVE_EXTRACTION_PROMPT
except Exception:
    pass

try:
    from mem0 import Memory
except ImportError:
    Memory = None

class CustomerMemoryStore:

    def __init__(self, settings:Settings, llm:Any):
        if Memory is None:
            raise RuntimeError("mem0ai is not installed. Install dependencies with `uv sync`.")
        _ = llm

        config: dict[str, Any] = {
            "llm": {
                "provider": "groq",
                "config": {
                    "model": settings.groq_model,
                    "api_key": settings.groq_api_key,
                    "temperature": settings.llm_temperature,
                },
            },
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "path": str(settings.chroma_mem0_path),
                },
            },
        }

        if settings.google_api_key:
            config["embedder"] = {
                "provider": "gemini",
                "config": {
                    "api_key": settings.google_api_key,
                    "model": settings.effective_google_embedding_model,
                },
            }

        elif settings.openai_api_key:
            config["embedder"] = {
                "provider": "openai",
                "config": {
                    "api_key": settings.openai_api_key,
                },
            }

        elif settings.enable_local_embeddings:
            config["embedder"] = {
                "provider": "huggingface",
                "config": {
                    "model": "all-MiniLM-L6-v2",
                },
            }

        else:
            raise RuntimeError(
                "No embedding provider configured for Mem0. Set GOOGLE_API_KEY (recommended) "
                "or OPENAI_API_KEY. Set ENABLE_LOCAL_EMBEDDINGS=true to use local HuggingFace embeddings."
            )

        self._memory = Memory.from_config(config)

    def search(self, query: str, user_id: str, limit: int = 5) -> list[dict[str, Any]]:
        try:
            raw = self._memory.search(query, filters={"user_id": user_id}, limit=limit)
        except Exception:
            try:
                raw = self._memory.search(query, user_id=user_id, limit=limit)
            except TypeError:
                raw = self._memory.search(query, user_id=user_id)
        return self._normalize_results(raw, limit)

    def list_memories(self, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        if hasattr(self._memory, "get_all"):
            try:
                raw = self._memory.get_all(filters={"user_id": user_id})
            except Exception:
                raw = self._memory.get_all(user_id=user_id)
            return self._normalize_results(raw, limit)

    def add_interaction(
        self,
        user_id: str,
        user_input: str,
        assistant_response: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        messages = [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_response},
        ]

        self._add_messages(messages=messages, user_id=user_id, metadata=metadata)

    def add_resolution(
        self,
        user_id: str,
        ticket_subject: str,
        ticket_description: str,
        accepted_draft: str,
        entity_links: list[str] | None = None,
    ) -> None:
        entity_text = ""
        if entity_links:
            entity_text = "\nLinked entities: " + ", ".join(entity_links)

        messages = [
            {
                "role": "user",
                "content": f"Ticket subject: {ticket_subject}\nProblem: {ticket_description}",
            },
            {
                "role": "assistant",
                "content": (
                    "Resolution accepted by support agent:\n"
                    f"{accepted_draft}{entity_text}"
                ),
            },
        ]

        metadata = {"type": "resolution"}

        self._add_messages(messages=messages, user_id=user_id, metadata=metadata)

    def _add_messages(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        try:
            self._memory.add(messages, user_id=user_id, metadata=metadata or {})
        except TypeError:
            self._memory.add(messages, user_id=user_id)


    def _normalize_results(self, raw: Any, limit: int) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []

        if isinstance(raw, dict) and "results" in raw:
            iterable = raw.get("results") or []
        elif isinstance(raw, list):
            iterable = raw
        else:
            iterable = []

        for entry in iterable[:limit]:
            if isinstance(entry, dict):
                memory_text = entry.get("memory") or entry.get("content") or ""
                if memory_text:
                    items.append(
                        {
                            "memory": memory_text,
                            "score": entry.get("score"),
                            "metadata": entry.get("metadata") or {},
                        }
                    )
            elif entry:
                items.append({"memory": str(entry), "score": None, "metadata": {}})

        return items

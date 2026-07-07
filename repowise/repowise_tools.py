import json
from repowise.safe_mcp_call import safe_mcp_call
class RepoWiseTools:
    def __init__(self, repowise_client):
        self.client = repowise_client

    @safe_mcp_call
    async def repository_overview(self):
        """
        Returns a concise architecture overview.
        """
        result = await self.client.call_mcp_tool("get_overview", {})


        data = result["result"]
        content = data["content"]
        isError = data["isError"]

        text = content[0]["text"]
        text = json.loads(text)

        return {
            "title": text.get("title"),
            "summary": text.get("content_md"),
            "entry_points": text.get("entry_points", []),
            "layers": text.get("architecture", {}).get("layers", []),
            "key_modules": text.get("key_modules", []),
            "reading_order": text.get("reading_order", []),
            "guided_tour": text.get("guided_tour", []),
            "index_age_days": text.get("_meta", {}).get("index_age_days"),
        }

    @safe_mcp_call
    async def search(self, query: str, limit: int = 5):
        """
        Search the repository.
        """
        result = await self.client.call_mcp_tool(
            "search_codebase",
            {
                "query": query,
                "limit": limit,
            },
        )

        hits = []

        for hit in result["result"].get("results", []):
            hits.append({
                "symbol_id": hit.get("symbol_id"),
                "file": hit.get("file"),
                "summary": hit.get("summary"),
                "kind": hit.get("kind"),
            })

        return hits

    @safe_mcp_call
    async def explain(self, question: str):
        """
        Answer repository questions. 

        Args:
            question: Question to be answered.
        """
        result = await self.client.call_mcp_tool(
            "get_answer",
            {
                "question": question,
            },
        )

        
        data = result["result"]
        content = data["content"]
        isError = data["isError"]

        text = content[0]["text"]
        text = json.loads(text)


        return {
            "answer": text.get("answer"),
            "confidence": text.get("confidence"),
            "retrival_quality": text.get("retrival_quality"),
            "best_guesses": text.get("best_guesses"),
            "citations": text.get("citations", []),
            "next_action_hint": text.get("next_action_hint"),
            "fallback_targets": text.get("fallback_targets"),
            "retrieval": text.get("retrieval"),
        }

    @safe_mcp_call
    async def context(
        self,
        targets: list[str],
        include: list[str] | None = None,
        compact: bool = False,
    ):
        """
        Get context for files, modules, or symbols.

        Args:
            targets: File paths, module paths, or symbol IDs.
            include: Optional sections to include (e.g. callers, callees,
                decisions, metrics, ownership, community, skeleton,
                full_doc, last_change).
            compact: Whether to return the compact view.
        """

        payload = {
            "targets": targets,
            "compact": compact,
        }

        if include:
            payload["include"] = include

        result = await self.client.call_mcp_tool(
            "get_context",
            payload,
        )

        data = result["result"]
        content = data["content"]
        isError = data["isError"]

        text = content[0]["text"]
        text = json.loads(text)

        contexts = []


        for key,target in text.get("targets", []).items():
            item = target
            if isinstance(item,str):
                item = json.loads(item)
            docs = item.get("docs")
            if isinstance(docs,str):
                docs = json.loads(docs)

            context = {
                "target": item.get("target"),
                "summary": docs.get("summary"),
                "signatures": docs.get("symbols"),
                "symbols": docs.get("symbols"),
                "structure": docs.get("structure"),
                "hotspot": item.get("hotspot"),
                "freshness": item.get("freshness"),
                "truncated": item.get("truncated"),
                "dropped_targets": item.get("dropped_targets"),
                "dropped_symbols": item.get("dropped_symbols"),
            }

            optional_fields = [
                "callers",
                "callees",
                "decision_records",
                "ownership",
                "last_change",
                "metrics",
                "community",
                "structure",
                "imports",
                "docstrings",
                "hotspot",
                "verified",
                "mostly_full",
            ]

            for field in optional_fields:
                if field in item:
                    context[field] = item[field]

            contexts.append(context)

        return contexts
    @safe_mcp_call
    async def symbol(self, symbol_id: str):
        """
        Read implementation of one symbol.
        """
        result = await self.client.call_mcp_tool(
            "get_symbol",
            {
                "symbol_id": symbol_id,
            },
        )

        symbol = result["result"]

        return {
            "name": symbol.get("name"),
            "verified": symbol.get("verified"),
            "signature": symbol.get("signature"),
            "source": symbol.get("source"),
            "file": symbol.get("file"),
            "line_start": symbol.get("line_start"),
            "line_end": symbol.get("line_end"),
        }

    @safe_mcp_call
    async def why(self, query: str = None, targets=None):
        """
        Explain architectural decisions.
        """
        result = await self.client.call_mcp_tool(
            "get_why",
            {
                "query": query,
                "targets": targets,
            },
        )

        data = result["result"]

        return {
            "summary": data.get("summary"),
            "decisions": data.get("decisions", []),
            "evidence": data.get("evidence", []),
        }
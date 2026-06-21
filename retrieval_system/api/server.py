from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from retrieval_system.search.pipeline import RetrievalPipeline
from retrieval_system.settings import load_config
from retrieval_system.vectordb.store import SearchResult


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SFT-BE Retrieval</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #111; }
    main { max-width: 920px; margin: 0 auto; }
    textarea { width: 100%; min-height: 90px; font: inherit; padding: 10px; box-sizing: border-box; }
    button { margin-top: 10px; padding: 10px 14px; font: inherit; cursor: pointer; }
    label { display: inline-flex; gap: 6px; align-items: center; margin-right: 16px; }
    pre { white-space: pre-wrap; background: #f5f5f5; padding: 12px; overflow: auto; }
    .result { border-top: 1px solid #ddd; padding: 12px 0; }
    .citation { color: #555; font-size: 14px; }
  </style>
</head>
<body>
<main>
  <h1>SFT-BE Retrieval</h1>
  <textarea id="query" placeholder="Nhập câu hỏi..."></textarea>
  <div>
    <label><input id="use_llm" type="checkbox"> LLM verify</label>
    <label><input id="answer" type="checkbox"> Answer</label>
    <label><input id="context" type="checkbox"> Context</label>
  </div>
  <button onclick="runQuery()">Search</button>
  <pre id="answerBox"></pre>
  <div id="results"></div>
</main>
<script>
async function runQuery() {
  const query = document.getElementById("query").value;
  const body = {
    query,
    use_llm: document.getElementById("use_llm").checked,
    answer: document.getElementById("answer").checked,
    show_context: document.getElementById("context").checked
  };
  document.getElementById("answerBox").textContent = "Running...";
  document.getElementById("results").innerHTML = "";
  const response = await fetch("/query", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body)
  });
  const data = await response.json();
  document.getElementById("answerBox").textContent = data.error || data.answer || "";
  const results = data.results || [];
  document.getElementById("results").innerHTML = results.map((item) => `
    <div class="result">
      <div><b>#${item.rank}</b> score=${item.score.toFixed(4)} vector=${item.vector_score.toFixed(4)} cross=${item.cross_score?.toFixed(4) ?? "n/a"}</div>
      <div class="citation">${escapeHtml(item.citation)}</div>
      ${item.reason ? `<div>${escapeHtml(item.reason)}</div>` : ""}
      <pre>${escapeHtml(item.context || item.text)}</pre>
    </div>
  `).join("");
}
function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[ch]));
}
</script>
</body>
</html>
"""


class RetrievalHTTPServer(ThreadingHTTPServer):
    pipeline: RetrievalPipeline
    defaults: dict[str, int]


class RetrievalHandler(BaseHTTPRequestHandler):
    server: RetrievalHTTPServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(INDEX_HTML)
            return
        if parsed.path == "/health":
            self._send_json({"status": "ok"})
            return
        if parsed.path == "/search":
            params = parse_qs(parsed.query)
            query = params.get("q", [""])[0].strip()
            payload = {
                "query": query,
                "use_llm": _parse_bool(params.get("use_llm", ["false"])[0]),
                "answer": _parse_bool(params.get("answer", ["false"])[0]),
                "show_context": _parse_bool(params.get("show_context", ["false"])[0]),
            }
            self._handle_query(payload)
            return
        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/query":
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return

        try:
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except (json.JSONDecodeError, ValueError) as exc:
            self._send_json({"error": f"invalid json: {exc}"}, status=HTTPStatus.BAD_REQUEST)
            return
        self._handle_query(payload)

    def _handle_query(self, payload: dict) -> None:
        query = str(payload.get("query", "")).strip()
        if not query:
            self._send_json({"error": "query is required"}, status=HTTPStatus.BAD_REQUEST)
            return

        defaults = self.server.defaults
        results = self.server.pipeline.retrieve(
            query,
            vector_top_k=int(payload.get("vector_top_k", defaults["vector_top_k"])),
            cross_rerank_k=int(payload.get("cross_rerank_k", defaults["cross_rerank_k"])),
            llm_rerank_k=int(payload.get("llm_rerank_k", defaults["llm_rerank_k"])),
            final_k=int(payload.get("final_k", defaults["final_context_k"])),
            use_llm=_parse_bool(payload.get("use_llm", False)),
        )

        answer = None
        if _parse_bool(payload.get("answer", False)):
            answer = self.server.pipeline.answer(query, results)

        show_context = _parse_bool(payload.get("show_context", False))
        self._send_json(
            {
                "query": query,
                "answer": answer,
                "results": [
                    _result_to_dict(self.server.pipeline, item, idx, show_context)
                    for idx, item in enumerate(results, start=1)
                ],
            }
        )

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, content: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}")


def _result_to_dict(
    pipeline: RetrievalPipeline,
    result: SearchResult,
    rank: int,
    show_context: bool,
) -> dict:
    score = result.final_score if result.final_score is not None else result.vector_score
    metadata = result.chunk.metadata or {}
    output = {
        "rank": rank,
        "score": score,
        "vector_score": result.vector_score,
        "lexical_score": result.lexical_score,
        "cross_score": result.cross_score,
        "llm_score": result.llm_score,
        "reason": result.reason,
        "citation": pipeline.citation_for_result(result),
        "paper": metadata.get("paper") or metadata.get("book"),
        "section": metadata.get("section"),
        "page": metadata.get("page"),
        "source": result.chunk.source,
        "chunk_index": result.chunk.chunk_index,
        "text": result.chunk.text,
    }
    if show_context:
        output["context"] = pipeline.context_for_result(result)
    return output


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description="Serve SFT-BE retrieval over HTTP")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--index-dir", default=config["index_dir"])
    parser.add_argument("--checkpoint", default=config["checkpoint_path"])
    parser.add_argument("--ollama-host", default=config["ollama_host"])
    parser.add_argument("--ollama-model", default=config["ollama_model"])
    parser.add_argument("--cross-encoder-model", default=config["cross_encoder_model"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--cross-encoder-batch-size",
        type=int,
        default=int(config["cross_encoder_batch_size"]),
    )
    args = parser.parse_args()

    pipeline = RetrievalPipeline.from_paths(
        index_dir=args.index_dir,
        checkpoint_path=args.checkpoint,
        ollama_host=args.ollama_host,
        ollama_model=args.ollama_model,
        cross_encoder_model=args.cross_encoder_model,
        batch_size=args.batch_size,
        cross_encoder_batch_size=args.cross_encoder_batch_size,
    )
    server = RetrievalHTTPServer((args.bind, args.port), RetrievalHandler)
    server.pipeline = pipeline
    server.defaults = {
        "vector_top_k": int(config["vector_top_k"]),
        "cross_rerank_k": int(config["cross_rerank_k"]),
        "llm_rerank_k": int(config["llm_rerank_k"]),
        "final_context_k": int(config["final_context_k"]),
    }
    print(f"Serving retrieval UI at http://{args.bind}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlencode, urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from retrieval_system.indexing.chunker import load_chunks
from retrieval_system.search.pipeline import RetrievalPipeline
from retrieval_system.settings import load_config
from retrieval_system.vectordb.store import FileVectorStore
from retrieval_system.vectordb.store import SearchResult


STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML_PATH = STATIC_DIR / "index.html"
PAPERS_INPUT_PATH = Path("retrieval_system/papers")
PAPERS_ROOT = Path("retrieval_system/papers").resolve()


class RetrievalHTTPServer(ThreadingHTTPServer):
    pipeline: RetrievalPipeline
    defaults: dict[str, int]
    config: dict
    index_lock: threading.RLock
    index_status: dict


class RetrievalHandler(BaseHTTPRequestHandler):
    server: RetrievalHTTPServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(INDEX_HTML_PATH.read_text(encoding="utf-8"))
            return
        if parsed.path == "/health":
            self._send_json({"status": "ok"})
            return
        if parsed.path == "/papers":
            self._send_json({"papers": _list_papers()})
            return
        if parsed.path == "/folders":
            self._send_json({"folders": _list_folders()})
            return
        if parsed.path == "/index/status":
            self._send_json(self.server.index_status)
            return
        if parsed.path == "/paper":
            params = parse_qs(parsed.query)
            source = params.get("source", [""])[0]
            self._send_paper(source)
            return
        if parsed.path == "/arxiv/search":
            params = parse_qs(parsed.query)
            query = params.get("q", [""])[0].strip()
            max_results = int(params.get("max_results", ["8"])[0])
            self._handle_arxiv_search(query, max_results)
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
        try:
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except (json.JSONDecodeError, ValueError) as exc:
            self._send_json({"error": f"invalid json: {exc}"}, status=HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/query":
            self._handle_query(payload)
            return
        if parsed.path == "/papers/download":
            self._handle_download_paper(payload)
            return
        if parsed.path == "/papers/delete":
            self._handle_delete_paper(payload)
            return
        if parsed.path == "/folders/create":
            self._handle_create_folder(payload)
            return
        if parsed.path == "/folders/delete":
            self._handle_delete_folder(payload)
            return
        if parsed.path == "/index/rebuild":
            self._handle_rebuild_index()
            return
        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def _handle_query(self, payload: dict) -> None:
        query = str(payload.get("query", "")).strip()
        if not query:
            self._send_json({"error": "query is required"}, status=HTTPStatus.BAD_REQUEST)
            return

        defaults = self.server.defaults
        try:
            source = str(payload.get("source", "")).strip()
            retrieve_kwargs = {
                "vector_top_k": int(payload.get("vector_top_k", defaults["vector_top_k"])),
                "cross_rerank_k": int(payload.get("cross_rerank_k", defaults["cross_rerank_k"])),
                "llm_rerank_k": int(payload.get("llm_rerank_k", defaults["llm_rerank_k"])),
                "final_k": int(payload.get("final_k", defaults["final_context_k"])),
                "use_llm": _parse_bool(payload.get("use_llm", False)),
            }
            if source:
                _safe_paper_path(source)
                results = self.server.pipeline.retrieve_in_source(
                    query,
                    source=source,
                    **retrieve_kwargs,
                )
            else:
                results = self.server.pipeline.retrieve(query, **retrieve_kwargs)
        except Exception as exc:
            self._send_json(
                {"error": f"retrieval failed: {type(exc).__name__}: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        answer = None
        if _parse_bool(payload.get("answer", False)):
            try:
                answer = self.server.pipeline.answer(query, results)
            except Exception as exc:
                answer = f"Không sinh được answer từ LLM: {type(exc).__name__}: {exc}"

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

    def _handle_arxiv_search(self, query: str, max_results: int) -> None:
        if not query:
            self._send_json({"error": "q is required"}, status=HTTPStatus.BAD_REQUEST)
            return
        max_results = max(1, min(max_results, 20))
        try:
            self._send_json({"papers": _search_arxiv(query, max_results=max_results)})
        except Exception as exc:
            self._send_json(
                {"error": f"arXiv search failed: {type(exc).__name__}: {exc}"},
                status=HTTPStatus.BAD_GATEWAY,
            )

    def _handle_download_paper(self, payload: dict) -> None:
        arxiv_id = str(payload.get("arxiv_id", "")).strip()
        title = str(payload.get("title", "")).strip()
        category = str(payload.get("category", "")).strip()
        pdf_url = str(payload.get("pdf_url", "")).strip()
        if not arxiv_id or not category:
            self._send_json(
                {"error": "arxiv_id and category are required"},
                status=HTTPStatus.BAD_REQUEST,
            )
            return
        try:
            path = _download_arxiv_pdf(arxiv_id, title, category, pdf_url)
            self._send_json({"ok": True, "paper": _paper_to_dict(path)})
        except Exception as exc:
            self._send_json(
                {"error": f"download failed: {type(exc).__name__}: {exc}"},
                status=HTTPStatus.BAD_GATEWAY,
            )

    def _handle_delete_paper(self, payload: dict) -> None:
        source = str(payload.get("source", "")).strip()
        try:
            path = _safe_paper_path(source)
            path.unlink()
            self._send_json({"ok": True, "deleted": source, "papers": _list_papers()})
        except Exception as exc:
            self._send_json(
                {"error": f"delete failed: {type(exc).__name__}: {exc}"},
                status=HTTPStatus.BAD_REQUEST,
            )

    def _handle_create_folder(self, payload: dict) -> None:
        name = str(payload.get("name", "")).strip()
        try:
            folder = _safe_folder_path(name, must_exist=False)
            folder.mkdir(parents=True, exist_ok=True)
            self._send_json({"ok": True, "folder": folder.name, "folders": _list_folders()})
        except Exception as exc:
            self._send_json(
                {"error": f"create folder failed: {type(exc).__name__}: {exc}"},
                status=HTTPStatus.BAD_REQUEST,
            )

    def _handle_delete_folder(self, payload: dict) -> None:
        name = str(payload.get("name", "")).strip()
        try:
            folder = _safe_folder_path(name)
            if any(folder.iterdir()):
                self._send_json(
                    {"error": "folder is not empty; delete papers first"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            folder.rmdir()
            self._send_json({"ok": True, "deleted": folder.name, "folders": _list_folders()})
        except Exception as exc:
            self._send_json(
                {"error": f"delete folder failed: {type(exc).__name__}: {exc}"},
                status=HTTPStatus.BAD_REQUEST,
            )

    def _handle_rebuild_index(self) -> None:
        if not self.server.index_lock.acquire(blocking=False):
            self._send_json({"error": "index rebuild is already running"}, status=HTTPStatus.CONFLICT)
            return
        try:
            start = time.time()
            self.server.index_status = {
                "state": "running",
                "message": "Rebuilding vector index",
                "started_at_unix": start,
            }
            chunks = load_chunks(
                PAPERS_INPUT_PATH,
                max_chars=int(self.server.config["chunk_max_chars"]),
                overlap_chars=int(self.server.config["chunk_overlap_chars"]),
            )
            chunks = [chunk for chunk in chunks if str(chunk.text or "").strip()]
            if not chunks:
                raise RuntimeError("No chunks found under retrieval_system/papers")

            vectors = self.server.pipeline.encoder.encode(chunk.text for chunk in chunks)
            store = FileVectorStore(self.server.config["index_dir"])
            store.save(
                chunks,
                vectors,
                metadata={
                    "input_path": str(PAPERS_INPUT_PATH),
                    "checkpoint_path": str(self.server.config["checkpoint_path"]),
                    "num_chunks": len(chunks),
                    "embedding_dim": self.server.pipeline.encoder.dim,
                    "chunk_max_chars": int(self.server.config["chunk_max_chars"]),
                    "chunk_overlap_chars": int(self.server.config["chunk_overlap_chars"]),
                    "created_at_unix": time.time(),
                    "elapsed_sec": time.time() - start,
                    "rebuilt_from_api": True,
                },
            )
            self.server.pipeline.store = store.load()
            self.server.index_status = {
                "state": "ready",
                "message": f"Indexed {len(chunks)} chunks",
                "num_chunks": len(chunks),
                "elapsed_sec": time.time() - start,
                "finished_at_unix": time.time(),
            }
            self._send_json({"ok": True, **self.server.index_status, "papers": _list_papers()})
        except Exception as exc:
            self.server.index_status = {
                "state": "error",
                "message": f"{type(exc).__name__}: {exc}",
                "finished_at_unix": time.time(),
            }
            self._send_json(
                {"error": f"rebuild failed: {type(exc).__name__}: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
        finally:
            self.server.index_lock.release()

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

    def _send_paper(self, source: str) -> None:
        try:
            path = _safe_paper_path(source)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        if not path.exists() or not path.is_file():
            self._send_json({"error": "paper not found"}, status=HTTPStatus.NOT_FOUND)
            return

        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Disposition", f'inline; filename="{path.name}"')
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


def _safe_paper_path(source: str) -> Path:
    if not source:
        raise ValueError("source is required")
    path = Path(source)
    if not path.is_absolute():
        path = Path.cwd() / path
    resolved = path.resolve()
    if not resolved.is_relative_to(PAPERS_ROOT):
        raise ValueError("source must be inside retrieval_system/papers")
    if resolved.suffix.lower() != ".pdf":
        raise ValueError("only PDF papers can be opened")
    return resolved


def _safe_folder_path(name: str, must_exist: bool = True) -> Path:
    safe_name = _sanitize_slug(name)
    if not safe_name:
        raise ValueError("folder name is required")
    path = (PAPERS_ROOT / safe_name).resolve()
    if not path.is_relative_to(PAPERS_ROOT):
        raise ValueError("folder must be inside retrieval_system/papers")
    if must_exist and not path.is_dir():
        raise ValueError("folder not found")
    return path


def _list_papers() -> list[dict]:
    papers = []
    if not PAPERS_ROOT.exists():
        return papers
    for path in sorted(PAPERS_ROOT.glob("*/*.pdf")):
        rel = path.relative_to(Path.cwd()).as_posix()
        category = path.parent.name
        paper = _paper_to_dict(path)
        paper["source"] = rel
        paper["category"] = category
        papers.append(paper)
    return papers


def _paper_to_dict(path: Path) -> dict:
    paper_id, title = _paper_label(path)
    return {
        "paper_id": paper_id,
        "title": title,
        "category": path.parent.name,
        "filename": path.name,
        "source": path.relative_to(Path.cwd()).as_posix(),
    }


def _list_folders() -> list[str]:
    if not PAPERS_ROOT.exists():
        return []
    return sorted(path.name for path in PAPERS_ROOT.iterdir() if path.is_dir())


def _paper_label(path: Path) -> tuple[str, str]:
    stem = path.stem
    parts = stem.split("_", 1)
    if len(parts) == 1:
        return stem, stem.replace("_", " ").title()
    paper_id, title = parts
    return paper_id, f"{paper_id} · {title.replace('_', ' ')}"


def _search_arxiv(query: str, max_results: int = 8) -> list[dict]:
    params = urlencode(
        {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    request = Request(
        f"https://export.arxiv.org/api/query?{params}",
        headers={"User-Agent": "SFT-BE-Retrieval-Demo/1.0"},
    )
    with urlopen(request, timeout=30) as response:
        xml = response.read()

    root = ElementTree.fromstring(xml)
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    papers = []
    for entry in root.findall("atom:entry", ns):
        entry_id = _text(entry.find("atom:id", ns))
        arxiv_id = entry_id.rsplit("/", 1)[-1]
        title = re.sub(r"\s+", " ", _text(entry.find("atom:title", ns))).strip()
        summary = re.sub(r"\s+", " ", _text(entry.find("atom:summary", ns))).strip()
        published = _text(entry.find("atom:published", ns))[:10]
        authors = [_text(item.find("atom:name", ns)) for item in entry.findall("atom:author", ns)]
        primary = entry.find("arxiv:primary_category", ns)
        primary_category = primary.attrib.get("term", "") if primary is not None else ""
        pdf_url = f"https://arxiv.org/pdf/{quote(arxiv_id)}.pdf"
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href", pdf_url)
                break
        papers.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "summary": summary,
                "published": published,
                "authors": authors[:6],
                "primary_category": primary_category,
                "pdf_url": pdf_url,
            }
        )
    return papers


def _download_arxiv_pdf(arxiv_id: str, title: str, category: str, pdf_url: str) -> Path:
    folder = _safe_folder_path(category, must_exist=False)
    folder.mkdir(parents=True, exist_ok=True)
    clean_id = _sanitize_slug(arxiv_id.replace("/", "_").replace(".", "_"))
    clean_title = _sanitize_slug(title)[:90] or "arxiv_paper"
    filename = f"{clean_id}_{clean_title}.pdf"
    path = (folder / filename).resolve()
    if not path.is_relative_to(PAPERS_ROOT):
        raise ValueError("download path escapes papers root")
    if not pdf_url:
        pdf_url = f"https://arxiv.org/pdf/{quote(arxiv_id)}.pdf"
    request = Request(pdf_url, headers={"User-Agent": "SFT-BE-Retrieval-Demo/1.0"})
    with urlopen(request, timeout=90) as response:
        content_type = response.headers.get("Content-Type", "")
        data = response.read()
    if len(data) < 1000 or (content_type and "pdf" not in content_type.lower()):
        raise RuntimeError("downloaded response does not look like a PDF")
    path.write_bytes(data)
    return path


def _sanitize_slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("._-")


def _text(element) -> str:
    if element is None or element.text is None:
        return ""
    return element.text.strip()


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
    server.config = {
        **config,
        "index_dir": args.index_dir,
        "checkpoint_path": args.checkpoint,
    }
    server.index_lock = threading.RLock()
    server.index_status = {
        "state": "ready",
        "message": "Index loaded",
    }
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

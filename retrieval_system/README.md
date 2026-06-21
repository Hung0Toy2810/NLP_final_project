# Retrieval System

Engineering layer above the SFT-BE embedding model.

## Role

SFT-BE is used as a low-cost, high-recall first-stage retriever. A pretrained
MS MARCO cross-encoder reranks its candidates before optional LLM verification.
This layer owns document ingestion, vector indexing, reranking, verification,
and the API/demo surface.

## Modules

- `indexing/`: chunk documents, encode text, and write vectors/metadata.
- `vectordb/`: vector store abstraction and persistence.
- `search/`: top-k retrieval, hybrid scoring, context-window assembly.
- `reranking/`: pretrained cross-encoder scoring between retrieval and LLM.
- `llm/`: LLM API client and answer verification.
- `api/`: user-facing service endpoints.
- `demo/`: runnable demos and evaluation scenarios.
- `configs/`: runtime configuration.
- `tests/`: focused tests for retrieval behavior.

## Paper DB

Put source papers under `retrieval_system/papers`. Supported formats are `.txt`,
`.md`, `.rst`, and text-based `.pdf`.

Build the local index:

```bash
python3 -m retrieval_system.indexing.build_index retrieval_system/papers \
  --index-dir retrieval_system/papers_index \
  --checkpoint checkpoints/stage0_final.pt
```

If PDFs are placed directly under `retrieval_system/`, build only those PDFs:

```bash
python3 -m retrieval_system.indexing.build_index retrieval_system/*.pdf \
  --index-dir retrieval_system/papers_index \
  --checkpoint checkpoints/stage0_final.pt
```

Retrieval and cross-encoder reranking without LLM verification:

```bash
python3 -m retrieval_system.demo.query "your question" \
  --index-dir retrieval_system/papers_index \
  --checkpoint checkpoints/stage0_final.pt \
  --show-context
```

The default flow is:

```text
SFT-BE top 100 -> MS MARCO cross-encoder top 10 -> optional LLM top 8 -> final top 5
```

Answer generation with Ollama. Add `--llm-rerank` only when you want the local
LLM to verify and rescore the cross-encoder results.

```bash
python3 -m retrieval_system.demo.query "your question" \
  --index-dir retrieval_system/papers_index \
  --checkpoint checkpoints/stage0_final.pt \
  --ollama-model gemma3:4b \
  --answer
```

Serve a browser/API gateway:

```bash
python3 -m retrieval_system.api.server \
  --index-dir retrieval_system/papers_index \
  --checkpoint checkpoints/stage0_final.pt \
  --bind 127.0.0.1 \
  --port 8088
```

Open `http://127.0.0.1:8088` on the same machine. For LAN/mobile access, bind to
`0.0.0.0` and open the Mac firewall for that port.

Every retrieved result carries citation metadata:

```text
paper=<paper>; section=<section>; page <n>; source=<path>; chunk=<index>
```

If the source text does not expose section or page information, the system keeps
`section unknown` or `page unknown` instead of inventing metadata. Scanned PDFs
need OCR before indexing because `pypdf` can only extract embedded text.

# SchemeSaathi — a RAG project that solves a real problem

**The problem.** Around 3,000 welfare schemes exist across India's centre and
states, and most people who are entitled to one cannot find out which one, or
what the conditions are, or which office to walk into. The information exists,
but it is spread over dozens of PDFs written in bureaucratic English.

**What this builds.** A question-answering assistant over scheme documents. You
ask in plain language — *"I am a SC student doing B.Com, family income 1.8 lakh,
is there a scholarship?"* — and it answers **only** from the documents, quoting
the passage and linking the official portal, or refuses to answer when the
documents don't cover it.

**Why it is a good RAG project.** The domain punishes guessing. A hallucinated
₹5 lakh or a wrong income ceiling is worse than no answer, so every piece of
the pipeline has to earn its place: retrieval quality, citation enforcement, and
an explicit "I don't know" path. That is the whole point of RAG, and here it is
not decoration.

Nothing is hidden behind a framework. Every step is a file you can read in one
sitting, and `eval/` measures it so you can prove a change helped.

---

## 1. Quickstart

```bash
pip install -r requirements.txt

python ingest.py                        # build the index (about 1 second)
python ask.py "I am a small farmer, what money can I get?"
python ask.py --debug "how many days of work does MGNREGA guarantee?"
streamlit run app.py                    # the web UI
```

That is the whole setup. No API key, no model download, no Docker — the default
configuration uses a zero-dependency embedder and quotes passages instead of
calling an LLM, so the project runs on any laptop, offline, in about a second.

When you want better answers, add a model (section 6). The evaluation harness
tells you how much it actually helped.

Sample output:

```
$ python ask.py "How much does PM-KISAN pay and who is excluded?"

An income support of Rs 6,000 per year, paid in three equal instalments of
Rs 2,000 each, directly into the bank account of the farmer family. [S1]
The following categories are explicitly excluded, even if they own farmland:
Institutional landholders, and farmer families where a member is a
constitutional post holder ... [S3]

Sources
  [S1] PM-KISAN Samman Nidhi > What this scheme gives you
        https://pmkisan.gov.in
  [S3] PM-KISAN Samman Nidhi > Who is NOT eligible
```

---

## 2. What RAG actually is

A language model knows what it was trained on and nothing else. It does not know
your documents, and it will happily invent an answer instead of admitting that.

RAG fixes this by splitting the job in two:

```
    question
       |
       v
  RETRIEVE  the few passages from your documents most likely to contain the answer
       |
       v
  AUGMENT   paste those passages into the prompt as numbered sources
       |
       v
  GENERATE  the model writes an answer using only those passages, with citations
```

The model's job shrinks from "remember everything" to "read these five
paragraphs and answer carefully". That is why RAG can answer about documents it
has never seen, and why the citation you get back is checkable.

The thing beginners get wrong: **RAG quality is mostly retrieval quality.** If
the right passage is not in the five you retrieved, no prompt can save you. So
this project spends its effort on chunking, hybrid retrieval, metadata filters,
and measuring recall — not on prompt poetry.

---

## 3. The pipeline, file by file

There are two phases, and keeping them separate in your head is the single most
useful mental model for RAG.

### Phase 1 — Ingestion (run once per document set)

```
data/schemes/*.md
   |  rag/loaders.py     read the file, parse the metadata header
   v
Document(doc_id, title, body, meta)
   |  rag/chunker.py     split on headings, then by size, keeping the heading
   v                      in the text that gets embedded
Chunk(embed_text, section, meta)
   |  rag/embedder.py    text -> vector
   v
(70 x 512) matrix
   |  rag/store.py       write to SQLite: text, metadata, float32 vector
   v
data/index/index.sqlite3
```

Run it with `python ingest.py`. It is idempotent, so you can re-run it after
editing a document.

### Phase 2 — Answering (per question)

```
question
   |  rag/embedder.py + rag/bm25.py    two retrievers, in parallel
   v
dense hits + keyword hits
   |  rag/retrieve.py                  Reciprocal Rank Fusion -> one ranked list
   v
top-5 passages
   |  rag/retrieve.py                  answerability gate: do we know enough?
   |                                   -> if not, abstain and say so
   v
   |  rag/generate.py                  prompt with [S1]..[S5], then a citation
   |                                   check on the way out
   v
Answer(text, citations, sources, confidence)
```

`rag/pipeline.py` is the only file that knows this order. Read `build_index()`
and `RAG.ask()` first — forty lines that call everything else in sequence.

| File | Lines | What it does | Why it exists |
|---|---|---|---|
| `config.py` | ~60 | every knob in one place | so you can change chunk size and re-measure |
| `rag/loaders.py` | ~90 | file → Document + metadata | metadata is what makes filtering possible |
| `rag/chunker.py` | ~120 | Document → Chunks | the step that most affects quality |
| `rag/embedder.py` | ~150 | text → vector, 3 backends | shows what embeddings buy you |
| `rag/bm25.py` | ~70 | keyword retrieval | embeddings miss exact names and numbers |
| `rag/store.py` | ~200 | SQLite + numpy | inspectable, no server to run |
| `rag/retrieve.py` | ~180 | fusion, filters, abstain gate | where "hybrid" happens |
| `rag/llm.py` | ~180 | 4 LLM providers | one HTTP POST each, no SDK |
| `rag/generate.py` | ~200 | prompt, citations, extractive mode | the part you must not trust blindly |
| `rag/pipeline.py` | ~180 | `build_index()` and `ask()` | the two public entry points |

---

## 4. The five decisions that make this more than a demo

**1. Chunk on structure, and embed the heading with the text.**
`chunker.py` splits on markdown headings first, so a chunk never begins in the
middle of a rule. It then prefixes every chunk with `Title | Section` before
embedding. A bare sentence *"The applicant must be a landholding farmer family"*
is unfindable; *"PM-KISAN Samman Nidhi | Who is eligible — The applicant..."* is
not. Try `--size 300` vs `--size 1200` in `ingest.py` and watch recall move.

**2. Hybrid retrieval, not just embeddings.**
Vector search is bad at exactly what citizens type: scheme names, acronyms and
numbers. `PM-JAY`, `14555`, `Rs 10,000` are near-invisible to an embedding model
but trivial for BM25. `retrieve.py` runs both and fuses them with Reciprocal
Rank Fusion, which needs no score normalisation and no tuning. Compare:

```bash
python ask.py --mode dense "PM-JAY 5 lakh"
python ask.py --mode bm25  "PM-JAY 5 lakh"
python ask.py              "PM-JAY 5 lakh"     # hybrid, usually best
```

**3. Metadata filters, because "eligible" is relative.**
Each document carries `who: rural|urban|all` and a `category`. `--filter
who=rural` restricts retrieval before ranking. Values of `all` match any filter,
so a central scheme is visible to everyone. This is how you would stop a rural
land question from retrieving a street-vendor loan.

**4. An explicit abstain path.**
`answerability()` in `retrieve.py` asks whether the question's words are present
in what we retrieved, weighted by how rare those words are. If not, `ask()`
returns "I could not answer that from the documents I have" plus the closest
titles. A system that says "I don't know" is more useful than one that is
confidently wrong, especially about money.

**5. Check the answer after you get it.**
`validate_citations()` counts how many sentences carry a `[S#]` marker and drops
markers pointing at sources that don't exist. An answer where fewer than half
the sentences are cited is flagged `grounded: False` and the UI warns the user.
Never trust a generated answer because it looks well-written; check it in code.

---

## 5. Measuring it

```bash
python eval/run_eval.py
python eval/run_eval.py --k 3 --mode bm25
```

30 questions in `eval/questions.jsonl`: 23 answerable (with the document that
should be retrieved and a key figure the answer must contain) and 7 that must be
refused. Metrics:

| Metric | Meaning |
|---|---|
| `hit@k` | did the correct document appear in the top k at all? |
| `MRR` | how high did it rank? 1.0 = always first |
| `answer/abstain` | answered when it should, refused when it should |
| `key fact present` | does the answer contain the expected figure (e.g. `6,000`)? |
| `citation grounded` | does the answer carry valid `[S#]` markers? |

**Baseline, measured in this repository** (hashing embedder, extractive mode,
`k=5`, 70 chunks):

```
hit@5              100%   (23/23)
MRR                 1.000
answer/abstain        93%   (28/30)
key fact present      83%   (19/23)
citation grounded    100%   (23/23)
```

Two questions are wrongly answered rather than refused: *"What is the capital of
France?"* (it matches the word "capital" inside "working **capital** loan") and
*"What is the income tax slab for a freelancer?"* (it matches "income tax" from
the PM-KISAN exclusions). Four answered questions miss their key figure because
extractive mode quotes a sentence that shares words with the question instead of
the sentence holding the number.

That is the honest state of a lexical system, and those six failures are your
worklist. Each one has a specific fix:

| Failure | Fix | Where |
|---|---|---|
| Lexical gap ("no PF" ≠ "unorganised sector") | semantic embeddings | `EMBED_BACKEND=sentence-transformers` |
| Quotes the wrong sentence | a real LLM writes the answer | `LLM_PROVIDER=ollama` |
| Off-topic question answered | rerank, or an LLM relevance check | add a cross-encoder, or ask the model to answer `NOT_FOUND` |
| Chunk too big/small | tune `CHUNK_SIZE` | `config.py`, then re-run eval |

---

## 6. Making the answers better

### A real embedding model

```bash
pip install sentence-transformers
python ingest.py --backend sentence-transformers --reset
python eval/run_eval.py
```

This downloads a ~90 MB model once. Expect the "no PF" style failures to
disappear: the model matches meaning, not spelling. Note that the index must be
rebuilt — the vectors have a different dimension, and `RAG` will refuse to
silently mix them.

### A real LLM

```bash
cp .env.example .env
```

Then either run a free local model:

```bash
# install ollama from ollama.com, then:
ollama pull llama3.2
ollama serve
# and in .env:  LLM_PROVIDER=ollama
```

or put a hosted key in `.env` (`LLM_PROVIDER=openai|gemini|anthropic`). Re-run
`eval/run_eval.py` and compare `key fact present` with the baseline above. If it
does not improve, your retrieval is the bottleneck, not the model — which is the
most useful lesson in this whole project.

### Real documents

`data/schemes/` contains ten short summaries **written for this project**. They
are plain-language, and they are not official. Replace them:

1. Download official scheme guidelines (PDF) from `myscheme.gov.in` or the
   ministry site.
2. `pip install pypdf`, drop the PDFs into `data/schemes/`.
3. Add a small `.md` wrapper with the metadata header if you want filters:
   ```
   ---
   id: pm-kisan
   title: PM-KISAN Samman Nidhi
   category: farmer-income-support
   who: rural
   source: https://pmkisan.gov.in
   ---
   ```
4. `python ingest.py --reset`

Ten real PDFs will give you a few hundred chunks — still instant. At a few
thousand chunks, swap `rag/store.py` for FAISS or pgvector; nothing else changes.

---

## 7. Known limitations (read this before you show it to anyone)

- **The documents are summaries, not the law.** Every figure can change with a
  cabinet decision. The UI carries a disclaimer; keep it.
- **The abstain gate is lexical.** It catches obviously off-topic questions, and
  it can be fooled by a single shared word. A semantic embedder or a reranker
  fixes most of it.
- **Extractive mode is not an answer.** It is a stack of quotations. It exists so
  the pipeline runs with no keys, and so you can see what the LLM adds.
- **One language.** English only. Real users ask in Hindi, Tamil, Bengali. Adding
  a multilingual embedder plus a translation step at the edges is the obvious
  next project.
- **No memory, no follow-ups.** "What about my wife?" will not resolve "my wife"
  against the previous turn. Query rewriting with the chat history is a
  well-known fix and a good exercise.
- **Personal data.** If you deploy this, do not log the questions. People type
  their caste, income and health conditions into a box like this, and under
  India's DPDP Act that is personal data with real obligations.
- **It is not advice.** An eligibility answer from this tool is a starting point
  for a conversation at the Panchayat or the bank, not a decision.

---

## 8. Where to take it next

1. **Conversational memory** — rewrite the follow-up question using the last two
   turns before retrieving. Biggest single UX win.
2. **A reranker** — retrieve 20, cross-encode to pick 5. Costs one model, buys a
   lot of precision.
3. **Answer-level self-check** — ask the model to mark each sentence as
   supported or not by its cited source, and drop the unsupported ones.
4. **Multilingual** — embed with a multilingual model, answer in the user's
   language, keep the citation markers.
5. **A feedback loop** — a thumbs-down button that logs the question and the
   retrieved doc ids, so your eval set grows from real failures.
6. **Deployment** — `streamlit run app.py` on a small VM is enough for a pilot.
   Put it behind a login before real citizens touch it.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No index at ...` | never ingested | `python ingest.py` |
| `index was built with 'hashing' but you are querying with ...` | changed embedder | `python ingest.py --reset` |
| Answers are always "could not answer" | gate too strict, or wrong corpus | try `--debug`, then lower `COVERAGE_MIN` |
| `fell back to extractive mode` | Ollama not running, or bad API key | check the note in `--debug` output |
| Everything is slow on first run | downloading the embedding model | once only, then cached |
| `libtorch_cpu.so: failed to map segment` | torch on a low-memory/odd-filesystem machine | stay on `EMBED_BACKEND=hashing`, or use the OpenAI embedder |

---

## 10. Learn it in a week

`LEARN.md` is a seven-day path: what to read in what order, what to break on
purpose, and one exercise per day that changes the numbers in section 5. Read it
alongside the code, not after it.

---

*Built as a learning project. The scheme summaries are illustrative; the
engineering is real.*
#   R A G - l e a r n i n g  
 
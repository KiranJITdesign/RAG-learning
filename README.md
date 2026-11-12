SchemeSaathi
A RAG system that answers questions about Indian government welfare schemes — and refuses when it doesn't know.

Built from scratch, no framework, every step readable in one sitting.

Python 3.9+ · runs offline, no API key · 17 tests passing · hit@5 100% · abstain accuracy 93%

Quickstart · How it works · Results · Design decisions · Roadmap · Learn RAG in a week [blocked]

The problem
India runs thousands of welfare schemes across the centre and the states. Most people who are entitled to one cannot find out which one applies to them, what the conditions are, or which office to walk into. The information exists — spread across dozens of PDFs written in bureaucratic English.

What this builds
A question-answering assistant over scheme documents. You ask in plain language:

"I am a SC student doing B.Com, family income 1.8 lakh, is there a scholarship?"

and it answers only from the documents, quoting the passage it came from, or it says it doesn't know.



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


$ python ask.py "What is the capital of France?"

I could not answer that from the documents I have. The closest material I
found is: PM SVANidhi street vendor working capital loan, PM-KISAN Samman
Nidhi. This assistant only answers from its own scheme summaries, so it will
not guess.
Why this is a good RAG project
The domain punishes guessing. A hallucinated ₹5 lakh or a wrong income ceiling is worse than no answer, so every part of the pipeline has to earn its place: retrieval quality, citation enforcement, and an explicit "I don't know" path. RAG is not decoration here — it is the only way the thing can work.

It also forces the problems that tutorials skip:




What a tutorial does	What this project has to do
Embed and cosine-search	Add BM25 — embeddings are blind to PM-JAY, 14555, ₹10,000
One big index	Metadata filters — rural vs urban, scheme category
Always answers	An abstain gate with a measured false-positive rate
"The answer looks right"	Citation validation in code, not vibes
Trust the demo	A 30-question eval harness with 5 metrics
Quickstart
bash


git clone https://github.com/KiranJITdesign/RAG-learning.git
cd RAG-learning
pip install -r requirements.txt

python ingest.py                                       # build the index (~1 s)
python ask.py "I am a small farmer, what money can I get?"
python ask.py --debug "how many days of work does MGNREGA guarantee?"
streamlit run app.py                                   # the web UI
No API key, no model download, no Docker. The default configuration uses a zero-dependency embedder and quotes passages instead of calling an LLM, so the project runs on any laptop, offline, in about a second.

Two commands you will use constantly:

bash


python -m pytest -q          # 17 unit tests, under a second, no network
python eval/run_eval.py      # the numbers in the Results section below
How it works
Two phases. Keeping them separate in your head is the single most useful mental model for RAG.

Phase 1 — Ingestion (once per document set)


data/schemes/*.md
   │  rag/loaders.py      read the file, parse the metadata header
   ▼
Document(doc_id, title, body, meta)
   │  rag/chunker.py      split on headings, then by size, keeping the heading
   ▼                     inside the text that gets embedded
Chunk(embed_text, section, meta)
   │  rag/embedder.py     text → vector
   ▼
(60 × 512) matrix
   │  rag/store.py        SQLite: text, metadata, float32 vector
   ▼
data/index/index.sqlite3
Phase 2 — Answering (per question)


question
   │  rag/embedder.py + rag/bm25.py    two retrievers, in parallel
   ▼
dense hits + keyword hits
   │  rag/retrieve.py                  Reciprocal Rank Fusion → one ranked list
   ▼
top-5 passages
   │  rag/retrieve.py                  answerability gate
   │                                   → not enough evidence? abstain, say so
   ▼
   │  rag/generate.py                  prompt with [S1]..[S5]
   │                                   → citation check on the way out
   ▼
Answer(text, citations, sources, confidence)
rag/pipeline.py is the only file that knows this order. Read build_index() and RAG.ask() first — forty lines that call everything else in sequence.

Project structure


RAG-learning/
├── config.py              every knob in one place
├── ingest.py              CLI: build the index
├── ask.py                 CLI: ask a question
├── app.py                 Streamlit UI
├── rag/
│   ├── loaders.py         file → Document + metadata
│   ├── chunker.py         Document → Chunks  (structure-aware)
│   ├── embedder.py        text → vector (3 interchangeable backends)
│   ├── bm25.py            keyword retrieval, no dependency
│   ├── store.py           SQLite + numpy vector store
│   ├── retrieve.py        fusion, filters, abstain gate
│   ├── llm.py             4 LLM providers, one HTTP POST each
│   ├── generate.py        prompt, citations, extractive mode
│   └── pipeline.py        build_index() and ask()
├── data/schemes/          10 scheme documents (markdown + metadata header)
├── eval/
│   ├── questions.jsonl    30 labelled questions, 7 must be refused
│   └── run_eval.py        hit@k, MRR, abstain accuracy, grounding
├── tests/test_rag.py      17 tests
├── LEARN.md               7-day guided build-it-yourself path
└── README.md
Results
Measured on this repository — hashing embedder, extractive mode, k=5, 60 chunks, 30 questions:




Metric	Score	What it means
hit@5	100% (23/23)	the right document always made the top 5
MRR	1.000	and it was ranked first every time
answer/abstain	93% (28/30)	answered when it should, refused when it should
key fact present	83% (19/23)	the answer contained the expected figure (6,000, 5 lakh, …)
citation grounded	100% (23/23)	every answer carried valid [S#] markers
Reproduce it:

bash


python ingest.py && python eval/run_eval.py
The six failures are documented, not hidden — each one is a specific lesson:




Failure	Cause	Fix
"no PF" doesn't match "unorganised sector"	lexical gap	EMBED_BACKEND=sentence-transformers
Quotes a sentence that shares words but not the number	no generator	LLM_PROVIDER=ollama
"capital of France" answered via "working capital loan"	one shared word	reranker or an LLM relevance check
"income tax slab for a freelancer" answered	same	same
That is the honest state of a lexical system with no LLM — and it is the baseline your improvements have to beat.

Five decisions that make this more than a demo
1. Chunk on structure, and embed the heading with the text. chunker.py splits on markdown headings first, so a chunk never begins mid-rule. Every chunk is prefixed with Title | Section before embedding. A bare sentence "The applicant must be a landholding farmer family" is unfindable; "PM-KISAN Samman Nidhi | Who is eligible — The applicant..." is not.

2. Hybrid retrieval, not just embeddings. Vector search is bad at exactly what citizens type: scheme names, acronyms and numbers. retrieve.py runs dense + BM25 and fuses them with Reciprocal Rank Fusion, which needs no score normalisation and no tuning.

bash


python ask.py --mode dense "PM-JAY 5 lakh"
python ask.py --mode bm25  "PM-JAY 5 lakh"
python ask.py              "PM-JAY 5 lakh"     # hybrid, usually best
3. Metadata filters, because "eligible" is relative. Each document carries who: rural|urban|all and a category. --filter who=rural restricts retrieval before ranking, so a rural land question can't retrieve a street-vendor loan. Values of all match any filter.

4. An explicit abstain path. answerability() asks whether the question's words appear in what was retrieved, weighted by how rare those words are — plus a rule that one incidental word is never enough unless it matches a scheme name. If not, ask() returns "I could not answer that" plus the closest titles. A system that says I don't know is more useful than one that is confidently wrong about money.

5. Check the answer after you get it. validate_citations() counts how many sentences carry a [S#] marker and strips markers pointing at sources that don't exist. An answer where fewer than half the sentences are cited is flagged grounded: False and the UI warns the user. Never trust a generated answer because it reads well — check it in code.

Making it better
A real embedding model
bash


pip install sentence-transformers
python ingest.py --backend sentence-transformers --reset
python eval/run_eval.py
Downloads a ~90 MB model once. Expect the "no PF" style failures to disappear. The index must be rebuilt — the vectors have a different dimension, and RAG refuses to silently mix them.

A real LLM
bash


cp .env.example .env
Free and local (recommended first step):

bash


# install ollama from ollama.com, then:
ollama pull llama3.2
ollama serve
# in .env:  LLM_PROVIDER=ollama
Or a hosted key: LLM_PROVIDER=openai | gemini | anthropic. Then re-run the eval and compare key fact present with the baseline above. If it doesn't improve, your retrieval is the bottleneck, not the model — which is the most useful lesson in the whole project.

Your own documents
The ten files in data/schemes/ are short plain-language summaries written for this project. They are not official. To use real guidelines:

Download official PDFs from myscheme.gov.in or the ministry site.
pip install pypdf, drop them into data/schemes/.
Optionally add a metadata header for filtering:
markdown


---
id: pm-kisan
title: PM-KISAN Samman Nidhi
category: farmer-income-support
who: rural
source: https://pmkisan.gov.in
---
python ingest.py --reset
Ten real PDFs give a few hundred chunks — still instant. At a few thousand, swap rag/store.py for FAISS or pgvector; nothing else changes.

Known limitations
The documents are summaries, not the law. Every figure can change with a cabinet decision. The UI carries a disclaimer; keep it.
The abstain gate is lexical. It catches obviously off-topic questions and can be fooled by a single shared word. A semantic embedder or reranker fixes most of it.
Extractive mode is not an answer. It is a stack of quotations, and it exists so the pipeline runs with no keys.
English only. Real users ask in Hindi, Tamil, Bengali. A multilingual embedder plus translation at the edges is the obvious next project.
No memory. "What about my wife?" won't resolve against the previous turn. Query rewriting is a well-known fix and a good exercise.
Personal data. If you deploy this, don't log the questions. People type their caste, income and health conditions into a box like this; under India's DPDP Act that is personal data with real obligations.
It is not advice. An eligibility answer is a starting point for a conversation at the Panchayat or the bank, not a decision.
Roadmap
 Conversational memory — rewrite the follow-up using the last two turns before retrieving. Biggest UX win per line of code.
 Reranker — retrieve 20, cross-encode to pick 5.
 Answer-level self-check — mark each sentence supported or not by its cited source; drop the unsupported ones.
 Multilingual — multilingual embedder, answer in the user's language, keep [S#] intact.
 Feedback loop — a thumbs-down button logging the question and retrieved doc ids, so the eval set grows from real failures.
 Real corpus — official PDFs from myscheme.gov.in, including the tables and footnotes that break naive parsers.
Learn RAG with this repo
LEARN.md [blocked] is a seven-day path: what to read in what order, what to break on purpose, and one experiment per day that moves the numbers in the Results table. Each day has a measurable task — four chunk sizes, three retrieval modes, trying to fool the abstain gate, swapping in a real LLM.

Five sentences worth memorising

Retrieval quality caps answer quality. Fix retrieval first, always.
Chunk on structure, and embed the heading with the text.
Hybrid retrieval beats either retriever alone, because names and meaning fail differently.
An abstain path is a feature, not a bug.
If you did not measure it, you do not know whether it improved.
Built with
Python 3.9+ · numpy · Streamlit · SQLite (stdlib) · pytest

No LangChain, no LlamaIndex, no vector database server. Deliberately — so you can see every step, and so there is no magic to debug.

Disclaimer
A learning project. The scheme summaries are illustrative and may be out of date; verify every figure on the official portal before acting on it. Not affiliated with any government body.


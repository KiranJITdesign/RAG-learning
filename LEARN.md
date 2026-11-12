# Learn RAG by building this, in seven days

This is the study plan that goes with the code. Each day has a reading target, a
hands-on task, and one experiment where you change a number and watch a metric
move. The point is not to finish; it is to stop believing that RAG is magic.

Rule for the week: **after every change, run `python eval/run_eval.py` and write
the four numbers down.** By Friday you will have a table of your own and you
will know which knob matters, which is a thing most people who write about RAG
cannot say.

Keep a file called `lab-notes.md`. One line per experiment:

```
exp | change | hit@5 | MRR | abstain | key fact | grounded
```

---

## Day 1 — See it work, then look inside

**Read:** `README.md` sections 1–3.

**Do:**

```bash
pip install -r requirements.txt
python ingest.py
python ask.py "how much does PM-KISAN pay?"
python ask.py --debug "how many days of work does MGNREGA guarantee?"
python ask.py --json "my mother is 72, is she covered for health?"
```

**Understand the debug output.** Every question prints the five retrieved
passages with `dense_rank` and `bm25_rank`. Notice that they disagree often, and
that the fused order is usually better than either.

**Experiment:** open `data/index/index.sqlite3` with any SQLite browser (DB
Browser for SQLite is free) and look at the `chunks` table. That is your whole
knowledge base: text, metadata, and a vector. Nothing else is stored.

**Write down:** how many chunks the ten documents produced (should be ~60), and
why it is not ten.

---

## Day 2 — Ingestion is the whole ballgame

**Read:** `rag/loaders.py`, `rag/chunker.py`, then `rag/embedder.py`.

**Do:** run the pipeline four times with different chunk sizes and record the
numbers each time.

```bash
python ingest.py --size 300 --overlap 50  && python eval/run_eval.py
python ingest.py --size 500 --overlap 100 && python eval/run_eval.py
python ingest.py --size 800 --overlap 150 && python eval/run_eval.py
python ingest.py --size 1500 --overlap 200 && python eval/run_eval.py
```

**Questions to answer for yourself:**

- Why do tiny chunks lose on `key fact present`?
- Why do huge chunks hurt `hit@5`? (Hint: one vector now has to represent three
  unrelated ideas, so it matches everything weakly and nothing strongly.)
- What does the overlap actually protect against? Find a sentence that would be
  cut in half without it.

**Break it on purpose:** in `chunker.py`, delete the line that puts `doc.title`
and the section heading into `embed_text`, rebuild, re-evaluate. Write down the
drop. Then put it back and never forget it.

---

## Day 3 — Retrieval: two weak methods beat one

**Read:** `rag/bm25.py` and `rag/retrieve.py`.

**Do:** pick five questions and run each in all three modes.

```bash
for q in "PM-JAY 5 lakh" "how much does Sukanya Samriddhi allow per year" "MGNREGA 15 days wages"; do
  for m in dense bm25 hybrid; do
    python ask.py --mode $m --debug "$q" | head -6
  done
done
```

**Understand:** BM25 wins on names, acronyms and numbers. Dense wins on
paraphrase ("money for a farmer" → "income support"). RRF (`1/(60+rank)`) is why
you do not have to normalise the two score scales — a rank is already a common
currency.

**Break it on purpose:** set `RRF_K = 1` in `config.py` and re-evaluate. Now the
top hit of either list dominates. Notice how much worse it gets, and explain why
to yourself in one sentence.

**Write down:** one query where hybrid beat both single modes, and why.

---

## Day 4 — The "I don't know" path

**Read:** `answerability()` and `should_abstain()` in `rag/retrieve.py`, then
`RAG.ask()` in `rag/pipeline.py`.

**Do:** try to fool it.

```
What is the capital of France?
How do I renew my passport?
Is there a scheme for a boat owner in Kerala?
My landlord will not return my deposit, what can I do?
Can I get Ayushman for dental implants?
```

Then look at the `--debug` line `coverage ... matched terms ... title hits ...`
and work out *why* each one was answered or refused.

**This is the most important day.** A RAG system's value is not what it answers;
it is what it refuses. Count how many of the five it got right. Fix at least one
of the wrong ones — either by adding a document, or by tightening the gate, and
check you did not break a correct answer elsewhere.

**Experiment:** change `COVERAGE_MIN` from 0.25 to 0.10 and to 0.50, and watch
`answer/abstain` fall and rise. There is no free lunch: every point of recall you
buy costs you a hallucination somewhere.

---

## Day 5 — Generation, and not trusting it

**Read:** `rag/generate.py` — the system prompt, `validate_citations()`, and
`extractive_answer()`.

**Do:** get a real LLM in the loop, ideally a local one.

```bash
# install ollama, then:
ollama pull llama3.2
ollama serve
echo "LLM_PROVIDER=ollama" >> .env
python ask.py "I am a SC student doing B.Com, income 1.8 lakh, any scholarship?"
python eval/run_eval.py
```

Compare `key fact present` with the baseline in `README.md` section 5. This is
the number that shows what a generator adds over quotation.

**Then attack it.** Ask questions the documents half-answer and see whether the
model invents a figure or says `NOT_FOUND`. Try:

- "What is the PM-KISAN amount for 2027?" (the docs give no year)
- "Is there a scheme that pays 12,000 a year to farmers?" (there isn't, in these
  documents — the answer is 6,000)

**Understand:** the citation check is a smoke detector, not a fire door. It
catches uncited sentences; it cannot tell whether the cited sentence really
supports the claim. That gap is why the roadmap in the README mentions an
answer-level self-check.

---

## Day 6 — Put it in front of a person

**Read:** `app.py`.

**Do:**

```bash
streamlit run app.py
```

Open it on your phone too (Streamlit prints a network URL). Ask the four example
questions. Toggle "Show retrieval internals".

**Then do the design work:** show it to one person who is not a programmer, and
watch where they get stuck. Not what they say — where their thumb pauses. Two
things will probably come up:

1. They will type a follow-up ("and my wife?") that the system cannot resolve.
2. They will want to know what to *do* next, and the answer only says what the
   scheme is.

Both are retrieval and prompt problems, not UI problems. Write them down as your
next two tasks.

**Experiment:** add one more document to `data/schemes/` — a real scheme you know
something about, with its metadata header — and confirm it gets found. That is
the whole "add knowledge" workflow in this design: drop a file, re-ingest, done.
No retraining, no fine-tuning. Say that out loud; it is the reason RAG won.

---

## Day 7 — Make it yours

Pick one and ship it. In order of value per hour of work:

1. **Follow-up questions.** Keep the last two turns, and rewrite the new
   question into a standalone one before retrieval ("and my wife?" → "is my wife
   covered under PM-JAY?"). Ten lines and a huge UX win.
2. **A reranker.** Retrieve 20 with the current code, then score each (question,
   passage) pair with a cross-encoder and keep 5. Measure `MRR` before and after.
3. **Real PDFs.** Ten official scheme guidelines from `myscheme.gov.in`, with
   `pypdf`. Watch what breaks on real documents — tables, footnotes, headers
   repeated on every page — because that is what production RAG is actually
   about.
4. **Multilingual.** Ask in Hindi, retrieve with a multilingual embedder, answer
   in Hindi, keep `[S#]` markers intact.
5. **A thumbs-down loop.** Log the question and the retrieved doc ids when a user
   marks an answer wrong. That log is your next eval set, and it will be better
   than the one I wrote, because it comes from real questions.

When you finish one, add a line to `README.md` section 8 and re-run the eval.
You now have a project you can explain, defend, and extend — which is what
"learned RAG" actually means.

---

## The five sentences worth memorising

1. Retrieval quality caps answer quality. Fix retrieval first, always.
2. Chunk on structure, and embed the heading with the text.
3. Hybrid retrieval beats either retriever alone, because names and meaning fail
   differently.
4. An abstain path is a feature, not a bug.
5. If you did not measure it, you do not know whether it improved.

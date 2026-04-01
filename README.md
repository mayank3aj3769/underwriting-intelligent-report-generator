# Underwriting Intelligence Report System

This project builds **automated underwriting-style intelligence reports** for UK companies. You give it a company name or a Companies House registration number; it pulls structured data from official APIs, runs targeted web and news search, and returns a **citation-backed** report that calls out gaps and uncertainty instead of guessing.

**Stack:** Python 3.11 · Streamlit · LangGraph · OpenAI · SerpAPI · Companies House API  

There is **no custom web scraping** — evidence comes from API responses and search snippets only.

---

## What it does

Roughly, a full run looks like this:

1. **Resolve** the company through Companies House (with disambiguation when the name is ambiguous).
2. **Fetch** the official company profile and officers list. The profile includes flags such as whether charges or insolvency history exist; the pipeline does not call separate filings endpoints for those.
3. **Ask an LLM** for up to **six** focused Google-style queries tailored to that company (status, SIC codes, context).
4. **Search in parallel** via SerpAPI: those web queries plus a dedicated Google News query on the first pass.
5. **Summarise** everything into a structured briefing with `[Source: URL]` citations.
6. **Synthesise** the final JSON report that merges Companies House context with the briefing.

The report covers business model (including revenue where public data exists), competitive landscape, quality signals with strength ratings, outlook, and explicit uncertainty flags. Follow-up questions against an existing report are supported through the same `ReportGenerator` path (intent classification + optional follow-up handler).

---

## Running it locally

You will need **Python 3.11+** and API keys for [Companies House](https://developer.company-information.service.gov.uk/) (free), [SerpAPI](https://serpapi.com/), and [OpenAI](https://platform.openai.com/).

**1. Clone and install**

```bash
git clone <repo-url>
cd "Intelligent Report generation system"

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

**2. Environment**

```bash
cp .env.example .env
```

Edit `.env` — typical live settings:

```env
COMPANIES_HOUSE_API_KEY=your_key_here
COMPANIES_HOUSE_API_URL=https://api.company-information.service.gov.uk
SERP_API_KEY=your_serpapi_key
OPENAI_API_KEY=your_openai_key
```

Use `https://api-sandbox.company-information.service.gov.uk` for sandbox testing.

**3. Streamlit (recommended for everyday use)**  

This mode calls the pipeline **in process** — no separate backend.

```bash
streamlit run app.py --server.port 8080
```

Open http://localhost:8080

**4. FastAPI + Streamlit (optional split stack)**  

Useful if you want a REST API or Swagger while developing the UI separately.

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then, in another terminal:

```bash
streamlit run streamlit_app.py --server.port 8501
```

API base: http://localhost:8000 · Docs: http://localhost:8000/docs

---

## Docker and production

The Dockerfile runs **standalone Streamlit** (`app.py`) on **port 8080**. Health checks can use Streamlit’s `/_stcore/health` endpoint — that pattern fits platforms (e.g. Digital Ocean App Platform) that expect a single HTTP port.

```bash
docker build -t underwriting-intel .
docker run -p 8080:8080 --env-file .env underwriting-intel
```

**Why one process in production?** Running FastAPI and Streamlit as two services on two ports often breaks managed-platform health checks. `app.py` talks to `ReportGenerator` directly, so you avoid an extra HTTP hop and keep deployment simple.

---

## REST API (when `main.py` is running)

Not used by the standalone `app.py` deployment.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Status and key configuration |
| POST | `/api/v1/search` | Search Companies House by name |
| POST | `/api/v1/report` | Report by name or number (disambiguation aware) |
| POST | `/api/v1/report/by-number` | Report by company number |

Example:

```bash
curl -X POST http://localhost:8000/api/v1/report \
  -H "Content-Type: application/json" \
  -d '{"identifier": "REVOLUT LTD"}'
```

---

## Tests

```bash
python -m tests.test_e2e
```

Output also lands in `tests/test_results.txt`.

---

## Configuration

| Variable | Required | Default | Notes |
|----------|----------|---------|--------|
| `COMPANIES_HOUSE_API_KEY` | Yes | — | |
| `COMPANIES_HOUSE_API_URL` | No | Sandbox URL in `config.py` | Use live `https://api.company-information.service.gov.uk` when ready |
| `SERP_API_KEY` | Yes | — | Google Search + News via SerpAPI |
| `OPENAI_API_KEY` | Yes | — | |
| `OPENAI_MODEL` | No | `gpt-4o` | |
| `REQUEST_TIMEOUT` | No | `30` | Seconds |

`MAX_AGENT_ITERATIONS` may still appear in `.env.example` / `config.py`, but the **LangGraph pipeline does not use it**. Evidence-gathering repeats are capped by `MAX_SUFFICIENCY_ITERATIONS` in `agents/nodes.py` (currently **2**).

---

## Architecture (how it fits together)

### External services

| Service | Role | Auth |
|---------|------|------|
| Companies House REST | Profile (`/company/{number}`) + officers (`/officers`); charge/insolvency **flags** on the profile | API key as HTTP Basic **username**, empty password |
| SerpAPI | Web + news search snippets | API key |
| OpenAI | Query generation, summarisation, final synthesis | API key |

### Repository layout

```
.
├── app.py                  # Streamlit, production-style (in-process pipeline)
├── main.py                 # FastAPI
├── streamlit_app.py        # Streamlit UI → HTTP → FastAPI
├── config.py
├── schemas/                # Pydantic models (company, evidence, report)
├── tools/                  # companies_house.py, search_api.py
├── agents/                 # LangGraph: state.py, nodes.py, graph.py
├── services/               # resolver, report_generator, formatter, intent, follow-up
└── tests/
```

`report_generator.py` is the front door: it classifies intent, resolves entities, runs the graph (or handles follow-ups). The **seven** graph nodes live in `nodes.py`; `sufficiency_router` handles the conditional branch after evaluation.

### Production shape (conceptually)

```
┌─────────────────────────────────────────────┐
│  Container :8080                            │
│  Streamlit (app.py) → ReportGenerator       │
│  LangGraph pipeline in-process              │
│  Health: /_stcore/health                    │
└─────────────────────────────────────────────┘
```

### LangGraph pipeline

The graph is a `StateGraph` over `PipelineState`. After the first summarisation, **`evaluate_sufficiency`** decides whether to **synthesise** or loop through **`generate_gap_queries` → execute searches → summarise again**, up to two sufficiency iterations. If the iteration limit is hit, the graph **proceeds anyway** so a report always completes.

```
                         External calls
┌──────────────────────┐
│ 1. fetch_companies_  │ ──► Companies House (profile + officers)
│    house             │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 2. generate_queries  │ ──► OpenAI (JSON), up to 6 web queries
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 3. execute_searches  │ ──► SerpAPI parallel (6 web + 1 news on iter 0)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 4. summarize_searches│ ──► OpenAI briefing, [Source: URL]
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 5. evaluate_         │     Six section flags, confidence score
│    sufficiency       │ ──┐
└──────────┬───────────┘   │ if insufficient & iter < 2
           │               ▼
           │      ┌──────────────────────┐
           │      │ 6. generate_gap_     │ ──► OpenAI, up to 5 queries
           │      │    queries           │     → back to step 3
           │      └──────────────────────┘
           ▼ (sufficient or max iter)
┌──────────────────────┐
│ 7. synthesize_report │ ──► OpenAI → UnderwritingReport JSON
└──────────────────────┘
```

**Sufficiency (high level).** Coverage is tracked for six keys: `business_model`, `revenue`, `competition`, `quality_signals`, `news`, `financial_regulatory`, using keyword-style checks on snippets (and sometimes the summary). The score blends source count, how many sections fired, and summary length. To pass early you need confidence ≥ 0.6, at least **8** sources, and **3+** of the **6** sections — see `agents/nodes.py` for the exact formula.

**Performance habits:** SerpAPI calls are gathered concurrently (sync client wrapped with `asyncio.to_thread`). Gap iterations add cost but are bounded.

### Entity resolution

Before the graph runs, `EntityResolver` checks whether the input matches a Companies House number pattern (`^[A-Z]{0,2}\d{6,8}$`, case-insensitive). Otherwise it searches by name: single match, lone active company among many, or a single exact active name match auto-resolve; anything ambiguous returns candidates for the UI.

### State and metrics (for developers)

Pipeline state includes inputs, CH text/metadata, queries, merged search results, the summary string, evidence metrics, iteration count, sufficiency flag, final report dict, and an `errors` list accumulated with `operator.add`.

`EvidenceMetrics` carries flags such as `has_business_info`, `has_revenue_data`, `section_coverage`, `confidence_score`, `missing_sections`, and a short `reasoning` string.

### Report shape

The Pydantic `UnderwritingReport` in `schemas/report.py` is the source of truth. In words: **business model** (optional revenue fields), **competitive landscape** (including SIC codes and citations), **quality signals** (with counts and citations), plus outlook strings, uncertainty flags, and metadata like `evidence_confidence_score` / `evidence_iterations`.

### Design choices worth knowing

- **Single container in production** trades a bit of coupling for simpler ops and reliable health checks.
- **LangGraph** gives an explicit graph, typed state, and easy inspection of each step versus a hand-rolled agent loop.
- **Dedicated summariser node** shrinks noisy raw hits before the final synthesis call.
- **No alternate search or LLM providers** in code — fewer moving parts for auditability.

If you are extending the system, start with `agents/graph.py` and `services/report_generator.py`; model changes belong in `schemas/report.py`.

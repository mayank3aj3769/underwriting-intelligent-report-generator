# Architecture

Technical reference for the Underwriting Intelligence Report System.

## System Overview

The system generates structured underwriting intelligence reports for UK companies by orchestrating three external services through a LangGraph state-machine pipeline:

| Service | Role | Auth |
|---|---|---|
| Companies House REST API | Company profile, officers, filings, charges, insolvency | API key (Basic auth) |
| SerpAPI (Google Search + News) | Web search snippets, Knowledge Graph, news articles | API key |
| OpenAI GPT (gpt-4o) | Query generation, search summarisation, report synthesis | API key |

No web scraping is performed. Every data point originates from a structured API response or a search-engine snippet.

## Directory Structure

```
.
├── app.py                     Streamlit standalone entrypoint (production)
├── main.py                    FastAPI REST API entrypoint
├── config.py                  Environment variable loader (Settings singleton)
├── streamlit_app.py           Streamlit frontend (calls FastAPI over HTTP)
├── Dockerfile                 Single-container image (runs app.py on port 8080)
├── requirements.txt           Python dependencies
├── .env.example               Environment variable template
├── .streamlit/config.toml     Streamlit server configuration
│
├── schemas/                   Pydantic data models
│   ├── company.py             CompanyCandidate, CompanyProfile, Officer, CompanySearchResponse
│   ├── evidence.py            SourceType, Evidence, EvidenceCollection
│   └── report.py              UnderwritingReport and all nested models
│
├── tools/                     External API clients
│   ├── base.py                Abstract BaseTool interface
│   ├── companies_house.py     Companies House REST API client
│   └── search_api.py          SerpAPI Google Search + News client
│
├── agents/                    LangGraph pipeline
│   ├── state.py               PipelineState TypedDict (shared state contract)
│   ├── nodes.py               5 pipeline node functions
│   └── graph.py               StateGraph construction and compilation
│
├── services/                  Business logic
│   ├── entity_resolver.py     Company name/number disambiguation and resolution
│   ├── report_generator.py    Orchestrates resolution + pipeline invocation
│   └── report_formatter.py    SIC code map + Markdown report renderer
│
└── tests/                     Test suite
    └── test_e2e.py            End-to-end integration tests (7 tests)
```

## Deployment Architecture

The production deployment uses a **single-container, single-process** model:

```
┌─────────────────────────────────────────────┐
│  Docker Container (port 8080)               │
│                                             │
│  Streamlit (app.py)                         │
│    ├── Renders chat UI                      │
│    ├── Calls ReportGenerator directly       │
│    │   (no HTTP, no FastAPI)                │
│    └── Pipeline runs in-process             │
│                                             │
│  Health check: /_stcore/health              │
└─────────────────────────────────────────────┘
```

`app.py` integrates the agent pipeline directly into the Streamlit process, eliminating the need for a separate FastAPI backend. This avoids port conflicts on single-container platforms like Digital Ocean App Platform.

The FastAPI backend (`main.py`) and HTTP-based frontend (`streamlit_app.py`) remain available for local development and headless API access.

## Pipeline Architecture

The pipeline is a linear LangGraph `StateGraph` with 5 nodes. Each node receives the full `PipelineState` and returns a partial state update dict.

```
                                External Service
                                ────────────────
┌──────────────────────┐
│ 1. fetch_companies_  │ ──►    Companies House REST API
│    house             │        GET /company/{number}
│                      │        GET /company/{number}/officers
└──────────┬───────────┘
           │ company_profile_text, company_metadata
           ▼
┌──────────────────────┐
│ 2. generate_queries  │ ──►    OpenAI GPT (temp=0.4, JSON mode)
│                      │        Generates exactly 5 targeted queries
│                      │        informed by SIC codes, status, insolvency
└──────────┬───────────┘
           │ search_queries: list[str] (max 5)
           ▼
┌──────────────────────┐
│ 3. execute_searches  │ ──►    SerpAPI Google Search + News
│    (parallel)        │        5 web queries + 1 news query
│                      │        ALL executed concurrently via asyncio.gather
│                      │        Returns snippets, Knowledge Graph, news
└──────────┬───────────┘
           │ search_results: list[dict]
           ▼
┌──────────────────────┐
│ 4. summarize_searches│ ──►    OpenAI GPT (temp=0.2)
│                      │        Structured briefing with [Source: URL]
│                      │        7 sections: business, competition,
│                      │        quality, news, outlook, sector, risk
└──────────┬───────────┘
           │ search_summary: str
           ▼
┌──────────────────────┐
│ 5. synthesize_report │ ──►    OpenAI GPT (temp=0.2, JSON mode)
│                      │        Final UnderwritingReport JSON
│                      │        Merges CH filing + search briefing
└──────────────────────┘
```

### Latency Optimisations

| Optimisation | Impact |
|---|---|
| **Parallel SerpAPI execution** — all 6 queries (5 web + 1 news) run concurrently via `asyncio.gather` | ~30-35s saved vs sequential |
| **Reduced query count** — LLM generates 5 focused queries (hard-capped) instead of 8-10 | ~10-15s saved on search + summarisation |
| **Single-process deployment** — `app.py` calls pipeline directly, no HTTP round-trip to FastAPI | ~1-2s saved per request |

## State Schema

```python
class PipelineState(TypedDict):
    company_number: str                          # Input
    company_name: str                            # Input
    company_profile_text: str                    # Node 1: formatted CH profile
    company_metadata: dict                       # Node 1: structured metadata
    search_queries: list[str]                    # Node 2: LLM-generated queries (max 5)
    search_results: list[dict]                   # Node 3: normalised SerpAPI results
    search_summary: str                          # Node 4: LLM briefing with citations
    final_report: dict                           # Node 5: serialised UnderwritingReport
    errors: Annotated[list[str], operator.add]   # Accumulated across all nodes
```

## Report Schema

The final `UnderwritingReport` contains three core sections, each designed to be **explainable and grounded**:

### Section A: Business Model Summary

| Field | Type | Description |
|---|---|---|
| `description` | str | What the company does, grounded in evidence |
| `revenue_model` | str | How it generates revenue (subscriptions, fees, etc.) |
| `key_products_services` | list[str] | Core product/service offerings |
| `customer_segments` | list[str] | Who the customers are |
| `geographies` | list[str] | Key markets/regions served |
| `evidence_basis` | str | Which sources informed this section and their reliability |
| `citations` | list[Citation] | Source URLs backing claims |

### Section B: Competitive Landscape

| Field | Type | Description |
|---|---|---|
| `industry` | str | Industry classification |
| `market_position` | str | Leader, challenger, niche player — with reasoning |
| `competitors` | list[Competitor] | Named competitors with relevance explanation |
| `competition_degree` | enum | low / medium / high |
| `competitive_advantages` | list[str] | What gives this company an edge |
| `competitive_disadvantages` | list[str] | Where it is weaker vs peers |
| `reasoning` | str | Evidence-backed analysis of WHY that degree matters |
| `evidence_basis` | str | Source reliability assessment |

### Section C: Company Quality Signals

| Field | Type | Description |
|---|---|---|
| `signals` | list[QualitySignal] | Each with sentiment, strength (strong/moderate/weak), source, URL, detail |
| `confidence` | enum | Overall confidence: high / medium / low |
| `signal_coverage_assessment` | str | Which categories are well-covered vs thin |
| `data_gaps` | list[str] | Where evidence is thin, absent, or conflicting |
| `conflicting_signals` | list[str] | Contradictions between sources |

### Additional Sections

- **Business Outlook** — Evidence-backed trajectory assessment
- **Sectoral Outlook** — Industry trends and market direction
- **Uncertainty Flags** — Missing data, conflicting evidence, low-confidence areas

## Entity Resolution

Before the pipeline runs, `EntityResolver` handles company identification:

1. **Input looks like a company number** (regex: `^[A-Z]{0,2}\d{6,8}$`) → direct profile lookup
2. **Input is a name** → Companies House search API, then:
   - Single result → auto-resolve
   - Multiple results but only one active → auto-resolve to the active company
   - Multiple active but only one exact name match → auto-resolve
   - Otherwise → return disambiguation response with candidate list

## API Endpoints (FastAPI)

Available when running `main.py`. Not used by the standalone `app.py` deployment.

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | System health and API key status |
| `POST` | `/api/v1/search` | Search Companies House by name |
| `POST` | `/api/v1/report` | Generate report by name or number (handles disambiguation) |
| `POST` | `/api/v1/report/by-number` | Generate report by company number (direct) |

## Frontend

Two Streamlit entrypoints are available:

| File | Mode | Backend |
|---|---|---|
| `app.py` | Standalone (production) | Calls `ReportGenerator` directly in-process |
| `streamlit_app.py` | Client-server (dev) | Calls FastAPI backend via HTTP |

Both provide the same chat-based UI with report rendering, disambiguation handling, and sidebar quick examples.

## Key Design Decisions

### Single-container deployment

`app.py` integrates the pipeline directly into the Streamlit process. This eliminates the FastAPI HTTP hop and resolves health check failures on single-port platforms (Digital Ocean, Railway, Render). The tradeoff is tighter coupling between UI and pipeline code.

### Parallel search execution

All SerpAPI queries (5 web + 1 news) execute concurrently via `asyncio.gather`. Each query runs in a separate thread (`asyncio.to_thread`) since the SerpAPI client is synchronous. Failures in individual queries are caught and logged without blocking others.

### 5 focused queries instead of 8-10

The LLM generates exactly 5 queries (hard-capped with `[:5]`), each targeting a distinct area: business model, competitors, reviews, news, and sectoral outlook. This reduces SerpAPI cost and latency while maintaining coverage across all report sections.

### LangGraph over custom agent loop

The explicit `StateGraph` provides a typed state contract (`PipelineState`), per-node inspectable I/O, automatic error accumulation via `Annotated[list[str], operator.add]`, and trivial extensibility (add a node + an edge).

### Separate summariser node

Raw search results from 5 queries can exceed 15,000 tokens. The summariser distils them into a ~3,000 token briefing with enforced `[Source: URL]` citations, keeping the synthesis prompt focused and debuggable.

### No fallbacks

SerpAPI is the sole search provider. OpenAI is the sole synthesis engine. For a production underwriting system, reliability and auditability outweigh zero-cost fallback coverage.

### Signal strength assessment

Each quality signal carries a `strength` field (strong/moderate/weak) based on source reliability and corroboration. The `signal_coverage_assessment` field provides a meta-assessment of whether enough data exists to be confident. This explicitly surfaces uncertainty rather than hiding it.

## Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `COMPANIES_HOUSE_API_KEY` | Yes | — | Companies House API key |
| `COMPANIES_HOUSE_API_URL` | No | `https://api-sandbox.company-information.service.gov.uk` | API base URL (sandbox or live) |
| `SERP_API_KEY` | Yes | — | SerpAPI key |
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `OPENAI_MODEL` | No | `gpt-4o` | OpenAI model identifier |
| `MAX_AGENT_ITERATIONS` | No | `8` | Max pipeline iterations |
| `REQUEST_TIMEOUT` | No | `30` | HTTP request timeout (seconds) |

## Future Improvements

1. **Caching** — Redis or in-memory cache for CH profiles (24h TTL) and search results (1h TTL)
2. **Conditional graph edges** — Insolvency flag triggers a dedicated risk deep-dive node
3. **FCA register integration** — Automated FCA authorisation check for financial services companies
4. **Streaming responses** — Progressive report rendering as each node completes
5. **Merge summarise + synthesise** — Single LLM call to further reduce latency
6. **API authentication** — OAuth2 or API key auth on FastAPI endpoints
7. **Rate limiting** — Per-user SerpAPI quota protection
8. **Report persistence** — PostgreSQL storage with version comparison over time

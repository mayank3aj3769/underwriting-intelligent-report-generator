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
├── main.py                    FastAPI application entry point
├── config.py                  Environment variable loader (Settings singleton)
├── cli.py                     CLI tool for report generation
├── streamlit_app.py           Streamlit chat-based frontend
├── Dockerfile                 Container image definition
├── requirements.txt           Python dependencies
├── .env.example               Environment variable template
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
│                      │        Generates 8-10 targeted Google queries
│                      │        informed by SIC codes, status, insolvency
└──────────┬───────────┘
           │ search_queries: list[str]
           ▼
┌──────────────────────┐
│ 3. execute_searches  │ ──►    SerpAPI Google Search (per query, num=5)
│                      │        SerpAPI Google News (company name)
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

## State Schema

```python
class PipelineState(TypedDict):
    company_number: str                          # Input
    company_name: str                            # Input
    company_profile_text: str                    # Node 1: formatted CH profile
    company_metadata: dict                       # Node 1: structured metadata
    search_queries: list[str]                    # Node 2: LLM-generated queries
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

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | System health and API key status |
| `POST` | `/api/v1/search` | Search Companies House by name |
| `POST` | `/api/v1/report` | Generate report by name or number (handles disambiguation) |
| `POST` | `/api/v1/report/by-number` | Generate report by company number (direct) |

## Frontend

`streamlit_app.py` provides a chat-based interface that:

- Calls the FastAPI backend over HTTP
- Validates all responses with Pydantic models (client-side mirrors of `schemas/report.py`)
- Renders reports with structured sections, signal strength badges, source links
- Handles disambiguation with selectable company cards
- Shows system health in the sidebar

## Key Design Decisions

### LangGraph over custom agent loop

The explicit `StateGraph` provides a typed state contract (`PipelineState`), per-node inspectable I/O, automatic error accumulation via `Annotated[list[str], operator.add]`, and trivial extensibility (add a node + an edge). The tradeoff is an additional dependency tree.

### LLM-generated search queries

The LLM sees Companies House context (SIC codes, incorporation date, status, insolvency flag) and generates queries specific to the company. A fintech company gets queries about banking licences and FCA regulation, not generic "UK company overview". One additional LLM call (~$0.01) for significantly better search relevance.

### Separate summariser node

Raw results from 8-10 queries can exceed 30,000 tokens. The summariser distils them into a ~3,000 token briefing with enforced `[Source: URL]` citations, keeping the synthesis prompt focused and debuggable.

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

1. **Parallel search execution** — `asyncio.gather` over SerpAPI queries instead of sequential
2. **Caching** — Redis cache for CH profiles (24h TTL) and search results (1h TTL)
3. **Conditional graph edges** — Insolvency flag triggers a dedicated risk deep-dive node
4. **FCA register integration** — Automated FCA authorisation check for financial services companies
5. **Streaming responses** — SSE for progressive report rendering
6. **API authentication** — OAuth2 or API key auth on endpoints
7. **Rate limiting** — Per-user SerpAPI quota protection
8. **Report persistence** — PostgreSQL storage with version comparison over time

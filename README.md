# Underwriting Intelligence Report System

Automated underwriting intelligence reports for UK companies. Accepts a company name or Companies House registration number, gathers evidence from structured APIs, and produces a grounded, citation-backed report.

**Stack:** Python 3.11 | Streamlit | LangGraph | OpenAI GPT | SerpAPI | Companies House API

## What It Does

Given a UK company identifier, the system:

1. Resolves the company via Companies House (with disambiguation when names are ambiguous)
2. Fetches the official filing data — profile, officers, SIC codes, charges, insolvency history
3. Uses an LLM to generate 5 targeted search queries based on the company's context
4. Executes all queries in parallel via SerpAPI (Google Search + Google News)
5. Summarises all search results into a structured, citation-backed intelligence briefing
6. Synthesises the final report combining Companies House data and the research briefing

The report includes:

- **Business Model Summary** — What the company does, how it makes money, who its customers are
- **Competitive Landscape** — Degree of competition, market position, specific competitors with relevance, advantages and disadvantages
- **Company Quality Signals** — Each signal with strength rating (strong/moderate/weak), sentiment, source URL, and supporting detail
- **Business & Sectoral Outlook** — Evidence-backed trajectory assessments
- **Uncertainty Flags** — Missing data, conflicting evidence, low-confidence areas explicitly surfaced

Every claim references the source it came from. The system surfaces uncertainty rather than papering over it.

## Entrypoints

| Entrypoint | What it runs | Use case |
|---|---|---|
| `app.py` | **Streamlit (standalone)** — calls pipeline directly, no backend needed | Production deployment, Digital Ocean |
| `streamlit_app.py` | Streamlit frontend that calls FastAPI backend via HTTP | Local dev with separate API |
| `main.py` | FastAPI REST API | Headless API access, curl/Postman |
| `cli.py` | CLI report generator | Quick local testing |

## Quick Start (Local)

### Prerequisites

- Python 3.11+
- API keys for:
  - [Companies House](https://developer.company-information.service.gov.uk/) (free)
  - [SerpAPI](https://serpapi.com/) (free tier: 100 searches/month)
  - [OpenAI](https://platform.openai.com/) (paid)

### 1. Clone and set up

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

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
COMPANIES_HOUSE_API_KEY=your_key_here
COMPANIES_HOUSE_API_URL=https://api.company-information.service.gov.uk
SERP_API_KEY=your_serpapi_key
OPENAI_API_KEY=your_openai_key
```

Set `COMPANIES_HOUSE_API_URL` to `https://api-sandbox.company-information.service.gov.uk` for sandbox testing.

### 3. Run the standalone Streamlit app

```bash
streamlit run app.py --server.port 8080
```

Open http://localhost:8080 for the chat interface. No separate backend required.

### 4. Run the FastAPI backend (alternative)

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- API: http://localhost:8000
- Swagger docs: http://localhost:8000/docs

Then in a separate terminal:

```bash
streamlit run streamlit_app.py --server.port 8501
```

### 5. Run via CLI

```bash
python cli.py 08804411
python cli.py "REVOLUT LTD"
python cli.py "Barclays"
```

## Deployment (Digital Ocean App Platform)

The app is designed for single-container deployment. The Dockerfile runs `app.py` (standalone Streamlit) on port **8080** with the agent pipeline integrated directly — no separate backend service.

### Build and run locally

```bash
docker build -t underwriting-intel .
docker run -p 8080:8080 --env-file .env underwriting-intel
```

Open http://localhost:8080

### Deploy to Digital Ocean

1. Push your repo to GitHub.

2. In Digital Ocean App Platform, create a new app from your repo.

3. Set the **HTTP port** to `8080` in the app settings.

4. Add environment variables in the Digital Ocean dashboard:

```
COMPANIES_HOUSE_API_KEY=your_key
COMPANIES_HOUSE_API_URL=https://api.company-information.service.gov.uk
SERP_API_KEY=your_key
OPENAI_API_KEY=your_key
```

5. The health check will automatically hit `http://localhost:8080/_stcore/health` (Streamlit's built-in health endpoint).

6. Deploy. The container starts Streamlit on port 8080, which is the same port Digital Ocean health checks will probe.

### Why a single service?

The previous architecture ran FastAPI (port 8000) + Streamlit (port 8501) as two processes. This caused health check failures on platforms that expect a single port. `app.py` eliminates the HTTP hop by calling `ReportGenerator` directly from the Streamlit process.

## API Reference (FastAPI backend)

These endpoints are available when running `main.py` (not used by `app.py`).

### `GET /health`

Returns system status and API key configuration.

### `POST /api/v1/search`

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Revolut"}'
```

### `POST /api/v1/report`

```bash
curl -X POST http://localhost:8000/api/v1/report \
  -H "Content-Type: application/json" \
  -d '{"identifier": "REVOLUT LTD"}'
```

### `POST /api/v1/report/by-number`

```bash
curl -X POST http://localhost:8000/api/v1/report/by-number \
  -H "Content-Type: application/json" \
  -d '{"company_number": "08804411"}'
```

## Running Tests

```bash
python -m tests.test_e2e
```

Results are written to `tests/test_results.txt`.

## Configuration Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `COMPANIES_HOUSE_API_KEY` | Yes | — | Companies House API key |
| `COMPANIES_HOUSE_API_URL` | No | Sandbox URL | `https://api.company-information.service.gov.uk` for live |
| `SERP_API_KEY` | Yes | — | SerpAPI key for Google Search + News |
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `OPENAI_MODEL` | No | `gpt-4o` | Model identifier |
| `REQUEST_TIMEOUT` | No | `30` | HTTP timeout in seconds |

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full technical reference including pipeline design, state schema, report schema, and design decisions.

# Tech Market-Check

An autonomous agent that searches LinkedIn for jobs using a given **technology** (any
keyword you set, e.g. `langchain`, `react`, `rust`) and saves each posting as a clean
markdown file, including a short summary of *how* the role uses that tech. Read-only: it
collects market data, it does **not** apply.

Built as a tool-calling agent with [LangChain `create_agent`](https://docs.langchain.com/oss/python/langchain/agents),
DeepSeek, and async Playwright, the LLM drives a real browser through tools.

## Why

The market signal for a technology, who's hiring for it, where, and how they use it, is
scattered across LinkedIn job posts and locked behind its UI. This is the **first, raw
extraction step**: it pulls that signal out of LinkedIn into plain, portable markdown files
that live outside LinkedIn.

Once the data is in your own `.md` files, it's yours to transform downstream, analysis,
trend tracking, dashboards, or feeding another tool. This repo intentionally does only that
one job: clean extraction.

## Features

- **Filter-driven**: technology (keyword), location, radius, and recency come from `instructions.md`.
- **One markdown file per job**: posting date, title, company, full description, and a
  tech-usage summary.
- **Real posting dates**: converts LinkedIn's "2 weeks ago" into an actual date.
- **Cross-run dedup**: skips jobs already collected, so re-runs only add new ones.

## Requirements

- **Python 3.10+** (tested on 3.13)
- **A DeepSeek API key**: set as `DEEPSEEK_API_KEY` in a `.env` file
- **A LinkedIn account**: logged in once in the browser profile the agent opens (it runs as you)
- **Python packages:** `langchain`, `langchain-deepseek`, `langgraph`, `langchain-core`, `playwright`, `python-dotenv`
- **Playwright's Chromium browser**: installed with `playwright install chromium`
- An internet connection

## Setup

```bash
pip install -r requirements.txt
playwright install chromium
echo DEEPSEEK_API_KEY=your-key-here > .env
```

Then run the agent once and log into LinkedIn in the Chromium window it opens, the session
persists for future runs.

## Usage

```bash
python agent-tech-market-check.py
```

## Configuration: `instructions.md`

```
## Search filter
keywords: langchain
location: United States
distance: 0
date_posted: 1 month
```

Change `keywords` to check the market for any technology.

## Output

```
data/<tech>-<location>/<job_id>-<company>.md
```

```md
Posted: 2026-06-17 (2 weeks ago)

# Senior AI Engineer
Company: Acme

## Company description
...

## Job description
...

## langchain usage
Uses LangChain + LangGraph to build the customer-support agent; RAG over the docs...
```

## How it works

A single agent loop: `list_jobs → open_job → read_job → resolve_posting_date → save_job`.
The tools handle the browser; the LLM decides what to open, extracts the data, and writes
the tech-usage summary.

# Indeed Apply Now Agent

An autonomous agent that finds Indeed **Apply now** (Indeed Apply) jobs matching your search
and applies to them for you, filling the application form (including screening questions) from
your own instructions, and submitting. It saves a markdown record of every confirmed
application.

Built as a tool-calling agent with [LangChain `create_agent`](https://docs.langchain.com/oss/python/langchain/agents),
DeepSeek, and async Playwright, the LLM drives a real browser through tools. It is the Indeed
twin of the LinkedIn Easy Apply agent and shares the same structure and `instructions.md` format.

## Why

Applying to jobs one by one on Indeed is slow and repetitive, and the same screening questions
come up again and again. This agent automates the mechanical part: it searches, opens each new
job, walks Indeed's multi-step apply flow, answers the questions from a policy you control, and
submits, so you can cover far more of the market in far less time. It only applies to jobs that
offer **Apply now** (never external "Apply on company site" links), and never touches ones
you've already applied to.

## Features

- **Filter-driven**: keyword, location, recency, and remote come from `instructions.md`.
- **Answers screening questions**: from an answering policy + screening answers you set.
- **Skips already-applied jobs**: reads the "Applied" state from the results list.
- **Skips external applications**: only completes Indeed Apply ("Apply now"), not company-site links.
- **Walks the SmartApply wizard**: handles Indeed's multi-step flow, including when it opens in a new tab.
- **Saves a record**: one markdown file per confirmed application, under `data/<keyword>-<location>/`.

## Requirements

- **Python 3.10+** (tested on 3.13)
- **A DeepSeek API key**: set as `DEEPSEEK_API_KEY` in a `.env` file
- **An Indeed account**: logged in once in the browser profile the agent opens (it runs as you)
- **Your resume**: placed under `data/input/` and referenced by `resume:` in `instructions.md`
- **Python packages:** `langchain`, `langchain-deepseek`, `langgraph`, `langchain-core`, `playwright`, `python-dotenv`, `pypdf`
- **Playwright + real Chrome**: `playwright install chromium` (the agent launches the `chrome` channel)
- An internet connection

## Setup

```bash
pip install -r requirements.txt
playwright install chromium
cp .env.example .env                          # then add your DEEPSEEK_API_KEY
cp instructions.example.md instructions.md    # then fill in your filter + answers
# put your resume under data/input/ and point `resume:` at it in instructions.md
```

Run the agent once and log into Indeed in the Chrome window it opens — the session persists in
`../.indeed-profile` for future runs. If Indeed shows a Cloudflare / "verify you are human"
check, solve it in that window and the agent will continue.

## Usage

```bash
python agent-indeed-apply-now.py --out .
```

`--out` is the base folder for results; files land under
`<out>/agent-indeed-apply-now/data/<keyword>-<location>/`.

## Configuration: `instructions.md`

```
## Search filter
keywords: react
location: United States
date_posted: 1 week
remote: true
resume: data/input/resume.pdf
limit: 10
```

The `## Answering policy` and `## Screening question answers` sections tell the agent how to
answer the application questions. See `instructions.example.md`.

## Output

```
data/<keyword>-<location>/<job_id>-<company>.md
```

## How it works

A single agent loop: `list_jobs → open_job → read_job → open_apply → read_form → fill →
click_next → submit_application`. The tools handle the browser; the LLM decides what to open,
how to answer each field, and when the application is submitted. Because Indeed's apply flow is
a multi-step SmartApply wizard (contact info → resume → employer questions → review) that can
open in a new tab, the form tools act on a tracked "apply page" that may differ from the search
page.

## Notes

- **It submits real applications.** Review your `instructions.md` before running.
- **Selectors are best-effort.** Indeed changes its markup and runs aggressive bot detection;
  if the layout shifts, the DOM selectors in `agent-indeed-apply-now.py` may need updating.
- Nothing personal is committed: `.env`, `resume.pdf`, `instructions.md`, `data/`, and the
  browser profile are all git-ignored.

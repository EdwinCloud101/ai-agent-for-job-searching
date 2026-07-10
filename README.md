# AI Agent for Job Searching

Autonomous agents that drive a real browser to search and apply to jobs. A supervisor
coordinates three specialists — **tech market-check** (collects market data on a
technology), **LinkedIn Easy Apply**, and **Indeed Apply Now** — built with LangChain,
LangGraph, and Playwright.

## Demo

https://github.com/user-attachments/assets/911574ad-c56d-41b8-816a-05406d9884fd



## Run

```bash
pip install -r job-opportunity-manager/requirements.txt
playwright install chrome

python job-opportunity-manager/job-opportunity-manager.py \
  --out ./output \
  "only apply to Easy Apply jobs on LinkedIn"
```

The manager reads the goal and delegates to the right agent(s). First set
`DEEPSEEK_API_KEY` in a `.env` file, and give each agent an `instructions.md`
(copy its `instructions.example.md`) with your search filters and answers.

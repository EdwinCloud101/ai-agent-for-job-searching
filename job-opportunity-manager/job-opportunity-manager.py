"""Job Opportunity Manager — LangGraph supervisor over two worker agents.

tech_market_check collects LinkedIn market data; linkedin_easy_apply applies.
Both share one Playwright browser launched here and run sequentially.

Run:  python job-opportunity-manager.py --out <base-folder> "collect market data, then apply"
"""

import asyncio
import importlib.util
import json
import os
import re
import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
from langgraph_supervisor import create_supervisor
from playwright.async_api import async_playwright

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)  # ai-agent-for-job-searching/: workers + shared .env/profile live here

load_dotenv(os.path.join(REPO_ROOT, ".env"))

USER_DATA_DIR = os.path.join(REPO_ROOT, ".linkedin-profile")   # LinkedIn agents share this
INDEED_DATA_DIR = os.path.join(REPO_ROOT, ".indeed-profile")   # Indeed agent uses its own login
RECURSION_LIMIT = 1000


def load_worker(folder: str, filename: str):
    # Dash-named standalone scripts can't be imported normally, so load by path. Both workers
    # share the one library at the ai-agent-for-job-searching root (each adds it to sys.path itself).
    path = os.path.join(REPO_ROOT, folder, filename)
    mod_name = folder.replace("-", "_")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


market = load_worker("agent-tech-market-check", "agent-tech-market-check.py")
applyer = load_worker("agent-linkedin-easy-apply", "agent-linkedin-easy-apply.py")
indeed = load_worker("agent-indeed-apply-now", "agent-indeed-apply-now.py")


async def _launch(p, profile_dir):
    # one real-Chrome persistent context per profile (LinkedIn + Indeed have separate logins)
    ctx = await p.chromium.launch_persistent_context(
        profile_dir,
        headless=False,
        channel="chrome",
        args=["--disable-blink-features=AutomationControlled"],
        ignore_default_args=["--enable-automation"],
    )
    await ctx.add_init_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return ctx

SUPERVISOR_PROMPT = (
    "You are the Job Opportunity Manager, a supervisor coordinating three specialist agents "
    "(the two LinkedIn agents share one browser; Indeed uses its own). You do NOT browse or use "
    "any tools yourself — you only delegate, then report.\n\n"
    "Your team:\n"
    "  - tech_market_check: SEARCHES LinkedIn and COLLECTS market data on job postings that use "
    "a given technology, saving one file per job. It NEVER applies to anything.\n"
    "  - linkedin_easy_apply: APPLIES to LinkedIn 'Easy Apply' jobs and saves a confirmation file "
    "per submitted application.\n"
    "  - indeed_apply_now: APPLIES to Indeed 'Apply now' jobs and saves a confirmation file per "
    "submitted application, using its OWN separate Indeed browser/login.\n\n"
    "Each specialist already has its own fixed configuration (keyword, location, targets) from its "
    "own instructions.md. You do not configure them — you decide WHICH to delegate to and in what "
    "ORDER, one at a time.\n\n"
    "Default plan, unless the user's request says otherwise: FIRST delegate to tech_market_check to "
    "collect current market data, THEN delegate to linkedin_easy_apply to apply. Delegate to exactly "
    "one agent at a time and let it fully finish (it will reply DONE) before the next step. When all "
    "requested work is complete, STOP and give a short final summary of what each agent accomplished."
)

async def select_agents(goal: str):
    # Read the goal and decide which agents are needed, so only the required browsers open.
    # Uses the LLM (handles negations like "do not use linkedin"); defaults to all on failure.
    prompt = (
        "Decide which agents this request needs. Respect negations.\n"
        "- tech_market_check: collects LinkedIn market data (no applying)\n"
        "- linkedin_easy_apply: applies to jobs on LinkedIn\n"
        "- indeed_apply_now: applies to jobs on Indeed\n"
        f'Request: "{goal}"\n'
        'Reply with ONLY a JSON object, no prose, e.g. '
        '{"tech_market_check": false, "linkedin_easy_apply": false, "indeed_apply_now": true}'
    )
    try:
        resp = await market.build_llm(market.LLM_MODEL, market.LLM_PROVIDER).ainvoke(prompt)
        m = re.search(r"\{.*\}", getattr(resp, "content", "") or "", re.S)
        sel = json.loads(m.group(0))
        want = (bool(sel.get("tech_market_check")), bool(sel.get("linkedin_easy_apply")),
                bool(sel.get("indeed_apply_now")))
        return want if any(want) else (True, True, True)
    except Exception as e:
        print(f"[manager] router failed ({e!r}); running all agents", flush=True)
        return (True, True, True)


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Job Opportunity Manager supervisor")
    parser.add_argument("--out", required=True,
                        help="base output folder; each agent writes under <out>/<agent-name>/data")
    parser.add_argument("goal", nargs="*", help="what to do; if omitted, nothing runs")
    args = parser.parse_args()
    goal = " ".join(args.goal).strip()
    if not goal:
        print(
            'no goal given — nothing to do. Pass one, e.g.:\n'
            '  python job-opportunity-manager.py --out ./results "collect market data, then apply"',
            flush=True,
        )
        return
    out_base = os.path.abspath(args.out)
    print(f"[manager] goal: {goal}", flush=True)
    print(f"[manager] output base: {out_base}", flush=True)

    want_market, want_linkedin, want_indeed = await select_agents(goal)
    need_li = want_market or want_linkedin
    print(f"[manager] using -> market:{want_market} linkedin:{want_linkedin} indeed:{want_indeed}", flush=True)

    async with async_playwright() as p:
        agents, li_ctx, indeed_ctx = [], None, None

        # Only open the browser a selected agent actually needs (LinkedIn agents share one page).
        if need_li:
            li_ctx = await _launch(p, USER_DATA_DIR)
            li_page = li_ctx.pages[0] if li_ctx.pages else await li_ctx.new_page()
            await li_page.goto("https://www.linkedin.com", wait_until="domcontentloaded", timeout=60000)
            if want_market:
                agents.append(market.build_agent(li_page, out_base, li_ctx))
            if want_linkedin:
                agents.append(applyer.build_agent(li_page, out_base, li_ctx))
        if want_indeed:
            indeed_ctx = await _launch(p, INDEED_DATA_DIR)
            indeed_page = indeed_ctx.pages[0] if indeed_ctx.pages else await indeed_ctx.new_page()
            await indeed_page.goto("https://www.indeed.com", wait_until="domcontentloaded", timeout=60000)
            agents.append(indeed.build_agent(indeed_page, out_base, indeed_ctx))

        supervisor = create_supervisor(
            agents,
            model=market.build_llm(market.LLM_MODEL, market.LLM_PROVIDER),
            prompt=SUPERVISOR_PROMPT,
            output_mode="last_message",
            add_handoff_back_messages=True,
        ).compile()

        # one shared trace logger so every sub-agent's thoughts + tool calls/results
        # print to this console (workers' own loggers are bypassed under the supervisor)
        trace = market.TokenLogger(os.path.join(BASE_DIR, "tokens.txt"))
        try:
            result = await supervisor.ainvoke(
                {"messages": [{"role": "user", "content": goal}]},
                config={"recursion_limit": RECURSION_LIMIT, "callbacks": [trace]},
            )
            msgs = result.get("messages", []) if isinstance(result, dict) else []
            if msgs:
                print("\n[manager] FINAL:", str(getattr(msgs[-1], "content", ""))[:1000], flush=True)
        except Exception as e:
            print("\n[manager] error:", repr(e), flush=True)
        finally:
            parts = []
            if want_market:
                parts.append(f"tech_market_check collected {market.S.get('collected')}/{market.S.get('target')}")
            if want_linkedin:
                parts.append(f"linkedin_easy_apply applied {applyer.S.get('applied')}/{applyer.S.get('target')}")
            if want_indeed:
                parts.append(f"indeed_apply_now applied {indeed.S.get('applied')}/{indeed.S.get('target')}")
            print("\n[manager] " + "; ".join(parts), flush=True)
            if li_ctx:
                await li_ctx.close()
            if indeed_ctx:
                await indeed_ctx.close()


if __name__ == "__main__":
    asyncio.run(main())

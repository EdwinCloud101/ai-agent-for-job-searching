"""Tech Market-Check — a tool-calling agent that COLLECTS data on LinkedIn jobs using a
given technology (the search keyword); it never applies. Writes one markdown file per job
under data/<tech>-<location>/. Self-contained: reads its own instructions.md."""

import asyncio
import datetime
import os
import re
import sys
from urllib.parse import urlencode

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import tool
from playwright.async_api import async_playwright

# shared `library` lives at the ai-agent-for-job-searching root (one level up), used by both sub-agents
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from library import build_llm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)  # ai-agent-for-job-searching/: shared .env + .linkedin-profile live here
load_dotenv(os.path.join(REPO_ROOT, ".env"))
INSTRUCTION_FILE = os.path.join(BASE_DIR, "instructions.md")
TOKEN_LOG = os.path.join(BASE_DIR, "tokens.txt")
USER_DATA_DIR = os.path.join(REPO_ROOT, ".linkedin-profile")  # shared, logged-in LinkedIn session

COLLECT_TARGET = 50               # hardcoded for now
RECURSION_LIMIT = 500

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire",
    "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee",
    "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}


# --- Token logging -----------------------------------------------------------

class TokenLogger(BaseCallbackHandler):
    """Prepend 'in N out N' to TOKEN_LOG after every LLM call (newest on top)."""

    def __init__(self, path):
        self.path = path
        self.total_in = 0
        self.total_out = 0

    def on_llm_end(self, response, **kwargs):
        tin = tout = 0
        try:
            usage = getattr(response.generations[0][0].message, "usage_metadata", None) or {}
            tin = usage.get("input_tokens", 0)
            tout = usage.get("output_tokens", 0)
        except Exception:
            pass
        self.total_in += tin
        self.total_out += tout
        old = ""
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                old = f.read()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(f"in {tin} out {tout}\n" + old)


# --- Plain helpers (no page / no AI) -----------------------------------------

def slugify(name: str) -> str:
    name = name.strip().lower().replace(" ", "-")
    name = re.sub(r'[\\/:*?"<>|]', "", name)
    name = re.sub(r"-{2,}", "-", name)
    return name.strip("-. ") or "unknown"


def location_slug(location: str) -> str:
    """'San Francisco, CA' -> 'san-francisco-california' (expand the state abbrev)."""
    parts = [p.strip() for p in location.split(",")]
    if len(parts) >= 2:
        city, st = parts[0], parts[1]
        full = US_STATES.get(st.lower(), st)
        return slugify(f"{city} {full}")
    return slugify(location)


def parse_duration_seconds(text: str) -> int:
    m = re.search(r"(\d+)\s*([a-z]*)", text.lower())
    if not m:
        return 0
    n, unit = int(m.group(1)), (m.group(2) or "h")
    if unit.startswith("mo"):
        return n * 2592000   # month(s) = 30d
    if unit.startswith("w"):
        return n * 604800    # week(s)
    if unit.startswith("d"):
        return n * 86400     # day(s)
    return n * 3600          # hour(s), default


def extract_section(text, header):
    out, cap = [], False
    for line in text.splitlines():
        if line.strip().startswith("## "):
            if cap:
                break
            cap = line.strip() == header
            continue
        if cap:
            out.append(line)
    return "\n".join(out).strip()


def read_setting(key: str, default: str = "", section: str = "## Search filter") -> str:
    """Read a single `key: value` line from the given instructions section."""
    if not os.path.isfile(INSTRUCTION_FILE):
        return default
    text = open(INSTRUCTION_FILE, encoding="utf-8").read()
    for line in extract_section(text, section).splitlines():
        line = line.strip()
        if ":" in line:
            k, _, v = line.partition(":")
            if k.strip().lower() == key:
                return v.strip() or default
    return default


def read_instruction(path: str) -> dict:
    """Parse the '## Search filter' section into LinkedIn URL params."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing instruction file: {path}")
    text = open(path, encoding="utf-8").read()
    raw = {}
    for line in extract_section(text, "## Search filter").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, _, v = line.partition(":")
        raw[k.strip().lower()] = v.strip()
    params = {}
    if raw.get("keywords"):
        params["keywords"] = raw["keywords"]
    if raw.get("location"):
        params["location"] = raw["location"]
    if raw.get("distance") not in (None, ""):
        params["distance"] = raw["distance"]  # radius in miles (0 = the exact city)
    sec = parse_duration_seconds(raw.get("date_posted", ""))
    if sec:
        params["f_TPR"] = f"r{sec}"
    return params


def collection_guide() -> str:
    return extract_section(open(INSTRUCTION_FILE, encoding="utf-8").read(), "## What to collect")


def relative_to_date(relative: str) -> str:
    """'2 weeks ago' -> absolute date 'YYYY-MM-DD' relative to today."""
    if not relative:
        return ""
    today = datetime.date.today()
    low = relative.lower()
    if "yesterday" in low:
        return (today - datetime.timedelta(days=1)).isoformat()
    if "today" in low or "just now" in low:
        return today.isoformat()
    m = re.search(r"(\d+)\s*(minute|hour|day|week|month|year)", low)
    if not m:
        return ""
    n, unit = int(m.group(1)), m.group(2)
    days = {"minute": 0, "hour": 0, "day": n, "week": 7 * n, "month": 30 * n, "year": 365 * n}[unit]
    return (today - datetime.timedelta(days=days)).isoformat()


# --- Async page helpers ------------------------------------------------------

async def first_text(page, selectors):
    for sel in selectors:
        el = await page.query_selector(sel)
        if el:
            t = (await el.inner_text()).strip()
            if t:
                return t
    return ""


# --- Run-time config + shared state ------------------------------------------

SEARCH_PARAMS = read_instruction(INSTRUCTION_FILE)
JOBS_URL = "https://www.linkedin.com/jobs/search/?" + urlencode(SEARCH_PARAMS)
TECH = SEARCH_PARAMS.get("keywords", "the technology")  # the tech/keyword from instructions.md
LLM_PROVIDER = read_setting("provider", "deepseek", section="## LLM")
LLM_MODEL = read_setting("model", "deepseek-v4-flash", section="## LLM")
LOCATION_SLUG = location_slug(SEARCH_PARAMS.get("location", "unknown"))
DATA_SUBDIR = f"{slugify(TECH)}-{LOCATION_SLUG}"
COLLECTION_GUIDE = collection_guide()

# Output paths are unknown until a base path is injected. Results go under
# <OUTPUT_BASE>/<agent-folder>/data/<tech>-<location>/.
OUTPUT_BASE = None
DATA_DIR = None
LOCATION_DIR = None

S = {"page": None, "context": None, "collected": 0, "target": COLLECT_TARGET, "cur": {}}


def set_output_base(base: str):
    global OUTPUT_BASE, DATA_DIR, LOCATION_DIR
    OUTPUT_BASE = base
    DATA_DIR = os.path.join(base, os.path.basename(BASE_DIR), "data")
    LOCATION_DIR = os.path.join(DATA_DIR, DATA_SUBDIR)


def _page():
    return S["page"]


def write_job_md(cur, tech_summary: str) -> str:
    """One markdown file per collected job under data/<tech>-<location>/."""
    if not LOCATION_DIR:
        raise RuntimeError("output base not set — call set_output_base() first")
    os.makedirs(LOCATION_DIR, exist_ok=True)
    name = f"{cur.get('job_id', 'unknown')}-{slugify(cur.get('company', 'unknown'))}.md"
    path = os.path.join(LOCATION_DIR, name)
    rel = (cur.get("posted") or "").strip()
    abs_date = cur.get("posted_date") or relative_to_date(rel)  # fallback if agent skipped the tool
    if abs_date and rel:
        posted_line = f"{abs_date} ({rel})"
    else:
        posted_line = abs_date or rel or "(unknown)"
    content = (
        f"Posted: {posted_line}\n\n"
        f"# {(cur.get('title') or '').strip()}\n"
        f"Company: {(cur.get('company') or '').strip()}\n\n"
        f"## Company description\n{(cur.get('company_description') or '').strip()}\n\n"
        f"## Job description\n{(cur.get('description') or '').strip()}\n\n"
        f"## {TECH} usage\n{(tech_summary or '').strip()}\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def already_collected(job_id: str) -> bool:
    """True if a .md for this job was already saved (avoid re-collecting across runs)."""
    if not os.path.isdir(LOCATION_DIR):
        return False
    prefix = f"{job_id}-"
    return any(f.startswith(prefix) and f.endswith(".md") for f in os.listdir(LOCATION_DIR))


# --- Async tools -------------------------------------------------------------

@tool
async def go_to_search() -> str:
    """Open the LinkedIn jobs search (filtered per instructions.md). Call once at the start."""
    page = _page()
    await page.goto(JOBS_URL, wait_until="domcontentloaded", timeout=60000)
    try:
        await page.wait_for_selector("li[data-occludable-job-id]", timeout=20000)
    except Exception:
        return "search opened but no jobs visible"
    return f"search opened: {page.url}"


@tool
async def list_jobs() -> str:
    """List visible jobs as 'job_id | COLLECTED|new | title'. Collect ONLY the 'new' ones —
    'COLLECTED' jobs were already saved in a previous run and must be skipped."""
    page = _page()
    for _ in range(5):
        try:
            await page.evaluate("() => { const i=document.querySelectorAll('li[data-occludable-job-id]'); if(i.length) i[i.length-1].scrollIntoView(); }")
        except Exception:
            pass
        await page.wait_for_timeout(800)
    out = []
    for li in await page.query_selector_all("li[data-occludable-job-id]"):
        jid = await li.get_attribute("data-occludable-job-id")
        t = ((await li.inner_text()) or "").replace("\n", " ").strip()[:70]
        if jid:
            state = "COLLECTED" if already_collected(jid) else "new"
            out.append(f"{jid} | {state} | {t}")
    return "JOBS (collect ONLY the 'new' ones):\n" + "\n".join(out[:25]) if out else "no jobs visible"


@tool
async def next_page(start: int) -> str:
    """Load the next page of results. start = 25, 50, 75, ... 'no more results' when exhausted."""
    page = _page()
    await page.goto(JOBS_URL + f"&start={start}", wait_until="domcontentloaded", timeout=60000)
    try:
        await page.wait_for_selector("li[data-occludable-job-id]", timeout=15000)
    except Exception:
        return "no more results"
    return f"page start={start} loaded"


@tool
async def open_job(job_id: str) -> str:
    """Open a job's detail pane by its id (resets the current-job context)."""
    page = _page()
    await page.goto(JOBS_URL + f"&currentJobId={job_id}", wait_until="domcontentloaded", timeout=60000)
    try:
        await page.wait_for_selector(".jobs-description__content, #job-details", timeout=20000)
    except Exception:
        return "opened but no description pane"
    await page.wait_for_timeout(1200)
    for sel in (".jobs-description__footer-button", "button.show-more-less-html__button"):
        b = await page.query_selector(sel)
        if b:
            try:
                await b.click()
                await page.wait_for_timeout(400)
            except Exception:
                pass
            break
    S["cur"] = {"job_id": str(job_id)}
    return f"opened job {job_id}"


@tool
async def read_job() -> str:
    """Read the open job: posting date ('X ago'), title, company, company description, and
    the full job description. Call before save_job."""
    page = _page()
    title = await first_text(page, [".job-details-jobs-unified-top-card__job-title", ".jobs-unified-top-card__job-title", "h1"])
    company = await first_text(page, [".job-details-jobs-unified-top-card__company-name", ".jobs-unified-top-card__company-name"])
    company_desc = await first_text(page, [".jobs-company__company-description", ".jobs-company__box", ".jobs-company"])
    desc = await first_text(page, ["#job-details", ".jobs-description__content", ".jobs-box__html-content"])
    posted = await page.evaluate(
        """() => {
            const tc = document.querySelector('.job-details-jobs-unified-top-card__primary-description-container')
                || document.querySelector('.job-details-jobs-unified-top-card__tertiary-description-container')
                || document.querySelector('.jobs-unified-top-card') || document.body;
            const t = tc.innerText || '';
            const m = t.match(/\\b\\d+\\s+(minute|hour|day|week|month|year)s?\\s+ago\\b/i)
                || t.match(/\\b(reposted|posted)\\b[^\\n]*\\bago\\b/i);
            return m ? m[0].trim() : '';
        }"""
    )
    S["cur"].update({"title": title, "company": company, "company_description": company_desc,
                     "description": desc, "posted": posted})
    return f"POSTED: {posted}\nTITLE: {title}\nCOMPANY: {company}\nDESC: {desc[:1200]}"


@tool
async def resolve_posting_date(relative_ago: str) -> str:
    """Convert a relative posting time like '2 weeks ago' into an absolute date
    (YYYY-MM-DD) based on today, and store it as the job's posting date. Pass the 'X ago'
    text from read_job. Call after read_job, before save_job."""
    rel = relative_ago or S["cur"].get("posted", "")
    d = relative_to_date(rel)
    S["cur"]["posted_date"] = d
    return d or f"could not parse a date from '{rel}'"


@tool
async def save_job(tech_summary: str) -> str:
    """Save the open job to a markdown file, with your summary of HOW the role uses the
    target technology (the search keyword). Call read_job first. Counts toward the target."""
    cur = S["cur"]
    if not cur.get("description"):
        return "call read_job first — nothing captured yet"
    if already_collected(cur.get("job_id", "")):
        return "already collected in a previous run — skip this job and pick a 'new' one"
    path = write_job_md(cur, tech_summary)
    S["collected"] += 1
    base = os.path.basename(path)
    if S["collected"] >= S["target"]:
        return f"SAVED ({S['collected']}/{S['target']}) -> {base}. TARGET REACHED — reply DONE and stop."
    return f"SAVED ({S['collected']}/{S['target']}) -> {base}. Move on to the next job."


TOOLS = [go_to_search, list_jobs, next_page, open_job, read_job, resolve_posting_date, save_job]

SYSTEM = (
    f"You are a market-research agent that COLLECTS data on LinkedIn jobs using {TECH}. "
    "You do NOT apply to anything.\n\n"
    f"GOAL: collect {COLLECT_TARGET} NEW jobs, then reply DONE and stop.\n\n"
    "list_jobs marks each job 'COLLECTED' or 'new'. ONLY collect 'new' jobs — NEVER open a "
    "'COLLECTED' one (it was already saved in a previous run; it doesn't count and wastes "
    "time). Use next_page / list_jobs to get more 'new' jobs.\n\n"
    "Per new job: open_job -> read_job -> resolve_posting_date(the 'X ago' text from "
    f"read_job) -> write a short summary of HOW the role uses {TECH} (from the job "
    f"description) -> save_job(that summary). Every save_job counts toward the target. Do "
    f"not fabricate — base the {TECH} summary only on the job description you read.\n\n"
    f"WHAT TO COLLECT (per instructions):\n{COLLECTION_GUIDE}"
)

GOAL = (
    f"Collect {COLLECT_TARGET} LinkedIn '{SEARCH_PARAMS.get('keywords', '')}' jobs in "
    f"{SEARCH_PARAMS.get('location', '')}. Start by calling go_to_search, then list_jobs."
)

token_logger = TokenLogger(TOKEN_LOG)
llm = build_llm(LLM_MODEL, LLM_PROVIDER, callbacks=[token_logger])
agent = create_agent(llm, TOOLS, system_prompt=SYSTEM)


def build_agent(page, output_base, context=None):
    # Factory for the supervisor: set the injected output base, inject the shared browser,
    # then return the agent with a name (create_supervisor uses it for handoffs).
    set_output_base(output_base)
    S["page"] = page
    if context is not None:
        S["context"] = context
    return create_agent(llm, TOOLS, system_prompt=SYSTEM, name="tech_market_check")


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tech Market-Check agent")
    parser.add_argument("--out", required=True,
                        help="base output folder; results go under <out>/agent-tech-market-check/data")
    set_output_base(os.path.abspath(parser.parse_args().out))
    print(f"Search params: {SEARCH_PARAMS}", flush=True)
    print(f"Data dir: {LOCATION_DIR}\nTarget: {COLLECT_TARGET} jobs\n", flush=True)
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            USER_DATA_DIR,
            headless=False,
            channel="chrome",  # real Google Chrome (less bot-detectable)
            args=["--disable-blink-features=AutomationControlled"],
            ignore_default_args=["--enable-automation"],
        )
        await ctx.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        S["context"] = ctx
        S["page"] = ctx.pages[0] if ctx.pages else await ctx.new_page()
        await S["page"].goto("https://www.linkedin.com", wait_until="domcontentloaded", timeout=60000)
        print("Opened:", await S["page"].title(), flush=True)
        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": GOAL}]},
                config={"recursion_limit": RECURSION_LIMIT},
            )
            msgs = result.get("messages", []) if isinstance(result, dict) else []
            if msgs:
                print("\nAGENT FINAL:", str(getattr(msgs[-1], "content", ""))[:600], flush=True)
        except Exception as e:
            print("\nagent error:", repr(e), flush=True)
        print(
            f"\nCollected {S['collected']}/{S['target']}. "
            f"Tokens in {token_logger.total_in} out {token_logger.total_out}",
            flush=True,
        )
        await ctx.close()


if __name__ == "__main__":
    asyncio.run(main())

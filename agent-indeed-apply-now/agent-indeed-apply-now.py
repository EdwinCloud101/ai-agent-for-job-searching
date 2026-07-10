"""Indeed Apply Now — a tool-calling agent that APPLIES to "Apply now" (Indeed Apply) jobs,
submitting APPLY_TARGET new confirmed applications and saving one markdown file per submission
under data/<keyword>-<location>/. Config lives in instructions.md.

Indeed differs from LinkedIn in two ways that shape this file:
  * different search URL + DOM selectors (jobs are `[data-jk]` cards, 10 per page);
  * the "Apply now" flow is Indeed's multi-step SmartApply, which often opens in a NEW tab
    (smartapply.indeed.com) rather than a single in-page modal. The form tools therefore act
    on a tracked "apply page" (S['apply_page']) that may differ from the search page."""

import asyncio
import datetime
import os
import re
import sys
from urllib.parse import urlencode

# Force UTF-8: Indeed text and model replies contain non-cp1252 chars that crash the
# default Windows console encoding.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import tool
from playwright.async_api import async_playwright

# shared `library` lives at the ai-agent-for-job-searching root (one level up), used by all sub-agents
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from library import build_llm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)  # ai-agent-for-job-searching/: shared .env + .indeed-profile live here
load_dotenv(os.path.join(REPO_ROOT, ".env"))

INSTRUCTION_FILE = os.path.join(BASE_DIR, "instructions.md")  # search filter + answering policy
TOKEN_LOG = os.path.join(BASE_DIR, "tokens.txt")

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

def _short(text, n=300):
    return " ".join(str(text).split())[:n]


class TokenLogger(BaseCallbackHandler):
    """Log token usage to TOKEN_LOG, and print a full trace of the agent's actions:
    its reasoning (on_llm_end), each tool call (on_tool_start) and result (on_tool_end)."""

    def __init__(self, path):
        self.path = path
        self.total_in = 0
        self.total_out = 0
        self.step = 0

    def on_llm_end(self, response, **kwargs):
        tin = tout = 0
        try:
            msg = response.generations[0][0].message
            usage = getattr(msg, "usage_metadata", None) or {}
            tin = usage.get("input_tokens", 0)
            tout = usage.get("output_tokens", 0)
            text = (getattr(msg, "content", "") or "").strip()
            if text:
                print(f"[THINK] {_short(text, 500)}", flush=True)
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

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.step += 1
        name = (serialized or {}).get("name", "tool")
        arg = kwargs.get("inputs") or input_str
        print(f"[{self.step:03d}] -> {name}({_short(arg, 160)})", flush=True)

    def on_tool_end(self, output, **kwargs):
        out = getattr(output, "content", output)
        print(f"        = {_short(out, 300)}", flush=True)

    def on_tool_error(self, error, **kwargs):
        print(f"        ! tool error: {_short(error, 300)}", flush=True)


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
        return slugify(f"{city} {US_STATES.get(st.lower(), st)}")
    return slugify(location)


def parse_duration_days(text: str) -> int:
    """Indeed's date filter (`fromage`) is a number of days: 1, 3, 7, or 14. Convert any
    human duration to whole days (rounded up) so '1 week' -> 7, '48h' -> 2."""
    m = re.search(r"(\d+)\s*([a-z]*)", text.lower())
    if not m:
        return 0
    n, unit = int(m.group(1)), (m.group(2) or "d")
    if unit.startswith("mo"):
        days = n * 30
    elif unit.startswith("w"):
        days = n * 7
    elif unit.startswith("h"):
        days = max(1, (n + 23) // 24)
    else:
        days = n  # day(s), default
    return days


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
    """Parse the '## Search filter' section into Indeed URL params."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing instructions file: {path} (copy instructions.example.md)")
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
        params["q"] = raw["keywords"]      # Indeed keyword param
    if raw.get("location"):
        params["l"] = raw["location"]      # Indeed location param
    days = parse_duration_days(raw.get("date_posted", ""))
    if days:
        params["fromage"] = str(days)
    if raw.get("remote", "").lower() in ("true", "yes", "1", "on"):
        # Indeed encodes the Remote filter as a work-attribute in the `sc` param.
        params["sc"] = "0kf:attr(DSQF7);"
    return params


def read_resume_text(path: str) -> str:
    """Extract the resume's text so the agent can answer questions from real experience."""
    if not os.path.isfile(path):
        return ""
    try:
        reader = PdfReader(path)
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        return re.sub(r"\n{3,}", "\n\n", text).strip()
    except Exception as e:
        print(f"warning: could not read resume {path}: {e}", flush=True)
        return ""


def distill_instructions() -> str:
    """The answering policy + screening answers the agent uses to fill forms."""
    if not os.path.isfile(INSTRUCTION_FILE):
        return ""
    text = open(INSTRUCTION_FILE, encoding="utf-8").read()
    return (
        f"Blacklist (skip any job matching these):\n{extract_section(text, '## Blacklist')}\n\n"
        f"Answering policy:\n{extract_section(text, '## Answering policy')}\n\n"
        f"Screening answers:\n{extract_section(text, '## Screening question answers')}"
    ).strip()


# --- Async page helpers ------------------------------------------------------

async def first_text(page, selectors):
    for sel in selectors:
        el = await page.query_selector(sel)
        if el:
            t = (await el.inner_text()).strip()
            if t:
                return t
    return ""


async def applied_confirmed(page):
    # SmartApply lands on .../form/post-apply after a successful submission
    if "post-apply" in (page.url or ""):
        return True
    # scan every frame — the confirmation may render inside the SmartApply iframe
    for f in page.frames:
        try:
            hit = await f.evaluate(
                """() => {
                    const t = (document.body.innerText || '').toLowerCase();
                    return t.includes('application submitted')
                        || t.includes('application was submitted')
                        || t.includes('your application has been submitted')
                        || t.includes("we've sent your application")
                        || t.includes('was sent to')
                        || t.includes('has been submitted');
                }"""
            )
            if hit:
                return True
        except Exception:
            pass
    return False


async def close_any_modal(page):
    for sel in ("button[aria-label*='close' i]", "button[aria-label*='dismiss' i]"):
        b = await page.query_selector(sel)
        if b:
            try:
                await b.click()
                await page.wait_for_timeout(600)
            except Exception:
                pass


# deepAll/deepFind: querySelectorAll that also pierces open shadow roots — SmartApply
# renders some steps (e.g. demographic questions) inside web components, invisible to a
# plain querySelectorAll.
_DEEP_JS = """
    const deepAll = (root, sel) => {
        const out = [];
        const walk = (node) => {
            if (!node || !node.querySelectorAll) return;
            node.querySelectorAll(sel).forEach(e => out.push(e));
            node.querySelectorAll('*').forEach(e => { if (e.shadowRoot) walk(e.shadowRoot); });
        };
        walk(root);
        return out;
    };
    const deepFind = (root, sel) => deepAll(root, sel)[0] || null;
"""

# Tags visible form fields with data-fill-idx inside the SmartApply page (or any dialog).
# Radios/checkboxes are kept even when the native input is hidden, and are labeled with
# their question (fieldset legend / group aria-label).
_FIELDS_JS = "(maxN) => {" + _DEEP_JS + """
    // scan the whole document: SmartApply's beta layout keeps the form OUTSIDE the old
    // .ia-BasePage-content container, which made every step report zero fields. Prefer a
    // dialog only when it really holds form controls (inline-apply modal case).
    const dlg = document.querySelector('[role=dialog]');
    const root = (dlg && dlg.querySelector('input, select, textarea, [role=radio], [role=checkbox]'))
        ? dlg : document;
    const els = deepAll(root, 'input, select, textarea');
    const out = []; let idx = 0;
    for (const e of els) {
        const tag = e.tagName.toLowerCase();
        const type = (e.getAttribute('type')||'').toLowerCase();
        if (tag==='input' && type==='hidden') continue;
        const isChoice = (type==='radio' || type==='checkbox');
        const st = getComputedStyle(e); const r = e.getBoundingClientRect();
        const visible = (r.width>0 && r.height>0 && st.visibility!=='hidden' && st.display!=='none');
        if (!visible && !isChoice) continue;
        let kind = tag==='select'?'select':(tag==='textarea'?'textarea':(type||'text'));
        let own='';
        if (e.id){const l=(e.getRootNode()||document).querySelector('label[for="'+e.id+'"]'); if(l)own=l.innerText;}
        if (!own) own = e.getAttribute('aria-label')||e.getAttribute('placeholder')||e.value||'';
        own = own.replace(/\\s+/g,' ').trim();
        let label = own;
        if (isChoice) {
            let q = '';
            const fs = e.closest('fieldset');
            if (fs && fs.querySelector('legend')) q = fs.querySelector('legend').innerText;
            if (!q) { const g = e.closest('[role=group],[role=radiogroup]'); if (g) q = g.getAttribute('aria-label')||''; }
            q = (q||'').replace(/\\s+/g,' ').trim();
            label = (q ? q + ' = ' : '') + (own || 'option');
        }
        label = label.slice(0,140);
        let options=[];
        if (tag==='select') options=Array.from(e.options).map(o=>o.text.trim()).filter(Boolean).slice(0,20);
        e.setAttribute('data-fill-idx', idx);
        out.push({i:idx,kind,label,value:(e.value||'').slice(0,40),options,
                  checked:isChoice?!!e.checked:null});
        idx++; if(idx>=maxN)break;
    }
    // SmartApply demographic/screening steps use custom ARIA widgets, not native inputs
    for (const e of deepAll(root, '[role=radio], [role=checkbox]')) {
        if (idx>=maxN) break;
        if (e.tagName.toLowerCase()==='input') continue;  // native ones handled above
        const st = getComputedStyle(e); const r = e.getBoundingClientRect();
        if (!(r.width>0&&r.height>0) || st.visibility==='hidden' || st.display==='none') continue;
        let own = (e.getAttribute('aria-label') || e.innerText || '').replace(/\\s+/g,' ').trim();
        let q = '';
        const g = e.closest('[role=radiogroup],[role=group],fieldset');
        if (g) {
            const h = g.querySelector('legend, h1, h2, h3, [class*="label" i]');
            q = (g.getAttribute('aria-label') || (h ? h.innerText : '') || '').replace(/\\s+/g,' ').trim();
        }
        e.setAttribute('data-fill-idx', idx);
        out.push({i:idx, kind: e.getAttribute('role'),
                  label: ((q ? q + ' = ' : '') + (own || 'option')).slice(0,140),
                  value:'', options:[],
                  checked: e.getAttribute('aria-checked')==='true'});
        idx++;
    }
    return out;
}"""

# --- Run-time config + shared state ------------------------------------------

SEARCH_PARAMS = read_instruction(INSTRUCTION_FILE)
JOBS_URL = "https://www.indeed.com/jobs?" + urlencode(SEARCH_PARAMS)
KEYWORD = SEARCH_PARAMS.get("q", "jobs")
LOCATION = SEARCH_PARAMS.get("l", "")
RESUME_PDF = os.path.join(BASE_DIR, read_setting("resume", "data/input/resume.pdf"))
USER_DATA_DIR = os.path.join(REPO_ROOT, ".indeed-profile")  # shared, logged-in Indeed session
APPLY_TARGET = int(read_setting("limit", "10"))  # how many jobs to apply to this run
LLM_PROVIDER = read_setting("provider", "deepseek", section="## LLM")
LLM_MODEL = read_setting("model", "deepseek-v4-flash", section="## LLM")
DISTILLED = distill_instructions()
RESUME_TEXT = read_resume_text(RESUME_PDF)  # fed to the agent so it answers from real experience

# Output paths are unknown until a base path is injected. Results go under
# <OUTPUT_BASE>/<agent-folder>/data/<keyword>-<location>/.
OUTPUT_BASE = None
DATA_DIR = None
OUT_DIR = None

# 'page'      -> the Indeed search/viewjob page
# 'apply_page'-> the SmartApply page (may be a popup); form tools act here
S = {"page": None, "apply_page": None, "context": None, "applied": 0, "target": APPLY_TARGET, "cur": {}}


def set_output_base(base: str):
    global OUTPUT_BASE, DATA_DIR, OUT_DIR
    OUTPUT_BASE = base
    DATA_DIR = os.path.join(base, os.path.basename(BASE_DIR), "data")
    OUT_DIR = os.path.join(DATA_DIR, f"{slugify(KEYWORD)}-{location_slug(LOCATION)}")


def _page():
    return S["page"]


def _apage():
    """The page the apply form lives on: the tracked SmartApply popup if one opened,
    otherwise the main page (Indeed sometimes applies inline)."""
    return S.get("apply_page") or S["page"]


async def _aframe():
    """The FRAME hosting the apply form. SmartApply often renders inside an iframe
    (smartapply/indeedapply) — top-document queries then see no fields/buttons at all —
    so pick the frame by URL, else the frame with the most form controls."""
    page = _apage()
    for f in page.frames:
        u = (f.url or "").lower()
        if "smartapply" in u or "indeedapply" in u:
            return f
    best, best_n = page.main_frame, -1
    for f in page.frames:
        try:
            n = await f.evaluate(
                "() => document.querySelectorAll('input:not([type=hidden]), select, textarea, button').length")
        except Exception:
            continue
        if n > best_n:
            best, best_n = f, n
    return best


async def _goto(page, url, wait_sel=None, sel_timeout=20000):
    """Navigate, returning True if wait_sel appeared (or none was requested)."""
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
    except Exception as e:
        if "ERR_ABORTED" not in str(e):
            raise
        await page.wait_for_timeout(1200)
    if not wait_sel:
        return True
    try:
        await page.wait_for_selector(wait_sel, timeout=sel_timeout)
        return True
    except Exception:
        return False


def write_applied_md(cur) -> str:
    """One markdown confirmation per successful application under data/<keyword>-<location>/."""
    if not OUT_DIR:
        raise RuntimeError("output base not set — call set_output_base() first")
    os.makedirs(OUT_DIR, exist_ok=True)
    name = f"{cur.get('job_id', 'unknown')}-{slugify(cur.get('company', 'unknown'))}.md"
    path = os.path.join(OUT_DIR, name)
    when = datetime.date.today().isoformat()
    parts = [f"Applied: {when}", "", f"# {(cur.get('title') or '').strip()}",
             f"Company: {(cur.get('company') or '').strip()}", ""]
    cdesc = (cur.get("company_description") or "").strip()
    if cdesc:
        parts += [cdesc, ""]
    parts += ["---", "", (cur.get("description") or "").strip(), ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return path


# --- Async tools -------------------------------------------------------------

@tool
async def go_to_search() -> str:
    """Open the Indeed jobs search (filtered per instructions.md). Call once at the start."""
    page = _page()
    if not await _goto(page, JOBS_URL, "#mosaic-provider-jobcards, [data-jk]", 20000):
        return ("search opened but no jobs visible — Indeed may be showing a Cloudflare / "
                "'verify you are human' check; solve it in the browser window, then retry list_jobs")
    return f"search opened: {page.url}"


@tool
async def list_jobs() -> str:
    """List visible jobs as 'job_id | APPLIED|new | title'. ONLY open the 'new' ones —
    'APPLIED' jobs are already done and must be skipped."""
    page = _page()
    for _ in range(4):
        try:
            await page.evaluate("() => window.scrollBy(0, document.body.scrollHeight)")
        except Exception:
            pass
        await page.wait_for_timeout(700)
    data = await page.evaluate(
        """() => {
            const seen = new Set(); const out = [];
            document.querySelectorAll('[data-jk]').forEach(el => {
                const id = el.getAttribute('data-jk');
                if (!id || seen.has(id)) return;
                seen.add(id);
                const card = el.closest('.job_seen_beacon, li, .cardOutline') || el;
                const applied = /\\bApplied\\b/.test(card.innerText || '');
                const a = card.querySelector('h2.jobTitle a, a.jcs-JobTitle, h2 a') || el;
                const title = (a.innerText || card.innerText || '').replace(/\\s+/g,' ').trim().slice(0, 60);
                out.push({ id, applied, title });
            });
            const noResults = !!document.querySelector('.jobsearch-NoResult, [class*="NoResult"]');
            return { noResults, items: out };
        }"""
    )
    items = data["items"]
    if data["noResults"] and not items:
        return ("NO RESULTS on this page for the filter. Try next_page; if that is also empty, "
                "reply DONE and stop.")
    if not items:
        return ("no jobs visible — Indeed may be showing a human-verification / Cloudflare "
                "check. Solve it in the browser window, then retry list_jobs.")
    lines = [f"{it['id']} | {'APPLIED' if it['applied'] else 'new'} | {it['title']}" for it in items[:25]]
    return "JOBS (open ONLY the 'new' ones):\n" + "\n".join(lines)


@tool
async def next_page(start: int) -> str:
    """Load the next page of results. start = 10, 20, 30, ... 'no more results' when exhausted."""
    page = _page()
    if not await _goto(page, JOBS_URL + f"&start={start}", "[data-jk]", 15000):
        return "no more results"
    return f"page start={start} loaded"


@tool
async def open_job(job_id: str) -> str:
    """Open a job's full page by its id / job key (resets the current-job context)."""
    page = _page()
    url = f"https://www.indeed.com/viewjob?jk={job_id}"
    if not await _goto(page, url, "#jobDescriptionText, .jobsearch-JobComponent", 20000):
        return "opened but no description pane"
    await page.wait_for_timeout(1000)
    S["cur"] = {"job_id": str(job_id)}
    S["submit_disabled"] = 0
    print(f"[OPEN] job {job_id}", flush=True)
    return f"opened job {job_id}"


@tool
async def read_job() -> str:
    """Read the open job (title, company, company blurb, description). Call before applying."""
    page = _page()
    title = await first_text(page, [".jobsearch-JobInfoHeader-title", "h1.jobsearch-JobInfoHeader-title", "h1"])
    company = await first_text(page, ["[data-testid='inlineHeader-companyName']",
                                      ".jobsearch-CompanyInfoContainer a",
                                      "[data-company-name]"])
    company_desc = await first_text(page, [".jobsearch-CompanyInfoContainer",
                                           "[data-testid='companyInfo']"])
    desc = await first_text(page, ["#jobDescriptionText", ".jobsearch-JobComponent-description"])
    S["cur"].update({"title": title, "company": company, "company_description": company_desc, "description": desc})
    return f"TITLE: {title}\nCOMPANY: {company}\nDESC: {desc[:1000]}"


@tool
async def open_apply() -> str:
    """Open the 'Apply now' (Indeed Apply) flow for the current job. Reports if it's external
    ('Apply on company site' -> skip) or if there's no apply button. SmartApply may open in a
    new tab; this tool tracks it so the form tools act on the right page."""
    page = _page()
    ctx = S.get("context")
    before = set(ctx.pages) if ctx else set()

    # External applications render an anchor / button labelled 'Apply on company site'.
    ext = await page.evaluate(
        """() => {
            const t = (document.body.innerText || '').toLowerCase();
            return t.includes('apply on company site') && !t.includes('apply now');
        }"""
    )
    if ext:
        return "EXTERNAL application (apply on company site) — skip this job"

    btn = None
    for sel in ("#indeedApplyButton", "button.ia-IndeedApplyButton",
                "#applyButtonLinkContainer button", "button[aria-label*='Apply now' i]"):
        btn = await page.query_selector(sel)
        if btn:
            break
    if not btn:
        # Fall back to any visible button whose text is exactly 'Apply now'.
        btn_found = await page.evaluate(
            """() => {
                for (const b of document.querySelectorAll('button, a')) {
                    const t = (b.innerText || '').replace(/\\s+/g,' ').trim().toLowerCase();
                    if (t === 'apply now') { b.setAttribute('data-apply-btn','1'); return true; }
                }
                return false;
            }"""
        )
        if btn_found:
            btn = await page.query_selector("[data-apply-btn='1']")
    if not btn:
        return "no 'Apply now' button (maybe already applied or external) — skip this job"

    try:
        await btn.click()
        await page.wait_for_timeout(2500)
    except Exception as e:
        return f"could not open Apply: {e}"

    # SmartApply frequently opens in a popup. Track whichever page hosts the form.
    S["apply_page"] = None
    if ctx:
        new_pages = [p for p in ctx.pages if p not in before]
        for p in new_pages:
            try:
                await p.wait_for_load_state("domcontentloaded", timeout=15000)
            except Exception:
                pass
            if "smartapply" in (p.url or "") or "apply" in (p.url or ""):
                S["apply_page"] = p
                break
        if S["apply_page"] is None and new_pages:
            S["apply_page"] = new_pages[-1]
    await _apage().wait_for_timeout(1500)
    frames = [(f.url or "")[:70] for f in _apage().frames]
    print(f"[APPLY] page={_short(_apage().url, 80)} frames={frames}", flush=True)
    return "Apply flow opened — call read_form next"


@tool
async def read_form() -> str:
    """List the open apply form's fields and all visible buttons (Continue / Submit / etc.)."""
    page = _apage()
    frame = await _aframe()
    fields = await frame.evaluate(_FIELDS_JS, 40)
    lines = [
        f"{f['i']} | {f['kind']} | {f['label']} | value='{f['value']}'"
        + ("" if f.get("checked") is None else (" | CHECKED" if f["checked"] else " | unchecked"))
        + (f" | options={f['options']}" if f["options"] else "")
        for f in fields
    ]
    all_btns = await frame.evaluate(
        """() => {
            const out = [];
            document.querySelectorAll('button, [role=button]').forEach(b => {
                const st = getComputedStyle(b); const r = b.getBoundingClientRect();
                if (!(r.width>0&&r.height>0)||st.visibility==='hidden'||st.display==='none') return;
                const t = (b.innerText || b.getAttribute('aria-label') || '').replace(/\\s+/g,' ').trim();
                if (t && t.length <= 40) out.push(t);
            });
            return [...new Set(out)].slice(0, 15);
        }"""
    )
    print(f"[FORM] page={_short(page.url, 80)} frame={_short(frame.url, 80)} "
          f"fields={len(fields)} buttons={all_btns}", flush=True)
    for ln in lines:
        print(f"[FIELD] {ln}", flush=True)
    page_text = ""
    if not fields:
        dbg = await frame.evaluate(
            "() => {" + _DEEP_JS + """
                const main = document.querySelector('main') || document.body;
                const hosts = [];
                const walk = (n) => n.querySelectorAll('*').forEach(e => {
                    if (e.shadowRoot) { hosts.push(e.tagName.toLowerCase()); walk(e.shadowRoot); }
                });
                walk(document);
                return { text: (main.innerText || '').replace(/\\s+/g, ' ').slice(0, 700),
                         hosts: hosts.slice(0, 15) };
            }"""
        )
        print(f"[FORM-DEBUG] shadow-hosts={dbg['hosts']} text='{_short(dbg['text'], 400)}'", flush=True)
        page_text = "\nPAGE TEXT (no fields found; questions may be optional): " + dbg["text"]
    return (
        "FIELDS:\n" + ("\n".join(lines) if lines else "(none)")
        + "\nBUTTONS: " + (", ".join(all_btns) or "none")
        + page_text
    )


@tool
async def fill_field(index: int, value: str) -> str:
    """Type text into the text/number/textarea field with this index."""
    try:
        frame = await _aframe()
        await frame.fill(f'[data-fill-idx="{index}"]', str(value)[:200], timeout=6000)
        return f"filled {index}"
    except Exception as e:
        return f"fill failed: {e}"


@tool
async def choose_option(index: int) -> str:
    """Select the radio/checkbox with this index and VERIFY it took. Only trust a 'checked'
    reply — a FAILED reply means the input is still unchecked."""
    page = _apage()
    frame = await _aframe()
    sel = f'[data-fill-idx="{index}"]'

    async def checked():
        try:
            return await frame.evaluate(
                "(s) => {" + _DEEP_JS + """
                    const e = deepFind(document, s);
                    if (!e) return null;
                    if (e.tagName === 'INPUT') return e.checked;
                    return e.getAttribute('aria-checked') === 'true';
                }""", sel)
        except Exception:
            return None

    state = await checked()
    if state is None:
        return f"no element with index {index} — call read_form again (indices reset per step)"
    if state:
        return f"{index} already checked"

    # 1) real mouse click on the label (or the input itself) — most human-like
    try:
        h = await frame.evaluate_handle(
            "(s) => {" + _DEEP_JS + """
                const e = deepFind(document, s);
                if (!e) return null;
                let lab = e.id ? (e.getRootNode()||document).querySelector('label[for="'+e.id+'"]') : null;
                if (!lab) lab = e.closest('label');
                return lab || e;
            }""",
            sel,
        )
        el = h.as_element()
        if el:
            await el.click(timeout=4000)
            await page.wait_for_timeout(400)
    except Exception as e:
        print(f"[RADIO] {index}: label click failed: {_short(e, 160)}", flush=True)
    if await checked():
        print(f"[RADIO] {index}: checked via label click", flush=True)
        return f"checked {index} (label click, verified)"

    # 2) Playwright force-check the input directly
    try:
        await frame.check(sel, timeout=3000, force=True)
        await page.wait_for_timeout(300)
    except Exception as e:
        print(f"[RADIO] {index}: force check failed: {_short(e, 160)}", flush=True)
    if await checked():
        print(f"[RADIO] {index}: checked via force", flush=True)
        return f"checked {index} (force, verified)"

    # 3) React-safe: native checked setter + input/change/click events (SmartApply is React;
    # plain DOM clicks on styled widgets sometimes never reach its state)
    try:
        await frame.evaluate(
            "(s) => {" + _DEEP_JS + """
                const e = deepFind(document, s);
                if (!e) return;
                if (e.tagName === 'INPUT') {
                    const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'checked').set;
                    set.call(e, true);
                }
                e.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
                for (const t of ['input', 'change'])
                    e.dispatchEvent(new Event(t, { bubbles: true }));
            }""",
            sel,
        )
        await page.wait_for_timeout(400)
    except Exception as e:
        print(f"[RADIO] {index}: event dispatch failed: {_short(e, 160)}", flush=True)
    if await checked():
        print(f"[RADIO] {index}: checked via events", flush=True)
        return f"checked {index} (events, verified)"
    print(f"[RADIO] {index}: ALL strategies failed, still unchecked", flush=True)
    return (f"FAILED to check {index} — still unchecked after 3 strategies. Call read_form: "
            "the real control may be a different index or a button-styled option (try click_button "
            "with the option's text).")


@tool
async def select_dropdown(index: int, value: str) -> str:
    """Pick the option (by exact visible text) in the dropdown <select> with this index."""
    try:
        frame = await _aframe()
        await frame.select_option(f'[data-fill-idx="{index}"]', label=str(value), timeout=6000)
        return f"selected '{value}' in {index}"
    except Exception as e:
        return f"select failed: {e}"


# Reports whether the resume with this filename is present/selected on the apply page.
_RESUME_CHECK_JS = """(fname) => {
    const root = document.querySelector('.ia-BasePage-content')
        || document.querySelector('form')
        || document.querySelector('[role=dialog]')
        || document.querySelector('main') || document;
    const f = fname.toLowerCase();
    const cards = Array.from(root.querySelectorAll(
        '[data-testid*="resume"], [class*="resume"], label, li'
    ));
    for (const c of cards) {
        const t = (c.innerText || '').replace(/\\s+/g, ' ').trim();
        if (!t.toLowerCase().includes(f)) continue;
        let input = null;
        if (c.htmlFor) input = document.getElementById(c.htmlFor);
        if (!input) input = c.querySelector('input[type=radio], input[type=checkbox]');
        const selected = input ? input.checked
            : /selected|active|checked/i.test(c.className || '');
        return { found: true, selected: !!selected, text: t.slice(0, 120) };
    }
    if ((root.innerText || '').toLowerCase().includes(f))
        return { found: true, selected: true, text: fname };
    return { found: false, selected: false, text: '' };
}"""

_RESUME_PICK_JS = """(fname) => {
    const root = document.querySelector('.ia-BasePage-content')
        || document.querySelector('form')
        || document.querySelector('[role=dialog]')
        || document.querySelector('main') || document;
    for (const c of root.querySelectorAll(
        '[data-testid*="resume"], [class*="resume"], label, li'
    )) {
        const t = (c.innerText || '').toLowerCase();
        if (t.includes(fname.toLowerCase())) { c.click(); return true; }
    }
    return false;
}"""


@tool
async def upload_resume() -> str:
    """Ensure the resume from instructions.md is the attached one, comparing by filename:
    if it already matches, nothing is uploaded; if its card exists but isn't selected, it is
    selected; only when absent is the file uploaded."""
    page = _apage()
    frame = await _aframe()
    fname = os.path.basename(RESUME_PDF)
    state = await frame.evaluate(_RESUME_CHECK_JS, fname)
    if state["found"] and state["selected"]:
        return f"correct resume already attached ({state['text']}) — no upload needed"
    if state["found"]:
        await frame.evaluate(_RESUME_PICK_JS, fname)
        await page.wait_for_timeout(800)
        return f"'{fname}' was present but not selected — selected its card"
    n = 0
    for h in await frame.query_selector_all("input[type=file]"):
        try:
            await h.set_input_files(RESUME_PDF)
            n += 1
        except Exception:
            pass
    await page.wait_for_timeout(2500)
    state = await frame.evaluate(_RESUME_CHECK_JS, fname)
    if state["found"] and not state["selected"]:
        await frame.evaluate(_RESUME_PICK_JS, fname)
        await page.wait_for_timeout(800)
    if state["found"]:
        return f"uploaded '{fname}' ({n} input(s)) and it is now the attached resume"
    if n == 0:
        return (f"NO file input found and nothing shows '{fname}' — the correct resume is "
                "NOT attached. Indeed is using its stored profile resume. Look for an "
                "'Upload resume' / 'Replace' / 'Change' option via read_form and click_button.")
    return (f"uploaded '{fname}' to {n} input(s) but could not confirm it shows on the page "
            "— call read_form and verify before continuing.")


@tool
async def click_next() -> str:
    """Advance the apply form: click 'Continue' / 'Review your application' / 'Next'."""
    page = _apage()
    frame = await _aframe()
    hit = await frame.evaluate(
        """() => {
            const wants = ['continue', 'review your application', 'review', 'next', 'save and continue'];
            for (const b of document.querySelectorAll('button, [role=button]')) {
                const st = getComputedStyle(b); const r = b.getBoundingClientRect();
                if (!(r.width>0&&r.height>0)||st.visibility==='hidden'||st.display==='none') continue;
                if (b.disabled) continue;
                const t = (b.innerText || b.getAttribute('aria-label') || '').replace(/\\s+/g,' ').trim().toLowerCase();
                if (t.includes('submit')) continue;  // never fire submit from here
                if (wants.some(w => t === w || t.startsWith(w))) { b.click(); return t; }
            }
            return '';
        }"""
    )
    await page.wait_for_timeout(1800)
    print(f"[NEXT] clicked='{hit or 'none'}' now-at={_short(page.url, 90)}", flush=True)
    if hit:
        return f"advanced ('{hit}')"
    return "no next/continue button — maybe the Submit button is available now"


async def _dump_blocker(page, frame):
    """Capture everything that could explain a frozen/disabled submit: any dialog/modal,
    the submit button's own state, and the full review frame HTML. Prints a summary and
    writes the raw HTML to a file next to this script for inspection."""
    info = await frame.evaluate(
        "() => {" + _DEEP_JS + """
            const pick = (e) => e ? e.outerHTML.slice(0, 6000) : '';
            const dialogs = deepAll(document, '[role=dialog], [role=alertdialog], .modal, [class*="Modal" i], [class*="overlay" i]')
                .map(d => ({ cls: d.className || '', text: (d.innerText||'').replace(/\\s+/g,' ').slice(0,400), html: pick(d) }));
            let submit = null;
            for (const b of deepAll(document, 'button, [role=button]')) {
                const t = (b.innerText || b.getAttribute('aria-label') || '').toLowerCase();
                if (t.includes('submit')) {
                    submit = { text: (b.innerText||'').trim(), disabled: b.disabled,
                               ariaDisabled: b.getAttribute('aria-disabled'), html: pick(b) };
                    break;
                }
            }
            const main = document.querySelector('main') || document.body;
            return { url: location.href, dialogs, submit,
                     bodyText: (main.innerText||'').replace(/\\s+/g,' ').slice(0,1200),
                     frameHtml: (main.outerHTML||'').slice(0, 20000) };
        }"""
    )
    print(f"[BLOCKER] url={_short(info['url'], 100)}", flush=True)
    print(f"[BLOCKER] submit={info['submit']}", flush=True)
    print(f"[BLOCKER] dialogs found: {len(info['dialogs'])}", flush=True)
    for i, d in enumerate(info["dialogs"]):
        print(f"[BLOCKER] dialog[{i}] class='{d['cls']}' text='{_short(d['text'], 300)}'", flush=True)
    print(f"[BLOCKER] page text: {_short(info['bodyText'], 600)}", flush=True)
    # also list all frames (a fresh iframe — e.g. a captcha challenge — may have appeared)
    print(f"[BLOCKER] frames: {[_short(f.url, 80) for f in page.frames]}", flush=True)
    try:
        out = os.path.join(BASE_DIR, "blocker-dump.html")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(f"<!-- url: {info['url']} -->\n")
            fh.write(f"<!-- submit: {info['submit']} -->\n\n")
            for i, d in enumerate(info["dialogs"]):
                fh.write(f"<!-- DIALOG {i} class={d['cls']} -->\n{d['html']}\n\n")
            fh.write(f"<!-- MAIN FRAME -->\n{info['frameHtml']}\n")
        print(f"[BLOCKER] full HTML written to {out}", flush=True)
    except Exception as e:
        print(f"[BLOCKER] could not write dump file: {e}", flush=True)


# How long to wait for a human to solve a reCAPTCHA challenge before giving up on a job.
CAPTCHA_WAIT_SECONDS = 240


async def _show_human_banner(page, text="HUMAN NEEDED! Solve the CAPTCHA"):
    """Inject a large fixed RED bar across the top of the page so the human notices."""
    try:
        await page.evaluate(
            """(text) => {
                let el = document.getElementById('__human_needed_bar');
                if (!el) {
                    el = document.createElement('div');
                    el.id = '__human_needed_bar';
                    document.body.appendChild(el);
                }
                el.textContent = '🚨 ' + text + ' 🚨';
                el.style.cssText = [
                    'position:fixed','top:0','left:0','right:0','z-index:2147483647',
                    'background:#d40000','color:#fff','font-size:34px','font-weight:900',
                    'font-family:Arial,sans-serif','text-align:center','padding:18px 12px',
                    'letter-spacing:1px','box-shadow:0 3px 12px rgba(0,0,0,.5)',
                    'text-transform:uppercase','animation:hnblink 1s step-start infinite'
                ].join(';');
                if (!document.getElementById('__human_needed_style')) {
                    const s = document.createElement('style');
                    s.id = '__human_needed_style';
                    s.textContent = '@keyframes hnblink{50%{background:#7a0000}}';
                    document.head.appendChild(s);
                }
            }""",
            text,
        )
    except Exception:
        pass


async def _hide_human_banner(page):
    try:
        await page.evaluate(
            "() => { const e = document.getElementById('__human_needed_bar'); if (e) e.remove(); }")
    except Exception:
        pass

_SUBMIT_JS = """() => {
    let disabled = false;
    for (const b of document.querySelectorAll('button, [role=button]')) {
        const st = getComputedStyle(b); const r = b.getBoundingClientRect();
        if (!(r.width>0&&r.height>0)||st.visibility==='hidden'||st.display==='none') continue;
        const t = (b.innerText || b.getAttribute('aria-label') || '').replace(/\\s+/g,' ').trim().toLowerCase();
        if (!(t.includes('submit') && t.includes('application') || t === 'submit')) continue;
        if (b.disabled || b.getAttribute('aria-disabled') === 'true') { disabled = true; continue; }
        b.click(); return 'clicked';
    }
    return disabled ? 'disabled' : '';
}"""


@tool
async def submit_application() -> str:
    """Click the final 'Submit application'. Only a CONFIRMED submission counts and saves the .md file."""
    page = _apage()
    frame = await _aframe()
    result = await frame.evaluate(_SUBMIT_JS)
    if result == "disabled":
        # Indeed disables Submit until a reCAPTCHA challenge is solved. The agent can't
        # solve it, so pause for a HUMAN to solve it in the browser, polling until Submit
        # enables (then click it), or until we time out.
        print("[SUBMIT] submit DISABLED — likely a reCAPTCHA challenge. Dumping + waiting "
              "for a human to solve it in the browser.", flush=True)
        await _dump_blocker(page, frame)
        await _show_human_banner(page)
        print(f"\n[CAPTCHA] >>> SOLVE THE reCAPTCHA in the Chrome window. Waiting up to "
              f"{CAPTCHA_WAIT_SECONDS}s; I'll submit automatically once it enables. <<<\n", flush=True)
        waited = 0
        while waited < CAPTCHA_WAIT_SECONDS:
            await page.wait_for_timeout(3000)
            waited += 3
            frame = await _aframe()
            try:
                r = await frame.evaluate(_SUBMIT_JS)
            except Exception:
                r = ""
            if r == "clicked":
                print(f"[CAPTCHA] solved after ~{waited}s — Submit clicked.", flush=True)
                await _hide_human_banner(page)
                result = "clicked"
                break
            if waited % 30 == 0:
                print(f"[CAPTCHA] still waiting… {waited}/{CAPTCHA_WAIT_SECONDS}s", flush=True)
                await _show_human_banner(page)  # re-assert in case the page re-rendered
        if result != "clicked":
            await _hide_human_banner(page)
            print("[CAPTCHA] not solved in time — giving up on this job.", flush=True)
            return ("Submit stayed disabled — the reCAPTCHA was not solved in time. Call "
                    "skip_job and move on to the next job.")
    elif result != "clicked":
        print("[SUBMIT] no submit button found", flush=True)
        return "no Submit button yet — fill required fields and click_next until it appears"
    if result != "clicked":
        print("[SUBMIT] no submit button found", flush=True)
        return "no Submit button yet — fill required fields and click_next until it appears"
    await page.wait_for_timeout(2800)
    confirmed = await applied_confirmed(page)
    print(f"[SUBMIT] clicked, confirmed={confirmed} now-at={_short(page.url, 90)}", flush=True)
    if not confirmed:
        return "submitted but NOT confirmed — a required field may be unanswered. Call read_form and fix it."
    if not S["cur"].get("description"):
        return "CONFIRMED but read_job was not called first, so nothing was saved. Call read_job before applying next time."
    S["applied"] += 1
    base = os.path.basename(write_applied_md(S["cur"]))
    # Close the SmartApply popup (if any) and return focus to the search page.
    ap = S.get("apply_page")
    if ap and ap is not S["page"]:
        try:
            await ap.close()
        except Exception:
            pass
    S["apply_page"] = None
    print(f"[APPLIED {S['applied']}/{S['target']}] {S['cur'].get('company','?')} — "
          f"{S['cur'].get('title','?')}  ({base})", flush=True)
    if S["applied"] >= S["target"]:
        return f"APPLIED + CONFIRMED ({S['applied']}/{S['target']}), saved {base}. TARGET REACHED — reply DONE and stop."
    return f"APPLIED + CONFIRMED ({S['applied']}/{S['target']}), saved {base}. Move on to the next job."


@tool
async def click_button(text: str) -> str:
    """Click a visible apply-form button whose text contains `text` (e.g. 'Continue
    applying', 'Got it'). Use to get PAST reminders/interstitials — never to back out."""
    page = _apage()
    frame = await _aframe()
    hit = await frame.evaluate(
        """(text) => {
            const t = text.toLowerCase();
            for (const b of document.querySelectorAll('button, [role=button]')) {
                const st = getComputedStyle(b); const r = b.getBoundingClientRect();
                if (!(r.width>0&&r.height>0)||st.visibility==='hidden'||st.display==='none') continue;
                const bt = (b.innerText || b.getAttribute('aria-label') || '').replace(/\\s+/g,' ').trim();
                if (bt.toLowerCase().includes(t)) { b.click(); return bt; }
            }
            return '';
        }""",
        text,
    )
    await page.wait_for_timeout(1500)
    return f"clicked '{hit}'" if hit else f"no button matching '{text}'"


@tool
async def inspect_element(index: int) -> str:
    """Show the raw HTML of the field with this index and its surrounding group. Use when a
    field won't fill/check or read_form looks wrong, to SEE the real widget and decide how
    to interact (e.g. click_button with the option's visible text)."""
    frame = await _aframe()
    html = await frame.evaluate(
        "(s) => {" + _DEEP_JS + """
            const e = deepFind(document, s);
            if (!e) return '';
            const g = e.closest('[role=radiogroup],[role=group],fieldset,label') || e.parentElement || e;
            return g.outerHTML.slice(0, 3000);
        }""",
        f'[data-fill-idx="{index}"]',
    )
    return html or f"no element with index {index} — call read_form again"


@tool
async def skip_job(reason: str) -> str:
    """Skip the current job (external or can't complete) and move on. Nothing is saved."""
    ap = S.get("apply_page")
    if ap and ap is not S["page"]:
        try:
            await ap.close()
        except Exception:
            pass
    else:
        await close_any_modal(_page())
    S["apply_page"] = None
    cur = S["cur"]
    print(f"[SKIP] job {cur.get('job_id','?')} {cur.get('company','')}: {reason}", flush=True)
    return f"skipped: {reason}"


TOOLS = [
    go_to_search, list_jobs, next_page, open_job, read_job,
    open_apply, read_form, fill_field, choose_option, select_dropdown,
    upload_resume, click_next, click_button, submit_application, skip_job,
    inspect_element,
]

SYSTEM = (
    "You are an autonomous Indeed 'Apply now' (Indeed Apply) agent driving a real browser via "
    f"tools. GOAL: submit {APPLY_TARGET} NEW, CONFIRMED job applications, then stop.\n\n"
    "list_jobs marks each job 'APPLIED' or 'new'. ONLY work on 'new' jobs — NEVER open a "
    "job marked 'APPLIED'. Use next_page / list_jobs to get more 'new' jobs (Indeed pages "
    "are 10 jobs each: start = 10, 20, 30, ...).\n\n"
    "For each 'new' job: open_job -> read_job (ALWAYS, before applying) -> open_apply "
    "(if it reports EXTERNAL / 'apply on company site', skip_job) -> loop read_form -> "
    "fill_field / choose_option / select_dropdown / upload_resume as needed -> click_next, "
    "until a Submit button exists, then submit_application. Use choose_option for "
    "radio/checkbox, select_dropdown for real dropdowns. The Indeed apply form is a "
    "multi-step wizard: keep calling read_form + click_next to walk through every step "
    "(contact info, resume, employer questions, review) before the final Submit appears.\n\n"
    "RESUME: on ANY step that shows, mentions, or asks for a resume, call upload_resume — "
    "it verifies by FILENAME that the attached resume is the one from instructions.md, and "
    "only selects/uploads when it isn't (Indeed pre-selects your stored profile resume, "
    "which may be outdated). If it reports the correct file is NOT attached, find the "
    "'Upload resume'/'Replace'/'Change' control (read_form / click_button), then call "
    "upload_resume again. Never advance past a resume step until upload_resume confirms "
    "the correct file is attached.\n\n"
    "RADIO BUTTONS: each option of a radio question is its OWN index in read_form, labeled "
    "'question = option' — call choose_option with the index of the exact OPTION you want "
    "(e.g. for 'Authorized to work = Yes' pick the index whose label ends in '= Yes'), never "
    "fill_field on a radio and never the question itself. After choosing, call read_form to "
    "VERIFY it took; if it didn't, try choose_option once more on that index, then on the "
    "other index of the same question and back (toggling can wake the widget). If click_next "
    "does not advance and re-shows the same step, an unanswered radio is the usual cause — "
    "re-read the form and answer every radio group before advancing. If a field refuses to "
    "fill/check or a step seems to have invisible controls, call inspect_element on it to "
    "see the real HTML and pick another way in (often click_button with the option's text).\n\n"
    "YOU MUST ACTUALLY APPLY to every 'new' job — carry it through submit_application until "
    "CONFIRMED. skip_job ONLY when: open_apply reported EXTERNAL, OR the job matches the "
    "Blacklist below (blacklisted company, or the form asks for education dates / an "
    "unselectable school-degree). Judge the blacklist yourself from the job, company, and "
    "each form you read — the moment a job matches, call skip_job and move on; never guess "
    "education dates. If read_form shows a reminder/interstitial (e.g. 'Continue applying'), "
    "use click_button to proceed. After a CONFIRMED submission, move on. Only a CONFIRMED "
    "submission counts. When you reach the target, reply DONE and stop.\n\n"
    "Answer screening questions from the candidate's RESUME below (real experience: years, "
    "skills, titles, employers, education), guided by the answering policy and screening "
    "answers. The questions vary every time — reason from the resume for anything factual "
    "(e.g. 'how many years of X', 'do you know Y'); use the screening answers for fixed "
    "items (work authorization, salary, EEO). Never leave a required field blank.\n\n"
    f"CANDIDATE RESUME:\n{RESUME_TEXT or '(no resume text available)'}\n\n"
    f"POLICY AND SCREENING ANSWERS:\n{DISTILLED}"
)

GOAL = (
    f"Apply to {APPLY_TARGET} '{KEYWORD}' Indeed 'Apply now' jobs"
    + (f" in {LOCATION}" if LOCATION else "")
    + ". Start by calling go_to_search, then list_jobs."
)

token_logger = TokenLogger(TOKEN_LOG)
llm = build_llm(LLM_MODEL, LLM_PROVIDER)
agent = create_agent(llm, TOOLS, system_prompt=SYSTEM)


def build_agent(page, output_base, context=None):
    # Factory for the supervisor: set the injected output base, inject the shared browser,
    # then return the agent with a name (create_supervisor uses it for handoffs).
    set_output_base(output_base)
    S["page"] = page
    if context is not None:
        S["context"] = context
    return create_agent(llm, TOOLS, system_prompt=SYSTEM, name="indeed_apply_now")


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Indeed Apply Now agent")
    parser.add_argument("--out", required=True,
                        help="base output folder; results go under <out>/agent-indeed-apply-now/data")
    set_output_base(os.path.abspath(parser.parse_args().out))
    print(f"Search params: {SEARCH_PARAMS}", flush=True)
    print(f"Output dir: {OUT_DIR}\nTarget: {APPLY_TARGET} applications\n", flush=True)
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
        await S["page"].goto("https://www.indeed.com", wait_until="domcontentloaded", timeout=60000)
        print("Opened:", await S["page"].title(), flush=True)
        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": GOAL}]},
                config={"recursion_limit": RECURSION_LIMIT, "callbacks": [token_logger]},
            )
            msgs = result.get("messages", []) if isinstance(result, dict) else []
            if msgs:
                print("\nAGENT FINAL:", str(getattr(msgs[-1], "content", ""))[:600], flush=True)
        except Exception as e:
            print("\nagent error:", repr(e), flush=True)
        print(
            f"\nApplied {S['applied']}/{S['target']}. "
            f"Tokens in {token_logger.total_in} out {token_logger.total_out}",
            flush=True,
        )
        await ctx.close()


if __name__ == "__main__":
    asyncio.run(main())

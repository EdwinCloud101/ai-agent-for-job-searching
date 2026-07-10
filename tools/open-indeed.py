r"""
Open the same Chrome + profile the Indeed agent uses (../.indeed-profile in the repo root),
visibly, so you can log into Indeed once. The session persists and the agent reuses it.
The folder is created automatically on first run.

Run:  python tools/open-indeed.py
"""

import os

from playwright.sync_api import sync_playwright

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # tools/ -> repo root
PROFILE_DIR = os.path.join(REPO_ROOT, ".indeed-profile")  # same profile the Indeed agent opens

with sync_playwright() as p:
    context = p.chromium.launch_persistent_context(
        PROFILE_DIR,
        headless=False,
        channel="chrome",
        args=["--disable-blink-features=AutomationControlled"],
        ignore_default_args=["--enable-automation"],
    )
    context.add_init_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    page = context.pages[0] if context.pages else context.new_page()
    page.goto("https://www.indeed.com")
    print(f"Profile: {PROFILE_DIR}", flush=True)
    print("Log into Indeed now, then CLOSE the window to save & exit.", flush=True)

    closed = {"v": False}
    context.on("close", lambda: closed.__setitem__("v", True))
    try:
        while not closed["v"]:
            page.wait_for_timeout(1000)
    except Exception:
        pass

os._exit(0)

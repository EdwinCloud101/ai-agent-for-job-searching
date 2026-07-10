# Tech Market-Check — Agent Instructions

Copy this file to `instructions.md` and fill in your own values. `instructions.md` is
git-ignored so your personal config never gets committed.

This agent SEARCHES LinkedIn and COLLECTS data on job postings that use a given technology
(the `keywords` below). It does **NOT** apply to anything. Self-contained.

## Search filter
keywords: langchain
location: United States
distance: 0
date_posted: 1 month

## LLM
provider: deepseek
model: deepseek-v4-flash

## What to collect
For each job posting, save ONE markdown file. Include, in this order:
1. The job's **POSTING date** exactly as shown on the page ("1 day ago", "2 weeks ago",
   "1 month ago", etc.) — the POSTING date, NOT today's collection date.
2. Job title.
3. Company name.
4. Company description (already shown on the job posting page).
5. The FULL job description, unchanged (exactly as on LinkedIn).
6. A short summary of HOW this position uses the target technology (the keyword above),
   derived from the job description: what they build with it, related tools, and how
   central it is to the role.

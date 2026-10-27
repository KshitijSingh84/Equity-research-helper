import os
import sys
import json
import re
import time
from collections import Counter

import pdfplumber
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()                      # loads your API key from the .env file
client = Anthropic()
MODEL = "claude-sonnet-5-5"

PROMPT = """You are a sell-side equity research analyst covering Indian defence companies.
Read the earnings call transcript below and produce a complete analyst note as JSON.

STRICT RULES:
1. Use ONLY information in the transcript. Never use outside knowledge.
2. Every item has a "point" (your statement) and a "source_quote".
   - For FACTUAL sections, "source_quote" must be copied EXACTLY, word for word, from the
     transcript (under 40 words). Pick the shortest snippet that contains the key numbers.
   - If something is not mentioned in the transcript, leave it out. Never guess.
3. Keep numbers and rupee units exactly as stated in the transcript. Write fiscal periods
   in short form (Q1 FY26, H2, FY27).
4. For "analyst_read" items, "source_quote" is optional: use null when the item is your own
   judgment, and give an exact quote when the item points to something management said.
5. Be thorough. Capture every number, target, timeline and claim that matters.
6. Respond with JSON only. No extra text, no code fences.

SECTIONS TO PRODUCE:

1. financial_table: headline numbers, one row per metric, current vs prior-year period.
   Include standalone and consolidated revenue, operational EBITDA (with margin), and PAT
   (with margin), plus any other metric stated. Each row needs a quote that contains the
   current and previous numbers.
2. financial_notes: margin analysis, revenue mix (equipment vs AMC etc.), other income
   effects, expense changes and one-offs, segment margin differences, subsidiary
   contribution.
3. why_quarter: management's explanation of why the quarter came out as it did (beat,
   miss, deferrals, timing, customer changes).
4. order_book: total order book and its composition, expected near-term orders, order
   inflow expectations, past order slowness, procurement timing, lead times.
5. guidance_outlook: what was guided (revenue, margins, subsidiaries, capex, multi-year
   targets), what management refused to quantify, and execution timelines.
6. strategy: segment-by-segment and thematic updates. Each item has a "theme"
   (for example "Anti-drone systems", "Simulators", "Exports", "Subsidiaries and M&A",
   "R&D and new products", "AI", "Competition and moat").
7. supply_chain_risks: external dependencies, supply chain exposure, regulatory or
   geopolitical factors and management's response.
8. analyst_read:
   - positives
   - concerns (timing risk, concentration, guidance gaps, earnings quality, etc.)
   - evasive_or_unverifiable: questions management dodged, would not quantify, or claims
     that cannot be verified from the transcript alone
   - inconsistencies: small contradictions or number differences within the call
9. follow_up_questions: what an analyst should dig into next (plain strings).

OUTPUT FORMAT:
{
  "company": "...",
  "quarter": "...",
  "call_date": "...",
  "financial_table": [
    {"metric": "...", "current": "...", "previous": "...", "change": "...", "source_quote": "..."}
  ],
  "financial_notes": [{"point": "...", "source_quote": "..."}],
  "why_quarter": [{"point": "...", "source_quote": "..."}],
  "order_book": [{"point": "...", "source_quote": "..."}],
  "guidance_outlook": [{"point": "...", "source_quote": "..."}],
  "strategy": [{"theme": "...", "point": "...", "source_quote": "..."}],
  "supply_chain_risks": [{"point": "...", "source_quote": "..."}],
  "analyst_read": {
    "positives": [{"point": "...", "source_quote": null}],
    "concerns": [{"point": "...", "source_quote": null}],
    "evasive_or_unverifiable": [{"point": "...", "source_quote": null}],
    "inconsistencies": [{"point": "...", "source_quote": null}]
  },
  "follow_up_questions": ["...", "..."]
}

TRANSCRIPT:
"""

# Sections where every item MUST carry a verifiable quote
FACTUAL_SECTIONS = [
    ("financial_notes", "Financial Performance: Notes"),
    ("why_quarter", "Why the Quarter Came Out as It Did"),
    ("order_book", "Order Book and Pipeline"),
    ("guidance_outlook", "Guidance and Outlook"),
    ("strategy", "Strategy and Business Themes"),
    ("supply_chain_risks", "Supply Chain and Risks"),
]

ANALYST_SUBSECTIONS = [
    ("positives", "Positives"),
    ("concerns", "Concerns"),
    ("evasive_or_unverifiable", "Evasive or Unverifiable Claims"),
    ("inconsistencies", "Small Inconsistencies"),
]


# ---------------------------------------------------------------- reading the PDF

def pdf_to_text(path):
    """Reads all text from a PDF and removes page headers and footers."""
    with pdfplumber.open(path) as pdf:
        pages = [(page.extract_text() or "").split("\n") for page in pdf.pages]

    # Lines that repeat on at least half the pages are headers/footers (company name, date)
    line_counts = Counter()
    for lines in pages:
        for line in set(l.strip() for l in lines):
            line_counts[line] += 1
    repeated = {l for l, c in line_counts.items() if l and c >= max(2, len(pages) / 2)}

    cleaned = []
    for lines in pages:
        for line in lines:
            s = line.strip()
            if s in repeated or re.fullmatch(r"Page \d+ of \d+", s):
                continue
            cleaned.append(line)
    return "\n".join(cleaned)


# ---------------------------------------------------------------- helpers

def loose(text):
    """Very forgiving form for quote matching: lowercase letters and digits only.
    This ignores spacing, punctuation, curly quotes and PDF line-break hyphens."""
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def numbers_in(text):
    """Finds numbers like '1,200' or '34.26'. Ignores Q1, FY2026, H2 style labels."""
    found = re.findall(r"(?<![A-Za-z\d])\d[\d,]*(?:\.\d+)?", text or "")
    return [n.replace(",", "") for n in found]


def ask_claude(transcript):
    """Sends the transcript to Claude and returns its JSON answer as a Python dict."""
    with client.messages.stream(
        model=MODEL,
        max_tokens=32000,
        messages=[{"role": "user", "content": PROMPT + transcript}],
    ) as stream:
        final = stream.get_final_message()

    if final.stop_reason == "max_tokens":
        print("WARNING: the answer was cut off. Raise max_tokens in ask_claude().")

    raw = "".join(b.text for b in final.content if b.type == "text")
    start, end = raw.find("{"), raw.rfind("}")
    return json.loads(raw[start:end + 1])


# ---------------------------------------------------------------- the checker (no AI)

def check_item(point, quote, transcript_loose, quote_required):
    """Plain-code check of one statement against the transcript."""
    if not quote:
        return "NO QUOTE" if quote_required else "ANALYST VIEW"
    if loose(quote) not in transcript_loose:
        return "UNVERIFIED QUOTE (possible hallucination)"
    quote_nums = numbers_in(quote)
    missing = [n for n in numbers_in(point) if n not in quote_nums]
    if missing:
        return "NUMBER MISMATCH: " + ", ".join(missing)
    return "VERIFIED"


def check_row(row, transcript_loose):
    """Checks one row of the financial table (current and previous values)."""
    quote = row.get("source_quote")
    if not quote:
        return "NO QUOTE"
    if loose(quote) not in transcript_loose:
        return "UNVERIFIED QUOTE (possible hallucination)"
    quote_nums = numbers_in(quote)
    claimed = numbers_in(row.get("current")) + numbers_in(row.get("previous"))
    missing = [n for n in claimed if n not in quote_nums]
    if missing:
        return "NUMBER MISMATCH: " + ", ".join(missing)
    return "VERIFIED"


def validate(data, transcript):
    """Returns a dict of check results, keyed the same way as the data."""
    t = loose(transcript)
    checks = {"financial_table": [check_row(r, t) for r in data.get("financial_table", [])]}

    for key, _ in FACTUAL_SECTIONS:
        checks[key] = [
            check_item(i.get("point"), i.get("source_quote"), t, True)
            for i in data.get(key, [])
        ]

    analyst = data.get("analyst_read", {})
    checks["analyst_read"] = {
        key: [
            check_item(i.get("point"), i.get("source_quote"), t, False)
            for i in analyst.get(key, [])
        ]
        for key, _ in ANALYST_SUBSECTIONS
    }
    return checks


# ---------------------------------------------------------------- writing the note

def tag(status):
    """Short marker shown next to each line. Only problems are spelled out."""
    return "" if status in ("VERIFIED", "ANALYST VIEW") else f" **[CHECK: {status}]**"


def cell(text):
    return str(text if text is not None else "").replace("|", "/").replace("\n", " ")


def render(data, checks, name, seconds):
    L = [f"# Earnings call note: {name}", ""]
    L.append(f"**{data.get('company', '')}** | {data.get('quarter', '')} | "
             f"Call date: {data.get('call_date', '')}")
    L.append("")

    # 1. Financial performance
    L.append("## 1. Financial Performance")
    L.append("")
    L.append("| Metric | Current | Previous | Change | Check |")
    L.append("|---|---|---|---|---|")
    for row, status in zip(data.get("financial_table", []), checks["financial_table"]):
        L.append(f"| {cell(row.get('metric'))} | {cell(row.get('current'))} | "
                 f"{cell(row.get('previous'))} | {cell(row.get('change'))} | {status} |")
    L.append("")
    for item, status in zip(data.get("financial_notes", []), checks["financial_notes"]):
        L.append(f"- {item.get('point')}{tag(status)}")
        L.append(f"  > {item.get('source_quote')}")
    L.append("")

    # 2 to 7. Factual sections
    number = 2
    for key, title in FACTUAL_SECTIONS[1:]:
        L.append(f"## {number}. {title}")
        L.append("")
        items = data.get(key, [])
        if key == "strategy":
            current_theme = None
            for item, status in zip(items, checks[key]):
                theme = item.get("theme") or "Other"
                if theme != current_theme:
                    L.append(f"**{theme}**")
                    current_theme = theme
                L.append(f"- {item.get('point')}{tag(status)}")
                L.append(f"  > {item.get('source_quote')}")
        else:
            for item, status in zip(items, checks[key]):
                L.append(f"- {item.get('point')}{tag(status)}")
                L.append(f"  > {item.get('source_quote')}")
        L.append("")
        number += 1

    # 8. Analyst's read
    L.append(f"## {number}. Analyst's Read")
    L.append("")
    analyst = data.get("analyst_read", {})
    for key, title in ANALYST_SUBSECTIONS:
        L.append(f"**{title}**")
        for item, status in zip(analyst.get(key, []), checks["analyst_read"][key]):
            L.append(f"- {item.get('point')}{tag(status)}")
            if item.get("source_quote"):
                L.append(f"  > {item.get('source_quote')}")
        L.append("")
    number += 1

    # 9. Follow-ups
    L.append(f"## {number}. Follow-up Questions")
    L.append("")
    for q in data.get("follow_up_questions", []):
        L.append(f"- {q}")
    L.append("")

    # Verification summary
    flat = list(checks["financial_table"])
    for key, _ in FACTUAL_SECTIONS:
        flat += checks[key]
    verified = sum(1 for s in flat if s == "VERIFIED")
    flagged = [s for s in flat if s != "VERIFIED"]
    L.append("---")
    L.append(f"Verification: {verified} of {len(flat)} factual statements matched the "
             f"transcript exactly. {len(flagged)} flagged for manual review. "
             f"Analyst's Read items are judgments and are not counted. "
             f"Runtime: {seconds} seconds.")
    L.append("Based only on the call transcript. Cross-check against the results, "
             "investor presentation and filings. Not investment advice.")
    return "\n".join(L)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze.py transcripts/zen_1.pdf")
        return
    path = sys.argv[1]
    name = path.replace("\\", "/").split("/")[-1].replace(".pdf", "")

    transcript = pdf_to_text(path)
    if len(transcript) < 1000:
        print("Could not read text from this PDF. It may be a scanned image.")
        return

    started = time.time()
    data = ask_claude(transcript)
    checks = validate(data, transcript)
    seconds = round(time.time() - started)

    note = render(data, checks, name, seconds)

    os.makedirs("results", exist_ok=True)
    with open(f"results/{name}_note.md", "w", encoding="utf-8") as f:
        f.write(note)
    with open(f"results/{name}_raw.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(note)


main()
"""
Earnings call analyzer.

  1. Single call note:
         python analyze.py transcripts/zen_1.pdf

  2. Compare two calls (first file = EARLIER call, second file = LATER call):
         python analyze.py compare transcripts/zen_1.pdf transcripts/zen_2.pdf

     With readable labels in the note headings:
         python analyze.py compare transcripts/zen_1.pdf transcripts/zen_2.pdf --labels "Q1 FY25" "Q1 FY26"

  3. Compare a run of calls, neighbouring pairs in the order you list them
     (list them oldest first):
         python analyze.py compare-all transcripts/zen_1.pdf transcripts/zen_2.pdf transcripts/zen_3.pdf
"""
import os
import sys
import json
import re
import time
from collections import Counter
from functools import lru_cache

import pdfplumber
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()                     
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




COMPARE_PROMPT = """You are a sell-side equity research analyst covering Indian defence companies.
You are given two earnings call transcripts of the same company.
EARLIER call = __EARLIER__
LATER call = __LATER__
Compare them and produce a comparison note as JSON.

STRICT RULES:
1. Use ONLY information in the two transcripts. Never use outside knowledge.
2. Every comparison item has an "earlier" statement, a "later" statement, and one quote
   for each side.
   - "quote_earlier" must be copied EXACTLY, word for word, from the EARLIER transcript only.
   - "quote_later" must be copied EXACTLY, word for word, from the LATER transcript only.
   - Each quote must be under 40 words. Pick the shortest snippet that holds the key numbers.
   - Never take a quote from the wrong transcript.
3. If a topic appears in only one call, write "Not mentioned" for the other side and set
   that side's quote to null. Never guess.
4. Keep numbers and rupee units exactly as stated. Write fiscal periods in short form
   (Q1 FY26, H2, FY27).
5. The "change" field is your short conclusion for that item (see each section below).
6. Be thorough but skip trivia. Focus on what moved, what was promised, what slipped,
   and what changed in tone.
7. Respond with JSON only. No extra text, no code fences.

SECTIONS TO PRODUCE:

1. financial_table: headline numbers side by side (revenue, operational EBITDA and margin,
   PAT and margin, standalone and consolidated where given). "change" = the movement
   in percent or points if both numbers are available.
2. guidance_changes: revenue, margin, capex, multi-year target and any other guidance.
   "change" = raised, lowered, maintained, new, dropped or unclear.
3. order_book_changes: order book size and mix, expected orders, order inflow
   commentary, procurement timing. "change" = short conclusion.
4. promises_tracker: things management said would happen in the EARLIER call (orders
   expected, timelines, launches, targets) versus what the LATER call says about them.
   "earlier" = the promise, "later" = what is said now.
   "change" = delivered, partly delivered, slipped, changed or not addressed.
5. strategy_shifts: new themes, dropped themes, changed priorities (segments, exports,
   acquisitions, R&D, hard kill, AI, etc.). "change" = new, dropped, expanded, narrowed
   or unchanged.
6. risks_and_concerns: risks raised by management or analysts. "change" = new, resolved,
   persisting, worsened or eased.
7. tone_and_language: shifts in confidence, hedging, wording, defensiveness, or topics
   avoided. Each item has a "point". Quotes are optional: give "quote_earlier" and/or
   "quote_later" (exact, from the right transcript) when the point rests on wording,
   otherwise null.
8. analyst_read: your judgment, with the same optional quote fields.
   - improved: what got better
   - deteriorated: what got worse
   - red_flags: repeated slippage, changing explanations, widening gaps
   - inconsistencies: contradictions between the two calls
9. bottom_line: 3 to 4 sentences on how the story changed between the calls.
10. follow_up_questions: what an analyst should dig into next (plain strings).

OUTPUT FORMAT:
{
  "company": "...",
  "earlier_label": "...",
  "later_label": "...",
  "financial_table": [
    {"topic": "...", "earlier": "...", "later": "...", "change": "...",
     "quote_earlier": "...", "quote_later": "..."}
  ],
  "guidance_changes": [ same item shape as financial_table ],
  "order_book_changes": [ same item shape ],
  "promises_tracker": [ same item shape ],
  "strategy_shifts": [ same item shape ],
  "risks_and_concerns": [ same item shape ],
  "tone_and_language": [{"point": "...", "quote_earlier": null, "quote_later": null}],
  "analyst_read": {
    "improved": [{"point": "...", "quote_earlier": null, "quote_later": null}],
    "deteriorated": [{"point": "...", "quote_earlier": null, "quote_later": null}],
    "red_flags": [{"point": "...", "quote_earlier": null, "quote_later": null}],
    "inconsistencies": [{"point": "...", "quote_earlier": null, "quote_later": null}]
  },
  "bottom_line": "...",
  "follow_up_questions": ["...", "..."]
}

=== EARLIER TRANSCRIPT (__EARLIER__) ===
__TEXT_EARLIER__

=== LATER TRANSCRIPT (__LATER__) ===
__TEXT_LATER__
"""


PAIR_SECTIONS = [
    ("guidance_changes", "Guidance Changes"),
    ("order_book_changes", "Order Book and Pipeline"),
    ("promises_tracker", "Promises Tracker: What Was Said Then vs Now"),
    ("strategy_shifts", "Strategy Shifts"),
    ("risks_and_concerns", "Risks and Concerns"),
]

COMPARE_ANALYST_SUBSECTIONS = [
    ("improved", "What Improved"),
    ("deteriorated", "What Deteriorated"),
    ("red_flags", "Red Flags"),
    ("inconsistencies", "Inconsistencies Between the Calls"),
]




def pdf_to_text(path):
    """Reads all text from a PDF and removes page headers and footers."""
    with pdfplumber.open(path) as pdf:
        pages = [(page.extract_text() or "").split("\n") for page in pdf.pages]

   
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


@lru_cache(maxsize=None)
def load_transcript(path):
    """Reads a PDF (once per run) and stops with a clear message if it has no readable text."""
    text = pdf_to_text(path)
    if len(text) < 1000:
        sys.exit(f"Could not read text from {path}. It may be a scanned image.")
    return text


def label_from_path(path):
    return path.replace("\\", "/").split("/")[-1].replace(".pdf", "")




def loose(text):
    """Very forgiving form for quote matching: lowercase letters and digits only.
    This ignores spacing, punctuation, curly quotes and PDF line-break hyphens."""
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def numbers_in(text):
    """Finds numbers like '1,200' or '34.26'. Ignores Q1, FY2026, H2 style labels."""
    found = re.findall(r"(?<![A-Za-z\d])\d[\d,]*(?:\.\d+)?", text or "")
    return [n.replace(",", "") for n in found]


def call_claude(prompt):
    """Sends a prompt to Claude and returns its JSON answer as a Python dict."""
    with client.messages.stream(
        model=MODEL,
        max_tokens=32000,   
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        final = stream.get_final_message()

    if final.stop_reason == "max_tokens":
        sys.exit("The answer was cut off before it finished (thinking plus the note "
                 "used up max_tokens). Raise max_tokens in call_claude() and run again.")

    
    raw = "".join(b.text for b in final.content if b.type == "text")
    start, end = raw.find("{"), raw.rfind("}")
    return json.loads(raw[start:end + 1])


def ask_claude(transcript):
    return call_claude(PROMPT + transcript)


def ask_claude_compare(text_earlier, text_later, label_earlier, label_later):
    prompt = (COMPARE_PROMPT
              .replace("__TEXT_EARLIER__", text_earlier)
              .replace("__TEXT_LATER__", text_later)
              .replace("__EARLIER__", label_earlier)
              .replace("__LATER__", label_later))
    return call_claude(prompt)




def check_item(point, quote, transcript_loose, quote_required):
    """Checks one statement against one transcript."""
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
    """Checks one row of the single-call financial table."""
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
    """Check results for the single-call note, keyed the same way as the data."""
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




def check_side(text, quote, transcript_loose):
    """Checks one side (earlier or later) of a comparison item."""
    if not quote:
        if (text or "").strip().lower().startswith("not mentioned"):
            return "NOT MENTIONED"
        return "NO QUOTE"
    if loose(quote) not in transcript_loose:
        return "UNVERIFIED QUOTE (possible hallucination)"
    quote_nums = numbers_in(quote)
    missing = [n for n in numbers_in(text) if n not in quote_nums]
    if missing:
        return "NUMBER MISMATCH: " + ", ".join(missing)
    return "VERIFIED"


def check_pair(item, t_early, t_late):
    """Checks both sides of a comparison item. Each quote is checked against its OWN
    transcript, so a quote taken from the wrong call is caught."""
    sides = [
        ("earlier", check_side(item.get("earlier"), item.get("quote_earlier"), t_early)),
        ("later", check_side(item.get("later"), item.get("quote_later"), t_late)),
    ]
    problems = [f"{name}: {s}" for name, s in sides if s not in ("VERIFIED", "NOT MENTIONED")]
    return "; ".join(problems) if problems else "VERIFIED"


def check_view(item, t_early, t_late):
    """Judgment items: quotes are optional, but any quote given must be real."""
    problems = []
    if item.get("quote_earlier") and loose(item["quote_earlier"]) not in t_early:
        problems.append("earlier: UNVERIFIED QUOTE (possible hallucination)")
    if item.get("quote_later") and loose(item["quote_later"]) not in t_late:
        problems.append("later: UNVERIFIED QUOTE (possible hallucination)")
    if problems:
        return "; ".join(problems)
    if item.get("quote_earlier") or item.get("quote_later"):
        return "VERIFIED"
    return "ANALYST VIEW"


def validate_compare(data, text_earlier, text_later):
    te, tl = loose(text_earlier), loose(text_later)
    checks = {"financial_table": [check_pair(r, te, tl) for r in data.get("financial_table", [])]}
    for key, _ in PAIR_SECTIONS:
        checks[key] = [check_pair(i, te, tl) for i in data.get(key, [])]
    checks["tone_and_language"] = [check_view(i, te, tl) for i in data.get("tone_and_language", [])]
    analyst = data.get("analyst_read", {})
    checks["analyst_read"] = {
        key: [check_view(i, te, tl) for i in analyst.get(key, [])]
        for key, _ in COMPARE_ANALYST_SUBSECTIONS
    }
    return checks




def tag(status):
    """Short marker shown next to each line. Only problems are spelled out."""
    ok = ("VERIFIED", "ANALYST VIEW", "NOT MENTIONED")
    return "" if status in ok else f" **[CHECK: {status}]**"


def cell(text):
    return str(text if text is not None else "").replace("|", "/").replace("\n", " ")


def render(data, checks, name, seconds):
    """Markdown for the single-call note."""
    L = [f"# Earnings call note: {name}", ""]
    L.append(f"**{data.get('company', '')}** | {data.get('quarter', '')} | "
             f"Call date: {data.get('call_date', '')}")
    L.append("")

    
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

    
    L.append(f"## {number}. Follow-up Questions")
    L.append("")
    for q in data.get("follow_up_questions", []):
        L.append(f"- {q}")
    L.append("")

    
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


def render_pair_item(L, item, status):
    """One comparison item: topic, earlier vs later, change, and both quotes."""
    L.append(f"- **{item.get('topic')}**{tag(status)}")
    L.append(f"  - Earlier: {item.get('earlier')}")
    L.append(f"  - Later: {item.get('later')}")
    L.append(f"  - Change: **{item.get('change')}**")
    if item.get("quote_earlier"):
        L.append(f"  - > Earlier quote: {item.get('quote_earlier')}")
    if item.get("quote_later"):
        L.append(f"  - > Later quote: {item.get('quote_later')}")


def render_view_item(L, item, status):
    """One judgment item with optional quotes."""
    L.append(f"- {item.get('point')}{tag(status)}")
    if item.get("quote_earlier"):
        L.append(f"  > Earlier: {item.get('quote_earlier')}")
    if item.get("quote_later"):
        L.append(f"  > Later: {item.get('quote_later')}")


def render_compare(data, checks, label_e, label_l, seconds):
    """Markdown for the comparison note."""
    L = [f"# Earnings call comparison: {label_e} vs {label_l}", ""]
    L.append(f"**{data.get('company', '')}** | Earlier: {label_e} | Later: {label_l}")
    L.append("")

    # Bottom line first, so the reader gets the answer up front
    L.append("## Bottom Line")
    L.append("")
    L.append(data.get("bottom_line", ""))
    L.append("")

    
    L.append("## 1. Financial Performance Side by Side")
    L.append("")
    L.append(f"| Metric | {label_e} | {label_l} | Change | Check |")
    L.append("|---|---|---|---|---|")
    for row, status in zip(data.get("financial_table", []), checks["financial_table"]):
        L.append(f"| {cell(row.get('topic'))} | {cell(row.get('earlier'))} | "
                 f"{cell(row.get('later'))} | {cell(row.get('change'))} | {status} |")
    L.append("")

    
    number = 2
    for key, title in PAIR_SECTIONS:
        L.append(f"## {number}. {title}")
        L.append("")
        for item, status in zip(data.get(key, []), checks[key]):
            render_pair_item(L, item, status)
        L.append("")
        number += 1

    
    L.append(f"## {number}. Tone and Language")
    L.append("")
    for item, status in zip(data.get("tone_and_language", []), checks["tone_and_language"]):
        render_view_item(L, item, status)
    L.append("")
    number += 1

    
    L.append(f"## {number}. Analyst's Read")
    L.append("")
    analyst = data.get("analyst_read", {})
    for key, title in COMPARE_ANALYST_SUBSECTIONS:
        L.append(f"**{title}**")
        for item, status in zip(analyst.get(key, []), checks["analyst_read"][key]):
            render_view_item(L, item, status)
        L.append("")
    number += 1

    
    L.append(f"## {number}. Follow-up Questions")
    L.append("")
    for q in data.get("follow_up_questions", []):
        L.append(f"- {q}")
    L.append("")

    
    flat = list(checks["financial_table"])
    for key, _ in PAIR_SECTIONS:
        flat += checks[key]
    verified = sum(1 for s in flat if s == "VERIFIED")
    flagged = len(flat) - verified
    L.append("---")
    L.append(f"Verification: {verified} of {len(flat)} comparison items matched both "
             f"transcripts exactly. {flagged} flagged for manual review. "
             f"Tone and Analyst's Read items are judgments; any quotes in them are still "
             f"checked. Runtime: {seconds} seconds.")
    L.append("Based only on the two call transcripts. Cross-check against the results, "
             "investor presentations and filings. Not investment advice.")
    return "\n".join(L)




def run_single(path):
    name = label_from_path(path)
    transcript = load_transcript(path)

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


def run_compare(path_earlier, path_later, label_earlier=None, label_later=None):
    stem_e = label_from_path(path_earlier)
    stem_l = label_from_path(path_later)
    label_e = label_earlier or stem_e
    label_l = label_later or stem_l
    text_e = load_transcript(path_earlier)
    text_l = load_transcript(path_later)

    started = time.time()
    data = ask_claude_compare(text_e, text_l, label_e, label_l)
    checks = validate_compare(data, text_e, text_l)
    seconds = round(time.time() - started)

    note = render_compare(data, checks, label_e, label_l, seconds)

    os.makedirs("results", exist_ok=True)
    base = f"results/{stem_e}_vs_{stem_l}"      # file names always use the PDF names
    with open(f"{base}_comparison.md", "w", encoding="utf-8") as f:
        f.write(note)
    with open(f"{base}_raw.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(note)


def run_compare_all(paths):
    """Compares each file with the next one, in the order given (oldest first)."""
    for i in range(len(paths) - 1):
        print(f"\n{'=' * 70}\nPair {i + 1} of {len(paths) - 1}: "
              f"{label_from_path(paths[i])} -> {label_from_path(paths[i + 1])}\n{'=' * 70}\n")
        run_compare(paths[i], paths[i + 1])


def print_usage():
    print("Usage:")
    print("  Single call note:   python analyze.py transcripts/zen_1.pdf")
    print("  Compare two calls:  python analyze.py compare transcripts/zen_1.pdf "
          "transcripts/zen_2.pdf")
    print("  With labels:        python analyze.py compare transcripts/zen_1.pdf "
          "transcripts/zen_2.pdf --labels \"Q1 FY25\" \"Q1 FY26\"")
    print("  Compare a series:   python analyze.py compare-all transcripts/zen_1.pdf "
          "transcripts/zen_2.pdf transcripts/zen_3.pdf")
    print("  For compare and compare-all, list the files OLDEST FIRST.")


def main():
    args = sys.argv[1:]

    
    labels = None
    if "--labels" in args:
        i = args.index("--labels")
        labels = args[i + 1:i + 3]
        args = args[:i] + args[i + 3:]
        if len(labels) != 2:
            print("--labels needs two values: the earlier label and the later label.")
            return

    if args and args[0].lower() == "compare" and len(args) == 3:
        run_compare(args[1], args[2], *(labels or []))
    elif args and args[0].lower() == "compare-all" and len(args) >= 3 and not labels:
        run_compare_all(args[1:])
    elif len(args) == 1 and not labels and args[0].lower() not in ("compare", "compare-all"):
        run_single(args[0])
    else:
        print_usage()


if __name__ == "__main__":
    main()
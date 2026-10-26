import sys
import json
import re
import time
import pdfplumber 
import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()                      
client = Anthropic()
MODEL = "claude-sonnet-5-5"

PROMPT = """You are a sell-side equity research analyst covering Indian defence companies.
Read the earnings call transcript below and extract the fields listed.

STRICT RULES:
1. Use ONLY information in the transcript. Never use outside knowledge.
2. If a field is not mentioned, set both "value" and "source_quote" to null. Never guess.
3. Every "source_quote" must be copied EXACTLY, word for word, from the transcript (under 40 words).
4. Keep numbers and rupee units exactly as stated in the transcript.
5. Respond with JSON only. No extra text, no code fences.

Fields to extract:
- order_book: current order book size and change versus the previous quarter
- revenue_guidance: management's revenue guidance or targets
- margin_guidance: management's margin guidance or comments on margins
- capex_capacity_plans: capex, capacity expansion or new facility plans
- key_risks: top risks or concerns raised by management or analysts
- key_analyst_question: the single most important analyst question and the gist of the answer

Output format:
{
  "order_book": {"value": "...", "source_quote": "..."},
  "revenue_guidance": {"value": "...", "source_quote": "..."},
  "margin_guidance": {"value": "...", "source_quote": "..."},
  "capex_capacity_plans": {"value": "...", "source_quote": "..."},
  "key_risks": {"value": "...", "source_quote": "..."},
  "key_analyst_question": {"value": "...", "source_quote": "..."}
}

TRANSCRIPT:
"""


def pdf_to_text(path):
    """Reads all the text out of a PDF."""
    with pdfplumber.open(path) as pdf:
        return "\n".join((page.extract_text() or "") for page in pdf.pages)


def normalize(text):
    """Makes text comparable: lowercase, one space between words, plain quote marks."""
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def ask_claude(transcript):
    """Sends the transcript to Claude and returns its JSON answer as a Python dict."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        messages=[{"role": "user", "content": PROMPT + transcript}],
    )
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Response was cut off. Increase max_tokens.")

  
    raw = "".join(b.text for b in response.content if b.type == "text")

    start, end = raw.find("{"), raw.rfind("}")
    return json.loads(raw[start:end + 1])


def numbers_in(text):
    """Finds all numbers in a piece of text, e.g. '1,200' -> '1200'."""
    found = re.findall(r"\d[\d,\.]*", text)
    return [n.replace(",", "").strip(".") for n in found]


def validate(data, transcript):
    """The checker. It does NOT use AI. It uses plain code to catch hallucinations."""
    transcript_norm = normalize(transcript)
    results = {}
    for field, item in data.items():
        value = (item or {}).get("value")
        quote = (item or {}).get("source_quote")
        if value is None or quote is None:
            results[field] = "NOT FOUND"
        elif normalize(quote) not in transcript_norm:
            results[field] = "UNVERIFIED QUOTE (possible hallucination)"
        else:
            quote_numbers = numbers_in(quote)
            missing = [n for n in numbers_in(value) if n not in quote_numbers]
            if missing:
                results[field] = "NUMBER MISMATCH: " + ", ".join(missing)
            else:
                results[field] = "VERIFIED"
    return results


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

    lines = [f"# Earnings call note: {name}", ""]
    for field, item in data.items():
        lines.append(f"## {field.replace('_', ' ').title()}")
        lines.append(f"{(item or {}).get('value')}")
        lines.append(f"> Quote: {(item or {}).get('source_quote')}")
        lines.append(f"**Check: {checks[field]}**")
        lines.append("")
    verified = sum(1 for c in checks.values() if c == "VERIFIED")
    lines.append(f"Verified {verified} of {len(checks)} fields. Runtime: {seconds} seconds.")

    note = "\n".join(lines) 
    os.makedirs("results", exist_ok=True)
    with open(f"results/{name}_note.md", "w", encoding="utf-8") as f:
        f.write(note)
    print(note)


main() 

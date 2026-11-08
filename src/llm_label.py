"""Phase 2c: pre-label dev_sample.csv with an open-weights LLM on Groq, using the 8 README rules.

OFFLINE use only (labelling/auditing). Only messages from this repo are sent. Raw responses are saved to
results/llm_raw/<model>__<input>.jsonl (one JSON line per message: request id, raw text, parsed label, usage).

Run: python -m src.llm_label --model openai/gpt-oss-120b [--limit 5] [--input data/dev_sample.csv]
"""
import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from baseline.baseline import DATA, load_taxonomy

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "llm_raw"
URL = "https://api.groq.com/openai/v1/chat/completions"

DEFINITIONS = {
    "order_status": "asks where the order is / its status / ETA / tracking, with no complaint",
    "delivery_delay": "complains the order is late or delayed",
    "cancel_order": "wants to cancel an order that is not yet delivered",
    "refund_request": "wants money back / refund status / compensation for an order (delivered or already paid)",
    "missing_item": "an item or part of the order is absent / not received",
    "wrong_item": "received something different from what was ordered (wrong or someone else's order)",
    "payment_failed": "a payment failed, was declined, deducted twice, or money was taken without an order",
    "promo_code_issue": "a coupon / promo code / discount does not work or was not applied",
    "change_address": "wants to change or fix the delivery address",
    "account_help": "login, password, OTP, profile, phone/email, delete account or other account problems",
    "talk_to_agent": "asks ONLY for a human agent / escalation, with no other request",
    "out_of_scope": "anything not about QuickCart orders, accounts or payments (small talk, gibberish, other companies, "
                    "services QuickCart does not offer such as gift cards, loyalty points, tax invoices, tipping riders, "
                    "restaurant feedback); a greeting or thanks alone",
}

RULES = """1. Each message gets exactly one label, the intent for the action the customer wants done. If a message asks for several things, the first one asked is the primary request and decides the label. Example: "where is my order, and also I want a refund" is order_status. Rules 2 and 5 are the exceptions.
2. If the customer changes their mind, the last request wins ("cancel it... actually never mind, where is it?" is order_status).
3. Cancel plus refund: an order not yet delivered is cancel_order, a delivered one is refund_request.
4. A complaint with no request is labelled by the problem: late is delivery_delay, something absent is missing_item, something different is wrong_item.
5. Asking only for a human or escalation is talk_to_agent. If another request is also present, the other request wins.
6. Anything not about QuickCart orders, accounts or payments is out_of_scope. That includes small talk, gibberish, other companies' services, and services QuickCart does not offer (for example gift cards, loyalty points, tax invoices, tipping riders, restaurant feedback). A greeting or thanks alone is out_of_scope. A greeting plus a real request takes the request's label.
7. Filler, emotion and unrelated detail never change the label.
8. Order IDs, amounts and item names never change the label."""


def build_prompt(intents):
    defs = "\n".join(f"- {i}: {DEFINITIONS[i]}" for i in intents)
    system = (
        "You label customer-support messages for QuickCart, a food-delivery and grocery app. "
        "Choose exactly one intent from the list and follow the labelling rules strictly.\n\n"
        f"INTENTS:\n{defs}\n\nLABELLING RULES:\n{RULES}\n\n"
        'Reply with ONLY a JSON object: {"intent": "<one intent name>", "rule": <number of the main rule you applied, 1-8>, '
        '"reason": "<one short sentence>"}. Messages contain typos, emojis and Hindi-English mixing; read through them.'
    )
    return system


def call(model, system, message, key, retries=6):
    body = {"model": model, "temperature": 0, "seed": 42, "max_completion_tokens": 2048,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": f"Message: {message}"}]}
    if model.startswith("openai/gpt-oss"):
        body["reasoning_effort"] = "medium"
    for a in range(retries):
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={
            "Authorization": "Bearer " + key, "Content-Type": "application/json", "User-Agent": "quickcart-task"})
        try:
            return json.load(urllib.request.urlopen(req, timeout=120))
        except urllib.error.HTTPError as e:
            txt = e.read().decode()[:300]
            if e.code in (429, 500, 502, 503):
                wait = float(e.headers.get("retry-after") or 2 ** a * 2)
                print(f"  HTTP {e.code}, waiting {wait:.0f}s", flush=True)
                time.sleep(min(wait, 60))
                continue
            return {"error": f"HTTP {e.code}: {txt}"}
        except Exception as e:  # network blip
            time.sleep(2 ** a)
    return {"error": "gave up after retries"}


def parse(resp, intents):
    try:
        text = resp["choices"][0]["message"]["content"] or ""
    except Exception:
        return None, "", None
    m = re.search(r"\{.*\}", text, re.S)
    try:
        obj = json.loads(m.group(0))
        lab = str(obj.get("intent", "")).strip()
        return (lab if lab in intents else None), text, obj.get("rule")
    except Exception:
        return None, text, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", default=str(DATA / "dev_sample.csv"))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    load_dotenv(dotenv_path=ROOT / ".env", override=True)
    key = os.environ["GROQ_API_KEY"].strip()
    intents, _ = load_taxonomy()
    system = build_prompt(intents)
    df = pd.read_csv(a.input)
    if a.limit:
        df = df.head(a.limit)
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"{a.model.replace('/', '_')}__{Path(a.input).stem}.jsonl"
    done = {}
    if out.exists():                       # resume
        for line in open(out, encoding="utf-8"):
            r = json.loads(line)
            if r.get("label"):
                done[r["dev_row"]] = r
    with open(out, "w", encoding="utf-8") as f:
        for r in done.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for k, row in enumerate(df.itertuples()):
            if row.dev_row in done:
                continue
            resp = call(a.model, system, row.utterance, key)
            label, text, rule = parse(resp, intents)
            rec = {"dev_row": int(row.dev_row), "model": a.model, "label": label, "rule": rule, "raw_text": text,
                   "error": resp.get("error"), "usage": resp.get("usage")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if k % 25 == 0:
                print(k, row.silver_intent, "->", label, "|", text[:100].replace("\n", " "), flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()

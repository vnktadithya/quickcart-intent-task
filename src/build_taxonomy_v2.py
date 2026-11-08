"""Phase 3a: build data/taxonomy_v2.json from data/taxonomy.json with an explicit, reproducible change list.

Every edit is recorded in results/taxonomy_changes.csv (intent, example, action, reason).
Changes (reason = README rule or the exact defect found in Phase 1, results/eval_bugs.md #4 and #6):
  * same text under two intents -> keep it only where the rules put it
  * examples whose label breaks rules 1 / 4 -> move to the intent the rules give
  * promo_code_issue and account_help had NO examples -> add hand-written ones (paraphrases, not copied from any data file)
  * add a one-line description and a 'not' note per intent (for humans / LLM prompts; the TF-IDF matcher ignores them)

Run: python -m src.build_taxonomy_v2
"""
import json
from pathlib import Path

import pandas as pd

from baseline.baseline import DATA
from src.llm_label import DEFINITIONS

RESULTS = Path(__file__).resolve().parent.parent / "results"

REMOVE = [  # (intent, example, reason)
    ("delivery_delay", "where's my food", "duplicate of order_status example; a plain 'where is it' with no complaint is order_status (rule 4)"),
    ("wrong_item", "my order is incomplete", "duplicate of missing_item example; incomplete = something absent -> missing_item (rule 4)"),
    ("wrong_item", "an item is missing from my order", "absent item is missing_item, not wrong_item (rule 4)"),
    ("delivery_delay", "cancel my food order, taking too long", "first request is the cancellation -> cancel_order (rules 1 and 3: not delivered)"),
    ("payment_failed", "charged twice, please refund one payment", "the actual ask is a refund and the complaint comes first: ambiguous under rules 1/4, dropped rather than guessed"),
]
MOVE = [  # (from, to, example, reason)
    ("wrong_item", "missing_item", "an item is missing from my order", "absent item is missing_item (rule 4)"),
    ("delivery_delay", "cancel_order", "cancel my food order, taking too long", "first request is the cancellation (rules 1, 3)"),
]
ADD = {
    "promo_code_issue": [
        "my coupon code is not working",
        "the discount was not applied to my order",
        "promo code says invalid or expired",
        "where do I enter my promo code",
        "free delivery offer is not showing in the final amount",
        "promo kaam nahi kar raha",
        "I applied a coupon but the total did not change",
        "my first order discount is missing",
    ],
    "account_help": [
        "I forgot my password",
        "otp is not coming for login",
        "how do I delete my account",
        "my account has been locked",
        "update my phone number on my profile",
        "the app keeps logging me out",
        "my saved cards are missing from my account",
        "mera password reset nahi ho raha",
    ],
}
NOT_THIS = {
    "order_status": "a plain question about where the order is; if the customer complains it is late, use delivery_delay",
    "delivery_delay": "a complaint that it is late; if they only ask where it is, use order_status; if cancelling is the first ask, use cancel_order",
    "cancel_order": "cancel before delivery; for an already delivered order use refund_request (rule 3)",
    "refund_request": "money back for a delivered or paid order; for a failed payment use payment_failed",
    "missing_item": "something absent from the bag; if a different item arrived use wrong_item",
    "wrong_item": "something different arrived; if something is absent use missing_item",
    "payment_failed": "the payment itself failed or was duplicated; asking for the money back after delivery is refund_request",
    "promo_code_issue": "coupon / discount / offer problems; cashback or refunds of money are refund_request",
    "change_address": "delivery address only; phone number, name or profile edits are account_help",
    "account_help": "login, password, OTP, profile or saved-card problems; address changes are change_address",
    "talk_to_agent": "ONLY a request for a human; if another request is present, that request wins (rule 5)",
    "out_of_scope": "not about QuickCart orders, accounts or payments; greeting or thanks alone (rule 6)",
}


def main():
    tax = json.load(open(DATA / "taxonomy.json", encoding="utf-8"))
    by = {i["name"]: list(i["examples"]) for i in tax["intents"]}
    log = []
    moved = {(f, t, e) for f, t, e, _ in MOVE}
    for intent, ex, why in REMOVE:
        assert ex in by[intent], (intent, ex)
        by[intent].remove(ex)
        target = next((t for f, t, e, _ in MOVE if f == intent and e == ex), None)
        log.append({"intent": intent, "example": ex, "action": f"moved to {target}" if target else "removed", "reason": why})
    for f, t, ex, why in MOVE:
        by[t].append(ex)
        log.append({"intent": t, "example": ex, "action": f"added (moved from {f})", "reason": why})
    for intent, exs in ADD.items():
        assert not by[intent]
        by[intent] = exs
        for ex in exs:
            log.append({"intent": intent, "example": ex, "action": "added", "reason": "intent had no examples (eval_bugs.md #4)"})
    v2 = {"intents": [{"name": i["name"], "description": DEFINITIONS[i["name"]], "not_this": NOT_THIS[i["name"]],
                       "examples": by[i["name"]]} for i in tax["intents"]]}
    # sanity: no text under two intents
    seen = {}
    for i in v2["intents"]:
        for e in i["examples"]:
            assert e.lower() not in seen, (e, seen.get(e.lower()), i["name"])
            seen[e.lower()] = i["name"]
    json.dump(v2, open(DATA / "taxonomy_v2.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    pd.DataFrame(log).to_csv(RESULTS / "taxonomy_changes.csv", index=False)
    print({k: len(v) for k, v in by.items()})
    print(pd.DataFrame(log).action.value_counts().to_string())


if __name__ == "__main__":
    main()

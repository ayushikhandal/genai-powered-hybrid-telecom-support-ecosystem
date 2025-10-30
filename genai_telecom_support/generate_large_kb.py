"""
Generate a large synthetic KB as JSONL entries.
Usage (PowerShell):
python generate_large_kb.py --out large_kb.jsonl --size-gb 1

This script streams entries to disk and won't keep the whole dataset in memory.
Each entry shape:
{
  "id": "kb-0000001",
  "category": "Billing|Network Outage|Plan Change|Device Problem|Account|Security|Provisioning|IoT|Enterprise",
  "provider": "jio|airtel|vodafone|vi|bsnl|others",
  "query": "user query text",
  "response": "concise suggested response",
  "tags": ["billing","refund","roaming"]
}

Note: This is a synthetic-data generator for demo and load-testing only.
"""

import json
import random
import argparse
from itertools import count
from pathlib import Path

CATEGORIES = [
    "Billing",
    "Network Outage",
    "Plan Change",
    "Device Problem",
    "Account",
    "Security",
    "Provisioning",
    "IoT",
    "Enterprise",
    "Roaming",
    "Activation",
    "Refund",
]

PROVIDERS = ["jio", "airtel", "vodafone", "vi", "bsnl", "reliance", "airtelbiz", "idea"]

QUERIES = {
    "Billing": [
        "Why was I charged twice this month?",
        "There is a mysterious data charge on my bill.",
        "How do I get a refund for accidental recharge?",
        "Explain the taxes on my invoice.",
    ],
    "Network Outage": [
        "I have no signal since 10pm last night.",
        "My calls drop frequently in my area.",
        "Are there planned maintenance windows in my city?",
    ],
    "Plan Change": [
        "How do I change to unlimited plan?",
        "What happens to my remaining data if I downgrade my plan?",
    ],
    "Device Problem": [
        "Battery drains 50% overnight.",
        "SIM not detected after reboot.",
        "Mobile data slow even on 5G.",
    ],
    "Account": [
        "How to update my KYC?",
        "I lost access to my account — reset password?",
    ],
    "Security": [
        "I suspect someone ported my number.",
        "I received OTPs I didn't request.",
    ],
    "Roaming": [
        "How much do I get charged for roaming in Singapore?",
        "Enable international roaming for my number.",
    ],
    "IoT": [
        "My device SIM on APN corporate fails to attach.",
        "SIM deactivates after 30 days of inactivity.",
    ],
}

RESPONSES = {
    "Billing": [
        "Please check the detailed bill section in the app and if you find duplicate charges open a dispute.",
        "Refunds are processed within 5-7 business days once approved.",
    ],
    "Network Outage": [
        "We are investigating an outage in your area; please try restarting your device.",
    ],
    "Plan Change": [
        "Changing plans mid-cycle may prorate charges. Check 'Plan details' in app.",
    ],
    "Device Problem": [
        "Try restarting device and reseating SIM; if persists, visit service center.",
    ],
    "Account": [
        "To reset password use the 'Forgot Password' flow and verify via registered email/phone.",
    ],
}

TAG_POOL = ["billing","refund","roaming","activation","device","battery","5g","plan","security","iot","enterprise"]


def random_query_and_response(cat):
    q = random.choice(QUERIES.get(cat, [f"Generic question about {cat}"]))
    r = random.choice(RESPONSES.get(cat, [f"Suggested action for {cat}: contact support."]))
    # add small variations
    q = q.replace("my", random.choice(["my","the","your"]))
    return q, r


def generate(out_path: Path, target_bytes: int):
    cnt = count(1)
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        while written < target_bytes:
            i = next(cnt)
            cat = random.choice(CATEGORIES)
            prov = random.choice(PROVIDERS)
            q, r = random_query_and_response(cat)
            tags = random.sample(TAG_POOL, k=random.randint(1, 3))
            entry = {
                "id": f"kb-{i:07d}",
                "category": cat,
                "provider": prov,
                "query": q,
                "response": r,
                "tags": tags
            }
            line = json.dumps(entry, ensure_ascii=False) + "\n"
            f.write(line)
            written += len(line.encode('utf-8'))
            if i % 10000 == 0:
                print(f"Generated {i} entries, {written//(1024*1024)} MB written")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="large_kb.jsonl")
    parser.add_argument("--size-gb", type=float, default=1.0, help="Target size in GB")
    args = parser.parse_args()
    out = Path(args.out)
    target_bytes = int(args.size_gb * 1024 * 1024 * 1024)
    print(f"Generating ~{args.size_gb}GB to {out}")
    generate(out, target_bytes)

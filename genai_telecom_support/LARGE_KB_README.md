How to generate a large synthetic KB (1GB+)

This project includes a generator script `generate_large_kb.py` that streams synthetic KB entries to a JSONL file.

Example (PowerShell):

python generate_large_kb.py --out c:\temp\large_kb.jsonl --size-gb 1

Notes and guidance:
- The script writes JSONL (one JSON object per line). For the app you can either: 
  1) Modify `chatbot_app.py` to read JSONL (stream line-by-line) instead of JSON array, or
  2) Convert JSONL to a JSON array using a simple script if you prefer a single JSON file (may consume lots of RAM).
- Keep the generated file on a fast local disk. Generating 1GB will take time and disk I/O.
- The generator uses simple templates and randomization to produce diverse queries and responses; extend `QUERIES`/`RESPONSES` dictionaries to cover more domain-specific cases.
- When the KB does not contain a matching entry (low retrieval/confidence), the app will escalate to human support per the safety policy.

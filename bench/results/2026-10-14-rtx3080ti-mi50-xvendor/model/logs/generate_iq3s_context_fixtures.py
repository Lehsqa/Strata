#!/usr/bin/env python3
"""Deterministic synthetic IQ3_S context ladder; never launches inference."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from ab_engine import long_prompt
from strata_tokenizer import Tokenizer

GGUF = Path("<TARGET_HOME>/Models/gguf/Qwen3.8-Flash-Next/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf")
OUT = Path(__file__).resolve().parent / "iq3s-context-fixtures"
TARGETS = (4096, 32768, 131072, 149900)
NEEDLE = "amber-falcon-731"
PREFIX = "<|im_start|>user\nRead this synthetic word list and remember its secret code.\n"
MID = f"\nThe secret code word in this text is: {NEEDLE}. Remember it.\n"
SUFFIX = "\n\nWhat secret code word was mentioned in the text above? Reply with the code word only.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    tk = Tokenizer.from_gguf(GGUF)
    filler_text = long_prompt(max(TARGETS) + 100)
    filler_text = filler_text.split("\n", 1)[1].rsplit("\n\n", 1)[0]
    filler = tk.encode(" " + filler_text)
    assert len(filler) >= max(TARGETS)
    p = tk.encode(PREFIX, parse_special=True)
    m = tk.encode(MID, parse_special=True)
    s = tk.encode(SUFFIX, parse_special=True)
    rows = []
    for target in TARGETS:
        available = target - len(p) - len(m) - len(s)
        left = target // 2 - len(p)
        for _ in range(12):
            raw = p + filler[:left] + m + filler[left:available] + s
            text = tk.decode(raw)
            ids = tk.encode(text, parse_special=True)
            delta = target - len(ids)
            if delta == 0:
                break
            available += delta
        else:
            raise AssertionError(f"could not achieve exact token count {target}")
        assert len(ids) == target
        assert tk.decode(ids) == text
        assert text.count(NEEDLE) == 1
        assert target + 100 <= 150000
        needle_at = len(tk.encode(text.split(MID)[0], parse_special=True))
        stem = f"iq3s-q4-context150000-prompt{target}"
        body = text.encode("utf-8")
        token_bytes = (",".join(map(str, ids)) + "\n").encode()
        (OUT / (stem + ".txt")).write_bytes(body)
        (OUT / (stem + ".tokens")).write_bytes(token_bytes)
        row = {
            "target_prompt_tokens": target,
            "actual_prompt_tokens": len(ids),
            "context_capacity": 150000,
            "response_room_tokens": 150000 - len(ids),
            "recommended_max_new": 40,
            "text_file": stem + ".txt",
            "tokens_file": stem + ".tokens",
            "text_bytes": len(body),
            "text_sha256": sha256(body),
            "tokens_sha256": sha256(token_bytes),
            "needle_start_token": needle_at,
            "needle_depth_fraction": needle_at / len(ids),
            "needle": MID.strip(),
            "expected_answer": NEEDLE,
            "tokenizer_roundtrip": "PASS",
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    output_ids = [19, 248046, 198, 248044] + [248045] * 4
    decoded = tk.decode(output_ids)
    assert tk.decode([19]) == "4"
    manifest = {
        "schema": 1,
        "source": "tools/ab_engine.py::long_prompt",
        "source_sha256": sha256((ROOT / "tools/ab_engine.py").read_bytes()),
        "seed": 7,
        "filler": "synthetic twenty-word vocabulary only",
        "generator": str(Path(__file__).relative_to(ROOT)),
        "generator_sha256": sha256(Path(__file__).read_bytes()),
        "reproduce": ".venv/bin/python bench-local/xvendor-2026-10-14/generate_iq3s_context_fixtures.py",
        "model_shard": str(GGUF),
        "kv": "q4_0",
        "thinking": False,
        "chat_template_suffix": SUFFIX,
        "needle_code": NEEDLE,
        "expected_answer": NEEDLE,
        "short_output_ids": output_ids,
        "short_output_decoded": decoded,
        "short_first_answer": "4",
        "fixtures": rows,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("short output decoded:", repr(decoded), flush=True)
    print("manifest:", OUT / "manifest.json", flush=True)


if __name__ == "__main__":
    main()


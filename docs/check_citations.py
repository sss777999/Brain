import re
import sys
from collections import Counter

tex_path = sys.argv[1]
bib_path = sys.argv[2]

tex = open(tex_path, "r", encoding="utf-8").read()
bib = open(bib_path, "r", encoding="utf-8").read()

# --- Extract citation keys from tex ---
cite_pattern = re.compile(r"\\cite[tpa]?\{([^}]+)\}")
tex_keys = []
for m in cite_pattern.finditer(tex):
    raw = m.group(1)
    parts = [k.strip() for k in raw.split(",") if k.strip()]
    tex_keys.extend(parts)

tex_key_set = set(tex_keys)

# --- Extract bib keys ---
bib_pattern = re.compile(r"@\w+\{([^,]+),")
bib_keys = bib_pattern.findall(bib)
bib_key_set = set(k.strip() for k in bib_keys)

# --- Duplicates in bib ---
bib_key_counts = Counter(k.strip() for k in bib_keys)
bib_dupes = [k for k, c in bib_key_counts.items() if c > 1]

# --- Missing and unused ---
missing = sorted(tex_key_set - bib_key_set)
unused = sorted(bib_key_set - tex_key_set)

print("\n=== Citation Consistency Report ===\n")

print(f"Total citations in .tex: {len(tex_keys)}")
print(f"Unique citation keys in .tex: {len(tex_key_set)}")
print(f"Unique keys in .bib: {len(bib_key_set)}\n")

if bib_dupes:
    print("!! DUPLICATE KEYS IN .bib:")
    for k in bib_dupes:
        print("  -", k)
    print()
else:
    print("OK: No duplicate keys in .bib\n")

if missing:
    print("!! KEYS USED IN .tex BUT MISSING IN .bib:")
    for k in missing:
        print("  -", k)
    print()
else:
    print("OK: No missing citation keys (tex -> bib)\n")

if unused:
    print(".. KEYS PRESENT IN .bib BUT UNUSED IN .tex (optional cleanup):")
    for k in unused:
        print("  -", k)
    print()
else:
    print("OK: No unused bib entries\n")

# Quick sanity: common LaTeX compile warnings
if "??" in tex:
    print("NOTE: Found '??' in tex content; ensure you compile LaTeX twice.")

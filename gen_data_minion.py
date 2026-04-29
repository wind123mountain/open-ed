"""
Convert MINION (BIO token labels) to EventKD format for multilingual event detection.

Input:  datasets/MINION/<lang>/{train,dev,test}.json
        Each line: {"tokens": [...], "labels": ["O", "B_X", "I_X", ...], ...}
Output: data/minion_<lang>/{train,dev,test}.jsonl
        Each line: {"system_prompt": ..., "user_prompt": ..., "response": "{\"events\": [[trig, type], ...]}"}

Usage:
  python3 gen_data_minion.py --lang spanish
  python3 gen_data_minion.py --lang portuguese
  python3 gen_data_minion.py --lang all   # process all available languages
"""
import argparse
import json
import os

SYSTEM_PROMPT = (
    "You are an event extraction system. Your task is to identify events "
    "expressed or clearly implied in a given text.\n"
    "IMPORTANT: Output ONLY valid JSON. No explanations, no markdown, no extra text.\n"
    "Output Format (JSON only, no markdown):\n\n"
    "{\"events\": [[<trigger span>, <event type>], "
    "[<trigger span 2>, <event type 2>]]}\n\n"
    "- If no events are detected, return: {\"events\": []}"
)

USER_TEMPLATE = (
    "Given an input text:\n"
    "<input>\n{text}\n</input>\n\n"
    "Your task is to extract all events present in the text. "
    "The text may contain zero, one, or multiple events.\n\n"
    "For each event:\n"
    "- Identify a trigger: the word or phrase that most clearly indicates the event.\n"
    "- Identify event type\n\n"
    "Constraints and Guidelines\n"
    "- Do not invent information not supported by the text.\n"
    "- Do not paraphrase triggers.\n"
    "- The trigger must exactly match a span in the original input "
    "(from <input>...</input>).\n"
)

# Event-type descriptions. Required by the span-extraction loss in
# data_utils/lm_datasets.py:get_span_offsets, which expects each event tuple
# to be at least 3 elements (trigger, event_type, description) like MAVEN.
EVENT_TYPE_DESC = {
    "Business:START-ORG": "A new organization is founded or formally established.",
    "Conflict:Attack": "A violent or aggressive act directed at a target.",
    "Conflict:Demonstrate": "A public protest or demonstration is held.",
    "Contact:Meet": "Two or more parties physically meet or interact.",
    "Contact:Phone-Write": "Communication occurs via phone, written message, or similar means.",
    "Justice:Arrest-Jail": "A person is arrested or jailed by authorities.",
    "Life:Be-Born": "A person is born.",
    "Life:Die": "A person dies, naturally or due to violence.",
    "Life:Divorce": "A formal divorce or separation occurs between people.",
    "Life:Injure": "A person is injured or physically harmed.",
    "Life:Marry": "A marriage or formal union takes place.",
    "Movement:Transport": "A person or thing is moved or transported from one location to another.",
    "Personnel:End-Position": "A person leaves or is removed from a position or role.",
    "Personnel:Start-Position": "A person begins a new position or role.",
    "Transaction:Transfer-Money": "Money or financial value is transferred between parties.",
    "Transaction:Transfer-Ownership": "Ownership of an asset is transferred between parties.",
}


def parse_bio(tokens, labels):
    """Extract list of [trigger_text, event_type, description] from BIO labels."""
    events = []
    i = 0
    while i < len(labels):
        lab = labels[i]
        if lab.startswith("B_"):
            event_type = lab[2:]
            j = i + 1
            while j < len(labels) and labels[j] == "I_" + event_type:
                j += 1
            trigger = " ".join(tokens[i:j])
            description = EVENT_TYPE_DESC.get(
                event_type, f"An event of type {event_type}.")
            events.append([trigger, event_type, description])
            i = j
        else:
            i += 1
    return events


def convert_split(in_path, out_path):
    n_in = n_out = n_with_event = 0
    with open(in_path, encoding="utf-8") as fi, \
         open(out_path, "w", encoding="utf-8") as fo:
        for line in fi:
            n_in += 1
            d = json.loads(line)
            tokens = d.get("tokens", [])
            labels = d.get("labels", [])
            if not tokens or len(tokens) != len(labels):
                continue
            text = " ".join(tokens)
            events = parse_bio(tokens, labels)
            if events:
                n_with_event += 1
            response = json.dumps({"events": events}, ensure_ascii=False)
            record = {
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": USER_TEMPLATE.format(text=text),
                "response": response,
            }
            fo.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_out += 1
    return n_in, n_out, n_with_event


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minion-root", default="datasets/MINION")
    ap.add_argument("--out-root", default="data")
    ap.add_argument("--lang", required=True,
                    help="Language name (subfolder of MINION) or 'all'")
    args = ap.parse_args()

    if args.lang == "all":
        langs = sorted(d for d in os.listdir(args.minion_root)
                       if os.path.isdir(os.path.join(args.minion_root, d)))
    else:
        langs = [args.lang]

    # MINION uses dev.json (not valid.json) — map to dev.jsonl in output
    split_map = {"train": "train", "dev": "dev", "test": "test"}

    for lang in langs:
        lang_in = os.path.join(args.minion_root, lang)
        lang_out = os.path.join(args.out_root, f"minion_{lang}")
        if not os.path.isdir(lang_in):
            print(f"[skip] {lang}: input dir not found at {lang_in}")
            continue
        os.makedirs(lang_out, exist_ok=True)
        print(f"=== {lang} ===")
        for src, dst in split_map.items():
            in_path = os.path.join(lang_in, f"{src}.json")
            out_path = os.path.join(lang_out, f"{dst}.jsonl")
            if not os.path.isfile(in_path):
                print(f"  [skip] {src}: not found at {in_path}")
                continue
            n_in, n_out, n_with = convert_split(in_path, out_path)
            pct = 100.0 * n_with / max(n_out, 1)
            print(f"  {src} -> {dst}.jsonl  N={n_out}  "
                  f"({n_with} with events, {pct:.1f}%)")


if __name__ == "__main__":
    main()

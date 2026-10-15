#!/usr/bin/env python3
"""Turn a BALROG episode CSV (Step, Action, Reasoning, Observation, Reward, Done)
into a readable table.

Usage:
    python csv_to_table.py episode.csv                 # Markdown to stdout
    python csv_to_table.py episode.csv -o episode.md   # Markdown to file
    python csv_to_table.py episode.csv -f html -o episode.html
"""
import argparse
import csv
import html
import sys

DEFAULT_MARKER = "Your previous output did not contain a valid action"


def clean_observation(text):
    """Collapse the multi-line observation into one line and flag defaulted actions."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    flagged = any(DEFAULT_MARKER in ln for ln in lines)
    # Drop the warning and the "Observation:" header the env injects
    lines = [ln for ln in lines
             if DEFAULT_MARKER not in ln and ln.rstrip(":") != "Observation"]
    obs = "; ".join(lines)
    return obs, flagged


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = []
        for r in csv.DictReader(f):
            obs, flagged = clean_observation(r["Observation"])
            action = r["Action"].strip()
            if flagged:
                action += " (invalid, defaulted)"
            rows.append({
                "Step": r["Step"].strip(),
                "Action": action,
                "Reasoning": " ".join(r["Reasoning"].split()),
                "Observation": obs,
                "Reward": r["Reward"].strip(),
                "Done": r["Done"].strip(),
            })
        return rows


COLUMNS = ["Step", "Action", "Reasoning", "Observation", "Reward", "Done"]


def to_markdown(rows):
    esc = lambda s: s.replace("|", "\\|")
    out = ["| " + " | ".join(COLUMNS) + " |",
           "| " + " | ".join("---" for _ in COLUMNS) + " |"]
    for r in rows:
        out.append("| " + " | ".join(esc(r[c]) for c in COLUMNS) + " |")
    return "\n".join(out)


def to_html(rows):
    head = "".join(f"<th>{c}</th>" for c in COLUMNS)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{html.escape(r[c])}</td>" for c in COLUMNS) + "</tr>"
        for r in rows
    )
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><style>"
        "body{font-family:system-ui,sans-serif;margin:1rem}"
        "table{border-collapse:collapse;font-size:14px}"
        "th,td{border:1px solid #ccc;padding:4px 8px;vertical-align:top}"
        "th{background:#eee;position:sticky;top:0}"
        "@media (prefers-color-scheme: dark){body{background:#1e1e1e;color:#ddd}"
        "th{background:#333}th,td{border-color:#555}}"
        "</style></head><body>"
        f"<table><thead><tr>{head}</tr></thead><tbody>\n{body}\n</tbody></table>"
        "</body></html>"
    )


def summary(rows):
    last = rows[-1]
    defaulted = sum("(invalid" in r["Action"] for r in rows)
    return (f"Steps: {len(rows)} | Final reward: {last['Reward']} | "
            f"Done: {last['Done']} | Invalid actions: {defaulted}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv_file")
    p.add_argument("-f", "--format", choices=["md", "html"], default="md")
    p.add_argument("-o", "--output", help="output file (default: stdout)")
    args = p.parse_args()

    rows = read_rows(args.csv_file)
    if not rows:
        sys.exit("CSV has no rows.")
    table = to_markdown(rows) if args.format == "md" else to_html(rows)
    if args.format == "md":
        table = summary(rows) + "\n\n" + table

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(table + "\n")
        print(f"Wrote {len(rows)} rows to {args.output}")
    else:
        print(table)


if __name__ == "__main__":
    main()

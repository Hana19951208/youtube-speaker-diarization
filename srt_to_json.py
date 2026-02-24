#!/usr/bin/env python3
"""
Convert SRT (with optional speaker prefix) to JSON without rerunning pipeline.

Usage:
  python srt_to_json.py --srt input.srt --json output.json
"""

import re
import json
import argparse
from pathlib import Path
from collections import Counter


def parse_ts(ts: str) -> float:
    # HH:MM:SS,mmm
    hms, ms = ts.split(',')
    h, m, s = [int(x) for x in hms.split(':')]
    return h * 3600 + m * 60 + s + int(ms) / 1000.0


def parse_srt(path: Path):
    text = path.read_text(encoding='utf-8', errors='ignore').strip()
    blocks = re.split(r'\n\s*\n', text)
    items = []

    for b in blocks:
        lines = [x.rstrip() for x in b.splitlines() if x.strip()]
        if len(lines) < 2:
            continue

        # Allow with/without index line
        if '-->' in lines[0]:
            ts_line = lines[0]
            content_lines = lines[1:]
        else:
            if len(lines) < 3 or '-->' not in lines[1]:
                continue
            ts_line = lines[1]
            content_lines = lines[2:]

        m = re.match(r'\s*(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*', ts_line)
        if not m:
            continue

        start = parse_ts(m.group(1))
        end = parse_ts(m.group(2))
        raw = ' '.join(content_lines).strip()

        speaker = 'unknown'
        text_only = raw
        # format: SPEAKER: text
        m2 = re.match(r'^([A-Za-z0-9_\-\u4e00-\u9fa5]+)\s*:\s*(.*)$', raw)
        if m2:
            speaker = m2.group(1)
            text_only = m2.group(2).strip()

        items.append({
            'start': start,
            'end': end,
            'speaker': speaker,
            'text': text_only,
        })

    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--srt', required=True, help='Input srt path')
    ap.add_argument('--json', required=True, help='Output json path')
    args = ap.parse_args()

    srt_path = Path(args.srt)
    json_path = Path(args.json)

    sentences = parse_srt(srt_path)
    counts = Counter(x['speaker'] for x in sentences)

    out = {
        'summary': {
            'processing_summary': {
                'total_sentences': len(sentences),
                'speaker_sentence_counts': dict(counts),
            }
        },
        'sentences': sentences,
        'source': {
            'generated_from': 'srt',
            'srt_path': str(srt_path),
        }
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'✅ JSON generated from SRT: {json_path}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Label Qwen35 records for agentic distillation buckets.')
    parser.add_argument('--input', required=True, help='Qwen35 export directory or a single qwen35 jsonl file.')
    parser.add_argument('--output-root', required=True, help='Output directory root for label artifacts.')
    parser.add_argument('--min-tool-calls', type=int, default=1, help='Minimum tool calls required to consider a record agentic.')
    parser.add_argument('--min-tool-messages', type=int, default=1, help='Minimum tool messages required to consider a record agentic.')
    parser.add_argument('--min-rounds', type=int, default=1, help='Minimum dialogue rounds required to consider a record agentic.')
    parser.add_argument('--min-reasoning-chars', type=int, default=1, help='Minimum reasoning chars required for cot_eligible.')
    return parser.parse_args(argv)


def iter_input_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f'Input path does not exist: {input_path}')
    files: List[Path] = []
    for name in ('qwen35-train.jsonl', 'qwen35-train-lossy.jsonl'):
        candidate = input_path / name
        if candidate.exists():
            files.append(candidate)
    if files:
        return files
    return sorted(input_path.rglob('qwen35-*.jsonl'))


def load_records(files: List[Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in files:
        bucket = 'lossy' if 'lossy' in path.name else 'strict'
        with path.open('r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record['_bucket'] = bucket
                record['_source_file'] = str(path)
                records.append(record)
    return records


def count_role(messages: List[Dict[str, Any]], role: str) -> int:
    return sum(1 for message in messages if isinstance(message, dict) and message.get('role') == role)


def tool_call_count(messages: List[Dict[str, Any]]) -> int:
    return sum(
        len(message.get('tool_calls') or [])
        for message in messages
        if isinstance(message, dict) and message.get('role') == 'assistant'
    )


def reasoning_chars(messages: List[Dict[str, Any]]) -> int:
    return sum(
        len(message.get('reasoning_content', ''))
        for message in messages
        if isinstance(message, dict)
        and message.get('role') == 'assistant'
        and isinstance(message.get('reasoning_content'), str)
    )


def label_record(record: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    messages = record.get('messages', []) if isinstance(record.get('messages'), list) else []
    user_count = count_role(messages, 'user')
    assistant_count = count_role(messages, 'assistant')
    tool_count = count_role(messages, 'tool')
    calls = tool_call_count(messages)
    reasoning = reasoning_chars(messages)
    has_reasoning = reasoning >= args.min_reasoning_chars
    agentic = calls >= args.min_tool_calls and tool_count >= args.min_tool_messages and user_count >= args.min_rounds

    if agentic and has_reasoning:
        label = 'cot_eligible'
    elif agentic:
        label = 'agent_only'
    else:
        label = 'discard'

    return {
        'id': record.get('id'),
        'request_id': record.get('request_id'),
        'label': label,
        'bucket': record.get('_bucket'),
        'source_file': record.get('_source_file'),
        'user_message_count': user_count,
        'assistant_message_count': assistant_count,
        'tool_message_count': tool_count,
        'dialogue_rounds_est': user_count,
        'tool_call_count': calls,
        'reasoning_chars': reasoning,
        'has_reasoning': has_reasoning,
        'lossy_source': bool(record.get('meta', {}).get('lossy_source')),
        'lossy_reasons': record.get('meta', {}).get('lossy_reasons', []),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or [])
    input_path = Path(args.input).expanduser().resolve()
    files = iter_input_files(input_path)
    if not files:
        raise SystemExit('No Qwen35 JSONL files found.')

    records = load_records(files)
    labels = [label_record(record, args) for record in records]

    out_dir = Path(args.output_root).expanduser().resolve() / f'qwen35-agentic-labels-{datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")}'
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / 'labels.jsonl'
    manifest_path = out_dir / 'manifest.json'

    stats = Counter(label['label'] for label in labels)
    stats['records'] = len(labels)
    stats['strict_records'] = sum(1 for label in labels if label['bucket'] == 'strict')
    stats['lossy_records'] = sum(1 for label in labels if label['bucket'] == 'lossy')

    with labels_path.open('w', encoding='utf-8') as handle:
        for label in labels:
            handle.write(json.dumps(label, ensure_ascii=False) + '\n')

    manifest = {
        'input': str(input_path),
        'output_dir': str(out_dir),
        'input_files': [str(path) for path in files],
        'rules': {
            'min_tool_calls': args.min_tool_calls,
            'min_tool_messages': args.min_tool_messages,
            'min_rounds': args.min_rounds,
            'min_reasoning_chars': args.min_reasoning_chars,
            'cot_eligible': 'agentic and has visible reasoning',
            'agent_only': 'agentic without visible reasoning',
            'discard': 'does not meet agentic thresholds',
        },
        'stats': dict(stats),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(__import__('sys').argv[1:]))

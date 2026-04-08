#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .export_qwen35_training import (
    append_parquet_rows,
    ensure_parquet_runtime,
    record_to_parquet_row,
    validate_record_payload,
)

BATCH_SIZE = 1000
DEFAULT_CODEX_HOME = Path.home() / '.codex'


class CodexConversionError(RuntimeError):
    pass


class TurnBuilder:
    def __init__(self, session_meta: Dict[str, Any], turn_id: str, start_ts: str) -> None:
        self.session_meta = session_meta
        self.turn_id = turn_id
        self.start_ts = start_ts
        self.messages: List[Dict[str, Any]] = []
        self.pending_text_parts: List[str] = []
        self.pending_reasoning_parts: List[str] = []
        self.pending_tool_calls: List[Dict[str, Any]] = []
        self.call_names: Dict[str, str] = {}
        self.tool_specs: Dict[str, Dict[str, Any]] = {}
        self.lossy_reasons: set[str] = set()
        self.error_messages: List[str] = []
        self.last_ts: str = start_ts
        self.completed = False
        self.last_agent_message: Optional[str] = None

    def ingest(self, entry: Dict[str, Any]) -> None:
        self.last_ts = entry.get('timestamp') or self.last_ts
        entry_type = entry.get('type')
        payload = entry.get('payload') if isinstance(entry.get('payload'), dict) else {}

        if entry_type == 'response_item':
            self._ingest_response_item(payload)
            return

        if entry_type == 'event_msg':
            event_type = payload.get('type')
            if event_type == 'exec_command_end':
                self._ingest_exec_command_end(payload)
            elif event_type == 'error':
                message = payload.get('message')
                if isinstance(message, str) and message:
                    self.error_messages.append(message)
                    self.lossy_reasons.add('turn_error')
            elif event_type == 'task_complete':
                last_agent_message = payload.get('last_agent_message')
                if isinstance(last_agent_message, str) and last_agent_message.strip():
                    self.last_agent_message = last_agent_message
            return

    def _ingest_response_item(self, payload: Dict[str, Any]) -> None:
        item_type = payload.get('type')
        if item_type == 'message':
            self._ingest_message(payload)
        elif item_type == 'reasoning':
            self._ingest_reasoning(payload)
        elif item_type == 'function_call':
            self._ingest_function_call(payload)
        elif item_type == 'function_call_output':
            self._ingest_function_call_output(payload)
        elif item_type == 'custom_tool_call':
            self._ingest_custom_tool_call(payload)
        elif item_type == 'custom_tool_call_output':
            self._ingest_custom_tool_call_output(payload)

    def _ingest_message(self, payload: Dict[str, Any]) -> None:
        role = payload.get('role')
        content = payload.get('content') if isinstance(payload.get('content'), list) else []
        text = extract_codex_text(content)
        if role == 'assistant':
            if text:
                self.pending_text_parts.append(text)
            return

        self.flush_assistant()
        if role == 'developer':
            if text:
                self.messages.append({'role': 'system', 'content': text})
            return
        if role == 'user':
            if is_environment_context(text):
                self.messages.append({'role': 'system', 'content': text})
            elif text:
                self.messages.append({'role': 'user', 'content': text})
            return
        if text:
            self.lossy_reasons.add(f'unsupported_message_role_{role}')
            self.messages.append({'role': 'assistant', 'content': text})

    def _ingest_reasoning(self, payload: Dict[str, Any]) -> None:
        summary = payload.get('summary')
        extracted: List[str] = []
        if isinstance(summary, list):
            for item in summary:
                if isinstance(item, dict):
                    if isinstance(item.get('text'), str) and item['text'].strip():
                        extracted.append(item['text'].strip())
                    elif isinstance(item.get('summary_text'), str) and item['summary_text'].strip():
                        extracted.append(item['summary_text'].strip())
        content = payload.get('content')
        if isinstance(content, str) and content.strip():
            extracted.append(content.strip())
        if extracted:
            self.pending_reasoning_parts.extend(extracted)
        elif payload.get('encrypted_content'):
            self.lossy_reasons.add('encrypted_reasoning_without_summary')

    def _ingest_function_call(self, payload: Dict[str, Any]) -> None:
        name = payload.get('name') or 'unknown_function'
        call_id = payload.get('call_id')
        arguments = parse_json_object(payload.get('arguments'))
        tool_call = {
            'type': 'function',
            'id': call_id,
            'function': {'name': name, 'arguments': arguments},
        }
        self.pending_tool_calls.append(tool_call)
        if isinstance(call_id, str):
            self.call_names[call_id] = name
        self.tool_specs.setdefault(name, {'name': name})

    def _ingest_custom_tool_call(self, payload: Dict[str, Any]) -> None:
        name = payload.get('name') or 'custom_tool'
        call_id = payload.get('call_id')
        arguments = {'input': payload.get('input'), 'status': payload.get('status')}
        tool_call = {
            'type': 'function',
            'id': call_id,
            'function': {'name': name, 'arguments': arguments},
        }
        self.pending_tool_calls.append(tool_call)
        if isinstance(call_id, str):
            self.call_names[call_id] = name
        self.tool_specs.setdefault(name, {'name': name})

    def _ingest_function_call_output(self, payload: Dict[str, Any]) -> None:
        self.flush_assistant()
        call_id = payload.get('call_id')
        tool_name = self.call_names.get(call_id or '', 'tool')
        output = payload.get('output')
        if not isinstance(output, str):
            output = json.dumps(output, ensure_ascii=False, sort_keys=True)
        self.messages.append({'role': 'tool', 'name': tool_name, 'tool_call_id': call_id, 'content': output})
        self.tool_specs.setdefault(tool_name, {'name': tool_name})

    def _ingest_custom_tool_call_output(self, payload: Dict[str, Any]) -> None:
        self.flush_assistant()
        call_id = payload.get('call_id')
        tool_name = self.call_names.get(call_id or '', 'custom_tool')
        output = payload.get('output')
        if not isinstance(output, str):
            output = json.dumps(output, ensure_ascii=False, sort_keys=True)
        self.messages.append({'role': 'tool', 'name': tool_name, 'tool_call_id': call_id, 'content': output})
        self.tool_specs.setdefault(tool_name, {'name': tool_name})

    def _ingest_exec_command_end(self, payload: Dict[str, Any]) -> None:
        self.flush_assistant()
        call_id = payload.get('call_id')
        tool_name = self.call_names.get(call_id or '', 'exec_command')
        content = json.dumps(
            {
                'command': payload.get('command'),
                'cwd': payload.get('cwd'),
                'aggregated_output': payload.get('aggregated_output'),
                'exit_code': payload.get('exit_code'),
                'status': payload.get('status'),
                'duration': payload.get('duration'),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        self.messages.append({'role': 'tool', 'name': tool_name, 'tool_call_id': call_id, 'content': content})
        self.tool_specs.setdefault(tool_name, {'name': tool_name})

    def flush_assistant(self) -> None:
        if not self.pending_text_parts and not self.pending_reasoning_parts and not self.pending_tool_calls:
            return
        content = '\n\n'.join(part for part in self.pending_text_parts if part.strip())
        message: Dict[str, Any] = {'role': 'assistant', 'content': content}
        if self.pending_reasoning_parts:
            message['reasoning_content'] = '\n\n'.join(self.pending_reasoning_parts)
        if self.pending_tool_calls:
            message['tool_calls'] = list(self.pending_tool_calls)
        self.messages.append(message)
        self.pending_text_parts = []
        self.pending_reasoning_parts = []
        self.pending_tool_calls = []

    def finalize(self) -> Optional[Dict[str, Any]]:
        if self.last_agent_message and not self.pending_text_parts:
            # Some turns surface only `last_agent_message` at completion; preserve it as lossy synthetic text.
            self.pending_text_parts.append(self.last_agent_message)
            self.lossy_reasons.add('synthetic_last_agent_message')
        self.flush_assistant()
        if not any(message.get('role') == 'user' for message in self.messages):
            return None
        meta = build_meta(self)
        record = {
            'id': f"{self.session_meta.get('id')}:{self.turn_id}",
            'request_id': self.turn_id,
            'messages': self.messages,
            'tools': list(self.tool_specs.values()),
            'meta': meta,
        }
        validate_record_payload(record)
        return record


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert Codex sessions into local Qwen3.5 training schema.')
    parser.add_argument('--codex-home', default=str(DEFAULT_CODEX_HOME), help='Codex home directory.')
    parser.add_argument('--input', help='Specific Codex session file or directory. Defaults to ~/.codex/sessions.')
    parser.add_argument('--output-root', required=True, help='Output directory root for Qwen3.5 export.')
    parser.add_argument('--output-format', choices=('jsonl', 'parquet', 'both'), default='jsonl')
    parser.add_argument('--limit', type=int, default=0, help='Limit the number of session files processed.')
    return parser.parse_args(argv)


def parse_json_object(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {'value': parsed}
    except Exception:
        return {'raw': raw}


def extract_codex_text(content: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get('type')
        if item_type in {'input_text', 'output_text'} and isinstance(item.get('text'), str):
            parts.append(item['text'])
        elif item_type == 'input_image':
            parts.append('[image]')
    return '\n'.join(part for part in parts if part)


def is_environment_context(text: str) -> bool:
    return text.strip().startswith('<environment_context>') if isinstance(text, str) else False


def iter_session_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise CodexConversionError(f'Input path does not exist: {input_path}')
    return sorted(input_path.rglob('*.jsonl'))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as handle:
        for line_number, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                items.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise CodexConversionError(f'Invalid JSON at {path}:{line_number}: {exc}') from exc
    return items


def build_meta(builder: TurnBuilder) -> Dict[str, Any]:
    assistant_messages = [message for message in builder.messages if message.get('role') == 'assistant']
    tool_messages = [message for message in builder.messages if message.get('role') == 'tool']
    user_messages = [message for message in builder.messages if message.get('role') == 'user']
    reasoning_count = sum(
        1
        for message in assistant_messages
        if isinstance(message.get('reasoning_content'), str) and message['reasoning_content'].strip()
    )
    return {
        'endpoint': 'codex/turn',
        'status': 200 if not builder.error_messages else 500,
        'ts': builder.last_ts or builder.start_ts or '',
        'key': builder.session_meta.get('id'),
        'source': (
            f"codex:{builder.session_meta.get('source') or 'cli'}:"
            f"session={builder.session_meta.get('id')}:turn={builder.turn_id}:"
            f"cwd={builder.session_meta.get('cwd')}:cli={builder.session_meta.get('cli_version')}"
        ),
        'requested_model': builder.session_meta.get('model'),
        'actual_model': builder.session_meta.get('model'),
        'stream': False,
        'thinking_level': builder.session_meta.get('reasoning_effort'),
        'reasoning_summary_mode': 'codex_reasoning_summary',
        'thinking_type': 'codex_turn',
        'thinking_budget_tokens': None,
        'max_output_tokens': None,
        'tool_spec_count': len(builder.tool_specs),
        'tool_choice': {'mode': 'session_trace'},
        'request_contains_non_text_content': False,
        'request_image_block_count': 0,
        'request_video_block_count': 0,
        'request_tool_call_block_count': 0,
        'request_tool_result_block_count': 0,
        'request_thinking_block_count': 0,
        'response_contains_non_text_content': False,
        'response_image_block_count': 0,
        'response_video_block_count': 0,
        'response_tool_call_block_count': sum(len(message.get('tool_calls') or []) for message in assistant_messages),
        'response_tool_result_block_count': len(tool_messages),
        'response_thinking_block_count': reasoning_count,
        'request_truncated': False,
        'response_truncated': False,
        'lossy_source': bool(builder.lossy_reasons),
        'lossy_reasons': sorted(builder.lossy_reasons),
    }


def convert_session_file(path: Path) -> Tuple[List[Dict[str, Any]], Counter]:
    entries = read_jsonl(path)
    if not entries:
        raise CodexConversionError(f'Empty session file: {path}')
    session_meta_payload = entries[0].get('payload') if entries[0].get('type') == 'session_meta' else None
    if not isinstance(session_meta_payload, dict):
        raise CodexConversionError(f'Missing session_meta in {path}')

    session_meta = dict(session_meta_payload)
    stats = Counter()
    stats['input_files'] += 1
    records: List[Dict[str, Any]] = []
    current: Optional[TurnBuilder] = None

    for entry in entries:
        entry_type = entry.get('type')
        payload = entry.get('payload') if isinstance(entry.get('payload'), dict) else {}
        if entry_type == 'turn_context':
            session_meta['model'] = payload.get('model') or session_meta.get('model')
            session_meta['reasoning_effort'] = payload.get('effort') or session_meta.get('reasoning_effort')
            continue
        if entry_type == 'event_msg' and payload.get('type') == 'task_started':
            turn_id = payload.get('turn_id') or entry.get('timestamp')
            current = TurnBuilder(session_meta, str(turn_id), entry.get('timestamp') or '')
            continue
        if current is None:
            continue
        current.ingest(entry)
        if entry_type == 'event_msg' and payload.get('type') == 'task_complete':
            record = current.finalize()
            if record is not None:
                records.append(record)
                if record['meta']['lossy_source']:
                    stats['lossy_records'] += 1
                else:
                    stats['strict_records'] += 1
            stats['turns_total'] += 1
            current = None

    return records, stats


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or [])
    ensure_parquet_runtime(args.output_format)
    codex_home = Path(args.codex_home).expanduser().resolve()
    input_path = Path(args.input).expanduser().resolve() if args.input else (codex_home / 'sessions')
    session_files = iter_session_files(input_path)
    if args.limit > 0:
        session_files = session_files[: args.limit]
    if not session_files:
        raise SystemExit('No Codex session files found.')

    out_dir = Path(args.output_root).expanduser().resolve() / f'qwen35-codex-session-{datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")}'
    out_dir.mkdir(parents=True, exist_ok=True)
    strict_path = out_dir / 'qwen35-train.jsonl'
    lossy_path = out_dir / 'qwen35-train-lossy.jsonl'
    invalid_path = out_dir / 'invalid-records.jsonl'
    strict_parquet_path = out_dir / 'qwen35-train.parquet'
    lossy_parquet_path = out_dir / 'qwen35-train-lossy.parquet'
    manifest_path = out_dir / 'manifest.json'

    jsonl_enabled = args.output_format in {'jsonl', 'both'}
    parquet_enabled = args.output_format in {'parquet', 'both'}
    strict_out = strict_path.open('w', encoding='utf-8') if jsonl_enabled else None
    lossy_out = lossy_path.open('w', encoding='utf-8') if jsonl_enabled else None
    invalid_out = invalid_path.open('w', encoding='utf-8')
    strict_writer = None
    lossy_writer = None
    strict_batch: List[Dict[str, Any]] = []
    lossy_batch: List[Dict[str, Any]] = []
    stats = Counter()

    try:
        for session_file in session_files:
            try:
                records, file_stats = convert_session_file(session_file)
                stats.update(file_stats)
            except Exception as exc:
                stats['invalid_files'] += 1
                invalid_out.write(json.dumps({'path': str(session_file), 'error': str(exc)}, ensure_ascii=False) + '\n')
                continue

            for record in records:
                bucket = 'lossy' if record['meta']['lossy_source'] else 'strict'
                if bucket == 'strict':
                    if strict_out is not None:
                        strict_out.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
                    if parquet_enabled:
                        strict_batch.append(record_to_parquet_row(record))
                        if len(strict_batch) >= BATCH_SIZE:
                            strict_writer = append_parquet_rows(strict_writer, strict_batch, strict_parquet_path)
                            strict_batch = []
                else:
                    if lossy_out is not None:
                        lossy_out.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
                    if parquet_enabled:
                        lossy_batch.append(record_to_parquet_row(record))
                        if len(lossy_batch) >= BATCH_SIZE:
                            lossy_writer = append_parquet_rows(lossy_writer, lossy_batch, lossy_parquet_path)
                            lossy_batch = []
            print(json.dumps({'processed_files': stats['input_files'], **dict(stats)}, ensure_ascii=False), flush=True)

        if parquet_enabled:
            strict_writer = append_parquet_rows(strict_writer, strict_batch, strict_parquet_path)
            lossy_writer = append_parquet_rows(lossy_writer, lossy_batch, lossy_parquet_path)
    finally:
        if strict_out is not None:
            strict_out.close()
        if lossy_out is not None:
            lossy_out.close()
        invalid_out.close()
        if strict_writer is not None:
            strict_writer.close()
        if lossy_writer is not None:
            lossy_writer.close()

    manifest = {
        'codex_home': str(codex_home),
        'input': str(input_path),
        'output_dir': str(out_dir),
        'input_files': [str(path) for path in session_files],
        'stats': dict(stats),
        'strict_records': stats.get('strict_records', 0),
        'lossy_records': stats.get('lossy_records', 0),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(__import__('sys').argv[1:]))

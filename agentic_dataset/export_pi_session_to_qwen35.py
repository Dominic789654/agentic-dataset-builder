#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .export_pi_session import ARTIFACT_KEY, compact_text, read_jsonl
from .export_qwen35_training import (
    append_parquet_rows,
    ensure_parquet_runtime,
    record_to_parquet_row,
    validate_record_payload,
)

FULL_GLOB = '*.full.jsonl'
RAW_GLOB = '*.raw.jsonl'
BATCH_SIZE = 1000


class ConversionError(RuntimeError):
    pass


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    argv_list = list(argv)
    command = 'convert'
    if argv_list and argv_list[0] == 'stats':
        command = 'stats'
        argv_list = argv_list[1:]

    parser = argparse.ArgumentParser(description='Convert exported Pi sessions into local Qwen3.5 training schema.')
    parser.add_argument('--input', required=True, help='Input full-session file or directory containing exported Pi sessions, or a generated Qwen35 export dir in stats mode.')

    if command == 'stats':
        parser.add_argument('--json', action='store_true', help='Emit machine-readable JSON.')
        args = parser.parse_args(argv_list)
        args.command = command
        return args

    parser.add_argument('--output-root', required=True, help='Output directory root for Qwen3.5 schema export.')
    parser.add_argument('--output-format', choices=('jsonl', 'parquet', 'both'), default='jsonl')
    parser.add_argument('--include-raw', action='store_true', help='Also read *.raw.jsonl files when scanning directories.')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of input files processed.')
    args = parser.parse_args(argv_list)
    args.command = command
    return args


def iter_input_files(input_path: Path, include_raw: bool) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise ConversionError(f'Input path does not exist: {input_path}')
    patterns = [FULL_GLOB]
    if include_raw:
        patterns.append(RAW_GLOB)
    files: List[Path] = []
    for pattern in patterns:
        files.extend(sorted(input_path.rglob(pattern)))
    deduped: Dict[str, Path] = {}
    for path in files:
        deduped[str(path.resolve())] = path.resolve()
    return list(deduped.values())


def build_tree(entries: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[Optional[str], List[str]]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    children: Dict[Optional[str], List[str]] = defaultdict(list)
    for entry in entries:
        entry_id = entry.get('id')
        if not isinstance(entry_id, str):
            continue
        by_id[entry_id] = entry
        children[entry.get('parentId')].append(entry_id)
    return by_id, children


def leaf_ids(by_id: Dict[str, Dict[str, Any]], children: Dict[Optional[str], List[str]]) -> List[str]:
    leaves = [entry_id for entry_id in by_id if not children.get(entry_id)]
    return sorted(leaves)


def path_to_leaf(leaf_id: str, by_id: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered: List[Dict[str, Any]] = []
    current_id: Optional[str] = leaf_id
    while current_id is not None:
        entry = by_id.get(current_id)
        if entry is None:
            break
        ordered.append(entry)
        parent_id = entry.get('parentId')
        current_id = parent_id if isinstance(parent_id, str) else None
    ordered.reverse()
    return ordered


def convert_content_blocks(
    content: Any,
    lossy_reasons: set[str],
    unsupported_reason_prefix: str,
) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        lossy_reasons.add(f'{unsupported_reason_prefix}_nonstandard_content')
        return json.dumps(content, ensure_ascii=False, sort_keys=True)

    blocks: List[Dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            lossy_reasons.add(f'{unsupported_reason_prefix}_non_dict_block')
            continue
        block_type = block.get('type')
        if block_type == 'text':
            blocks.append({'type': 'text', 'text': block.get('text', '')})
        elif block_type == 'image':
            metadata: Dict[str, Any] = {}
            if isinstance(block.get('mimeType'), str):
                metadata['mimeType'] = block['mimeType']
            if isinstance(block.get('data'), str):
                metadata['data'] = block['data']
            blocks.append(
                {
                    'type': 'image',
                    'placeholder': True,
                    'placeholder_token': '<image>',
                    'source_kind': 'pi_session_inline_image',
                    'metadata': metadata or None,
                }
            )
        elif block_type == 'video':
            metadata = {}
            if isinstance(block.get('mimeType'), str):
                metadata['mimeType'] = block['mimeType']
            if isinstance(block.get('data'), str):
                metadata['data'] = block['data']
            blocks.append(
                {
                    'type': 'video',
                    'placeholder': True,
                    'placeholder_token': '<video>',
                    'source_kind': 'pi_session_inline_video',
                    'metadata': metadata or None,
                }
            )
        else:
            lossy_reasons.add(f'{unsupported_reason_prefix}_unsupported_block_{block_type}')
            blocks.append({'type': 'text', 'text': json.dumps(block, ensure_ascii=False, sort_keys=True)})
    if not blocks:
        return ''
    if len(blocks) == 1 and blocks[0].get('type') == 'text':
        return blocks[0]['text']
    return blocks


def convert_assistant_message(message: Dict[str, Any], lossy_reasons: set[str], tools_seen: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    content = message.get('content')
    text_blocks: List[Dict[str, Any]] = []
    reasoning_chunks: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    if isinstance(content, str):
        text_blocks = [{'type': 'text', 'text': content}]
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                lossy_reasons.add('assistant_non_dict_block')
                continue
            block_type = block.get('type')
            if block_type == 'text':
                text_blocks.append({'type': 'text', 'text': block.get('text', '')})
            elif block_type == 'thinking':
                thinking = block.get('thinking')
                if isinstance(thinking, str) and thinking:
                    reasoning_chunks.append(thinking)
            elif block_type == 'toolCall':
                name = block.get('name') or 'unknown_tool'
                arguments = block.get('arguments') if isinstance(block.get('arguments'), dict) else {}
                tool_calls.append(
                    {
                        'type': 'function',
                        'id': block.get('id'),
                        'function': {'name': name, 'arguments': arguments},
                    }
                )
                tools_seen.setdefault(name, {'name': name})
            elif block_type in {'image', 'video'}:
                converted = convert_content_blocks([block], lossy_reasons, 'assistant')
                if isinstance(converted, list):
                    text_blocks.extend(converted)
                elif isinstance(converted, str):
                    text_blocks.append({'type': 'text', 'text': converted})
            else:
                lossy_reasons.add(f'assistant_unsupported_block_{block_type}')
                text_blocks.append({'type': 'text', 'text': json.dumps(block, ensure_ascii=False, sort_keys=True)})
    else:
        lossy_reasons.add('assistant_nonstandard_content')
        text_blocks = [{'type': 'text', 'text': json.dumps(content, ensure_ascii=False, sort_keys=True)}]

    assistant_content: Any
    if not text_blocks:
        assistant_content = ''
    elif len(text_blocks) == 1 and text_blocks[0].get('type') == 'text':
        assistant_content = text_blocks[0]['text']
    else:
        assistant_content = text_blocks

    payload: Dict[str, Any] = {'role': 'assistant', 'content': assistant_content}
    if reasoning_chunks:
        payload['reasoning_content'] = '\n\n'.join(reasoning_chunks)
    if tool_calls:
        payload['tool_calls'] = tool_calls
    return payload


def embedded_artifact_text(node: Dict[str, Any]) -> Optional[str]:
    embedded = node.get(f'{ARTIFACT_KEY}Embedded')
    if not isinstance(embedded, dict):
        return None
    if embedded.get('encoding') == 'utf-8' and isinstance(embedded.get('text'), str):
        return embedded['text']
    if embedded.get('encoding') == 'base64' and isinstance(embedded.get('base64'), str):
        return '[binary artifact embedded as base64]'
    return None


def format_bash_execution(message: Dict[str, Any], lossy_reasons: set[str]) -> str:
    output = message.get('output')
    if not isinstance(output, str):
        output = ''
    full_text = embedded_artifact_text(message)
    if full_text is None and isinstance(message.get('details'), dict):
        full_text = embedded_artifact_text(message['details'])
    truncated = bool(message.get('truncated'))
    if truncated and full_text is None and isinstance(message.get(ARTIFACT_KEY), str):
        lossy_reasons.add('missing_embedded_full_output')
    effective_output = full_text if full_text is not None else output
    payload = {
        'command': message.get('command'),
        'exit_code': message.get('exitCode'),
        'cancelled': message.get('cancelled', False),
        'truncated': truncated,
        'exclude_from_context': message.get('excludeFromContext', False),
        'output': effective_output,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def synthetic_message_from_entry(entry: Dict[str, Any], label: str, text: Optional[str], lossy_reasons: set[str]) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        return None
    lossy_reasons.add(f'synthetic_{label}_message')
    return {'role': 'assistant', 'content': f'[{label}]\n{text.strip()}'}


def entry_has_missing_artifact(node: Any) -> bool:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == ARTIFACT_KEY and isinstance(value, str) and f'{ARTIFACT_KEY}Embedded' not in node:
                return True
            if entry_has_missing_artifact(value):
                return True
    elif isinstance(node, list):
        return any(entry_has_missing_artifact(item) for item in node)
    return False


def convert_entry_to_messages(
    entry: Dict[str, Any],
    lossy_reasons: set[str],
    tools_seen: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    entry_type = entry.get('type')
    if entry_type == 'message':
        message = entry.get('message')
        if not isinstance(message, dict):
            lossy_reasons.add('message_entry_missing_payload')
            return []
        role = message.get('role')
        if role == 'user':
            return [{'role': 'user', 'content': convert_content_blocks(message.get('content'), lossy_reasons, 'user')}]
        if role == 'assistant':
            return [convert_assistant_message(message, lossy_reasons, tools_seen)]
        if role == 'toolResult':
            tool_name = message.get('toolName')
            if isinstance(tool_name, str) and tool_name:
                tools_seen.setdefault(tool_name, {'name': tool_name})
            return [
                {
                    'role': 'tool',
                    'content': convert_content_blocks(message.get('content'), lossy_reasons, 'tool_result'),
                    'tool_call_id': message.get('toolCallId'),
                    'name': tool_name,
                }
            ]
        if role == 'bashExecution':
            tools_seen.setdefault('bash', {'name': 'bash'})
            return [{'role': 'tool', 'content': format_bash_execution(message, lossy_reasons), 'name': 'bash'}]
        if role == 'custom':
            custom_type = message.get('customType') or 'custom'
            custom_content = message.get('content')
            converted = convert_content_blocks(custom_content, lossy_reasons, 'custom')
            lossy_reasons.add('synthetic_custom_message')
            return [{'role': 'assistant', 'content': f'[custom:{custom_type}]\n{converted}' if isinstance(converted, str) else converted}]
        if role == 'branchSummary':
            return [synthetic_message_from_entry(entry, 'branch_summary', message.get('summary'), lossy_reasons)] if message.get('summary') else []
        if role == 'compactionSummary':
            return [synthetic_message_from_entry(entry, 'compaction_summary', message.get('summary'), lossy_reasons)] if message.get('summary') else []
        lossy_reasons.add(f'unsupported_message_role_{role}')
        return [{'role': 'assistant', 'content': json.dumps(message, ensure_ascii=False, sort_keys=True)}]

    if entry_type == 'branch_summary':
        return [synthetic_message_from_entry(entry, 'branch_summary', entry.get('summary'), lossy_reasons)] if entry.get('summary') else []
    if entry_type == 'compaction':
        return [synthetic_message_from_entry(entry, 'compaction_summary', entry.get('summary'), lossy_reasons)] if entry.get('summary') else []
    if entry_type == 'custom_message':
        custom_type = entry.get('customType') or 'custom'
        converted = convert_content_blocks(entry.get('content'), lossy_reasons, 'custom_message')
        lossy_reasons.add('synthetic_custom_message')
        return [{'role': 'assistant', 'content': f'[custom:{custom_type}]\n{converted}' if isinstance(converted, str) else converted}]
    return []


def count_blocks(content: Any) -> Dict[str, int]:
    counts = {
        'contains_non_text_content': False,
        'image_block_count': 0,
        'video_block_count': 0,
    }
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get('type')
            if block_type == 'image':
                counts['contains_non_text_content'] = True
                counts['image_block_count'] += 1
            elif block_type == 'video':
                counts['contains_non_text_content'] = True
                counts['video_block_count'] += 1
    return counts


def compute_meta_counts(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {
        'request_contains_non_text_content': False,
        'request_image_block_count': 0,
        'request_video_block_count': 0,
        'request_tool_call_block_count': 0,
        'request_tool_result_block_count': 0,
        'request_thinking_block_count': 0,
        'response_contains_non_text_content': False,
        'response_image_block_count': 0,
        'response_video_block_count': 0,
        'response_tool_call_block_count': 0,
        'response_tool_result_block_count': 0,
        'response_thinking_block_count': 0,
    }
    for message in messages:
        role = message.get('role')
        side = 'request' if role in {'system', 'user'} else 'response'
        block_counts = count_blocks(message.get('content'))
        counts[f'{side}_contains_non_text_content'] = counts[f'{side}_contains_non_text_content'] or block_counts['contains_non_text_content']
        counts[f'{side}_image_block_count'] += block_counts['image_block_count']
        counts[f'{side}_video_block_count'] += block_counts['video_block_count']
        if role == 'assistant':
            counts['response_tool_call_block_count'] += len(message.get('tool_calls') or [])
            if isinstance(message.get('reasoning_content'), str) and message['reasoning_content'].strip():
                counts['response_thinking_block_count'] += 1
        if role == 'tool':
            counts['response_tool_result_block_count'] += 1
    return counts


def build_record_for_path(
    header: Dict[str, Any],
    path_entries: List[Dict[str, Any]],
    source_path: Path,
    leaf_id: str,
    branch_index: int,
    branch_count: int,
) -> Dict[str, Any]:
    lossy_reasons: set[str] = set()
    tools_seen: Dict[str, Dict[str, Any]] = {}
    messages: List[Dict[str, Any]] = []
    models_seen: List[str] = []
    thinking_levels: List[str] = []

    for entry in path_entries:
        if entry_has_missing_artifact(entry):
            lossy_reasons.add('missing_embedded_artifact')
        if entry.get('type') == 'model_change':
            model_id = entry.get('modelId')
            provider = entry.get('provider')
            if isinstance(model_id, str):
                models_seen.append(f'{provider}/{model_id}' if provider else model_id)
            continue
        if entry.get('type') == 'thinking_level_change':
            level = entry.get('thinkingLevel')
            if isinstance(level, str):
                thinking_levels.append(level)
            continue
        if entry.get('type') in {'session_info', 'label', 'custom'}:
            continue
        for message in convert_entry_to_messages(entry, lossy_reasons, tools_seen):
            if message:
                messages.append(message)

    if not any(message.get('role') == 'user' for message in messages):
        raise ConversionError(f'No user messages found on branch {leaf_id} from {source_path}')

    if branch_count > 1:
        lossy_reasons.add('session_tree_branch_selected')

    export_info = header.get('exportInfo') if isinstance(header.get('exportInfo'), dict) else None
    if isinstance(export_info, dict) and int(export_info.get('missingArtifactCount') or 0) > 0:
        lossy_reasons.add('source_export_missing_artifacts')

    if len(set(models_seen)) > 1:
        lossy_reasons.add('multiple_models_on_branch')
    if len(set(thinking_levels)) > 1:
        lossy_reasons.add('multiple_thinking_levels_on_branch')

    tools = list(tools_seen.values())
    counts = compute_meta_counts(messages)
    meta = {
        'endpoint': 'pi/session_branch',
        'status': 200,
        'ts': path_entries[-1].get('timestamp') or header.get('timestamp') or '',
        'key': header.get('id'),
        'source': f'{source_path}#leaf={leaf_id}',
        'requested_model': models_seen[0] if models_seen else None,
        'actual_model': models_seen[-1] if models_seen else None,
        'stream': False,
        'thinking_level': thinking_levels[-1] if thinking_levels else None,
        'reasoning_summary_mode': 'pi_session_branch',
        'thinking_type': 'pi_session',
        'thinking_budget_tokens': None,
        'max_output_tokens': None,
        'tool_spec_count': len(tools),
        'tool_choice': {'mode': 'session_trace'},
        'request_truncated': False,
        'response_truncated': 'missing_embedded_full_output' in lossy_reasons,
        'lossy_source': bool(lossy_reasons),
        'lossy_reasons': sorted(lossy_reasons),
        **counts,
    }

    record = {
        'id': f"{header.get('id')}:{leaf_id}",
        'request_id': header.get('id'),
        'messages': messages,
        'tools': tools,
        'meta': meta,
    }
    validate_record_payload(record)
    return record


def convert_file(path: Path) -> Tuple[List[Dict[str, Any]], Counter]:
    entries = read_jsonl(path)
    header = entries[0]
    body = entries[1:]
    by_id, children = build_tree(body)
    leaves = leaf_ids(by_id, children)
    stats = Counter()
    stats['input_files'] += 1
    stats['branches_total'] += len(leaves)
    records: List[Dict[str, Any]] = []
    for index, leaf_id in enumerate(leaves, start=1):
        branch_path = path_to_leaf(leaf_id, by_id)
        record = build_record_for_path(header, branch_path, path, leaf_id, index, len(leaves))
        records.append(record)
        if record['meta']['lossy_source']:
            stats['lossy_records'] += 1
        else:
            stats['strict_records'] += 1
    return records, stats


def load_qwen_records(input_path: Path) -> List[Dict[str, Any]]:
    files: List[Path] = []
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        for name in ('qwen35-train.jsonl', 'qwen35-train-lossy.jsonl'):
            candidate = input_path / name
            if candidate.exists():
                files.append(candidate)
    if not files:
        raise ConversionError(f'No Qwen35 jsonl files found under {input_path}')

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
                records.append(record)
    return records


def content_char_count(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, dict):
                if isinstance(block.get('text'), str):
                    total += len(block['text'])
                else:
                    total += len(json.dumps(block, ensure_ascii=False, sort_keys=True))
            else:
                total += len(str(block))
        return total
    return len(json.dumps(content, ensure_ascii=False, sort_keys=True))


def stat_summary(values: List[int]) -> Optional[Dict[str, Any]]:
    if not values:
        return None
    ordered = sorted(values)
    return {
        'min': ordered[0],
        'median': statistics.median(ordered),
        'mean': round(statistics.mean(ordered), 2),
        'max': ordered[-1],
    }


def build_stats_report(records: List[Dict[str, Any]], input_path: Path) -> Dict[str, Any]:
    message_counts: List[int] = []
    user_counts: List[int] = []
    assistant_counts: List[int] = []
    tool_counts: List[int] = []
    round_counts: List[int] = []
    tool_call_counts: List[int] = []
    reasoning_message_counts: List[int] = []
    reasoning_char_counts: List[int] = []
    total_char_counts: List[int] = []
    per_record: List[Dict[str, Any]] = []
    assistant_total = 0
    assistant_with_reasoning_total = 0

    for record in records:
        messages = record.get('messages', [])
        users = [message for message in messages if message.get('role') == 'user']
        assistants = [message for message in messages if message.get('role') == 'assistant']
        tools = [message for message in messages if message.get('role') == 'tool']
        reasoning_messages = [
            message
            for message in assistants
            if isinstance(message.get('reasoning_content'), str) and message['reasoning_content'].strip()
        ]
        reasoning_chars = sum(len(message['reasoning_content']) for message in reasoning_messages)
        tool_calls = sum(len(message.get('tool_calls') or []) for message in assistants)
        total_chars = sum(content_char_count(message.get('content')) for message in messages)

        message_counts.append(len(messages))
        user_counts.append(len(users))
        assistant_counts.append(len(assistants))
        tool_counts.append(len(tools))
        round_counts.append(len(users))
        tool_call_counts.append(tool_calls)
        reasoning_message_counts.append(len(reasoning_messages))
        reasoning_char_counts.append(reasoning_chars)
        total_char_counts.append(total_chars)
        assistant_total += len(assistants)
        assistant_with_reasoning_total += len(reasoning_messages)

        per_record.append(
            {
                'id': record.get('id'),
                'bucket': record.get('_bucket'),
                'messages': len(messages),
                'users': len(users),
                'assistants': len(assistants),
                'tools': len(tools),
                'dialogue_rounds_est': len(users),
                'tool_calls': tool_calls,
                'reasoning_messages': len(reasoning_messages),
                'reasoning_chars': reasoning_chars,
                'content_chars': total_chars,
                'lossy_reasons': record.get('meta', {}).get('lossy_reasons', []),
            }
        )

    records_with_reasoning = sum(1 for count in reasoning_message_counts if count > 0)
    report = {
        'input': str(input_path),
        'records': len(records),
        'strict_records': sum(1 for record in records if record.get('_bucket') == 'strict'),
        'lossy_records': sum(1 for record in records if record.get('_bucket') == 'lossy'),
        'message_count': stat_summary(message_counts),
        'user_messages': stat_summary(user_counts),
        'assistant_messages': stat_summary(assistant_counts),
        'tool_messages': stat_summary(tool_counts),
        'dialogue_rounds_est': stat_summary(round_counts),
        'assistant_tool_calls': stat_summary(tool_call_counts),
        'assistant_reasoning_messages': stat_summary(reasoning_message_counts),
        'reasoning_chars_total_per_record': stat_summary(reasoning_char_counts),
        'content_chars_total': stat_summary(total_char_counts),
        'records_with_reasoning': records_with_reasoning,
        'records_with_reasoning_ratio': round(records_with_reasoning / len(records), 4) if records else 0.0,
        'assistant_messages_with_reasoning': assistant_with_reasoning_total,
        'assistant_messages_total': assistant_total,
        'assistant_reasoning_coverage': round(assistant_with_reasoning_total / assistant_total, 4) if assistant_total else 0.0,
        'per_record': per_record,
    }
    return report


def print_stats_report(report: Dict[str, Any], as_json: bool) -> int:
    if as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    print(f"input: {report['input']}")
    print(f"records: {report['records']} (strict={report['strict_records']}, lossy={report['lossy_records']})")
    print(f"records with reasoning: {report['records_with_reasoning']} ({report['records_with_reasoning_ratio']:.2%})")
    print(
        f"assistant reasoning coverage: {report['assistant_messages_with_reasoning']}/"
        f"{report['assistant_messages_total']} ({report['assistant_reasoning_coverage']:.2%})"
    )
    print(f"message count: {report['message_count']}")
    print(f"dialogue rounds est: {report['dialogue_rounds_est']}")
    print(f"assistant tool calls: {report['assistant_tool_calls']}")
    print(f"assistant reasoning messages: {report['assistant_reasoning_messages']}")
    print(f"reasoning chars per record: {report['reasoning_chars_total_per_record']}")
    print('per record:')
    for item in report['per_record']:
        print(
            f"  - {item['id']} [{item['bucket']}] msgs={item['messages']} rounds={item['dialogue_rounds_est']} "
            f"tool_calls={item['tool_calls']} reasoning_msgs={item['reasoning_messages']} reasoning_chars={item['reasoning_chars']}"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or [])

    if args.command == 'stats':
        input_path = Path(args.input).expanduser().resolve()
        records = load_qwen_records(input_path)
        report = build_stats_report(records, input_path)
        return print_stats_report(report, args.json)

    ensure_parquet_runtime(args.output_format)

    input_path = Path(args.input).expanduser().resolve()
    input_files = iter_input_files(input_path, args.include_raw)
    if args.limit > 0:
        input_files = input_files[: args.limit]
    if not input_files:
        raise SystemExit('No exported Pi session files found.')

    out_dir = Path(args.output_root).expanduser().resolve() / f'qwen35-pi-session-{datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")}'
    out_dir.mkdir(parents=True, exist_ok=True)
    strict_path = out_dir / 'qwen35-train.jsonl'
    lossy_path = out_dir / 'qwen35-train-lossy.jsonl'
    invalid_path = out_dir / 'invalid-records.jsonl'
    strict_parquet_path = out_dir / 'qwen35-train.parquet'
    lossy_parquet_path = out_dir / 'qwen35-train-lossy.parquet'
    manifest_path = out_dir / 'manifest.json'

    stats = Counter()
    strict_out = strict_path.open('w', encoding='utf-8') if args.output_format in {'jsonl', 'both'} else None
    lossy_out = lossy_path.open('w', encoding='utf-8') if args.output_format in {'jsonl', 'both'} else None
    invalid_out = invalid_path.open('w', encoding='utf-8')
    strict_writer = None
    lossy_writer = None
    strict_batch: List[Dict[str, Any]] = []
    lossy_batch: List[Dict[str, Any]] = []

    try:
        for path in input_files:
            try:
                records, file_stats = convert_file(path)
                stats.update(file_stats)
            except Exception as exc:
                stats['invalid_files'] += 1
                invalid_out.write(json.dumps({'path': str(path), 'error': str(exc)}, ensure_ascii=False) + '\n')
                continue

            for record in records:
                bucket = 'lossy' if record['meta']['lossy_source'] else 'strict'
                if bucket == 'strict':
                    if strict_out is not None:
                        strict_out.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
                    if args.output_format in {'parquet', 'both'}:
                        strict_batch.append(record_to_parquet_row(record))
                        if len(strict_batch) >= BATCH_SIZE:
                            strict_writer = append_parquet_rows(strict_writer, strict_batch, strict_parquet_path)
                            strict_batch = []
                else:
                    if lossy_out is not None:
                        lossy_out.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
                    if args.output_format in {'parquet', 'both'}:
                        lossy_batch.append(record_to_parquet_row(record))
                        if len(lossy_batch) >= BATCH_SIZE:
                            lossy_writer = append_parquet_rows(lossy_writer, lossy_batch, lossy_parquet_path)
                            lossy_batch = []
            print(json.dumps({'processed_files': stats['input_files'], **dict(stats)}, ensure_ascii=False), flush=True)

        if args.output_format in {'parquet', 'both'}:
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
        'input': str(input_path),
        'output_dir': str(out_dir),
        'input_files': [str(path) for path in input_files],
        'stats': dict(stats),
        'strict_records': stats.get('strict_records', 0),
        'lossy_records': stats.get('lossy_records', 0),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(__import__('sys').argv[1:]))

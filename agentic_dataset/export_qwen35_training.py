#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - parquet is optional at runtime
    pa = None
    pq = None

PARQUET_SCHEMA = None
if pa is not None:  # pragma: no branch
    PARQUET_SCHEMA = pa.schema(
        [
            ('id', pa.string()),
            ('request_id', pa.string()),
            ('endpoint', pa.string()),
            ('status', pa.int64()),
            ('ts', pa.string()),
            ('key', pa.string()),
            ('source', pa.string()),
            ('requested_model', pa.string()),
            ('actual_model', pa.string()),
            ('stream', pa.bool_()),
            ('thinking_level', pa.string()),
            ('reasoning_summary_mode_json', pa.string()),
            ('thinking_type', pa.string()),
            ('thinking_budget_tokens', pa.int64()),
            ('max_output_tokens', pa.int64()),
            ('tool_spec_count', pa.int64()),
            ('tool_choice_json', pa.string()),
            ('request_contains_non_text_content', pa.bool_()),
            ('request_image_block_count', pa.int64()),
            ('request_video_block_count', pa.int64()),
            ('request_tool_call_block_count', pa.int64()),
            ('request_tool_result_block_count', pa.int64()),
            ('request_thinking_block_count', pa.int64()),
            ('response_contains_non_text_content', pa.bool_()),
            ('response_image_block_count', pa.int64()),
            ('response_video_block_count', pa.int64()),
            ('response_tool_call_block_count', pa.int64()),
            ('response_tool_result_block_count', pa.int64()),
            ('response_thinking_block_count', pa.int64()),
            ('request_truncated', pa.bool_()),
            ('response_truncated', pa.bool_()),
            ('lossy_source', pa.bool_()),
            ('lossy_reasons_json', pa.string()),
            ('user_message_count', pa.int64()),
            ('assistant_message_count', pa.int64()),
            ('tool_message_count', pa.int64()),
            ('dialogue_rounds_est', pa.int64()),
            ('tool_call_count', pa.int64()),
            ('has_reasoning', pa.bool_()),
            ('reasoning_chars', pa.int64()),
            ('content_chars_total', pa.int64()),
            ('messages_json', pa.string()),
            ('tools_json', pa.string()),
            ('meta_json', pa.string()),
        ]
    )

try:
    from .qwen35_training_record import Qwen35TrainingRecord
except Exception:  # pragma: no cover - remote runtime may not ship pydantic
    Qwen35TrainingRecord = None

MAIN_RE = re.compile(r'.*(?:🟢|⚠️ ?|❌|🟡)\s+(\d+)\s+(GET|POST|PUT|PATCH|DELETE)\s+(\S+)')
META_RE = re.compile(r'^\s*[├└]─\s+([^:]+):\s?(.*)$')
TARGET_PATHS = {
    'POST /openai/v1/responses',
    'POST /openai/v1/responses/compact',
    'POST /openai/v1/chat/completions',
    'POST /api/v1/messages',
}
TRUNCATED_MARKER = '...[truncated]'
TEXT_BLOCK_TYPES = {'text', 'input_text', 'output_text'}
IMAGE_BLOCK_TYPES = {'image', 'input_image', 'output_image', 'image_url'}
VIDEO_BLOCK_TYPES = {'video', 'input_video', 'output_video', 'video_url'}
TOOL_CALL_BLOCK_TYPES = {'tool_use', 'tool_call', 'function_call', 'custom_tool_call', 'web_search_call'}
TOOL_RESULT_BLOCK_TYPES = {'tool_result', 'tool_output', 'function_call_output'}
THINKING_BLOCK_TYPES = {'thinking', 'reasoning'}
VISION_IMAGE_TOKEN = '<|vision_start|><|image_pad|><|vision_end|>'
VISION_VIDEO_TOKEN = '<|vision_start|><|video_pad|><|vision_end|>'
THINK_INLINE_RE = re.compile(r'<think>\s*(.*?)\s*</think>', re.S)
TOOL_RESPONSE_RE = re.compile(r'<tool_response>\s*(.*?)\s*</tool_response>', re.S)
TOOL_CALL_RE = re.compile(r'<tool_call>\s*<function=([^>\n]+)>\s*(.*?)</function>\s*</tool_call>', re.S)
TOOL_PARAM_RE = re.compile(r'<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>', re.S)
VISION_TOKEN_RE = re.compile(
    f'({re.escape(VISION_IMAGE_TOKEN)}|{re.escape(VISION_VIDEO_TOKEN)})'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export relay logs into Qwen3.5-compatible JSONL.')
    parser.add_argument('--output-root', required=True)
    parser.add_argument(
        '--archive-root',
        default='/vePFS-Mindverse/share/yiwen/claude-relay-service/docker-json-logs/di-20260320122547-ws9d2/claude-relay-service-claude-relay-1',
    )
    parser.add_argument('--container', default='claude-relay-service-claude-relay-1')
    parser.add_argument('--include-current', action='store_true', default=True)
    parser.add_argument('--exclude-current', dest='include_current', action='store_false')
    parser.add_argument('--limit-sources', type=int, default=0)
    parser.add_argument('--dedupe-mode', choices=('requestid', 'content', 'none'), default='requestid')
    parser.add_argument('--workers', type=int, default=0, help='Thread workers for per-source staging; 0 means auto.')
    parser.add_argument(
        '--output-format',
        choices=('jsonl', 'parquet', 'both'),
        default='parquet',
        help='Emit JSONL, Parquet, or both. Parquet is the default and is optimized for analytics-first workflows.',
    )
    parser.add_argument('--keep-staging', action='store_true', help='Keep intermediate staged chunk files for debugging.')
    return parser.parse_args()


def resolve_current_log_path(container: str) -> str:
    cmd = ['bash', '-lc', f"export DOCKER_API_VERSION=1.43; docker inspect -f '{{{{.LogPath}}}}' {container}"]
    return subprocess.check_output(cmd, text=True).strip()


def sorted_archive_sources(archive_root: str) -> List[str]:
    if not os.path.isdir(archive_root):
        return []

    def sort_key(path: str) -> Tuple[int, str]:
        base = os.path.basename(path)
        head = base.split('_', 1)[0]
        try:
            return (int(head), base)
        except ValueError:
            return (0, base)

    files = [
        os.path.join(archive_root, name)
        for name in os.listdir(archive_root)
        if name.endswith('.gz') and os.path.isfile(os.path.join(archive_root, name))
    ]
    return sorted(files, key=sort_key)


def iter_sources(archive_root: str, current_log: Optional[str], limit: int) -> List[str]:
    sources = sorted_archive_sources(archive_root)
    if current_log:
        sources.append(current_log)
    if limit > 0:
        return sources[:limit]
    return sources


def open_log_file(path: str):
    if path.endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8', errors='replace')
    return open(path, 'r', encoding='utf-8', errors='replace')


def iter_events(paths: List[str]) -> Iterator[Dict[str, Any]]:
    current_event: Optional[Dict[str, Any]] = None
    current_key: Optional[str] = None
    current_source: Optional[str] = None

    for path in paths:
        current_source = path
        with open_log_file(path) as handle:
            for raw_line in handle:
                try:
                    obj = json.loads(raw_line)
                    log_line = obj.get('log', '').rstrip('\n')
                    ts = obj.get('time')
                except Exception:
                    continue

                main_match = MAIN_RE.match(log_line)
                if main_match:
                    if current_event is not None:
                        yield current_event
                    current_event = {
                        'source': current_source,
                        'ts': ts,
                        'status': int(main_match.group(1)),
                        'method': main_match.group(2),
                        'path': main_match.group(3),
                        'meta': {},
                    }
                    current_key = None
                    continue

                if current_event is None:
                    continue

                meta_match = META_RE.match(log_line)
                if meta_match:
                    current_key = meta_match.group(1)
                    current_event['meta'][current_key] = meta_match.group(2)
                    continue

                if current_key:
                    current_event['meta'][current_key] += log_line

    if current_event is not None:
        yield current_event


def parse_json_maybe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    try:
        return json.loads(value)
    except Exception:
        return None


def json_fallback(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def has_truncation(value: Any) -> bool:
    if isinstance(value, str):
        return TRUNCATED_MARKER in value
    if isinstance(value, list):
        return any(has_truncation(item) for item in value)
    if isinstance(value, dict):
        if value.get('_truncated'):
            return True
        return any(has_truncation(item) for item in value.values())
    return False


def normalize_role(role: Any) -> Optional[str]:
    mapping = {
        'system': 'system',
        'developer': 'system',
        'user': 'user',
        'assistant': 'assistant',
        'tool': 'tool',
        'model': 'assistant',
    }
    return mapping.get(role)


def get_text_from_block(block: Dict[str, Any]) -> str:
    if isinstance(block.get('text'), str):
        return block['text']
    if isinstance(block.get('content'), str):
        return block['content']
    if isinstance(block.get('reasoning'), str):
        return block['reasoning']
    if isinstance(block.get('thinking'), str):
        return block['thinking']
    if isinstance(block.get('content'), list):
        return flatten_text_only(block['content'])
    return ''


def flatten_text_only(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        block_type = value.get('type')
        if block_type in TEXT_BLOCK_TYPES | THINKING_BLOCK_TYPES:
            return get_text_from_block(value)
        if isinstance(value.get('content'), list):
            return flatten_text_only(value['content'])
        return ''
    if isinstance(value, list):
        parts = [flatten_text_only(item) for item in value]
        return '\n'.join(part for part in parts if part)
    return str(value)


def parse_parameter_value(value: str) -> Any:
    parsed = parse_json_maybe(value)
    if parsed is not None:
        return parsed
    return value


def split_inline_reasoning(text: str, lossy_reasons: set[str]) -> Tuple[Optional[str], str]:
    reasoning_parts: List[str] = []

    def replacer(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        if inner:
            reasoning_parts.append(inner)
        return ''

    cleaned = THINK_INLINE_RE.sub(replacer, text)
    if '<think>' in cleaned or '</think>' in cleaned:
        lossy_reasons.add('unbalanced_think_markup')
    reasoning = '\n\n'.join(part for part in reasoning_parts if part).strip()
    return (reasoning or None), cleaned.strip()


def split_vision_placeholder_text(text: str) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    parts = VISION_TOKEN_RE.split(text)
    for part in parts:
        if not part:
            continue
        if part == VISION_IMAGE_TOKEN:
            blocks.append(
                {
                    'type': 'image',
                    'placeholder': True,
                    'placeholder_token': VISION_IMAGE_TOKEN,
                    'source_kind': 'placeholder',
                }
            )
        elif part == VISION_VIDEO_TOKEN:
            blocks.append(
                {
                    'type': 'video',
                    'placeholder': True,
                    'placeholder_token': VISION_VIDEO_TOKEN,
                    'source_kind': 'placeholder',
                }
            )
        else:
            stripped = part.strip()
            if stripped:
                blocks.append({'type': 'text', 'text': stripped})
    return blocks


def extract_tool_calls_from_text(text: str, lossy_reasons: set[str]) -> Tuple[List[Dict[str, Any]], str]:
    tool_calls: List[Dict[str, Any]] = []

    def replacer(match: re.Match[str]) -> str:
        name = match.group(1).strip()
        body = match.group(2)
        arguments: Dict[str, Any] = {}
        for param_match in TOOL_PARAM_RE.finditer(body):
            param_name = param_match.group(1).strip()
            param_value = param_match.group(2).strip()
            if param_name:
                arguments[param_name] = parse_parameter_value(param_value)
        if not arguments and body.strip():
            lossy_reasons.add('tool_call_markup_without_parameters')
        tool_calls.append(
            {
                'type': 'function',
                'function': {
                    'name': name,
                    'arguments': arguments,
                },
            }
        )
        return ''

    cleaned = TOOL_CALL_RE.sub(replacer, text)
    if '<tool_call>' in cleaned or '<function=' in cleaned:
        lossy_reasons.add('unparsed_tool_call_markup')
    return tool_calls, cleaned.strip()


def extract_tool_responses_from_text(text: str) -> Tuple[List[str], str]:
    responses = [match.group(1).strip() for match in TOOL_RESPONSE_RE.finditer(text) if match.group(1).strip()]
    cleaned = TOOL_RESPONSE_RE.sub('', text).strip()
    return responses, cleaned


def parse_arguments_to_object(arguments: Any, lossy_reasons: set[str]) -> Dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        parsed = parse_json_maybe(arguments)
        if isinstance(parsed, dict):
            return parsed
        lossy_reasons.add('tool_arguments_not_object')
        return {}
    lossy_reasons.add('tool_arguments_not_object')
    return {}


def normalize_tool_call(call: Dict[str, Any], lossy_reasons: set[str]) -> Optional[Dict[str, Any]]:
    if not isinstance(call, dict):
        lossy_reasons.add('invalid_tool_call')
        return None

    call_type = call.get('type')
    call_id = call.get('id') or call.get('tool_call_id') or call.get('call_id')
    function_block = call.get('function') if isinstance(call.get('function'), dict) else None

    if function_block:
        name = function_block.get('name') or call.get('name')
        arguments = function_block.get('arguments')
    else:
        name = call.get('name')
        arguments = call.get('arguments')
        if arguments is None:
            arguments = call.get('input')

    if call_type == 'web_search_call' and (not isinstance(name, str) or not name):
        name = 'web_search'
        if arguments is None:
            payload: Dict[str, Any] = {}
            status = call.get('status')
            if isinstance(status, str):
                payload['status'] = status
            arguments = payload

    if call_type == 'custom_tool_call' and isinstance(arguments, str):
        arguments = {'input': arguments}

    if not isinstance(name, str) or not name:
        lossy_reasons.add('tool_call_missing_name')
        return None

    return {
        'type': 'function',
        'id': call_id,
        'function': {
            'name': name,
            'arguments': parse_arguments_to_object(arguments, lossy_reasons),
        },
    }


def normalize_tool_specs(raw_tools: Any) -> List[Dict[str, Any]]:
    if isinstance(raw_tools, str):
        raw_tools = parse_json_maybe(raw_tools)
    if not isinstance(raw_tools, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for tool in raw_tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get('name')
        if not isinstance(name, str) or not name:
            func = tool.get('function')
            if isinstance(func, dict):
                name = func.get('name')
        if not isinstance(name, str) or not name:
            continue
        item = dict(tool)
        item['name'] = name
        normalized.append(item)
    return normalized


def ensure_text_content(value: Any, lossy_reasons: set[str]) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    text = flatten_text_only(value)
    if text:
        return text
    lossy_reasons.add('non_text_tool_content')
    return json_fallback(value)


def normalize_image_block(item: Dict[str, Any]) -> Dict[str, Any]:
    image_url = None
    if isinstance(item.get('image_url'), str):
        image_url = item['image_url']
    elif isinstance(item.get('image_url'), dict):
        image_url = item['image_url'].get('url')
    elif isinstance(item.get('url'), str):
        image_url = item['url']
    return {
        'type': 'image',
        'image_url': image_url,
        'placeholder': image_url is None,
        'placeholder_token': item.get('placeholder_token') or '<|vision_start|><|image_pad|><|vision_end|>',
        'source_kind': item.get('type') or ('image_url' if image_url else 'placeholder'),
    }


def normalize_video_block(item: Dict[str, Any]) -> Dict[str, Any]:
    video_url = None
    if isinstance(item.get('video_url'), str):
        video_url = item['video_url']
    elif isinstance(item.get('video_url'), dict):
        video_url = item['video_url'].get('url')
    elif isinstance(item.get('url'), str):
        video_url = item['url']
    return {
        'type': 'video',
        'video_url': video_url,
        'placeholder': video_url is None,
        'placeholder_token': item.get('placeholder_token') or '<|vision_start|><|video_pad|><|vision_end|>',
        'source_kind': item.get('type') or ('video_url' if video_url else 'placeholder'),
    }


def finalize_content(blocks: List[Dict[str, Any]]) -> Any:
    if not blocks:
        return ''
    if all(block.get('type') == 'text' for block in blocks):
        return '\n'.join(block['text'] for block in blocks if block.get('text'))
    return blocks


def append_text_block(blocks: List[Dict[str, Any]], text: str) -> None:
    text = text.strip()
    if not text:
        return
    if blocks and blocks[-1].get('type') == 'text':
        blocks[-1]['text'] += '\n\n' + text
    else:
        blocks.append({'type': 'text', 'text': text})


def merge_initial_system_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    system_messages: List[Dict[str, Any]] = []
    rest: List[Dict[str, Any]] = []
    leading = True
    for message in messages:
        if leading and message.get('role') == 'system':
            system_messages.append(message)
        else:
            leading = False
            rest.append(message)

    if len(system_messages) <= 1:
        return messages

    merged_content_parts: List[str] = []
    for message in system_messages:
        content = render_content_for_system_merge(message.get('content'))
        if content.strip():
            merged_content_parts.append(content.strip())
    merged_system = {
        'role': 'system',
        'content': '\n\n'.join(merged_content_parts),
    }
    return [merged_system] + rest


def render_content_for_system_merge(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'text' and isinstance(block.get('text'), str):
                parts.append(block['text'])
        return '\n'.join(part for part in parts if part)
    return ''


def is_effectively_empty_content(content: Any) -> bool:
    if content is None:
        return True
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, list):
        if not content:
            return True
        for block in content:
            if not isinstance(block, dict):
                return False
            if block.get('type') != 'text':
                return False
            if isinstance(block.get('text'), str) and block['text'].strip():
                return False
        return True
    return False


def content_features(content: Any) -> Dict[str, int]:
    counts = Counter()
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get('type')
            if block_type == 'image':
                counts['image'] += 1
            elif block_type == 'video':
                counts['video'] += 1
            elif block_type not in (None, 'text'):
                counts['other'] += 1
    return counts


def parse_message(
    role: str,
    raw_content: Any,
    explicit_tool_calls: Any = None,
    explicit_reasoning: Optional[str] = None,
    explicit_tool_call_id: Optional[str] = None,
    explicit_tool_name: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Counter, set[str]]:
    lossy_reasons: set[str] = set()
    feature_counts = Counter()
    tool_messages: List[Dict[str, Any]] = []
    tool_calls: List[Dict[str, Any]] = []
    reasoning_parts: List[str] = []
    content_blocks: List[Dict[str, Any]] = []

    if explicit_tool_calls is not None:
        raw_tool_calls = explicit_tool_calls
        if isinstance(raw_tool_calls, list):
            for item in raw_tool_calls:
                normalized = normalize_tool_call(item, lossy_reasons)
                if normalized:
                    tool_calls.append(normalized)
                    feature_counts['tool_call'] += 1
    if explicit_reasoning:
        reasoning_parts.append(explicit_reasoning.strip())
        feature_counts['thinking'] += 1

    items: List[Any]
    if isinstance(raw_content, list):
        items = raw_content
    elif raw_content is None:
        items = []
    else:
        items = [raw_content]

    for item in items:
        if isinstance(item, str):
            text = item
            if role == 'assistant':
                inline_reasoning, text = split_inline_reasoning(text, lossy_reasons)
                if inline_reasoning:
                    reasoning_parts.append(inline_reasoning)
                    feature_counts['thinking'] += 1
                inline_tool_calls, text = extract_tool_calls_from_text(text, lossy_reasons)
                if inline_tool_calls:
                    tool_calls.extend(inline_tool_calls)
                    feature_counts['tool_call'] += len(inline_tool_calls)
            if role == 'user':
                tool_responses, text = extract_tool_responses_from_text(text)
                for payload in tool_responses:
                    tool_messages.append(
                        {
                            'role': 'tool',
                            'content': payload,
                            'tool_call_id': None,
                            'name': None,
                        }
                    )
                    feature_counts['tool_result'] += 1
            blocks = split_vision_placeholder_text(text)
            if role == 'system' and any(block['type'] in {'image', 'video'} for block in blocks):
                lossy_reasons.add('system_multimodal_not_supported')
                append_text_block(content_blocks, text)
            else:
                for block in blocks:
                    if block['type'] == 'text':
                        append_text_block(content_blocks, block['text'])
                    else:
                        content_blocks.append(block)
                        feature_counts[block['type']] += 1
            continue
        if not isinstance(item, dict):
            append_text_block(content_blocks, str(item))
            lossy_reasons.add('non_dict_content_item')
            continue

        block_type = item.get('type')
        if block_type in TEXT_BLOCK_TYPES or (
            'text' in item and block_type not in IMAGE_BLOCK_TYPES | VIDEO_BLOCK_TYPES | THINKING_BLOCK_TYPES
        ):
            text = item.get('text') if isinstance(item.get('text'), str) else None
            if text is None and isinstance(item.get('content'), str):
                text = item['content']
            if text is None:
                text = flatten_text_only(item)
            if role == 'assistant':
                inline_reasoning, text = split_inline_reasoning(text, lossy_reasons)
                if inline_reasoning:
                    reasoning_parts.append(inline_reasoning)
                    feature_counts['thinking'] += 1
                inline_tool_calls, text = extract_tool_calls_from_text(text, lossy_reasons)
                if inline_tool_calls:
                    tool_calls.extend(inline_tool_calls)
                    feature_counts['tool_call'] += len(inline_tool_calls)
            if role == 'user':
                tool_responses, text = extract_tool_responses_from_text(text)
                for payload in tool_responses:
                    tool_messages.append(
                        {
                            'role': 'tool',
                            'content': payload,
                            'tool_call_id': None,
                            'name': None,
                        }
                    )
                    feature_counts['tool_result'] += 1
            blocks = split_vision_placeholder_text(text)
            if role == 'system' and any(block['type'] in {'image', 'video'} for block in blocks):
                lossy_reasons.add('system_multimodal_not_supported')
                append_text_block(content_blocks, text)
            else:
                for block in blocks:
                    if block['type'] == 'text':
                        append_text_block(content_blocks, block['text'])
                    else:
                        content_blocks.append(block)
                        feature_counts[block['type']] += 1
        elif block_type in THINKING_BLOCK_TYPES:
            text = get_text_from_block(item)
            if text:
                reasoning_parts.append(text.strip())
                feature_counts['thinking'] += 1
        elif block_type in IMAGE_BLOCK_TYPES or 'image_url' in item or 'image' in item:
            if role == 'system':
                append_text_block(content_blocks, '[unsupported system image omitted]')
                lossy_reasons.add('system_multimodal_not_supported')
            else:
                content_blocks.append(normalize_image_block(item))
                feature_counts['image'] += 1
        elif block_type in VIDEO_BLOCK_TYPES or 'video_url' in item or 'video' in item:
            if role == 'system':
                append_text_block(content_blocks, '[unsupported system video omitted]')
                lossy_reasons.add('system_multimodal_not_supported')
            else:
                content_blocks.append(normalize_video_block(item))
                feature_counts['video'] += 1
        elif block_type in TOOL_CALL_BLOCK_TYPES:
            normalized = normalize_tool_call(item, lossy_reasons)
            if normalized:
                tool_calls.append(normalized)
                feature_counts['tool_call'] += 1
        elif block_type in TOOL_RESULT_BLOCK_TYPES:
            tool_messages.append(
                {
                    'role': 'tool',
                    'content': ensure_text_content(item.get('content') or item.get('text'), lossy_reasons),
                    'tool_call_id': item.get('tool_use_id') or item.get('tool_call_id') or item.get('id') or item.get('call_id'),
                    'name': item.get('name') or item.get('tool_name'),
                }
            )
            feature_counts['tool_result'] += 1
        else:
            if isinstance(item.get('content'), list):
                nested_messages, nested_features, nested_lossy = parse_message(
                    role,
                    item['content'],
                    item.get('tool_calls'),
                    item.get('reasoning_content') if isinstance(item.get('reasoning_content'), str) else None,
                    item.get('tool_call_id'),
                    item.get('name'),
                )
                feature_counts.update(nested_features)
                lossy_reasons.update(nested_lossy)
                if nested_messages:
                    primary = nested_messages[0]
                    primary_content = primary.get('content', '')
                    if isinstance(primary_content, str):
                        append_text_block(content_blocks, primary_content)
                    elif isinstance(primary_content, list):
                        content_blocks.extend(primary_content)
            else:
                append_text_block(content_blocks, json_fallback(item))
                lossy_reasons.add('unknown_content_block')

    message: Dict[str, Any] = {'role': role, 'content': finalize_content(content_blocks)}
    if role == 'assistant':
        reasoning = '\n\n'.join(part for part in reasoning_parts if part).strip()
        if reasoning:
            message['reasoning_content'] = reasoning
        if tool_calls:
            message['tool_calls'] = tool_calls
    if role == 'tool':
        if explicit_tool_call_id:
            message['tool_call_id'] = explicit_tool_call_id
        if explicit_tool_name:
            message['name'] = explicit_tool_name
        if not isinstance(message['content'], str):
            message['content'] = ensure_text_content(message['content'], lossy_reasons)
    messages = [message]
    messages.extend(tool_messages)
    return messages, feature_counts, lossy_reasons


def normalize_message_sequence(raw_messages: Any, endpoint: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if isinstance(raw_messages, str):
        parsed = parse_json_maybe(raw_messages)
        if parsed is not None:
            raw_messages = parsed
    messages: List[Dict[str, Any]] = []
    feature_counts = Counter()
    lossy_reasons: set[str] = set()

    if isinstance(raw_messages, str):
        msg_list, msg_features, msg_lossy = parse_message('user', raw_messages)
        messages.extend(msg_list)
        feature_counts.update(msg_features)
        lossy_reasons.update(msg_lossy)
    elif isinstance(raw_messages, dict):
        role = normalize_role(raw_messages.get('role') or raw_messages.get('type'))
        if role:
            msg_list, msg_features, msg_lossy = parse_message(
                role,
                raw_messages.get('content') if 'content' in raw_messages else raw_messages.get('text'),
                raw_messages.get('tool_calls'),
                raw_messages.get('reasoning_content') if isinstance(raw_messages.get('reasoning_content'), str) else None,
                raw_messages.get('tool_call_id'),
                raw_messages.get('name'),
            )
            messages.extend(msg_list)
            feature_counts.update(msg_features)
            lossy_reasons.update(msg_lossy)
    elif isinstance(raw_messages, list):
        for item in raw_messages:
            if isinstance(item, dict) and ('role' in item or item.get('type') == 'message'):
                role = normalize_role(item.get('role'))
                if not role:
                    lossy_reasons.add('unsupported_role')
                    continue
                msg_list, msg_features, msg_lossy = parse_message(
                    role,
                    item.get('content') if 'content' in item else item.get('text'),
                    item.get('tool_calls'),
                    item.get('reasoning_content') if isinstance(item.get('reasoning_content'), str) else None,
                    item.get('tool_call_id'),
                    item.get('name'),
                )
                messages.extend(msg_list)
                feature_counts.update(msg_features)
                lossy_reasons.update(msg_lossy)
            else:
                msg_list, msg_features, msg_lossy = parse_message('user', item)
                messages.extend(msg_list)
                feature_counts.update(msg_features)
                lossy_reasons.update(msg_lossy)

    # Merge consecutive messages with same role except tool role.
    merged: List[Dict[str, Any]] = []
    for message in messages:
        if (
            merged
            and merged[-1]['role'] == message['role']
            and message['role'] != 'tool'
            and 'tool_calls' not in merged[-1]
            and 'tool_calls' not in message
            and 'reasoning_content' not in merged[-1]
            and 'reasoning_content' not in message
        ):
            prev = merged[-1]
            if isinstance(prev['content'], str) and isinstance(message['content'], str):
                prev['content'] = (prev['content'] + '\n\n' + message['content']).strip()
            elif isinstance(prev['content'], list) and isinstance(message['content'], list):
                prev['content'].extend(message['content'])
            else:
                prev['content'] = ensure_text_content(prev['content'], lossy_reasons) + '\n\n' + ensure_text_content(message['content'], lossy_reasons)
        else:
            merged.append(message)

    system_messages = [message for message in merged if message['role'] == 'system']
    non_system_messages = [message for message in merged if message['role'] != 'system']
    if system_messages and merged[: len(system_messages)] != system_messages:
        lossy_reasons.add('system_reordered')
    merged = system_messages + non_system_messages

    flags = {
        'contains_non_text_content': feature_counts['image'] > 0 or feature_counts['video'] > 0,
        'image_block_count': feature_counts['image'],
        'video_block_count': feature_counts['video'],
        'tool_call_block_count': feature_counts['tool_call'],
        'tool_result_block_count': feature_counts['tool_result'],
        'thinking_block_count': feature_counts['thinking'],
        'lossy_reasons': sorted(lossy_reasons),
    }
    return merged, flags


def extract_request_meta(endpoint: str, req_obj: Dict[str, Any]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    requested_model = req_obj.get('model')
    if isinstance(requested_model, str) and requested_model:
        meta['requested_model'] = requested_model
    if isinstance(req_obj.get('stream'), bool):
        meta['stream'] = req_obj['stream']
    reasoning = req_obj.get('reasoning')
    if isinstance(reasoning, dict):
        if isinstance(reasoning.get('effort'), str):
            meta['thinking_level'] = reasoning['effort']
        if 'summary' in reasoning:
            meta['reasoning_summary_mode'] = reasoning['summary']
    thinking = req_obj.get('thinking')
    if isinstance(thinking, dict):
        if isinstance(thinking.get('type'), str):
            meta['thinking_type'] = thinking['type']
        if isinstance(thinking.get('budget_tokens'), int):
            meta['thinking_budget_tokens'] = thinking['budget_tokens']
    if isinstance(req_obj.get('max_output_tokens'), int):
        meta['max_output_tokens'] = req_obj['max_output_tokens']
    elif isinstance(req_obj.get('max_tokens'), int):
        meta['max_output_tokens'] = req_obj['max_tokens']
    tools = normalize_tool_specs(req_obj.get('tools'))
    if tools:
        meta['tool_spec_count'] = len(tools)
    tool_choice = req_obj.get('tool_choice') or req_obj.get('toolChoice')
    if tool_choice is not None:
        meta['tool_choice'] = tool_choice
    return meta


def extract_response_meta(endpoint: str, res_obj: Dict[str, Any]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    body = res_obj.get('response') if isinstance(res_obj.get('response'), dict) else res_obj
    if isinstance(body, dict):
        actual_model = body.get('model')
        if isinstance(actual_model, str) and actual_model:
            meta['actual_model'] = actual_model
        usage = body.get('usage')
        if isinstance(usage, dict):
            total_tokens = usage.get('total_tokens')
            if isinstance(total_tokens, int):
                meta['total_tokens'] = total_tokens
    return meta


def normalize_request_messages(endpoint: str, req_obj: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    flags_total = Counter()
    lossy_reasons: set[str] = set()

    def absorb(seq: List[Dict[str, Any]], flags: Dict[str, Any]) -> None:
        messages.extend(seq)
        flags_total['image'] += flags['image_block_count']
        flags_total['video'] += flags['video_block_count']
        flags_total['tool_call'] += flags['tool_call_block_count']
        flags_total['tool_result'] += flags['tool_result_block_count']
        flags_total['thinking'] += flags['thinking_block_count']
        if flags['contains_non_text_content']:
            flags_total['non_text'] += 1
        lossy_reasons.update(flags['lossy_reasons'])

    if endpoint in ('POST /openai/v1/responses', 'POST /openai/v1/responses/compact'):
        instructions = req_obj.get('instructions')
        if instructions:
            messages.append({'role': 'system', 'content': str(instructions)})
        seq, flags = normalize_message_sequence(req_obj.get('input'), endpoint)
        absorb(seq, flags)
    elif endpoint == 'POST /openai/v1/chat/completions':
        instructions = req_obj.get('instructions')
        if instructions:
            messages.append({'role': 'system', 'content': str(instructions)})
        seq, flags = normalize_message_sequence(req_obj.get('messages'), endpoint)
        absorb(seq, flags)
    elif endpoint == 'POST /api/v1/messages':
        system_content = req_obj.get('system')
        if system_content is not None:
            seq, flags = normalize_message_sequence([{'role': 'system', 'content': system_content}], endpoint)
            absorb(seq, flags)
        seq, flags = normalize_message_sequence(req_obj.get('messages'), endpoint)
        absorb(seq, flags)
    else:
        seq, flags = normalize_message_sequence(req_obj, endpoint)
        absorb(seq, flags)

    if not any(message['role'] == 'user' for message in messages):
        lossy_reasons.add('missing_user_query')

    merged_messages = merge_initial_system_messages(messages)
    if len(merged_messages) != len(messages):
        lossy_reasons.add('merged_initial_system_messages')
    messages = merged_messages

    return messages, {
        'contains_non_text_content': bool(flags_total['non_text']),
        'image_block_count': flags_total['image'],
        'video_block_count': flags_total['video'],
        'tool_call_block_count': flags_total['tool_call'],
        'tool_result_block_count': flags_total['tool_result'],
        'thinking_block_count': flags_total['thinking'],
        'lossy_reasons': sorted(lossy_reasons),
    }


def normalize_response_messages(endpoint: str, res_obj: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not isinstance(res_obj, dict):
        return [], {
            'contains_non_text_content': False,
            'image_block_count': 0,
            'video_block_count': 0,
            'tool_call_block_count': 0,
            'tool_result_block_count': 0,
            'thinking_block_count': 0,
            'lossy_reasons': [],
        }

    body = res_obj.get('response') if isinstance(res_obj.get('response'), dict) else res_obj
    if endpoint in ('POST /openai/v1/responses', 'POST /openai/v1/responses/compact') and isinstance(body, dict):
        output = body.get('output')
        if isinstance(output, list) and output:
            if all(isinstance(item, dict) and 'role' not in item and item.get('type') != 'message' for item in output):
                messages, features, lossy = parse_message('assistant', output)
                return messages, {
                    'contains_non_text_content': features['image'] > 0 or features['video'] > 0,
                    'image_block_count': features['image'],
                    'video_block_count': features['video'],
                    'tool_call_block_count': features['tool_call'],
                    'tool_result_block_count': features['tool_result'],
                    'thinking_block_count': features['thinking'],
                    'lossy_reasons': sorted(lossy),
                }
            messages, flags = normalize_message_sequence(output, endpoint)
            if messages:
                return messages, flags
        if isinstance(body.get('output_text'), str) and body.get('output_text').strip():
            messages, flags = normalize_message_sequence([{'role': 'assistant', 'content': body['output_text']}], endpoint)
            return messages, flags
    if endpoint == 'POST /openai/v1/chat/completions':
        choices = body.get('choices') if isinstance(body, dict) else None
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            msg = choices[0].get('message')
            messages, flags = normalize_message_sequence([msg], endpoint)
            return messages, flags
    if endpoint == 'POST /api/v1/messages':
        messages, flags = normalize_message_sequence([{'role': 'assistant', 'content': body.get('content')}], endpoint)
        return messages, flags
    return [], {
        'contains_non_text_content': False,
        'image_block_count': 0,
        'video_block_count': 0,
        'tool_call_block_count': 0,
        'tool_result_block_count': 0,
        'thinking_block_count': 0,
        'lossy_reasons': [],
    }


def record_hash(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> str:
    payload = json.dumps({'messages': messages, 'tools': tools}, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def dedupe_key(mode: str, record_id: str, request_id: Optional[str], messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Optional[str]:
    if mode == 'none':
        return None
    if mode == 'requestid':
        return request_id or record_id
    return record_hash(messages, tools)


def lightweight_validate_record(payload: Dict[str, Any]) -> None:
    messages = payload.get('messages') or []
    if not messages:
        raise ValueError('messages must not be empty')

    seen_user = False
    seen_non_system = False
    for message in messages:
        role = message.get('role')
        if role != 'system':
            seen_non_system = True
        elif seen_non_system:
            raise ValueError('system messages must appear only at the beginning')

        if role == 'user':
            seen_user = True
        if role == 'system' and isinstance(message.get('content'), list):
            if any(isinstance(block, dict) and block.get('type') in {'image', 'video'} for block in message['content']):
                raise ValueError('system messages cannot contain image/video blocks')
        if role == 'assistant':
            reasoning = message.get('reasoning_content')
            if isinstance(reasoning, str) and ('<think>' in reasoning or '</think>' in reasoning):
                raise ValueError('reasoning_content must not contain think wrappers')
            content = message.get('content')
            if isinstance(content, str) and ('<think>' in content or '</think>' in content):
                raise ValueError('assistant content must not contain inline think wrappers')

    if not seen_user:
        raise ValueError('at least one user message is required')

    meta = payload.get('meta') or {}
    if meta.get('lossy_source') and not meta.get('lossy_reasons'):
        raise ValueError('lossy_source requires lossy_reasons')


def validate_record_payload(payload: Dict[str, Any]) -> Any:
    if Qwen35TrainingRecord is not None:
        return Qwen35TrainingRecord.model_validate(payload)
    lightweight_validate_record(payload)
    return payload


def record_messages_and_tools(record: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if Qwen35TrainingRecord is not None and hasattr(record, 'messages'):
        return (
            [message.model_dump(exclude_none=True) for message in record.messages],
            [tool.model_dump(exclude_none=True) for tool in record.tools],
        )
    return record['messages'], record.get('tools', [])


def record_is_lossy(record: Any) -> bool:
    if Qwen35TrainingRecord is not None and hasattr(record, 'meta'):
        return bool(record.meta.lossy_source)
    return bool(record.get('meta', {}).get('lossy_source'))


def record_id_value(record: Any) -> str:
    return record.id if hasattr(record, 'id') else record['id']


def record_request_id_value(record: Any) -> Optional[str]:
    return record.request_id if hasattr(record, 'request_id') else record.get('request_id')


def record_dump_json(record: Any) -> str:
    if Qwen35TrainingRecord is not None and hasattr(record, 'model_dump_json'):
        return record.model_dump_json(exclude_none=True)
    return json.dumps(record, ensure_ascii=False, separators=(',', ':'))


def record_as_dict(record: Any) -> Dict[str, Any]:
    if Qwen35TrainingRecord is not None and hasattr(record, 'model_dump'):
        return record.model_dump(exclude_none=True)
    return record


def parquet_content_projection(content: Any) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    if isinstance(content, str):
        return content, []
    blocks: List[Dict[str, Any]] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            blocks.append(
                {
                    'type': block.get('type'),
                    'text': block.get('text'),
                    'image_url': block.get('image_url'),
                    'video_url': block.get('video_url'),
                    'placeholder': block.get('placeholder'),
                    'placeholder_token': block.get('placeholder_token'),
                    'source_kind': block.get('source_kind'),
                    'metadata_json': json.dumps(block.get('metadata'), ensure_ascii=False, sort_keys=True)
                    if isinstance(block.get('metadata'), dict)
                    else None,
                }
            )
    return None, blocks


def parquet_tool_calls_projection(tool_calls: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(tool_calls, list):
        return rows
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get('function') if isinstance(call.get('function'), dict) else {}
        rows.append(
            {
                'id': call.get('id'),
                'type': call.get('type'),
                'function_name': function.get('name'),
                'function_arguments_json': json.dumps(function.get('arguments', {}), ensure_ascii=False, sort_keys=True),
            }
        )
    return rows


def parquet_tools_projection(tools: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(tools, list):
        return rows
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        rows.append(
            {
                'name': tool.get('name'),
                'description': tool.get('description'),
                'parameters_json': json.dumps(tool.get('parameters'), ensure_ascii=False, sort_keys=True)
                if tool.get('parameters') is not None
                else None,
                'raw_json': json.dumps(tool, ensure_ascii=False, sort_keys=True),
            }
        )
    return rows


def record_to_parquet_row(record: Dict[str, Any]) -> Dict[str, Any]:
    meta = dict(record.get('meta', {}))
    messages = record.get('messages', []) if isinstance(record.get('messages'), list) else []
    user_message_count = sum(1 for message in messages if isinstance(message, dict) and message.get('role') == 'user')
    assistant_message_count = sum(1 for message in messages if isinstance(message, dict) and message.get('role') == 'assistant')
    tool_message_count = sum(1 for message in messages if isinstance(message, dict) and message.get('role') == 'tool')
    tool_call_count = sum(
        len(message.get('tool_calls') or [])
        for message in messages
        if isinstance(message, dict) and message.get('role') == 'assistant'
    )
    has_reasoning = any(
        isinstance(message, dict)
        and message.get('role') == 'assistant'
        and isinstance(message.get('reasoning_content'), str)
        and bool(message.get('reasoning_content').strip())
        for message in messages
    )
    reasoning_chars = sum(
        len(message.get('reasoning_content', ''))
        for message in messages
        if isinstance(message, dict)
        and message.get('role') == 'assistant'
        and isinstance(message.get('reasoning_content'), str)
    )
    content_chars_total = sum(
        len(message.get('content'))
        if isinstance(message.get('content'), str)
        else len(json.dumps(message.get('content'), ensure_ascii=False, sort_keys=True))
        for message in messages
        if isinstance(message, dict) and message.get('content') is not None
    )
    return {
        'id': record.get('id'),
        'request_id': record.get('request_id'),
        'endpoint': meta.get('endpoint'),
        'status': meta.get('status'),
        'ts': meta.get('ts'),
        'key': meta.get('key'),
        'source': meta.get('source'),
        'requested_model': meta.get('requested_model'),
        'actual_model': meta.get('actual_model'),
        'stream': meta.get('stream'),
        'thinking_level': meta.get('thinking_level'),
        'reasoning_summary_mode_json': json.dumps(meta.get('reasoning_summary_mode'), ensure_ascii=False, sort_keys=True),
        'thinking_type': meta.get('thinking_type'),
        'thinking_budget_tokens': meta.get('thinking_budget_tokens'),
        'max_output_tokens': meta.get('max_output_tokens'),
        'tool_spec_count': meta.get('tool_spec_count'),
        'tool_choice_json': json.dumps(meta.get('tool_choice'), ensure_ascii=False, sort_keys=True),
        'request_contains_non_text_content': meta.get('request_contains_non_text_content'),
        'request_image_block_count': meta.get('request_image_block_count'),
        'request_video_block_count': meta.get('request_video_block_count'),
        'request_tool_call_block_count': meta.get('request_tool_call_block_count'),
        'request_tool_result_block_count': meta.get('request_tool_result_block_count'),
        'request_thinking_block_count': meta.get('request_thinking_block_count'),
        'response_contains_non_text_content': meta.get('response_contains_non_text_content'),
        'response_image_block_count': meta.get('response_image_block_count'),
        'response_video_block_count': meta.get('response_video_block_count'),
        'response_tool_call_block_count': meta.get('response_tool_call_block_count'),
        'response_tool_result_block_count': meta.get('response_tool_result_block_count'),
        'response_thinking_block_count': meta.get('response_thinking_block_count'),
        'request_truncated': meta.get('request_truncated'),
        'response_truncated': meta.get('response_truncated'),
        'lossy_source': meta.get('lossy_source'),
        'lossy_reasons_json': json.dumps(meta.get('lossy_reasons', []), ensure_ascii=False, sort_keys=True),
        'user_message_count': user_message_count,
        'assistant_message_count': assistant_message_count,
        'tool_message_count': tool_message_count,
        'dialogue_rounds_est': user_message_count,
        'tool_call_count': tool_call_count,
        'has_reasoning': has_reasoning,
        'reasoning_chars': reasoning_chars,
        'content_chars_total': content_chars_total,
        'messages_json': json.dumps(record.get('messages', []), ensure_ascii=False, sort_keys=True),
        'tools_json': json.dumps(record.get('tools', []), ensure_ascii=False, sort_keys=True),
        'meta_json': json.dumps(record.get('meta', {}), ensure_ascii=False, sort_keys=True),
    }


def auto_worker_count(requested_workers: int, source_count: int) -> int:
    if requested_workers > 0:
        return max(1, requested_workers)
    cpu = os.cpu_count() or 4
    return max(1, min(source_count, min(cpu, 8)))


def ensure_parquet_runtime(output_format: str) -> None:
    if output_format in {'parquet', 'both'} and (pa is None or pq is None):
        raise RuntimeError('pyarrow is required for Parquet output')


def build_staged_entry_from_event(event: Dict[str, Any], endpoint: str, event_index: int, dedupe_mode: str) -> Tuple[Optional[Dict[str, Any]], Counter]:
    stats = Counter()
    stats[f'events:{endpoint}'] += 1

    req_obj = parse_json_maybe(event['meta'].get('req'))
    if not isinstance(req_obj, dict):
        stats[f'bad_req:{endpoint}'] += 1
        return None, stats

    tools = normalize_tool_specs(req_obj.get('tools'))
    request_messages, request_flags = normalize_request_messages(endpoint, req_obj)
    response_obj = parse_json_maybe(event['meta'].get('res'))
    response_messages, response_flags = normalize_response_messages(endpoint, response_obj)

    messages = request_messages + response_messages
    if not messages:
        stats[f'empty_messages:{endpoint}'] += 1
        return None, stats

    request_id = event['meta'].get('requestId')
    lossy_reasons = set(request_flags['lossy_reasons']) | set(response_flags['lossy_reasons'])
    request_truncated = has_truncation(req_obj)
    response_truncated = has_truncation(response_obj)
    if request_truncated:
        lossy_reasons.add('request_truncated')
    if response_truncated:
        lossy_reasons.add('response_truncated')

    record_id = f"{endpoint}:{event.get('ts')}:{event_index}"
    meta = {
        'endpoint': endpoint,
        'status': event['status'],
        'ts': event.get('ts') or '',
        'key': event['meta'].get('key'),
        'source': event.get('source'),
        'request_contains_non_text_content': request_flags['contains_non_text_content'],
        'request_image_block_count': request_flags['image_block_count'],
        'request_video_block_count': request_flags['video_block_count'],
        'request_tool_call_block_count': request_flags['tool_call_block_count'],
        'request_tool_result_block_count': request_flags['tool_result_block_count'],
        'request_thinking_block_count': request_flags['thinking_block_count'],
        'response_contains_non_text_content': response_flags['contains_non_text_content'],
        'response_image_block_count': response_flags['image_block_count'],
        'response_video_block_count': response_flags['video_block_count'],
        'response_tool_call_block_count': response_flags['tool_call_block_count'],
        'response_tool_result_block_count': response_flags['tool_result_block_count'],
        'response_thinking_block_count': response_flags['thinking_block_count'],
        'request_truncated': request_truncated,
        'response_truncated': response_truncated,
        'lossy_source': bool(lossy_reasons),
        'lossy_reasons': sorted(lossy_reasons),
    }
    meta.update(extract_request_meta(endpoint, req_obj))
    if isinstance(response_obj, dict):
        meta.update(extract_response_meta(endpoint, response_obj))

    try:
        record = validate_record_payload(
            {
                'id': record_id,
                'request_id': request_id,
                'messages': messages,
                'tools': tools,
                'meta': meta,
            }
        )
    except Exception as exc:
        stats[f'invalid_record:{endpoint}'] += 1
        stats['invalid_records_total'] += 1
        return {
            'bucket': 'invalid',
            'endpoint': endpoint,
            'error': str(exc),
            'event_index': event_index,
        }, stats

    record_dict = record_as_dict(record)
    key = dedupe_key(
        dedupe_mode,
        record_id_value(record),
        record_request_id_value(record),
        record_dict['messages'],
        record_dict.get('tools', []),
    )
    bucket = 'lossy' if record_is_lossy(record) else 'strict'
    stats[f'{bucket}_records_staged'] += 1
    return {'bucket': bucket, 'dedupe_key': key, 'record': record_dict}, stats


def process_source_to_stage(source_path: str, staging_dir: Path, dedupe_mode: str) -> Dict[str, Any]:
    stats = Counter()
    safe_name = hashlib.sha1(source_path.encode('utf-8')).hexdigest()[:16]
    chunk_path = staging_dir / f'{safe_name}.jsonl'
    invalid_path = staging_dir / f'{safe_name}.invalid.jsonl'

    with chunk_path.open('w', encoding='utf-8') as chunk_out, invalid_path.open('w', encoding='utf-8') as invalid_out:
        for local_index, event in enumerate(iter_events([source_path]), start=1):
            endpoint = f"{event['method']} {event['path']}"
            if endpoint not in TARGET_PATHS:
                continue
            staged_entry, event_stats = build_staged_entry_from_event(event, endpoint, local_index, dedupe_mode)
            stats.update(event_stats)
            if not staged_entry:
                continue
            if staged_entry['bucket'] == 'invalid':
                invalid_out.write(json.dumps(staged_entry, ensure_ascii=False) + '\n')
            else:
                chunk_out.write(json.dumps(staged_entry, ensure_ascii=False) + '\n')

    return {
        'source': source_path,
        'chunk_path': str(chunk_path),
        'invalid_path': str(invalid_path),
        'stats': dict(stats),
    }


def append_parquet_rows(writer: Any, rows: List[Dict[str, Any]], path: Path) -> Any:
    if not rows:
        return writer
    table = pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA)
    if writer is None:
        writer = pq.ParquetWriter(str(path), PARQUET_SCHEMA)
    writer.write_table(table)
    return writer


def main() -> int:
    args = parse_args()
    ensure_parquet_runtime(args.output_format)
    current_log = resolve_current_log_path(args.container) if args.include_current else None
    sources = iter_sources(args.archive_root, current_log, args.limit_sources)
    if not sources:
        raise SystemExit('No log sources found.')

    out_dir = Path(args.output_root) / f'qwen35-export-{datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")}'
    out_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = out_dir / 'staging'
    staging_dir.mkdir(parents=True, exist_ok=True)
    strict_path = out_dir / 'qwen35-train.jsonl'
    lossy_path = out_dir / 'qwen35-train-lossy.jsonl'
    strict_parquet_path = out_dir / 'qwen35-train.parquet'
    lossy_parquet_path = out_dir / 'qwen35-train-lossy.parquet'
    invalid_path = out_dir / 'invalid-records.jsonl'
    manifest_path = out_dir / 'manifest.json'

    stats = Counter()
    worker_count = auto_worker_count(args.workers, len(sources))
    worker_results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(process_source_to_stage, source_path, staging_dir, args.dedupe_mode): source_path
            for source_path in sources
        }
        files_done = 0
        for future in as_completed(futures):
            result = future.result()
            worker_results.append(result)
            stats.update(result['stats'])
            files_done += 1
            print(
                json.dumps(
                    {'files_done': files_done, 'sources_total': len(sources), **dict(stats)},
                    ensure_ascii=False,
                ),
                flush=True,
            )

    jsonl_enabled = args.output_format in {'jsonl', 'both'}
    parquet_enabled = args.output_format in {'parquet', 'both'}
    strict_seen: set[str] = set()
    lossy_seen: set[str] = set()
    strict_writer = None
    lossy_writer = None
    strict_batch: List[Dict[str, Any]] = []
    lossy_batch: List[Dict[str, Any]] = []
    batch_size = 1000

    strict_out = strict_path.open('w', encoding='utf-8') if jsonl_enabled else None
    lossy_out = lossy_path.open('w', encoding='utf-8') if jsonl_enabled else None
    invalid_out = invalid_path.open('w', encoding='utf-8')

    try:
        for result in sorted(worker_results, key=lambda item: item['source']):
            chunk_path = Path(result['chunk_path'])
            invalid_chunk_path = Path(result['invalid_path'])

            if invalid_chunk_path.exists():
                with invalid_chunk_path.open('r', encoding='utf-8') as invalid_in:
                    for line in invalid_in:
                        invalid_out.write(line)

            if chunk_path.exists():
                with chunk_path.open('r', encoding='utf-8') as chunk_in:
                    for line in chunk_in:
                        if not line.strip():
                            continue
                        staged = json.loads(line)
                        bucket = staged['bucket']
                        dedupe = staged.get('dedupe_key')
                        record = staged['record']
                        seen = strict_seen if bucket == 'strict' else lossy_seen
                        if dedupe is not None and dedupe in seen:
                            stats[f'{bucket}_records_deduped'] += 1
                            continue
                        if dedupe is not None:
                            seen.add(dedupe)

                        if bucket == 'strict':
                            if strict_out is not None:
                                strict_out.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
                            if parquet_enabled:
                                strict_batch.append(record_to_parquet_row(record))
                                if len(strict_batch) >= batch_size:
                                    strict_writer = append_parquet_rows(strict_writer, strict_batch, strict_parquet_path)
                                    strict_batch = []
                            stats['strict_records_written'] += 1
                        else:
                            if lossy_out is not None:
                                lossy_out.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
                            if parquet_enabled:
                                lossy_batch.append(record_to_parquet_row(record))
                                if len(lossy_batch) >= batch_size:
                                    lossy_writer = append_parquet_rows(lossy_writer, lossy_batch, lossy_parquet_path)
                                    lossy_batch = []
                            stats['lossy_records_written'] += 1

            if not args.keep_staging:
                if chunk_path.exists():
                    chunk_path.unlink()
                if invalid_chunk_path.exists():
                    invalid_chunk_path.unlink()

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

    if not args.keep_staging:
        try:
            staging_dir.rmdir()
        except OSError:
            pass

    manifest = {
        'output_dir': str(out_dir),
        'source_count': len(sources),
        'sources': sources,
        'workers': worker_count,
        'dedupe_mode': args.dedupe_mode,
        'output_format': args.output_format,
        'strict_records': stats['strict_records_written'],
        'lossy_records': stats['lossy_records_written'],
        'stats': dict(stats),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

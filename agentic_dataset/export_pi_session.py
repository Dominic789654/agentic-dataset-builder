#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

DEFAULT_SESSION_ROOT = Path.home() / '.pi' / 'agent' / 'sessions'
EXPORT_VERSION = 1
TEXT_SAMPLE_LIMIT = 4000
RAW_MODE = 'raw'
FULL_MODE = 'full'
SUPPORTED_MODES = (FULL_MODE, RAW_MODE)
ARTIFACT_KEY = 'fullOutputPath'


class SessionExportError(RuntimeError):
    pass


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    command_names = {'list', 'verify', 'health'}
    argv_list = list(argv)
    command = 'export'
    if argv_list and argv_list[0] in command_names:
        command = argv_list[0]
        argv_list = argv_list[1:]

    parser = argparse.ArgumentParser(
        description='Export Pi sessions into self-contained JSONL files.',
    )
    parser.add_argument('--session-root', default=str(DEFAULT_SESSION_ROOT), help='Pi session root directory.')
    parser.add_argument('--cwd', default=os.getcwd(), help='Project directory used for default session lookup.')

    if command == 'list':
        parser.add_argument('--all', action='store_true', help='List sessions across every project directory.')
        parser.add_argument('--json', action='store_true', help='Emit machine-readable JSON.')
        args = parser.parse_args(argv_list)
        args.command = command
        return args

    if command == 'verify':
        parser.add_argument('path', help='Exported full-session JSONL to verify.')
        parser.add_argument('--json', action='store_true', help='Emit machine-readable JSON.')
        args = parser.parse_args(argv_list)
        args.command = command
        return args

    if command == 'health':
        parser.add_argument('--all', action='store_true', help='Scan every project directory instead of only the current cwd.')
        parser.add_argument('--json', action='store_true', help='Emit machine-readable JSON.')
        args = parser.parse_args(argv_list)
        args.command = command
        return args

    parser.add_argument('--session', help='Session file path, session UUID, or unique UUID prefix.')
    parser.add_argument('--select', action='store_true', help='Interactively choose a session before exporting.')
    parser.add_argument('--mode', choices=SUPPORTED_MODES, default=FULL_MODE, help='Export mode.')
    parser.add_argument('--out', help='Output file or directory. Defaults to ./pi-session-exports/.')
    parser.add_argument('--json', action='store_true', help='Emit machine-readable JSON.')
    args = parser.parse_args(argv_list)
    args.command = command
    return args


def session_dir_name(cwd: Path) -> str:
    normalized = str(cwd.resolve()).strip('/')
    return f"--{normalized.replace('/', '-')}--"


def iter_session_files(session_root: Path) -> Iterator[Path]:
    if not session_root.exists():
        return iter(())
    return session_root.rglob('*.jsonl')


def session_sort_key(path: Path) -> Tuple[str, str]:
    return (path.name, str(path))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    lines: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            text = raw_line.strip()
            if not text:
                continue
            try:
                lines.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise SessionExportError(f'Invalid JSON at {path}:{line_number}: {exc}') from exc
    if not lines:
        raise SessionExportError(f'Session file is empty: {path}')
    return lines


def load_session_metadata(path: Path) -> Dict[str, Any]:
    entries = read_jsonl(path)
    header = entries[0]
    session_name: Optional[str] = None
    first_user_text: Optional[str] = None
    for entry in entries[1:]:
        if entry.get('type') == 'session_info' and isinstance(entry.get('name'), str) and entry['name'].strip():
            session_name = entry['name'].strip()
        if first_user_text is None and entry.get('type') == 'message':
            message = entry.get('message')
            if isinstance(message, dict) and message.get('role') == 'user':
                first_user_text = extract_message_text(message.get('content'))
    return {
        'path': path,
        'header': header,
        'session_id': header.get('id'),
        'cwd': header.get('cwd'),
        'timestamp': header.get('timestamp'),
        'name': session_name,
        'preview': first_user_text,
    }


def extract_message_text(content: Any) -> Optional[str]:
    if isinstance(content, str):
        return compact_text(content)
    if not isinstance(content, list):
        return None
    parts: List[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        text = block.get('text')
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
        if len(' '.join(parts)) >= 160:
            break
    if not parts:
        return None
    return compact_text(' '.join(parts))


def compact_text(value: str, limit: int = 80) -> str:
    single_line = ' '.join(value.split())
    if len(single_line) <= limit:
        return single_line
    return single_line[: limit - 3] + '...'


def list_project_sessions(session_root: Path, cwd: Path) -> List[Dict[str, Any]]:
    directory = session_root / session_dir_name(cwd)
    if not directory.exists():
        return []
    paths = sorted(directory.glob('*.jsonl'), key=session_sort_key, reverse=True)
    return [load_session_metadata(path) for path in paths]


def list_all_sessions(session_root: Path) -> List[Dict[str, Any]]:
    paths = sorted(iter_session_files(session_root), key=session_sort_key, reverse=True)
    return [load_session_metadata(path) for path in paths]


def resolve_session(session_root: Path, cwd: Path, session_value: Optional[str], select: bool) -> Dict[str, Any]:
    if select:
        candidates = list_project_sessions(session_root, cwd)
        if not candidates:
            raise SessionExportError(f'No sessions found for cwd {cwd}')
        return select_session_interactively(candidates)

    if session_value:
        resolved = resolve_session_value(session_root, cwd, session_value)
        return load_session_metadata(resolved)

    candidates = list_project_sessions(session_root, cwd)
    if not candidates:
        raise SessionExportError(
            f'No sessions found for cwd {cwd}. Use --session to specify one or --select to choose interactively.'
        )
    return candidates[0]


def resolve_session_value(session_root: Path, cwd: Path, value: str) -> Path:
    candidate_path = Path(value).expanduser()
    if candidate_path.exists():
        return candidate_path.resolve()

    cwd_relative = (cwd / value).expanduser()
    if cwd_relative.exists():
        return cwd_relative.resolve()

    matches: List[Path] = []
    for path in iter_session_files(session_root):
        name = path.name
        if name.endswith(f'_{value}.jsonl') or value in name:
            matches.append(path)
            continue
        try:
            header = read_jsonl(path)[0]
        except SessionExportError:
            continue
        session_id = header.get('id')
        if isinstance(session_id, str) and session_id.startswith(value):
            matches.append(path)
    unique_matches = dedupe_paths(matches)
    if not unique_matches:
        raise SessionExportError(f'Could not resolve session {value!r}')
    if len(unique_matches) > 1:
        raise SessionExportError(
            f'Session selector {value!r} matched multiple files: ' + ', '.join(str(path) for path in unique_matches[:5])
        )
    return unique_matches[0]


def dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    seen: set[str] = set()
    output: List[Path] = []
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        output.append(path.resolve())
    return output


def select_session_interactively(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    print('Choose a session to export:', file=sys.stderr)
    for index, meta in enumerate(candidates, start=1):
        label = session_display_label(meta)
        print(f'  {index:>2}. {label}', file=sys.stderr)
    while True:
        raw = input('Enter number: ').strip()
        if not raw:
            continue
        if not raw.isdigit():
            print('Please enter a numeric choice.', file=sys.stderr)
            continue
        choice = int(raw)
        if 1 <= choice <= len(candidates):
            return candidates[choice - 1]
        print(f'Please choose between 1 and {len(candidates)}.', file=sys.stderr)


def session_display_label(meta: Dict[str, Any]) -> str:
    header = meta['header']
    session_id = header.get('id', 'unknown')
    timestamp = meta.get('timestamp') or 'unknown-time'
    name = meta.get('name') or meta.get('preview') or 'unnamed'
    cwd = meta.get('cwd') or 'unknown-cwd'
    return f'{timestamp}  {compact_text(name, limit=48)}  id:{session_id}  cwd:{cwd}'


def file_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def make_artifact_payload(path: Path) -> Dict[str, Any]:
    data = path.read_bytes()
    payload: Dict[str, Any] = {
        'sourcePath': str(path),
        'size': len(data),
        'sha256': file_sha256_bytes(data),
    }
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        payload['encoding'] = 'base64'
        payload['base64'] = base64.b64encode(data).decode('ascii')
    else:
        payload['encoding'] = 'utf-8'
        payload['text'] = text
        payload['textPreview'] = compact_text(text, limit=TEXT_SAMPLE_LIMIT)
    return payload


def build_export_info(session_path: Path, source_sha256: str, mode: str, artifact_count: int, missing_count: int) -> Dict[str, Any]:
    return {
        'format': 'pi-session-full-export',
        'version': EXPORT_VERSION,
        'mode': mode,
        'exportedAt': datetime.now(timezone.utc).isoformat(),
        'sourceSessionPath': str(session_path),
        'sourceSessionSha256': source_sha256,
        'fullyEmbedded': mode == FULL_MODE,
        'artifactKey': ARTIFACT_KEY,
        'embeddedArtifactCount': artifact_count,
        'missingArtifactCount': missing_count,
    }


def export_session(session_meta: Dict[str, Any], mode: str) -> Dict[str, Any]:
    session_path = Path(session_meta['path']).resolve()
    entries = read_jsonl(session_path)
    source_sha = file_sha256(session_path)
    header = copy.deepcopy(entries[0])
    body = [copy.deepcopy(entry) for entry in entries[1:]]

    artifact_count = 0
    missing_artifacts: List[Dict[str, Any]] = []

    if mode == FULL_MODE:
        for entry in body:
            artifact_count += embed_entry_artifacts(entry, missing_artifacts)

    export_info = build_export_info(session_path, source_sha, mode, artifact_count, len(missing_artifacts))
    if missing_artifacts:
        export_info['missingArtifacts'] = missing_artifacts
    header['exportInfo'] = export_info

    return {
        'header': header,
        'entries': body,
        'artifactCount': artifact_count,
        'missingArtifacts': missing_artifacts,
        'sourceSha256': source_sha,
        'sessionPath': session_path,
        'sessionId': header.get('id'),
    }


def embed_entry_artifacts(entry: Dict[str, Any], missing_artifacts: List[Dict[str, Any]]) -> int:
    count = 0

    def walk(node: Any, trail: List[str]) -> None:
        nonlocal count
        if isinstance(node, dict):
            for key, value in list(node.items()):
                current_trail = trail + [key]
                if key == ARTIFACT_KEY and isinstance(value, str):
                    source = Path(value).expanduser()
                    embedded_key = f'{key}Embedded'
                    if source.is_file():
                        node[embedded_key] = make_artifact_payload(source)
                        count += 1
                    else:
                        missing_artifacts.append(
                            {
                                'path': value,
                                'reason': 'missing_or_not_file',
                                'entryType': entry.get('type'),
                                'entryId': entry.get('id'),
                                'field': '.'.join(current_trail),
                            }
                        )
                walk(value, current_trail)
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, trail + [str(index)])

    walk(entry, [])
    return count


def default_output_path(exported: Dict[str, Any], requested_out: Optional[str]) -> Path:
    session_id = exported.get('sessionId') or 'unknown-session'
    source_path = Path(exported['sessionPath'])
    base_name = source_path.stem
    suffix = '.full.jsonl' if exported['header'].get('exportInfo', {}).get('mode') == FULL_MODE else '.raw.jsonl'

    if requested_out:
        out_path = Path(requested_out).expanduser()
        if out_path.exists() and out_path.is_dir():
            return (out_path / f'{base_name}{suffix}').resolve()
        if requested_out.endswith(os.sep):
            return (out_path / f'{base_name}{suffix}').resolve()
        if out_path.suffix:
            return out_path.resolve()
        return (out_path / f'{base_name}{suffix}').resolve()

    default_dir = Path.cwd() / 'pi-session-exports'
    return (default_dir / f'{base_name}{suffix}').resolve()


def write_export(exported: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        handle.write(json.dumps(exported['header'], ensure_ascii=False) + '\n')
        for entry in exported['entries']:
            handle.write(json.dumps(entry, ensure_ascii=False) + '\n')


def verify_export(path: Path) -> Dict[str, Any]:
    entries = read_jsonl(path)
    header = entries[0]
    export_info = header.get('exportInfo')
    issues: List[str] = []
    embedded_count = 0
    missing_refs: List[Dict[str, Any]] = []

    if not isinstance(export_info, dict):
        issues.append('Missing exportInfo header.')
    else:
        if export_info.get('format') != 'pi-session-full-export':
            issues.append('Unexpected exportInfo.format.')
        if export_info.get('mode') == FULL_MODE and not export_info.get('fullyEmbedded'):
            issues.append('Full export is not marked fullyEmbedded.')

    for entry in entries[1:]:
        count, missing = verify_entry_artifacts(entry)
        embedded_count += count
        missing_refs.extend(missing)

    if export_info and export_info.get('mode') == FULL_MODE and missing_refs:
        issues.append('Found external artifact references without embedded payloads.')

    return {
        'path': str(path),
        'sessionId': header.get('id'),
        'mode': export_info.get('mode') if isinstance(export_info, dict) else None,
        'embeddedArtifactCount': embedded_count,
        'missingEmbeddedArtifacts': missing_refs,
        'issues': issues,
        'ok': not issues,
    }


def verify_entry_artifacts(entry: Dict[str, Any]) -> Tuple[int, List[Dict[str, Any]]]:
    embedded_count = 0
    missing: List[Dict[str, Any]] = []

    def walk(node: Any, trail: List[str]) -> None:
        nonlocal embedded_count
        if isinstance(node, dict):
            for key, value in node.items():
                current_trail = trail + [key]
                if key == ARTIFACT_KEY and isinstance(value, str):
                    embedded = node.get(f'{key}Embedded')
                    if embedded is None:
                        missing.append(
                            {
                                'entryId': entry.get('id'),
                                'entryType': entry.get('type'),
                                'field': '.'.join(current_trail),
                                'path': value,
                            }
                        )
                    else:
                        embedded_count += 1
                walk(value, current_trail)
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, trail + [str(index)])

    walk(entry, [])
    return embedded_count, missing


def format_export_result(exported: Dict[str, Any], output_path: Path) -> Dict[str, Any]:
    export_info = exported['header']['exportInfo']
    return {
        'source': str(exported['sessionPath']),
        'output': str(output_path),
        'sessionId': exported.get('sessionId'),
        'mode': export_info.get('mode'),
        'embeddedArtifacts': exported.get('artifactCount', 0),
        'missingArtifacts': exported.get('missingArtifacts', []),
        'sourceSha256': exported.get('sourceSha256'),
    }


def print_list(sessions: List[Dict[str, Any]], as_json: bool) -> int:
    rows = [
        {
            'path': str(meta['path']),
            'sessionId': meta['session_id'],
            'timestamp': meta['timestamp'],
            'cwd': meta['cwd'],
            'name': meta['name'],
            'preview': meta['preview'],
        }
        for meta in sessions
    ]
    if as_json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    if not rows:
        print('No sessions found.')
        return 0
    for index, row in enumerate(rows, start=1):
        name = row['name'] or row['preview'] or 'unnamed'
        print(f'{index:>2}. {row["timestamp"]}  id:{row["sessionId"]}  {compact_text(name, 60)}')
        print(f'    cwd: {row["cwd"]}')
        print(f'    path: {row["path"]}')
    return 0


def print_result(payload: Dict[str, Any], as_json: bool) -> int:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print(f"source: {payload['source']}")
    print(f"output: {payload['output']}")
    print(f"session id: {payload['sessionId']}")
    print(f"mode: {payload['mode']}")
    print(f"embedded artifacts: {payload['embeddedArtifacts']}")
    print(f"missing artifacts: {len(payload['missingArtifacts'])}")
    if payload['missingArtifacts']:
        for item in payload['missingArtifacts'][:10]:
            print(f"  - {item['path']} ({item['field']})")
    return 0


def print_verify_result(payload: Dict[str, Any], as_json: bool) -> int:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if payload['ok'] else 1
    print(f"path: {payload['path']}")
    print(f"session id: {payload['sessionId']}")
    print(f"mode: {payload['mode']}")
    print(f"embedded artifacts: {payload['embeddedArtifactCount']}")
    print(f"ok: {'yes' if payload['ok'] else 'no'}")
    if payload['issues']:
        print('issues:')
        for issue in payload['issues']:
            print(f'  - {issue}')
    if payload['missingEmbeddedArtifacts']:
        print('missing embedded artifacts:')
        for item in payload['missingEmbeddedArtifacts'][:10]:
            print(f"  - {item['path']} ({item['field']})")
    return 0 if payload['ok'] else 1


def analyze_session_health(path: Path) -> Dict[str, Any]:
    entries = read_jsonl(path)
    header = entries[0]
    refs: List[Dict[str, Any]] = []

    for entry in entries[1:]:
        refs.extend(find_artifact_refs(entry))

    existing_refs = [ref for ref in refs if Path(ref['path']).is_file()]
    missing_refs = [ref for ref in refs if not Path(ref['path']).is_file()]
    return {
        'path': str(path),
        'sessionId': header.get('id'),
        'cwd': header.get('cwd'),
        'timestamp': header.get('timestamp'),
        'totalRefs': len(refs),
        'existingRefs': len(existing_refs),
        'missingRefs': len(missing_refs),
        'ready': len(missing_refs) == 0,
        'refs': refs,
    }


def find_artifact_refs(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []

    def walk(node: Any, trail: List[str]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                current_trail = trail + [key]
                if key == ARTIFACT_KEY and isinstance(value, str):
                    refs.append(
                        {
                            'entryId': entry.get('id'),
                            'entryType': entry.get('type'),
                            'field': '.'.join(current_trail),
                            'path': value,
                        }
                    )
                else:
                    walk(value, current_trail)
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, trail + [str(index)])

    walk(entry, [])
    return refs


def build_health_report(session_root: Path, cwd: Path, include_all: bool) -> Dict[str, Any]:
    sessions = list_all_sessions(session_root) if include_all else list_project_sessions(session_root, cwd)
    session_paths = [Path(meta['path']) for meta in sessions]
    reports = [analyze_session_health(path) for path in session_paths]

    total_sessions = len(reports)
    sessions_with_refs = [report for report in reports if report['totalRefs'] > 0]
    sessions_ready = [report for report in reports if report['ready']]
    sessions_blocked = [report for report in reports if report['missingRefs'] > 0]
    total_refs = sum(report['totalRefs'] for report in reports)
    existing_refs = sum(report['existingRefs'] for report in reports)
    missing_refs = sum(report['missingRefs'] for report in reports)

    projects: Dict[str, Dict[str, Any]] = {}
    for report in reports:
        key = report.get('cwd') or 'unknown-cwd'
        project = projects.setdefault(
            key,
            {
                'cwd': key,
                'sessions': 0,
                'sessionsWithRefs': 0,
                'sessionsBlocked': 0,
                'totalRefs': 0,
                'missingRefs': 0,
            },
        )
        project['sessions'] += 1
        project['totalRefs'] += report['totalRefs']
        project['missingRefs'] += report['missingRefs']
        if report['totalRefs'] > 0:
            project['sessionsWithRefs'] += 1
        if report['missingRefs'] > 0:
            project['sessionsBlocked'] += 1

    return {
        'scope': 'all' if include_all else 'cwd',
        'sessionRoot': str(session_root),
        'cwd': str(cwd),
        'totalSessions': total_sessions,
        'sessionsReady': len(sessions_ready),
        'sessionsWithRefs': len(sessions_with_refs),
        'sessionsBlocked': len(sessions_blocked),
        'totalRefs': total_refs,
        'existingRefs': existing_refs,
        'missingRefs': missing_refs,
        'missingRatio': (missing_refs / total_refs) if total_refs else 0.0,
        'projects': sorted(projects.values(), key=lambda item: (item['sessionsBlocked'], item['missingRefs']), reverse=True),
        'blockedSessions': sorted(sessions_blocked, key=lambda item: (item['missingRefs'], item['timestamp'] or ''), reverse=True),
    }


def print_health_report(payload: Dict[str, Any], as_json: bool) -> int:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print(f"scope: {payload['scope']}")
    print(f"session root: {payload['sessionRoot']}")
    print(f"cwd: {payload['cwd']}")
    print(f"total sessions: {payload['totalSessions']}")
    print(f"sessions ready: {payload['sessionsReady']}")
    print(f"sessions with external refs: {payload['sessionsWithRefs']}")
    print(f"sessions blocked by missing refs: {payload['sessionsBlocked']}")
    print(f"total refs: {payload['totalRefs']}")
    print(f"existing refs: {payload['existingRefs']}")
    print(f"missing refs: {payload['missingRefs']}")
    print(f"missing ratio: {payload['missingRatio']:.2%}")
    if payload['projects']:
        print('projects:')
        for project in payload['projects'][:10]:
            print(
                f"  - {project['cwd']}: sessions={project['sessions']}, with_refs={project['sessionsWithRefs']}, "
                f"blocked={project['sessionsBlocked']}, missing_refs={project['missingRefs']}"
            )
    if payload['blockedSessions']:
        print('blocked sessions:')
        for report in payload['blockedSessions'][:10]:
            print(
                f"  - {report['timestamp']} id:{report['sessionId']} missing={report['missingRefs']} path={report['path']}"
            )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    session_root = Path(args.session_root).expanduser().resolve()
    cwd = Path(args.cwd).expanduser().resolve()

    try:
        if args.command == 'list':
            sessions = list_all_sessions(session_root) if args.all else list_project_sessions(session_root, cwd)
            return print_list(sessions, args.json)

        if args.command == 'verify':
            result = verify_export(Path(args.path).expanduser().resolve())
            return print_verify_result(result, args.json)

        if args.command == 'health':
            result = build_health_report(session_root, cwd, args.all)
            return print_health_report(result, args.json)

        session_meta = resolve_session(session_root, cwd, args.session, args.select)
        exported = export_session(session_meta, args.mode)
        output_path = default_output_path(exported, args.out)
        write_export(exported, output_path)
        result = format_export_result(exported, output_path)
        return print_result(result, args.json)
    except SessionExportError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

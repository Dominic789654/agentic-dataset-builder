#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .export_qwen35_training import append_parquet_rows, ensure_parquet_runtime, record_to_parquet_row

DEFAULT_PI_ROOT = Path.home() / '.pi' / 'agent' / 'sessions'
DEFAULT_CODEX_ROOT = Path.home() / '.codex' / 'sessions'


def python_entry(module_name: str, script_dir: Path) -> List[str]:
    if __package__:
        package_root = __package__.split('.')[0]
        return [sys.executable, '-m', f'{package_root}.{module_name}']
    return [sys.executable, str(script_dir / f'{module_name}.py')]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build one merged agentic dataset from local Pi and Codex sessions.')
    parser.add_argument('--pi-root', default=str(DEFAULT_PI_ROOT), help='Pi session root.')
    parser.add_argument('--codex-root', default=str(DEFAULT_CODEX_ROOT), help='Codex session root.')
    parser.add_argument('--output-root', required=True, help='Output root directory.')
    parser.add_argument('--include-sources', default='pi,codex', help='Comma-separated sources to include: pi,codex')
    parser.add_argument('--include-labels', default='cot_eligible,agent_only', help='Comma-separated labels to keep in final dataset.')
    parser.add_argument('--pi-session-root-override', help='Override session root passed to export_pi_session.py health check.')
    parser.add_argument('--skip-pi-health-check', action='store_true', help='Skip Pi health scan and export only by direct conversion assumptions.')
    parser.add_argument('--codex-limit', type=int, default=0, help='Optional limit for Codex session files.')
    parser.add_argument('--jsonl-only', action='store_true', help='Use JSONL intermediates only.')
    parser.add_argument('--final-format', choices=('jsonl', 'parquet', 'both'), default='parquet', help='Final merged dataset output format.')
    parser.add_argument('--keep-intermediates', action='store_true', help='Keep intermediate export and label directories instead of deleting them after a successful build.')
    return parser.parse_args(argv)


def print_progress(step: str, message: str) -> None:
    print(f'[{step}] {message}', flush=True)


def run_cmd(command: List[str], cwd: Path, step: str, log_path: Path) -> None:
    rendered_command = ' '.join(command)
    print_progress(step, 'running ' + rendered_command)
    with log_path.open('a', encoding='utf-8') as log_handle:
        log_handle.write(f'\n## {step}\n$ {rendered_command}\n')
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        for line in process.stdout:
            text = line.rstrip('\n')
            log_handle.write(line)
            print_progress(step, text)
        exit_code = process.wait()
        if exit_code != 0:
            raise subprocess.CalledProcessError(exit_code, command)


def latest_dir(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f'No directories match {pattern} under {root}')
    return matches[-1]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_pi_sessions(session_root: Path) -> List[Path]:
    return sorted(session_root.rglob('*.jsonl'))


def export_all_pi_sessions(script_dir: Path, output_root: Path, session_root: Path, log_path: Path) -> Path:
    sessions = find_pi_sessions(session_root)
    if not sessions:
        raise RuntimeError(f'No Pi sessions found under {session_root}')
    export_dir = output_root / 'pi-full-sessions'
    export_dir.mkdir(parents=True, exist_ok=True)
    print_progress('pi-export', f'exporting {len(sessions)} session files')
    for session_path in sessions:
        run_cmd(
            python_entry('export_pi_session', script_dir)
            + [
                '--session-root',
                str(session_root),
                '--session',
                str(session_path),
                '--out',
                str(export_dir),
            ],
            cwd=script_dir.parent,
            step='pi-export',
            log_path=log_path,
        )
    return export_dir


def convert_pi(script_dir: Path, full_export_dir: Path, output_root: Path, jsonl_only: bool, log_path: Path) -> Path:
    fmt = 'jsonl' if jsonl_only else 'both'
    run_cmd(
        python_entry('export_pi_session_to_qwen35', script_dir)
        + [
            '--input',
            str(full_export_dir),
            '--output-root',
            str(output_root),
            '--output-format',
            fmt,
        ],
        cwd=script_dir.parent,
        step='pi-convert',
        log_path=log_path,
    )
    return latest_dir(output_root, 'qwen35-pi-session-*')


def convert_codex(script_dir: Path, codex_root: Path, output_root: Path, jsonl_only: bool, limit: int, log_path: Path) -> Path:
    fmt = 'jsonl' if jsonl_only else 'both'
    command = python_entry('export_codex_session_to_qwen35', script_dir) + [
        '--input',
        str(codex_root),
        '--output-root',
        str(output_root),
        '--output-format',
        fmt,
    ]
    if limit > 0:
        command.extend(['--limit', str(limit)])
    run_cmd(command, cwd=script_dir.parent, step='codex-convert', log_path=log_path)
    return latest_dir(output_root, 'qwen35-codex-session-*')


def label_export(script_dir: Path, export_dir: Path, output_root: Path, log_path: Path, step_name: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    run_cmd(
        python_entry('label_qwen35_agentic', script_dir)
        + [
            '--input',
            str(export_dir),
            '--output-root',
            str(output_root),
        ],
        cwd=script_dir.parent,
        step=step_name,
        log_path=log_path,
    )
    return latest_dir(output_root, 'qwen35-agentic-labels-*')


def build_record_index(export_dir: Path) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for name in ('qwen35-train.jsonl', 'qwen35-train-lossy.jsonl'):
        path = export_dir / name
        if not path.exists():
            continue
        for record in load_jsonl(path):
            index[record['id']] = record
    return index


def merge_labeled_datasets(
    label_dirs: List[Tuple[str, Path]],
    export_dirs: Dict[str, Path],
    keep_labels: set[str],
    output_dir: Path,
    final_format: str,
) -> Dict[str, Any]:
    dataset_path = output_dir / 'dataset.jsonl'
    dataset_gzip_path = output_dir / 'dataset.jsonl.gz'
    parquet_path = output_dir / 'dataset.parquet'
    stats = Counter()
    source_stats: Dict[str, Counter] = {}
    jsonl_enabled = final_format in {'jsonl', 'both'}
    parquet_enabled = final_format in {'parquet', 'both'}
    parquet_writer = None
    parquet_batch: List[Dict[str, Any]] = []

    out = dataset_path.open('w', encoding='utf-8') if jsonl_enabled else None
    out_gzip = gzip.open(dataset_gzip_path, 'wt', encoding='utf-8') if jsonl_enabled else None
    try:
        for source_name, label_dir in label_dirs:
            labels = load_jsonl(label_dir / 'labels.jsonl')
            record_index = build_record_index(export_dirs[source_name])
            source_counter = Counter()
            for label in labels:
                source_counter['records_seen'] += 1
                stats[f"labels_seen:{label['label']}"] += 1
                if label['label'] not in keep_labels:
                    source_counter['records_skipped'] += 1
                    continue
                record = record_index.get(label['id'])
                if record is None:
                    source_counter['missing_records'] += 1
                    continue
                merged = dict(record)
                merged_meta = dict(merged.get('meta', {}))
                merged['label'] = label['label']
                merged['source_system'] = source_name
                merged['source_bucket'] = label.get('bucket')
                merged['source_file'] = label.get('source_file')
                merged['agentic_label'] = {
                    'label': label['label'],
                    'tool_call_count': label.get('tool_call_count'),
                    'tool_message_count': label.get('tool_message_count'),
                    'dialogue_rounds_est': label.get('dialogue_rounds_est'),
                    'reasoning_chars': label.get('reasoning_chars'),
                    'has_reasoning': label.get('has_reasoning'),
                    'lossy_source': label.get('lossy_source'),
                    'lossy_reasons': label.get('lossy_reasons', []),
                }
                merged_meta['dataset_label'] = label['label']
                merged_meta['dataset_source_system'] = source_name
                merged_meta['dataset_source_bucket'] = label.get('bucket')
                merged_meta['dataset_source_file'] = label.get('source_file')
                merged_meta['dataset_has_reasoning'] = label.get('has_reasoning')
                merged_meta['dataset_reasoning_chars'] = label.get('reasoning_chars')
                merged['meta'] = merged_meta
                if out is not None:
                    line = json.dumps(merged, ensure_ascii=False) + '\n'
                    out.write(line)
                    if out_gzip is not None:
                        out_gzip.write(line)
                if parquet_enabled:
                    parquet_batch.append(record_to_parquet_row(merged))
                    if len(parquet_batch) >= 1000:
                        parquet_writer = append_parquet_rows(parquet_writer, parquet_batch, parquet_path)
                        parquet_batch = []
                source_counter['records_kept'] += 1
                source_counter[f"kept:{label['label']}"] += 1
                stats['records_kept'] += 1
                stats[f"kept:{label['label']}"] += 1
            source_stats[source_name] = source_counter
        if parquet_enabled:
            parquet_writer = append_parquet_rows(parquet_writer, parquet_batch, parquet_path)
    finally:
        if out is not None:
            out.close()
        if out_gzip is not None:
            out_gzip.close()
        if parquet_writer is not None:
            parquet_writer.close()

    result = {
        'dataset_path': str(dataset_path) if jsonl_enabled else None,
        'stats': dict(stats),
        'source_stats': {name: dict(counter) for name, counter in source_stats.items()},
    }
    if jsonl_enabled:
        result['dataset_gzip_path'] = str(dataset_gzip_path)
    if parquet_enabled:
        result['dataset_parquet_path'] = str(parquet_path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.final_format in {'parquet', 'both'}:
        ensure_parquet_runtime('parquet')
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_root = output_root / f'agentic-dataset-{datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")}'
    run_root.mkdir(parents=True, exist_ok=True)
    log_path = run_root / 'run.log'

    include_sources = {item.strip() for item in args.include_sources.split(',') if item.strip()}
    keep_labels = {item.strip() for item in args.include_labels.split(',') if item.strip()}
    pi_root = Path(args.pi_root).expanduser().resolve()
    codex_root = Path(args.codex_root).expanduser().resolve()
    export_dirs: Dict[str, Path] = {}
    label_dirs: List[Tuple[str, Path]] = []
    manifest: Dict[str, Any] = {
        'run_dir': str(run_root),
        'run_log': str(log_path),
        'pi_root': str(pi_root),
        'codex_root': str(codex_root),
        'include_sources': sorted(include_sources),
        'keep_labels': sorted(keep_labels),
        'keep_intermediates': args.keep_intermediates,
        'steps': {},
    }
    cleanup_paths: List[str] = []

    if 'pi' in include_sources:
        pi_run_root = run_root / 'pi'
        pi_run_root.mkdir(parents=True, exist_ok=True)
        full_dir = export_all_pi_sessions(script_dir, pi_run_root, pi_root, log_path)
        pi_export_dir = convert_pi(script_dir, full_dir, pi_run_root, args.jsonl_only, log_path)
        pi_label_dir = label_export(script_dir, pi_export_dir, pi_run_root / 'labels', log_path, 'pi-label')
        export_dirs['pi'] = pi_export_dir
        label_dirs.append(('pi', pi_label_dir))
        cleanup_paths.append(str(pi_run_root))
        manifest['steps']['pi'] = {
            'full_export_dir': str(full_dir),
            'qwen35_export_dir': str(pi_export_dir),
            'label_dir': str(pi_label_dir),
            'label_manifest': load_json(pi_label_dir / 'manifest.json'),
            'export_manifest': load_json(pi_export_dir / 'manifest.json'),
        }

    if 'codex' in include_sources:
        codex_run_root = run_root / 'codex'
        codex_run_root.mkdir(parents=True, exist_ok=True)
        codex_export_dir = convert_codex(script_dir, codex_root, codex_run_root, args.jsonl_only, args.codex_limit, log_path)
        codex_label_dir = label_export(script_dir, codex_export_dir, codex_run_root / 'labels', log_path, 'codex-label')
        export_dirs['codex'] = codex_export_dir
        label_dirs.append(('codex', codex_label_dir))
        cleanup_paths.append(str(codex_run_root))
        manifest['steps']['codex'] = {
            'qwen35_export_dir': str(codex_export_dir),
            'label_dir': str(codex_label_dir),
            'label_manifest': load_json(codex_label_dir / 'manifest.json'),
            'export_manifest': load_json(codex_export_dir / 'manifest.json'),
        }

    merge_info = merge_labeled_datasets(label_dirs, export_dirs, keep_labels, run_root, args.final_format)
    manifest['final_dataset'] = merge_info
    if not args.keep_intermediates:
        removed: List[str] = []
        for path_str in cleanup_paths:
            path = Path(path_str)
            if path.exists():
                shutil.rmtree(path)
                removed.append(path_str)
        manifest['cleanup'] = {'enabled': True, 'removed_paths': removed}
        print_progress('cleanup', f'removed {len(removed)} intermediate directories')
    else:
        manifest['cleanup'] = {'enabled': False, 'removed_paths': []}
    manifest_path = run_root / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(
        json.dumps(
            {
                'run_dir': str(run_root),
                'dataset_path': merge_info.get('dataset_path'),
                'dataset_parquet_path': merge_info.get('dataset_parquet_path'),
                'stats': merge_info['stats'],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

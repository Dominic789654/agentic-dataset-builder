from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Iterable, List, Optional


def os_name() -> str:
    return platform.system().lower()


def home_dir() -> Path:
    return Path.home()


def env_path(name: str) -> Optional[Path]:
    value = os.environ.get(name)
    if not value:
        return None
    return Path(value).expanduser()


def existing_or_default(candidates: Iterable[Path]) -> Path:
    collected = [candidate.expanduser() for candidate in candidates]
    for candidate in collected:
        if candidate.exists():
            return candidate.resolve()
    return collected[0].resolve() if collected else home_dir().resolve()


def candidate_pi_session_roots() -> List[Path]:
    home = home_dir()
    appdata = os.environ.get('APPDATA')
    localappdata = os.environ.get('LOCALAPPDATA')
    candidates: List[Path] = []
    override = env_path('PI_SESSION_ROOT')
    if override is not None:
        candidates.append(override)
    candidates.append(home / '.pi' / 'agent' / 'sessions')
    if appdata:
        candidates.append(Path(appdata) / 'pi' / 'agent' / 'sessions')
        candidates.append(Path(appdata) / '.pi' / 'agent' / 'sessions')
    if localappdata:
        candidates.append(Path(localappdata) / 'pi' / 'agent' / 'sessions')
        candidates.append(Path(localappdata) / '.pi' / 'agent' / 'sessions')
    return dedupe(candidates)


def candidate_codex_session_roots() -> List[Path]:
    home = home_dir()
    appdata = os.environ.get('APPDATA')
    localappdata = os.environ.get('LOCALAPPDATA')
    candidates: List[Path] = []
    override = env_path('CODEX_SESSION_ROOT')
    if override is not None:
        candidates.append(override)
    candidates.append(home / '.codex' / 'sessions')
    if appdata:
        candidates.append(Path(appdata) / 'Codex' / 'sessions')
        candidates.append(Path(appdata) / '.codex' / 'sessions')
    if localappdata:
        candidates.append(Path(localappdata) / 'Codex' / 'sessions')
        candidates.append(Path(localappdata) / '.codex' / 'sessions')
    return dedupe(candidates)


def default_pi_session_root() -> Path:
    return existing_or_default(candidate_pi_session_roots())


def default_codex_session_root() -> Path:
    return existing_or_default(candidate_codex_session_roots())


def dedupe(paths: Iterable[Path]) -> List[Path]:
    output: List[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.expanduser())
        if key in seen:
            continue
        seen.add(key)
        output.append(path)
    return output

#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const venvDir = path.join(__dirname, '.venv');
const requirementsPath = path.join(__dirname, 'requirements.txt');
const runPy = path.join(__dirname, 'run.py');

function isWindows() {
  return process.platform === 'win32';
}

function venvPythonPath() {
  return isWindows()
    ? path.join(venvDir, 'Scripts', 'python.exe')
    : path.join(venvDir, 'bin', 'python');
}

function findSystemPython() {
  const candidates = isWindows()
    ? ['py', 'python', 'python3']
    : ['python3', 'python'];
  for (const candidate of candidates) {
    const probeArgs = candidate === 'py' ? ['-3', '--version'] : ['--version'];
    const result = spawnSync(candidate, probeArgs, { stdio: 'ignore' });
    if (result.status === 0) {
      return candidate;
    }
  }
  return null;
}

function run(cmd, args, options = {}) {
  const rendered = [cmd, ...args].join(' ');
  console.error(`[agentic-dataset-builder] ${rendered}`);
  const result = spawnSync(cmd, args, {
    stdio: 'inherit',
    cwd: __dirname,
    env: process.env,
    ...options,
  });
  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}

function ensureEnv() {
  const pythonInVenv = venvPythonPath();
  if (existsSync(pythonInVenv)) {
    return pythonInVenv;
  }

  const systemPython = findSystemPython();
  if (!systemPython) {
    console.error('Python 3.10+ is required but was not found in PATH.');
    process.exit(1);
  }

  if (systemPython === 'py') {
    run(systemPython, ['-3', '-m', 'venv', venvDir]);
  } else {
    run(systemPython, ['-m', 'venv', venvDir]);
  }

  const venvPython = venvPythonPath();
  run(venvPython, ['-m', 'pip', 'install', '--upgrade', 'pip']);
  run(venvPython, ['-m', 'pip', 'install', '-r', requirementsPath]);
  return venvPython;
}

const python = ensureEnv();
const args = process.argv.slice(2);
run(python, [runPy, ...args]);

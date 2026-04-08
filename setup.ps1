$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if (-not $env:PYTHON_BIN) { $env:PYTHON_BIN = 'python' }
if (-not $env:VENV_DIR) { $env:VENV_DIR = '.venv' }

& $env:PYTHON_BIN -m venv $env:VENV_DIR
& "$PSScriptRoot\$($env:VENV_DIR)\Scripts\python.exe" -m pip install --upgrade pip
& "$PSScriptRoot\$($env:VENV_DIR)\Scripts\pip.exe" install -r requirements.txt

Write-Host "Environment ready: $PSScriptRoot\$($env:VENV_DIR)"
Write-Host "Activate with: .\$($env:VENV_DIR)\Scripts\Activate.ps1"
Write-Host "Run builder with: python run.py --output-root .\out"

<#
.SYNOPSIS
Build script for GOD's EYE Smart City Command Center.
Uses the project VENV's PyInstaller with an explicit .spec file to
exclude TensorFlow/Keras and prevent stalled DLL collection.
IMPORTANT: Always run from the project root: .\build.ps1
#>

$VenvPyInstaller = ".\venv\Scripts\pyinstaller.exe"
$IsccCompiler = "C:\Users\Dell\AppData\Local\Programs\Inno Setup 6\iscc.exe"

if (-not (Test-Path $VenvPyInstaller)) {
    Write-Host "ERROR: venv not found. Please set up the venv first." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  GOD's EYE — Production Build Pipeline" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# 1. Run PyInstaller
Write-Host "[1/2] Building executable with PyInstaller..." -ForegroundColor Yellow
& $VenvPyInstaller --noconfirm GodsEye.spec

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: PyInstaller build failed." -ForegroundColor Red
    exit 1
}

Write-Host "SUCCESS: Executable built in dist\GodsEye\" -ForegroundColor Green
Write-Host ""

# 2. Run Inno Setup
Write-Host "[2/2] Generating Desktop Installer with Inno Setup..." -ForegroundColor Yellow
if (Test-Path $IsccCompiler) {
    & $IsccCompiler installer.iss
    if ($LASTEXITCODE -eq 0) {
        Write-Host "SUCCESS: Installer generated in Output\GodsEye_Setup.exe" -ForegroundColor Green
    } else {
        Write-Host "ERROR: Inno Setup compilation failed." -ForegroundColor Red
    }
} else {
    Write-Host "WARNING: Inno Setup Compiler (iscc.exe) not found at $IsccCompiler" -ForegroundColor Yellow
    Write-Host "Please install Inno Setup 6 or update the path in build.ps1" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Pipeline Complete." -ForegroundColor Cyan

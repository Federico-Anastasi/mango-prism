# build.ps1 - builds the `prism` engine (PrismML fork + Mango's changes) for Windows x64.
#
#   .\mango-ci\build.ps1 -Variant cpu|vulkan|cuda-12 -Out <dir> [-Zip <file.zip>]
#
# The same script runs in GitHub Actions (.github/workflows/mango-engine.yml) and locally
# (`launcher\setup.ps1 -Only llama-prism` in the mango-code repo): the build options live here and only here,
# so the published build and the development PC's build don't diverge.
#
# <Out> receives llama-server.exe, llama-bench.exe and the DLLs. The CUDA runtimes (cudart, cuBLAS) are not
# included: the app adds them from the llama.cpp zip, setup.ps1 from the toolkit. With -Zip it also writes the
# zip (flat files) and <zip>.sha256.

param(
    [Parameter(Mandatory)][ValidateSet("cpu", "vulkan", "cuda-12")][string]$Variant,
    [Parameter(Mandatory)][string]$Out,
    [string]$Zip,
    # cuda-12 only. By default Pascal (61), Turing (75) and Ampere (86) and the other common ones, with PTX for the last one so
    # newer GPUs can JIT it.
    [string]$Archs = "61-real;70-real;75-real;80-real;86-real;89",
    # GTX 16xx (TU116) without tensor cores: it applies to EVERY cc 7.5 GPU (RTX 20xx would lose mma), so the
    # published build doesn't have it; setup.ps1 turns it on for a PC with a 16xx.
    [switch]$Tu116,
    [int]$Jobs = [Environment]::ProcessorCount,
    [string]$Build
)

$ErrorActionPreference = "Stop"
$src = Split-Path $PSScriptRoot -Parent
if (-not $Build) { $Build = Join-Path $src "build\$Variant" }

function Enter-Msvc {
    if (Get-Command cl -ErrorAction SilentlyContinue) { return }
    $installer = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer"
    $vs = & "$installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vs) { throw "Visual Studio with the C++ tools (x64) is missing" }
    # vcvars64 calls vswhere from the PATH: without the installer folder it prints a warning.
    $env:PATH = "$installer;$env:PATH"
    cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') { Set-Item "env:$($Matches[1])" $Matches[2] }
    }
}

function Invoke-Native {
    & $args[0] $args[1..($args.Count - 1)]
    if ($LASTEXITCODE -ne 0) { throw "$($args[0]) exited with $LASTEXITCODE" }
}

Enter-Msvc
foreach ($c in @("cmake", "ninja")) {
    if (-not (Get-Command $c -ErrorAction SilentlyContinue)) { throw "$c is not in the PATH" }
}

# C++ and CUDA flags are given in full: CMAKE_CXX_FLAGS replaces MSVC's defaults, and without /EHsc a C++
# exception (even one the code catches) kills the engine. CPU code is always Haswell (AVX2, FMA): without
# options MSVC compiles for SSE2 only.
$def = if ($Tu116) { "-DGGML_CUDA_TU116" } else { "" }
$options = @(
    "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
    "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
    "-DGGML_NATIVE=OFF", "-DGGML_AVX=ON", "-DGGML_AVX2=ON", "-DGGML_FMA=ON", "-DGGML_F16C=ON", "-DGGML_BMI2=ON",
    "-DCMAKE_CXX_FLAGS=/DWIN32 /D_WINDOWS /EHsc /GR $def"
)
switch ($Variant) {
    "vulkan" { $options += "-DGGML_VULKAN=ON" }
    "cuda-12" { $options += @("-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=$Archs", "-DCMAKE_CUDA_FLAGS=-D_WINDOWS -Xcompiler=/EHsc $def") }
}

Push-Location $src
try {
    Invoke-Native cmake -S . -B $Build @options
    Invoke-Native cmake --build $Build -j $Jobs --target llama-server --target llama-bench
} finally { Pop-Location }

New-Item -ItemType Directory -Force -Path $Out | Out-Null
Copy-Item "$Build\bin\*.dll", "$Build\bin\llama-server.exe", "$Build\bin\llama-bench.exe" $Out -Force

if ($Zip) {
    Remove-Item $Zip -ErrorAction SilentlyContinue
    Compress-Archive -Path "$Out\*" -DestinationPath $Zip
    $hash = (Get-FileHash $Zip -Algorithm SHA256).Hash.ToLower()
    # no BOM and LF line ending, like `sha256sum`
    [System.IO.File]::WriteAllText("$Zip.sha256", "$hash  $(Split-Path $Zip -Leaf)`n", (New-Object System.Text.UTF8Encoding($false)))

    # The engine must start from the zip, not only from the folder it was built in.
    $check = Join-Path ([IO.Path]::GetTempPath()) "mango-zip-$([guid]::NewGuid())"
    Expand-Archive $Zip $check
    try { Invoke-Native "$check\llama-server.exe" --version } finally { Remove-Item $check -Recurse -Force }
}

# build.ps1 - builds the `prism` engine (PrismML fork + Mango's changes) for Windows x64.
#
#   .\mango-ci\build.ps1 -Variant cpu|vulkan|cuda-12|cuda-12-gtx16 -Out <dir> [-Zip <file.zip>]
#
# cuda-12-gtx16 is the CUDA 12 build for a PC with a GTX 16xx: the same as cuda-12 with -Tu116.
#
# The same script runs in GitHub Actions (.github/workflows/mango-engine.yml) and locally
# (`launcher\setup.ps1 -Only llama-prism` in the mango-code repo): the build options live here and only here,
# so the published build and the development PC's build don't diverge.
#
# <Out> receives llama-server.exe, llama-bench.exe and the DLLs. The CUDA runtimes (cudart, cuBLAS) are not
# included: the app adds them from the llama.cpp zip, setup.ps1 from the toolkit. With -Zip it also writes the
# zip (flat files) and <zip>.sha256.

param(
    [Parameter(Mandatory)][ValidateSet("cpu", "vulkan", "cuda-12", "cuda-12-gtx16")][string]$Variant,
    [Parameter(Mandatory)][string]$Out,
    [string]$Zip,
    # cuda-12 only. By default Pascal (61), Turing (75) and Ampere (86) and the other common ones, with PTX for the last one so
    # newer GPUs can JIT it.
    [string]$Archs = "61-real;70-real;75-real;80-real;86-real;89",
    # GTX 16xx (TU116) without tensor cores: it applies to EVERY cc 7.5 GPU (RTX 20xx would lose mma), so it is a
    # build of its own (cuda-12-gtx16), for a PC with a GTX 16xx and no RTX 20xx.
    [switch]$Tu116,
    [int]$Jobs = [Environment]::ProcessorCount,
    [string]$Build
)

$ErrorActionPreference = "Stop"
$cuda = $Variant -like "cuda-12*"
if ($Variant -eq "cuda-12-gtx16") { $Tu116 = $true }
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

# Every DLL that the engine's files load must be in the folder, in Windows (System32 or an api-ms-/ext-ms- API set),
# or be what the user's PC adds: the CUDA runtime the app brings (cudart, cuBLAS), the NVIDIA driver (nvcuda) or the
# Vulkan loader of the GPU driver (vulkan-1). Starting the engine is not enough: on the CI runner a DLL that users do
# not have (OpenSSL, for one) can be found in the PATH.
function Assert-Dependencies([string]$dir) {
    $outside = '^(cudart64_12|cublas64_12|cublasLt64_12|nvcuda|vulkan-1|api-ms-win-.+|ext-ms-.+)\.dll$'
    $system = Join-Path $env:SystemRoot "System32"
    Get-ChildItem $dir -File | Where-Object { $_.Extension -in ".exe", ".dll" } | ForEach-Object {
        $file = $_.Name
        $deps = dumpbin /nologo /dependents $_.FullName | Where-Object { $_ -match '^\s+(\S+\.dll)\s*$' } | ForEach-Object { $Matches[1] }
        foreach ($d in $deps) {
            if ((Test-Path (Join-Path $dir $d)) -or (Test-Path (Join-Path $system $d)) -or $d -match $outside) { continue }
            throw "$file needs $d, which is neither in the zip, nor in Windows, nor the CUDA runtime or driver"
        }
    }
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
    "-DLLAMA_CURL=OFF", "-DLLAMA_OPENSSL=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
    "-DGGML_NATIVE=OFF", "-DGGML_AVX=ON", "-DGGML_AVX2=ON", "-DGGML_FMA=ON", "-DGGML_F16C=ON", "-DGGML_BMI2=ON",
    "-DCMAKE_CXX_FLAGS=/DWIN32 /D_WINDOWS /EHsc /GR $def"
)
if ($Variant -eq "vulkan") { $options += "-DGGML_VULKAN=ON" }
if ($cuda) { $options += @("-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=$Archs", "-DCMAKE_CUDA_FLAGS=-D_WINDOWS -Xcompiler=/EHsc $def") }

Push-Location $src
try {
    Invoke-Native cmake -S . -B $Build @options
    Invoke-Native cmake --build $Build -j $Jobs --target llama-server --target llama-bench
} finally { Pop-Location }

New-Item -ItemType Directory -Force -Path $Out | Out-Null
Copy-Item "$Build\bin\*.dll", "$Build\bin\llama-server.exe", "$Build\bin\llama-bench.exe" $Out -Force

if ($Zip) {
    # The sizes go in the log. A file over 2 GiB is too big for a Windows DLL: fewer CUDA architectures are needed.
    $files = @(Get-ChildItem $Out -File | Sort-Object Length -Descending)
    $files | ForEach-Object { "{0,14:N0}  {1}" -f $_.Length, $_.Name }
    $big = @($files | Where-Object { $_.Length -ge 2GB })
    if ($big) { throw "$($big.Name -join ', ') over 2 GiB: build fewer CUDA architectures (-Archs)" }
    Remove-Item $Zip -ErrorAction SilentlyContinue
    # tar.exe (bsdtar, part of Windows) streams the files into the zip: Compress-Archive fails on large ones
    # ("Stream was too long").
    # Called directly: through Invoke-Native the dash options reach tar mangled.
    tar.exe -a -c -f $Zip -C $Out @($files.Name)
    if ($LASTEXITCODE -ne 0) { throw "tar.exe exited with $LASTEXITCODE" }
    $hash = (Get-FileHash $Zip -Algorithm SHA256).Hash.ToLower()
    # no BOM and LF line ending, like `sha256sum`
    [System.IO.File]::WriteAllText("$Zip.sha256", "$hash  $(Split-Path $Zip -Leaf)`n", (New-Object System.Text.UTF8Encoding($false)))

    # The engine must run from the zip, not only from the folder it was built in: its dependencies are checked, and
    # it is started where it can be (the CUDA build needs the NVIDIA driver, which the CI runner does not have).
    $check = Join-Path ([IO.Path]::GetTempPath()) "mango-zip-$([guid]::NewGuid())"
    Expand-Archive $Zip $check
    try {
        Assert-Dependencies $check
        if (-not $cuda) { Invoke-Native "$check\llama-server.exe" --version }
    } finally { Remove-Item $check -Recurse -Force }
}

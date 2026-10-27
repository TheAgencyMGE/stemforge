# Optional research engines and the MIDI renderer, each isolated from the main install.
#   powershell -ExecutionPolicy Bypass -File scripts\setup-research.ps1 [-Root $env:USERPROFILE]
#
# Creates:
#   <Root>\.sfvenv-transkun     Transkun (piano). Code MIT; checkpoint licence not stated.
#   <Root>\.sfvenv-yourmt3      YourMT3+ runtime; code checkout in <Root>\stemforge-tools\yourmt3 (GPL-3.0),
#                               checkpoint from huggingface.co/mimbres/YourMT3 (card: Apache-2.0), SHA-256 checked.
#   <Root>\.sfvenv-muscriptor   MuScriptor runtime. Its weights are CC BY-NC 4.0 and gated: this script does
#                               NOT accept the licence for you (see the message at the end).
#   <Root>\stemforge-tools      FluidSynth 2.6.1 (LGPL, official release) and MuseScore General SF3 (MIT).
# and prints the STEMFORGE_* lines to put in backend\.env. All of this is for non-commercial research.
param([string]$Root = $env:USERPROFILE)
$ErrorActionPreference = "Stop"

$main = "$Root\.sfvenv\Scripts\python.exe"
if (-not (Test-Path $main)) { throw "Main environment not found at $main. Run scripts\setup.ps1 first." }
$uv = @($main, "-m", "uv")
$tools = "$Root\stemforge-tools"
New-Item -ItemType Directory -Force $tools | Out-Null

function New-EngineVenv($name, $packages) {
  $venv = "$Root\.sfvenv-$name"
  if (-not (Test-Path "$venv\Scripts\python.exe")) { python -m venv $venv }
  & $uv[0] $uv[1] $uv[2] pip install --python "$venv\Scripts\python.exe" torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
  & $uv[0] $uv[1] $uv[2] pip install --python "$venv\Scripts\python.exe" @packages
  return "$venv\Scripts\python.exe"
}

$transkun = New-EngineVenv "transkun" @("transkun==2.0.1")
$muscriptor = New-EngineVenv "muscriptor" @("muscriptor==0.4.0")
$yourmt3 = New-EngineVenv "yourmt3" @("lightning>=2.2.1", "transformers==4.45.1", "numpy==1.26.4", "librosa", "einops",
                                      "mido", "deprecated", "mir_eval", "soundfile", "wandb", "python-dotenv")

# YourMT3+ code (no LFS) and the YPTF.MoE+Multi (noPS) checkpoint.
$ymt = "$tools\yourmt3"
if (-not (Test-Path $ymt)) { $env:GIT_LFS_SKIP_SMUDGE = "1"; git clone --depth 1 https://huggingface.co/spaces/mimbres/YourMT3 $ymt }
$ckDir = "$ymt\amt\logs\2024\mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops\checkpoints"
New-Item -ItemType Directory -Force $ckDir | Out-Null
$ck = "$ckDir\last.ckpt"
$expected = "ae38e415c79efd5592dcb9b658cdb99ddb11d4c4e1eaa364cab04a052473fc25"
if (-not (Test-Path $ck) -or (Get-FileHash $ck -Algorithm SHA256).Hash.ToLower() -ne $expected) {
  curl.exe -sSL --retry 5 -C - -o $ck "https://huggingface.co/mimbres/YourMT3/resolve/main/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt"
  if ((Get-FileHash $ck -Algorithm SHA256).Hash.ToLower() -ne $expected) { throw "YourMT3 checkpoint SHA-256 mismatch" }
}

# FluidSynth (official Windows build) and MuseScore General (MIT).
$fs = "$tools\fluidsynth\fluidsynth-v2.6.1-win10-x64-cpp11\bin\fluidsynth.exe"
if (-not (Test-Path $fs)) {
  curl.exe -sSL -o "$tools\fluidsynth.zip" https://github.com/FluidSynth/fluidsynth/releases/download/v2.6.1/fluidsynth-v2.6.1-win10-x64-cpp11.zip
  Expand-Archive -Force "$tools\fluidsynth.zip" "$tools\fluidsynth"
}
$sf3 = "$tools\MuseScore_General.sf3"
if (-not (Test-Path $sf3)) {
  curl.exe -sSL -o $sf3 https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf3
  curl.exe -sSL -o "$tools\MuseScore_General_License.md" https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General_License.md
}

Write-Host "`nAdd to backend\.env:"
Write-Host "STEMFORGE_TRANSKUN_PYTHON=$transkun"
Write-Host "STEMFORGE_YOURMT3_PYTHON=$yourmt3"
Write-Host "STEMFORGE_YOURMT3_DIR=$ymt"
Write-Host "STEMFORGE_MUSCRIPTOR_PYTHON=$muscriptor"
Write-Host "STEMFORGE_FLUIDSYNTH_PATH=$fs"
Write-Host "STEMFORGE_SOUNDFONT_PATH=$sf3"
Write-Host "STEMFORGE_RESEARCH_ENGINES=true   # let measured research engines into the automatic policy"
Write-Host "`nMuScriptor weights are gated (CC BY-NC 4.0 + Kyutai/Mirelo conditions). To enable them yourself:"
Write-Host "  1. Sign in at https://huggingface.co and accept the conditions on https://huggingface.co/MuScriptor/muscriptor-small"
Write-Host "  2. & '$Root\.sfvenv-muscriptor\Scripts\hf.exe' auth login"

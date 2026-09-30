# Start the review app and publish it on a public Cloudflare URL.
#   .\share.ps1              serve and tunnel
#   .\share.ps1 -Bootstrap   build the demo data first (first run)
param(
    [switch]$Bootstrap,
    [int]$Port = 8000,
    [int]$Businesses = 3
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $py)) { throw "No virtual environment. Run: py -3.12 -m venv .venv; $py -m pip install -e '.[dev]'" }

if ($Bootstrap) {
    Write-Host "Building the demo (this takes a few minutes)..." -ForegroundColor Cyan
    & $py -m gstpipe.webapp.cli --bootstrap --no-serve --businesses $Businesses
}

# A local copy takes precedence; cloudflared is a single portable binary.
$local = Join-Path $PSScriptRoot "tools\cloudflared.exe"
$cloudflared = if (Test-Path $local) { $local }
               elseif (Get-Command cloudflared -ErrorAction SilentlyContinue) { "cloudflared" }
               else { $null }

if (-not $cloudflared) {
    Write-Host "cloudflared not found. Fetch the single binary with:" -ForegroundColor Yellow
    Write-Host "  curl -L -o tools\cloudflared.exe https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" -ForegroundColor Yellow
    Write-Host "Serving locally only." -ForegroundColor Yellow
    & $py -m gstpipe.webapp.cli --host 127.0.0.1 --port $Port
    exit
}

$app = Start-Process -FilePath $py -PassThru -NoNewWindow `
    -ArgumentList "-m", "gstpipe.webapp.cli", "--host", "127.0.0.1", "--port", "$Port"

try {
    # Wait for the app to answer before opening the tunnel.
    $ready = $false
    foreach ($attempt in 1..40) {
        Start-Sleep -Milliseconds 500
        try {
            if ((Invoke-WebRequest "http://127.0.0.1:$Port/api/health" -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200) {
                $ready = $true; break
            }
        } catch { }
    }
    if (-not $ready) { throw "The app did not start. Try: $py -m gstpipe.webapp.cli --bootstrap" }

    Write-Host "App is up on http://127.0.0.1:$Port" -ForegroundColor Green
    Write-Host "Opening a public URL. Share the trycloudflare.com address below." -ForegroundColor Cyan
    & $cloudflared tunnel --url "http://localhost:$Port"
}
finally {
    if ($app -and -not $app.HasExited) { Stop-Process -Id $app.Id -Force }
    Write-Host "Stopped." -ForegroundColor DarkGray
}

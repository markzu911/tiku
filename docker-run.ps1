param(
  [string]$ImageName = "exam-bank:local",
  [string]$ContainerName = "exam-bank-local",
  [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$envFile = Join-Path $projectRoot ".env"
if (-not (Test-Path $envFile)) {
  throw "Missing .env. Create it from .env.example before starting Docker."
}

$databaseUrlLine = Get-Content $envFile | Where-Object { $_ -match '^\s*DATABASE_URL\s*=' } | Select-Object -Last 1
if (-not $databaseUrlLine) {
  throw "DATABASE_URL is missing from .env."
}

$databaseUrl = ($databaseUrlLine -split '=', 2)[1].Trim()
$dockerDatabaseUrl = $databaseUrl -replace '(@)(127\.0\.0\.1|localhost)(?=[:/])', '$1host.docker.internal'

docker build --tag $ImageName $projectRoot
$existingContainer = docker ps --all --quiet --filter "name=^/$ContainerName$"
if ($existingContainer) {
  docker rm --force $ContainerName
}
docker run --detach --name $ContainerName `
  --env-file $envFile `
  --env "DATABASE_URL=$dockerDatabaseUrl" `
  --env "MODEL_SETTINGS_FILE=/app/runtime/model-provider.json" `
  --add-host "host.docker.internal:host-gateway" `
  --publish "${Port}:8000" `
  --volume "exam-bank-runtime:/app/runtime" `
  $ImageName

Write-Host "Exam Bank is running at http://127.0.0.1:$Port/"

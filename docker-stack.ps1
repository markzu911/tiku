param(
  [switch]$MigrateHostData
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$envFile = Join-Path $projectRoot ".env"
$composeFile = Join-Path $projectRoot "docker-compose.yml"
$runtimeDirectory = Join-Path $projectRoot "runtime"
$stackEnvFile = Join-Path $runtimeDirectory "docker-stack.env"

if (-not (Test-Path $envFile)) {
  throw "Missing .env. Create it from .env.example before starting Docker."
}

$databaseUrlLine = Get-Content $envFile | Where-Object { $_ -match '^\s*DATABASE_URL\s*=' } | Select-Object -Last 1
if (-not $databaseUrlLine) {
  throw "DATABASE_URL is missing from .env."
}

$databaseUri = [uri](($databaseUrlLine -split '=', 2)[1].Trim())
$databaseName = $databaseUri.AbsolutePath.Trim('/')
$databaseUserInfo = $databaseUri.UserInfo.Split(':', 2)
if ($databaseUserInfo.Count -ne 2 -or $databaseUserInfo[0] -ne "root") {
  throw "The Docker MySQL stack currently requires a root DATABASE_URL."
}

$rootPassword = [uri]::UnescapeDataString($databaseUserInfo[1])
$containerDatabaseUrl = "$($databaseUri.Scheme)://$($databaseUri.UserInfo)@db:3306$($databaseUri.AbsolutePath)$($databaseUri.Query)"
New-Item -ItemType Directory -Force -Path $runtimeDirectory | Out-Null
@(
  "MYSQL_DATABASE=$databaseName"
  "MYSQL_ROOT_PASSWORD=$rootPassword"
  "CONTAINER_DATABASE_URL=$containerDatabaseUrl"
) | Set-Content -Encoding utf8 $stackEnvFile

$appRuntimeVolume = docker volume inspect exam-bank-runtime --format '{{.Name}}' 2>$null
if (-not $appRuntimeVolume) {
  docker volume create exam-bank-runtime | Out-Null
}

function Invoke-Compose([string[]]$Arguments) {
  & docker compose --env-file $stackEnvFile --project-name exam-bank -f $composeFile @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Docker Compose command failed."
  }
}

$legacyContainer = docker ps --all --quiet --filter "name=^/exam-bank-local$"
if ($legacyContainer) {
  docker rm --force exam-bank-local | Out-Null
}

Invoke-Compose @("up", "-d", "db")

for ($attempt = 1; $attempt -le 60; $attempt += 1) {
  $health = (& docker inspect --format '{{.State.Health.Status}}' exam-bank-db 2>$null).Trim()
  if ($health -eq "healthy") {
    break
  }
  Start-Sleep -Seconds 2
}
if ($health -ne "healthy") {
  throw "MySQL container did not become healthy."
}

if ($MigrateHostData) {
  $dumpFile = Join-Path $runtimeDirectory "exam-bank-host-backup.sql"
  $previousMysqlPassword = $env:MYSQL_PWD
  $env:MYSQL_PWD = $rootPassword
  try {
    & mysqldump --host=$($databaseUri.Host) --port=$($databaseUri.Port) --user=root --single-transaction --routines --events --triggers --default-character-set=utf8mb4 --databases $databaseName --result-file=$dumpFile
    if ($LASTEXITCODE -ne 0) {
      throw "Host database export failed."
    }
  } finally {
    $env:MYSQL_PWD = $previousMysqlPassword
  }

  $dbContainer = (& docker compose --env-file $stackEnvFile --project-name exam-bank -f $composeFile ps --quiet db).Trim()
  if (-not $dbContainer) {
    throw "MySQL container is not available for import."
  }
  & docker cp $dumpFile "$dbContainer`:/tmp/exam-bank-host-backup.sql"
  if ($LASTEXITCODE -ne 0) {
    throw "Copying the host database backup into Docker failed."
  }
  Invoke-Compose @("exec", "-T", "-e", "MYSQL_PWD=$rootPassword", "db", "sh", "-c", "mysql -uroot < /tmp/exam-bank-host-backup.sql")
  Remove-Item -LiteralPath $dumpFile
}

Invoke-Compose @("up", "-d", "--build", "app")
Write-Host "Exam Bank is running at http://127.0.0.1:8000/"

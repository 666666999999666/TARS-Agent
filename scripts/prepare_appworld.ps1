param()
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$buildRoot = Join-Path $projectRoot 'build/appworld'
$sourceRoot = Join-Path $buildRoot 'source'
$dataRoot = Join-Path $buildRoot 'data-root'
$revision = '42b5bcf3cd334fee33f0c37c02070a9f5807add5'
$baseImage = 'tars-appworld:42b5bcf'
$image = 'tars-appworld:42b5bcf-sqlmodel044'
$compatRoot = Join-Path $PSScriptRoot 'appworld-compat'

function Invoke-CheckedNative([string]$Executable, [string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Executable exited with $LASTEXITCODE" }
}

New-Item -ItemType Directory -Force -Path $buildRoot | Out-Null
if (-not (Test-Path -LiteralPath $sourceRoot)) {
    Invoke-CheckedNative git @('clone', '--no-checkout', 'https://github.com/StonyBrookNLP/appworld.git', $sourceRoot)
    Invoke-CheckedNative git @('-C', $sourceRoot, 'checkout', $revision)
}
$actualRevision = & git -C $sourceRoot rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $actualRevision -ne $revision) {
    throw 'Existing AppWorld source does not match the fixed revision; it was not changed.'
}
$imageText = & docker image inspect $baseImage 2>$null
if ($LASTEXITCODE -ne 0) {
    Invoke-CheckedNative docker @('build', '--build-arg', 'APPWORLD_VERSION=source',
        '--label', "org.opencontainers.image.revision=$revision", '--tag', $baseImage,
        '--file', (Join-Path $sourceRoot 'dockerfile'), $sourceRoot)
    $imageText = & docker image inspect $baseImage
    if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect the built AppWorld image' }
}
$imageInfo = $imageText | ConvertFrom-Json
if ($imageInfo[0].Config.Labels.'org.opencontainers.image.revision' -ne $revision) {
    throw 'AppWorld image tag belongs to a different source revision.'
}
$imageText | Set-Content -Encoding utf8 (Join-Path $buildRoot 'base-image-metadata.json')

# Preserve the upstream source and data. SQLModel 0.0.45 changed naive datetime storage.
$compatText = & docker image inspect $image 2>$null
if ($LASTEXITCODE -ne 0) {
    Invoke-CheckedNative docker @('build', '--tag', $image, '--file',
        (Join-Path $compatRoot 'Dockerfile'), $compatRoot)
    $compatText = & docker image inspect $image
    if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect the AppWorld compatibility image' }
}
$compatInfo = ($compatText | ConvertFrom-Json)[0]
if ($compatInfo.Config.Labels.'org.opencontainers.image.revision' -ne $revision -or
    $compatInfo.Config.Labels.'com.tars-agent.appworld.compatibility' -ne 'sqlmodel044-naive-datetime') {
    throw 'Existing compatibility image does not match the fixed source and dependency constraint.'
}
Invoke-CheckedNative docker @('run', '--rm', '--network', 'none', '--label',
    'com.tars-agent.appworld.owner=compatibility-version-check', '--entrypoint', 'python', $image,
    '-c', 'import sqlmodel; assert sqlmodel.__version__ == "0.0.44"; print("SQLModel compatibility pin verified")')
$compatText | Set-Content -Encoding utf8 (Join-Path $buildRoot 'compat-image-metadata.json')
$freezeText = & docker run --rm --network none --label 'com.tars-agent.appworld.owner=compatibility-freeze' --entrypoint uv $image pip freeze --system
if ($LASTEXITCODE -ne 0) { throw 'Cannot record the compatibility image dependency versions' }
$freezeText | Set-Content -Encoding utf8 (Join-Path $buildRoot 'requirements.sqlmodel044.freeze.txt')

$dataset = Join-Path $dataRoot 'data'
if (Test-Path -LiteralPath $dataset) {
    $versionFile = Join-Path $dataset 'version.txt'
    if (-not (Test-Path -LiteralPath $versionFile) -or (Get-Content -Raw $versionFile).Trim() -ne '0.2.0') {
        throw 'Existing data is incomplete or has a different version; it was not overwritten.'
    }
    Write-Output 'Reusing existing AppWorld 0.2.0 data snapshot.'
    exit 0
}
New-Item -ItemType Directory -Force -Path $dataRoot | Out-Null
$bundle = Join-Path $buildRoot 'data-0.2.0.bundle'
if (-not (Test-Path -LiteralPath $bundle)) {
    Invoke-WebRequest -Uri 'https://s3.us-west-2.amazonaws.com/appworld.dev/data-0.2.0.bundle' -OutFile $bundle
}
$bundleHash = (Get-FileHash -LiteralPath $bundle -Algorithm SHA256).Hash
$bundleHash | Set-Content (Join-Path $buildRoot 'data-bundle.sha256')
$unpackCode = 'from appworld.common.crypto import unpack_bundle; from appworld.common.constants import PASSWORD,SALT; unpack_bundle(bundle_file_path="/input/data.bundle",base_directory="/run",password=PASSWORD,salt=SALT); print("official data unpacked")'
Invoke-CheckedNative docker @('run', '--rm', '--network', 'none', '--label',
    'com.tars-agent.appworld.owner=setup', '--mount', "type=bind,source=$dataRoot,target=/run",
    '--mount', "type=bind,source=$bundle,target=/input/data.bundle,readonly", '--entrypoint',
    'python', $image, '-c', $unpackCode)
Write-Output 'AppWorld environment prepared. Verify train/dev before running test_normal.'

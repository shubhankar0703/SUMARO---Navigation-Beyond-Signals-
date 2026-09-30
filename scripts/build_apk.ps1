# SUMARO - Android APK Build Automation Script
# Configures Java 17 environment and builds debug APK

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$AndroidDir = Join-Path $ProjectRoot "android"

# Check JDK 17
$Jdk17Path = "C:\Users\shubh\.jdks\jdk-17.0.20.1+1"
if (Test-Path $Jdk17Path) {
    $env:JAVA_HOME = $Jdk17Path
} else {
    Write-Host "Warning: Default JDK 17 path not found. Checking system JAVA_HOME..."
}

# Android SDK
$SdkPath = "C:\Users\shubh\AppData\Local\Android\Sdk"
if (Test-Path $SdkPath) {
    $env:ANDROID_HOME = $SdkPath
    $env:ANDROID_SDK_ROOT = $SdkPath
}

Write-Host "Building SUMARO Android APK..."
Write-Host "JAVA_HOME: $env:JAVA_HOME"
Write-Host "ANDROID_HOME: $env:ANDROID_HOME"

Push-Location $AndroidDir
try {
    .\gradlew.bat assembleDebug
} finally {
    Pop-Location
}

# Copy output APK
$BuiltApk = "C:\Users\shubh\.android_builds\SUMARO\app\outputs\apk\debug\app-debug.apk"
if (Test-Path $BuiltApk) {
    $DistDir = Join-Path $ProjectRoot "dist"
    if (-not (Test-Path $DistDir)) { New-Item -ItemType Directory -Path $DistDir | Out-Null }
    
    $DistApk = Join-Path $DistDir "sumaro-navigation-debug.apk"
    $RootApk = Join-Path $ProjectRoot "sumaro-debug.apk"
    
    Copy-Item $BuiltApk $DistApk -Force
    Copy-Item $BuiltApk $RootApk -Force
    
    Write-Host "APK Built Successfully!" -ForegroundColor Green
    Write-Host "Root APK: $RootApk"
    Write-Host "Dist APK: $DistApk"
} else {
    Write-Host "APK build artifact not found at $BuiltApk" -ForegroundColor Red
}

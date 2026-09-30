# SUMARO Android Prototype

This is the Android technical demonstration for SUMARO, a smartphone-based GNSS-denied dead-reckoning system for 2-wheelers.

## Architecture Overview

The system comprises:
- `IMUSensorManager`: Captures Accelerometer and Gyroscope readings.
- `GNSSManager`: Retrieves Fused Location updates and manages the GNSS status.
- `SumaroEKF`: The core Extended Kalman Filter written in pure Kotlin without external Linear Algebra libraries. It performs the prediction and GNSS update steps exactly matching the Python prototype.
- `MLInertialCorrector`: Stubbed interface for providing inertial corrections via ONNX/TFLite models.
- `NavigationEngine`: The main orchestrator connecting sensors with the EKF and managing mode transitions (GNSS_AIDED <-> DR_ONLY).

## ML Model Export (Python -> ONNX)

To bring the Scikit-learn Gradient Boosting models to Android, you must first convert them to ONNX. 
We provide a utility script `scripts/export_model_onnx.py` to convert `.joblib` models to `.onnx`.

## APK Installation & Build

### Pre-Built APK
The ready-to-install Android APK is generated at:
- **`sumaro-debug.apk`** (Project root)
- **`dist/sumaro-navigation-debug.apk`**

### Direct Installation (via ADB)
Connect your Android device with USB Debugging enabled:
```bash
adb install -r sumaro-debug.apk
```
Or transfer the `.apk` file directly to your Android device (via WhatsApp, Google Drive, or USB) and tap to install (enable "Install unknown apps" if prompted).

### How to Rebuild
Run the automated build script:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_apk.ps1
```
Or build with Gradle:
```bash
cd android
.\gradlew.bat assembleDebug
```
Ensure Java 17 LTS and Android SDK (API 34) are installed.

## Limitations

- **ML Inference**: `MLInertialCorrector` currently returns zeros. It needs the ONNX model initialized with real shapes and windowed features (dependent on the Python model's input feature size).
- **Coordinate Frames**: The coordinate transformation uses a simple ENU projection suitable for demo purposes.
- **UI**: Barebones UI for validation of logical integration.

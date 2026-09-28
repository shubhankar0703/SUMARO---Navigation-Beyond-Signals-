package com.sumaro.nav.ui.viewmodel

import android.app.Application
import android.os.Handler
import android.os.Looper
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.sumaro.navigation.core.SensorFusionManager
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch

enum class GnssState {
    NORMAL, LOST, RESTORED
}

enum class NavTrackingMode {
    GPS, DEAD_RECKONING
}

enum class AppLanguage(val displayName: String, val code: String) {
    ENGLISH("English", "en"),
    HINDI("हिंदी", "hi"),
    MARATHI("मराठी", "mr")
}

data class PositionState(
    val x: Double = 0.0,
    val y: Double = 0.0,
    val latitude: Double = 18.5204,
    val longitude: Double = 73.8567,
    val heading: Float = 45f,
    val usedGps: Boolean = true,
    val uncertainty: Double = 12.0,
    val confidenceRadius: Float = 12f,
    val trackingMode: NavTrackingMode = NavTrackingMode.GPS,
    val gnssState: GnssState = GnssState.NORMAL,
    val satelliteCount: Int = 11,
    val signalStrengthDb: Int = 38,
    val drDriftEstimate: Float = 0.8f,
    val currentInstruction: String = "Turn right onto MG Road in 150m",
    val nextInstruction: String = "Then follow NH 48 for 2.4km",
    val currentLanguage: AppLanguage = AppLanguage.ENGLISH,
    val isCalibrating: Boolean = false,
    val calibrationStep: Int = 1,
    val calibrationProgress: Float = 0f,
    val isCalibrationComplete: Boolean = false
)

class NavigationViewModel(application: Application) : AndroidViewModel(application) {
    private val sensorFusionManager = SensorFusionManager(application)

    private val _uiState = MutableStateFlow(PositionState())
    val uiState: StateFlow<PositionState> = _uiState.asStateFlow()

    init {
        sensorFusionManager.start()

        viewModelScope.launch {
            sensorFusionManager.fusionState.collect { result ->
                _uiState.update { current ->
                    val trackingMode = if (result.usedGps) NavTrackingMode.GPS else NavTrackingMode.DEAD_RECKONING
                    val gnssState = if (result.usedGps) GnssState.NORMAL else GnssState.LOST
                    val baseRadius = 10f
                    val scaleFactor = 2.5f
                    val confidenceRadius = baseRadius + (result.uncertainty * scaleFactor).toFloat()

                    current.copy(
                        x = result.x,
                        y = result.y,
                        usedGps = result.usedGps,
                        uncertainty = result.uncertainty,
                        confidenceRadius = confidenceRadius,
                        trackingMode = trackingMode,
                        gnssState = gnssState,
                        satelliteCount = if (result.usedGps) 12 else 0,
                        signalStrengthDb = if (result.usedGps) 40 else 0
                    )
                }
            }
        }
    }

    fun setLanguage(language: AppLanguage) {
        _uiState.update { it.copy(currentLanguage = language) }
    }

    fun simulateGnssLoss() {
        _uiState.update {
            it.copy(
                gnssState = GnssState.LOST,
                trackingMode = NavTrackingMode.DEAD_RECKONING,
                confidenceRadius = 55f,
                satelliteCount = 0,
                signalStrengthDb = 0
            )
        }
    }

    fun simulateGnssRestore() {
        _uiState.update {
            it.copy(
                gnssState = GnssState.RESTORED,
                trackingMode = NavTrackingMode.GPS,
                confidenceRadius = 12f,
                satelliteCount = 12,
                signalStrengthDb = 40
            )
        }
        Handler(Looper.getMainLooper()).postDelayed({
            _uiState.update { if (it.gnssState == GnssState.RESTORED) it.copy(gnssState = GnssState.NORMAL) else it }
        }, 3000)
    }

    fun startCalibration() {
        _uiState.update {
            it.copy(
                isCalibrating = true,
                calibrationStep = 1,
                calibrationProgress = 0.25f,
                isCalibrationComplete = false
            )
        }
    }

    fun advanceCalibration() {
        _uiState.update { state ->
            val nextStep = state.calibrationStep + 1
            if (nextStep > 3) {
                state.copy(isCalibrating = false, isCalibrationComplete = true, calibrationProgress = 1f)
            } else {
                state.copy(calibrationStep = nextStep, calibrationProgress = nextStep * 0.33f)
            }
        }
    }

    fun resetCalibration() {
        _uiState.update {
            it.copy(isCalibrating = false, isCalibrationComplete = false, calibrationStep = 1, calibrationProgress = 0f)
        }
    }

    override fun onCleared() {
        super.onCleared()
        sensorFusionManager.stop()
    }
}

package com.sumaro.nav.ui.viewmodel

import android.os.Handler
import android.os.Looper
import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

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

data class NavigationUiState(
    val currentLanguage: AppLanguage = AppLanguage.ENGLISH,
    val gnssState: GnssState = GnssState.NORMAL,
    val trackingMode: NavTrackingMode = NavTrackingMode.GPS,
    val confidenceRadius: Float = 12f, // meters
    val latitude: Double = 18.5204,
    val longitude: Double = 73.8567,
    val heading: Float = 45f,
    val satelliteCount: Int = 11,
    val signalStrengthDb: Int = 38,
    val drDriftEstimate: Float = 0.8f,
    val currentInstruction: String = "Turn right onto MG Road in 150m",
    val nextInstruction: String = "Then follow NH 48 for 2.4km",
    val isCalibrating: Boolean = false,
    val calibrationStep: Int = 1,
    val calibrationProgress: Float = 0f,
    val isCalibrationComplete: Boolean = false
)

class NavigationViewModel : ViewModel() {
    private val _uiState = MutableStateFlow(NavigationUiState())
    val uiState: StateFlow<NavigationUiState> = _uiState.asStateFlow()

    fun setLanguage(language: AppLanguage) {
        _uiState.update { it.copy(currentLanguage = language) }
    }

    fun simulateGnssLoss() {
        _uiState.update {
            it.copy(
                gnssState = GnssState.LOST,
                trackingMode = NavTrackingMode.DEAD_RECKONING,
                confidenceRadius = 45f,
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
}

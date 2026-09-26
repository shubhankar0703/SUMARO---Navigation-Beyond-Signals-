package com.sumaro.nav

import android.content.Context
import android.util.Log
import com.sumaro.nav.ekf.SumaroEKF
import com.sumaro.nav.ml.MLInertialCorrector
import com.sumaro.nav.sensor.GNSSManager
import com.sumaro.nav.sensor.IMUSensorManager

enum class NavMode {
    GNSS_AIDED,
    DR_ONLY
}

data class NavState(
    val x: Double,
    val y: Double,
    val heading: Double,
    val mode: NavMode,
    val gnssAvailable: Boolean
)

class NavigationEngine(private val context: Context) {
    private val imuManager = IMUSensorManager(context)
    private val gnssManager = GNSSManager(context)
    private val ekf = SumaroEKF()
    private val mlCorrector = MLInertialCorrector()

    var onStateUpdate: ((NavState) -> Unit)? = null
    
    private var lastTimestampMs = -1L
    private var currentMode = NavMode.DR_ONLY

    init {
        imuManager.onReadingReady = { reading ->
            processIMU(reading.timestampMs, reading.accel, reading.gyro)
        }

        gnssManager.onLocationUpdate = { enu ->
            ekf.updateGNSS(enu.x, enu.y)
            Log.d("NavEngine", "GNSS Update applied")
        }

        gnssManager.onGNSSStatusChange = { hasFix ->
            currentMode = if (hasFix) NavMode.GNSS_AIDED else NavMode.DR_ONLY
            Log.d("NavEngine", "Mode switched to: $currentMode")
        }
    }

    fun start() {
        imuManager.start()
        gnssManager.start()
    }

    fun stop() {
        imuManager.stop()
        gnssManager.stop()
        mlCorrector.close()
    }
    
    // Call this periodically to check for GNSS outage
    fun checkStatus() {
        gnssManager.checkStatus()
    }

    private fun processIMU(timestampMs: Long, accel: FloatArray, gyro: FloatArray) {
        if (lastTimestampMs < 0) {
            lastTimestampMs = timestampMs
            return
        }

        val dt = (timestampMs - lastTimestampMs) / 1000.0
        lastTimestampMs = timestampMs

        // Get ML corrections
        val corrections = mlCorrector.addReading(accel, gyro)

        // Predict
        ekf.predict(
            dt, 
            accel[0].toDouble(), accel[1].toDouble(), gyro[2].toDouble(),
            corrections.deltaAccelX, corrections.deltaAccelY, corrections.deltaGyroZ
        )

        // Publish State
        val state = NavState(
            ekf.state[0],
            ekf.state[1],
            ekf.state[4],
            currentMode,
            gnssManager.hasFix
        )
        onStateUpdate?.invoke(state)
    }
}

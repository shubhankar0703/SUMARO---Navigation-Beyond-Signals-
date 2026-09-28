package com.sumaro.navigation.core

import android.annotation.SuppressLint
import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.os.Looper
import com.google.android.gms.location.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlin.math.cos

class SensorFusionManager(private val context: Context) : SensorEventListener {
    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val fusedLocationClient = LocationServices.getFusedLocationProviderClient(context)

    private val ekf = SumaroEKF()
    private val accelRolling = RollingStats(10)
    private val gyroRolling = RollingStats(10)

    private val _fusionState = MutableStateFlow(SumaroEKF.FusionResult(0.0, 0.0, false, 10.0))
    val fusionState: StateFlow<SumaroEKF.FusionResult> = _fusionState.asStateFlow()

    private var lastTimestampNs = 0L
    private var lastGpsLocation: Location? = null
    private var referenceLocation: Location? = null
    private var lastGpsTimeMs = 0L
    private val gpsTimeoutMs = 3000L // 3 seconds without GPS considered GNSS loss

    private val locationCallback = object : LocationCallback() {
        override fun onLocationResult(result: LocationResult) {
            val location = result.lastLocation ?: return
            if (referenceLocation == null) {
                referenceLocation = location
            }
            lastGpsLocation = location
            lastGpsTimeMs = System.currentTimeMillis()
        }
    }

    @SuppressLint("MissingPermission")
    fun start() {
        accelerometer?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        gyroscope?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }

        val request = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 1000)
            .setMinUpdateIntervalMillis(1000)
            .build()
        fusedLocationClient.requestLocationUpdates(request, locationCallback, Looper.getMainLooper())
    }

    fun stop() {
        sensorManager.unregisterListener(this)
        fusedLocationClient.removeLocationUpdates(locationCallback)
    }

    override fun onSensorChanged(event: SensorEvent?) {
        event ?: return
        val currentTimestampNs = event.timestamp
        if (lastTimestampNs == 0L) {
            lastTimestampNs = currentTimestampNs
            return
        }
        val dt = (currentTimestampNs - lastTimestampNs) / 1_000_000_000.0
        lastTimestampNs = currentTimestampNs

        if (dt <= 0.0 || dt > 0.5) return

        val accelX = event.values[0].toDouble()
        val gyroZ = if (event.sensor.type == Sensor.TYPE_GYROSCOPE) event.values[2].toDouble() else 0.0

        if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
            accelRolling.addSample(accelX)
        } else if (event.sensor.type == Sensor.TYPE_GYROSCOPE) {
            gyroRolling.addSample(gyroZ)
        }

        // Features: [accel_forward, gyro_z, rolling_std_accel_10sample, rolling_std_gyro_10sample]
        val features = doubleArrayOf(
            accelX,
            gyroZ,
            accelRolling.currentStd(),
            gyroRolling.currentStd()
        )

        val accelBias = BiasCorrection.predictAccelCorrection(features)
        val gyroBias = BiasCorrection.predictGyroCorrection(features)

        var gpsXY: Pair<Double, Double>? = null
        val gpsLoc = lastGpsLocation
        val refLoc = referenceLocation
        if (gpsLoc != null && refLoc != null && (System.currentTimeMillis() - lastGpsTimeMs < gpsTimeoutMs)) {
            val enu = convertToLocalENU(gpsLoc, refLoc)
            gpsXY = Pair(enu.first, enu.second)
        }

        val result = ekf.step(
            accelRaw = accelX,
            gyroRaw = gyroZ,
            dt = dt,
            gpsXY = gpsXY,
            biasCorrection = Pair(accelBias, gyroBias)
        )

        _fusionState.value = result
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun convertToLocalENU(current: Location, ref: Location): Pair<Double, Double> {
        val r = 6371000.0
        val latRef = Math.toRadians(ref.latitude)
        val dLat = Math.toRadians(current.latitude - ref.latitude)
        val dLon = Math.toRadians(current.longitude - ref.longitude)
        val y = r * dLat
        val x = r * dLon * cos(latRef)
        return Pair(x, y)
    }
}

package com.sumaro.nav.sensor

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import java.util.concurrent.ConcurrentLinkedQueue

data class IMUReading(
    val timestampMs: Long,
    val accel: FloatArray,
    val gyro: FloatArray
)

class IMUSensorManager(context: Context) : SensorEventListener {
    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

    private var lastAccel: FloatArray? = null
    private var lastGyro: FloatArray? = null
    
    // Thread-safe buffer for latest readings
    val readingsBuffer = ConcurrentLinkedQueue<IMUReading>()
    
    var onReadingReady: ((IMUReading) -> Unit)? = null

    fun start() {
        accelerometer?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        gyroscope?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
    }

    fun stop() {
        sensorManager.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent?) {
        event ?: return
        
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> lastAccel = event.values.clone()
            Sensor.TYPE_GYROSCOPE -> lastGyro = event.values.clone()
        }

        // When both are available, create a reading
        if (lastAccel != null && lastGyro != null) {
            val timestampMs = System.currentTimeMillis()
            val reading = IMUReading(timestampMs, lastAccel!!, lastGyro!!)
            readingsBuffer.add(reading)
            
            // Keep buffer size manageable
            if (readingsBuffer.size > 100) {
                readingsBuffer.poll()
            }
            
            onReadingReady?.invoke(reading)
            
            lastAccel = null
            lastGyro = null
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // No-op
    }
}

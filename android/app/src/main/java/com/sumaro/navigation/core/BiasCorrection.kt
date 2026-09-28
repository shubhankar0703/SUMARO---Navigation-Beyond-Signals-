package com.sumaro.navigation.core

import java.util.LinkedList
import kotlin.math.sqrt

/**
 * Rolling window statistics calculator for live feature extraction.
 */
class RollingStats(private val windowSize: Int) {
    private val samples = LinkedList<Double>()
    private var sum = 0.0
    private var sumSq = 0.0

    fun addSample(value: Double) {
        if (samples.size >= windowSize) {
            val removed = samples.removeFirst()
            sum -= removed
            sumSq -= removed * removed
        }
        samples.addLast(value)
        sum += value
        sumSq += value * value
    }

    fun currentStd(): Double {
        val n = samples.size
        if (n < 2) return 0.0
        val mean = sum / n
        val variance = (sumSq / n) - (mean * mean)
        return if (variance > 0.0) sqrt(variance) else 0.0
    }
}

/**
 * ML Bias Correction model using linear regression coefficients.
 */
object BiasCorrection {
    // --- REPLACE WITH TRAINED VALUES ---
    val ACCEL_WEIGHTS = doubleArrayOf(0.05, 0.02, 0.01, 0.005)
    const val ACCEL_INTERCEPT = 0.001

    val GYRO_WEIGHTS = doubleArrayOf(0.001, 0.03, 0.002, 0.001)
    const val GYRO_INTERCEPT = 0.0001
    // ----------------------------------

    /**
     * Predicts accelerometer bias correction.
     * Features: [accel_forward, gyro_z, rolling_std_accel_10sample, rolling_std_gyro_10sample]
     */
    fun predictAccelCorrection(features: DoubleArray): Double {
        var dot = ACCEL_INTERCEPT
        val len = minOf(features.size, ACCEL_WEIGHTS.size)
        for (i in 0 until len) {
            dot += features[i] * ACCEL_WEIGHTS[i]
        }
        return dot
    }

    /**
     * Predicts gyroscope bias correction.
     * Features: [accel_forward, gyro_z, rolling_std_accel_10sample, rolling_std_gyro_10sample]
     */
    fun predictGyroCorrection(features: DoubleArray): Double {
        var dot = GYRO_INTERCEPT
        val len = minOf(features.size, GYRO_WEIGHTS.size)
        for (i in 0 until len) {
            dot += features[i] * GYRO_WEIGHTS[i]
        }
        return dot
    }
}

package com.sumaro.navigation.core

import kotlin.math.*

/**
 * Extended Kalman Filter (EKF) for SUMARO Two-Wheeler Dead Reckoning Navigation.
 * 
 * State Vector (5D): [pos_x, pos_y, velocity, heading, gyro_bias]
 * 
 * This filter fuses accelerometer, gyroscope, and GNSS measurements to maintain 
 * continuous, smooth positioning even when GNSS signals are lost (tunnels, urban canyons, parking).
 */
class SumaroEKF(
    private val adaptiveWeightingEnabled: Boolean = true,
    private val confidenceRejectionEnabled: Boolean = true
) {
    // 5D State vector: [x, y, v, theta, gyro_bias]
    var state = DoubleArray(5) { 0.0 }
        private set

    // 5x5 Error Covariance Matrix P
    var P = Array(5) { DoubleArray(5) { 0.0 } }
        private set

    // Base Process Noise Covariance Q diagonal: [pos_x, pos_y, velocity, heading, gyro_bias]
    private val baseQ = doubleArrayOf(2.0, 2.0, 5.0, 0.1, 0.0001)

    // GPS Measurement Noise R (2x2 diagonal): (2.0 std)^2 = 4.0
    private val R = arrayOf(
        doubleArrayOf(4.0, 0.0),
        doubleArrayOf(0.0, 4.0)
    )

    private var previousAccel = 0.0

    init {
        // Initialize Covariance P with initial uncertainties
        P[0][0] = 10.0 // pos_x
        P[1][1] = 10.0 // pos_y
        P[2][2] = 4.0  // velocity
        P[3][3] = Math.toRadians(10.0).pow(2) // heading
        P[4][4] = 0.01 // gyro_bias
    }

    data class FusionResult(
        val x: Double,
        val y: Double,
        val usedGps: Boolean,
        val uncertainty: Double
    )

    /**
     * Executes one prediction and optional GNSS correction step.
     * 
     * @param accelRaw Raw forward acceleration (m/s^2)
     * @param gyroRaw Raw angular velocity (rad/s)
     * @param dt Time delta since last reading (seconds)
     * @param gpsXY Optional GPS ENU position (pos_x, pos_y) if available this step
     * @param biasCorrection Optional ML bias corrections (delta_accel, delta_gyro)
     */
    fun step(
        accelRaw: Double,
        gyroRaw: Double,
        dt: Double,
        gpsXY: Pair<Double, Double>?,
        biasCorrection: Pair<Double, Double>? = null
    ): FusionResult {
        if (dt <= 0.0) {
            return FusionResult(state[0], state[1], false, getUncertainty())
        }

        val px = state[0]
        val py = state[1]
        val velocity = state[2]
        val heading = state[3]
        val gyroBias = state[4]

        // Apply ML bias corrections if provided
        val correctedAccel = accelRaw + (biasCorrection?.first ?: 0.0)
        val correctedGyro = gyroRaw + (biasCorrection?.second ?: 0.0)

        // 1. PREDICT STEP
        val yawRate = correctedGyro - gyroBias
        val nextHeading = (heading + yawRate * dt) % (2 * PI)
        val nextVelocity = max(0.0, velocity + correctedAccel * dt)
        val nextPx = px + velocity * cos(heading) * dt
        val nextPy = py + velocity * sin(heading) * dt

        val predictedState = doubleArrayOf(nextPx, nextPy, nextVelocity, nextHeading, gyroBias)

        // Compute Jerk for Adaptive Weighting
        val jerk = abs(correctedAccel - previousAccel) / dt
        previousAccel = correctedAccel

        // Construct Q with adaptive weighting if jerk exceeds threshold
        val Q = Array(5) { i -> DoubleArray(5) { j -> if (i == j) baseQ[i] else 0.0 } }
        if (adaptiveWeightingEnabled && jerk > 1.0) {
            val multi = doubleArrayOf(1.0, 1.0, 8.0, 4.0, 1.0)
            for (i in 0..4) {
                Q[i][i] *= multi[i]
            }
        }

        // State Transition Jacobian F (5x5)
        val s = sin(heading)
        val c = cos(heading)
        val F = arrayOf(
            doubleArrayOf(1.0, 0.0, c * dt, -velocity * s * dt, 0.0),
            doubleArrayOf(0.0, 1.0, s * dt,  velocity * c * dt, 0.0),
            doubleArrayOf(0.0, 0.0, 1.0, 0.0, 0.0),
            doubleArrayOf(0.0, 0.0, 0.0, 1.0, -dt),
            doubleArrayOf(0.0, 0.0, 0.0, 0.0, 1.0)
        )

        // P_pred = F * P * F^T + Q
        val FP = multiplyMatrix(F, P)
        val F_T = transposeMatrix(F)
        val FPF_T = multiplyMatrix(FP, F_T)
        var predictedP = addMatrix(FPF_T, Q)

        state = predictedState
        P = predictedP

        var usedGps = false

        // 2. GPS CORRECTION STEP (if GPS fix available)
        if (gpsXY != null) {
            val z = doubleArrayOf(gpsXY.first, gpsXY.second)

            // Measurement matrix H (2x5): observes pos_x and pos_y
            val H = arrayOf(
                doubleArrayOf(1.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 1.0, 0.0, 0.0, 0.0)
            )

            // Innovation y = z - H * x_pred
            val zPred = doubleArrayOf(state[0], state[1])
            val y = doubleArrayOf(z[0] - zPred[0], z[1] - zPred[1])

            // S = H * P * H^T + R
            val HP = multiplyMatrix(H, P)
            val H_T = transposeMatrix(H)
            val HPH_T = multiplyMatrix(HP, H_T)
            val S = addMatrix(HPH_T, R)

            // Invert S (2x2)
            val SInv = invert2x2(S)

            // Confidence-based rejection using Mahalanobis distance
            // d_squared = y^T * S^-1 * y
            var acceptGps = true
            if (confidenceRejectionEnabled && SInv != null) {
                val sInvY = multiplyMatrixVector(SInv, y)
                val dSquared = y[0] * sInvY[0] + y[1] * sInvY[1]
                // Chi-square 99% threshold for 2 degrees of freedom is 9.21
                if (dSquared > 9.21) {
                    acceptGps = false
                }
            }

            if (acceptGps && SInv != null) {
                // Kalman Gain K = P * H^T * S^-1
                val PH_T = multiplyMatrix(P, H_T)
                val K = multiplyMatrix(PH_T, SInv)

                // x = x + K * y
                val Ky = multiplyMatrixVector(K, y)
                for (i in 0..4) {
                    state[i] += Ky[i]
                }

                // P = (I - K * H) * P
                val I = Array(5) { i -> DoubleArray(5) { j -> if (i == j) 1.0 else 0.0 } }
                val KH = multiplyMatrix(K, H)
                val I_KH = subtractMatrix(I, KH)
                P = multiplyMatrix(I_KH, P)
                usedGps = true
            }
        }

        return FusionResult(state[0], state[1], usedGps, getUncertainty())
    }

    /**
     * Uncertainty is defined as the trace of the top-left 2x2 position covariance block of P.
     */
    private fun getUncertainty(): Double {
        return P[0][0] + P[1][1]
    }

    // --- Matrix Math Helpers ---

    private fun multiplyMatrix(A: Array<DoubleArray>, B: Array<DoubleArray>): Array<DoubleArray> {
        val rows = A.size
        val cols = B[0].size
        val common = A[0].size
        val result = Array(rows) { DoubleArray(cols) { 0.0 } }
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                var sum = 0.0
                for (k in 0 until common) sum += A[i][k] * B[k][j]
                result[i][j] = sum
            }
        }
        return result
    }

    private fun multiplyMatrixVector(A: Array<DoubleArray>, V: DoubleArray): DoubleArray {
        val rows = A.size
        val cols = A[0].size
        val result = DoubleArray(rows) { 0.0 }
        for (i in 0 until rows) {
            var sum = 0.0
            for (j in 0 until cols) sum += A[i][j] * V[j]
            result[i] = sum
        }
        return result
    }

    private fun transposeMatrix(A: Array<DoubleArray>): Array<DoubleArray> {
        val rows = A.size
        val cols = A[0].size
        val result = Array(cols) { DoubleArray(rows) { 0.0 } }
        for (i in 0 until rows) {
            for (j in 0 until cols) result[j][i] = A[i][j]
        }
        return result
    }

    private fun addMatrix(A: Array<DoubleArray>, B: Array<DoubleArray>): Array<DoubleArray> {
        val rows = A.size
        val cols = A[0].size
        val result = Array(rows) { DoubleArray(cols) { 0.0 } }
        for (i in 0 until rows) {
            for (j in 0 until cols) result[i][j] = A[i][j] + B[i][j]
        }
        return result
    }

    private fun subtractMatrix(A: Array<DoubleArray>, B: Array<DoubleArray>): Array<DoubleArray> {
        val rows = A.size
        val cols = A[0].size
        val result = Array(rows) { DoubleArray(cols) { 0.0 } }
        for (i in 0 until rows) {
            for (j in 0 until cols) result[i][j] = A[i][j] - B[i][j]
        }
        return result
    }

    private fun invert2x2(A: Array<DoubleArray>): Array<DoubleArray>? {
        val det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
        if (abs(det) < 1e-9) return null
        val result = Array(2) { DoubleArray(2) { 0.0 } }
        result[0][0] = A[1][1] / det
        result[0][1] = -A[0][1] / det
        result[1][0] = -A[1][0] / det
        result[1][1] = A[0][0] / det
        return result
    }
}

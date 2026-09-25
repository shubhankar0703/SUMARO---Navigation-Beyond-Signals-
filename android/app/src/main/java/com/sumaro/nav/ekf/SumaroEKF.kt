package com.sumaro.nav.ekf

import kotlin.math.*

// EKF for SUMARO
// States: [px, py, vx, vy, heading, gyro_bias] (6x1)

class SumaroEKF {
    var state = DoubleArray(6) { 0.0 }
    var P = Array(6) { DoubleArray(6) { 0.0 } }
    
    // Q = diag([0.01, 0.01, 0.10, 0.10, 1e-5, 1e-7])
    private val Q = Array(6) { DoubleArray(6) { 0.0 } }
    
    // R_gnss = diag([16, 16])
    private val R_gnss = Array(2) { DoubleArray(2) { 0.0 } }
    
    var mountingAngle = 0.0 // Configurable mounting angle
    
    init {
        // Initialize Q
        Q[0][0] = 0.01; Q[1][1] = 0.01
        Q[2][2] = 0.10; Q[3][3] = 0.10
        Q[4][4] = 1e-5; Q[5][5] = 1e-7
        
        // Initialize R
        R_gnss[0][0] = 16.0; R_gnss[1][1] = 16.0
        
        // Initialize P0
        P[0][0] = 25.0; P[1][1] = 25.0
        P[2][2] = 25.0; P[3][3] = 25.0
        P[4][4] = Math.toRadians(5.0).pow(2)
        P[5][5] = 0.02.pow(2)
    }

    fun predict(dt: Double, accelX: Double, accelY: Double, gyroZ: Double, deltaAccelX: Double = 0.0, deltaAccelY: Double = 0.0, deltaGyroZ: Double = 0.0) {
        val px = state[0]
        val py = state[1]
        val vx = state[2]
        val vy = state[3]
        val heading = state[4]
        val gyroBias = state[5]

        // Adjust with ML corrections
        val corrAccelX = accelX + deltaAccelX
        val corrAccelY = accelY + deltaAccelY
        val corrGyroZ = gyroZ + deltaGyroZ

        // True yaw rate
        val yawRate = corrGyroZ - gyroBias
        
        // Transform accel to world frame
        val c = cos(heading + mountingAngle)
        val s = sin(heading + mountingAngle)
        val ax_w = corrAccelX * c - corrAccelY * s
        val ay_w = corrAccelX * s + corrAccelY * c

        // Predict State
        val nextState = DoubleArray(6)
        nextState[0] = px + vx * dt + 0.5 * ax_w * dt * dt
        nextState[1] = py + vy * dt + 0.5 * ay_w * dt * dt
        nextState[2] = vx + ax_w * dt
        nextState[3] = vy + ay_w * dt
        nextState[4] = (heading + yawRate * dt) % (2 * PI)
        nextState[5] = gyroBias

        // Jacobian F (6x6)
        val F = Array(6) { DoubleArray(6) { 0.0 } }
        for (i in 0..5) F[i][i] = 1.0
        F[0][2] = dt
        F[1][3] = dt
        F[2][4] = (-corrAccelX * s - corrAccelY * c) * dt
        F[3][4] = (corrAccelX * c - corrAccelY * s) * dt
        F[4][5] = -dt

        // P = F * P * F^T + Q
        val FP = multiplyMatrix(F, P)
        val F_T = transposeMatrix(F)
        val FPF_T = multiplyMatrix(FP, F_T)
        P = addMatrix(FPF_T, Q)
        state = nextState
    }

    fun updateGNSS(gnssX: Double, gnssY: Double) {
        // Measurement z
        val z = doubleArrayOf(gnssX, gnssY)
        
        // H = [1 0 0 0 0 0; 0 1 0 0 0 0]
        val H = Array(2) { DoubleArray(6) { 0.0 } }
        H[0][0] = 1.0; H[1][1] = 1.0

        // y = z - H * x
        val z_pred = doubleArrayOf(state[0], state[1])
        val y = doubleArrayOf(z[0] - z_pred[0], z[1] - z_pred[1])

        // S = H * P * H^T + R
        val HP = multiplyMatrix(H, P)
        val H_T = transposeMatrix(H)
        val HPH_T = multiplyMatrix(HP, H_T)
        val S = addMatrix(HPH_T, R_gnss)

        // K = P * H^T * S^-1
        val S_inv = invert2x2(S)
        val PH_T = multiplyMatrix(P, H_T)
        val K = multiplyMatrix(PH_T, S_inv)

        // x = x + K * y
        val Ky = multiplyMatrixVector(K, y)
        for (i in 0..5) state[i] += Ky[i]

        // P = (I - K * H) * P
        val I = Array(6) { DoubleArray(6) { 0.0 } }
        for (i in 0..5) I[i][i] = 1.0
        val KH = multiplyMatrix(K, H)
        val I_KH = subtractMatrix(I, KH)
        P = multiplyMatrix(I_KH, P)
    }

    // Matrix Math Helpers
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

    private fun invert2x2(A: Array<DoubleArray>): Array<DoubleArray> {
        val det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
        val result = Array(2) { DoubleArray(2) { 0.0 } }
        result[0][0] = A[1][1] / det
        result[0][1] = -A[0][1] / det
        result[1][0] = -A[1][0] / det
        result[1][1] = A[0][0] / det
        return result
    }
}

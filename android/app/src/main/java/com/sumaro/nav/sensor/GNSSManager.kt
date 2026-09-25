package com.sumaro.nav.sensor

import android.annotation.SuppressLint
import android.content.Context
import android.location.Location
import android.os.Looper
import com.google.android.gms.location.*
import kotlin.math.cos

data class ENUPosition(
    val x: Double,
    val y: Double,
    val timestampMs: Long
)

class GNSSManager(private val context: Context) {
    private val fusedLocationClient = LocationServices.getFusedLocationProviderClient(context)
    
    var hasFix: Boolean = false
        private set
        
    private var lastLocationTime: Long = 0
    private val outageTimeoutMs = 3000L // 3 seconds timeout
    
    private var startLocation: Location? = null
    
    var onLocationUpdate: ((ENUPosition) -> Unit)? = null
    var onGNSSStatusChange: ((Boolean) -> Unit)? = null

    private val locationCallback = object : LocationCallback() {
        override fun onLocationResult(result: LocationResult) {
            val location = result.lastLocation ?: return
            
            if (startLocation == null) {
                startLocation = location
            }
            
            lastLocationTime = System.currentTimeMillis()
            if (!hasFix) {
                hasFix = true
                onGNSSStatusChange?.invoke(true)
            }
            
            val enu = convertToLocalENU(location, startLocation!!)
            onLocationUpdate?.invoke(enu)
        }
    }

    @SuppressLint("MissingPermission")
    fun start() {
        val request = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 1000)
            .setMinUpdateIntervalMillis(1000)
            .build()
            
        fusedLocationClient.requestLocationUpdates(
            request,
            locationCallback,
            Looper.getMainLooper()
        )
    }

    fun stop() {
        fusedLocationClient.removeLocationUpdates(locationCallback)
        hasFix = false
    }

    fun checkStatus() {
        if (hasFix && System.currentTimeMillis() - lastLocationTime > outageTimeoutMs) {
            hasFix = false
            onGNSSStatusChange?.invoke(false)
        }
    }
    
    private fun convertToLocalENU(current: Location, ref: Location): ENUPosition {
        // Simple equirectangular projection for demo
        val r = 6371000.0 // Earth radius in meters
        val latRef = Math.toRadians(ref.latitude)
        
        val dLat = Math.toRadians(current.latitude - ref.latitude)
        val dLon = Math.toRadians(current.longitude - ref.longitude)
        
        val y = r * dLat
        val x = r * dLon * cos(latRef)
        
        return ENUPosition(x, y, current.time)
    }
}

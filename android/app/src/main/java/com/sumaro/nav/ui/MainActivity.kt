package com.sumaro.nav.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.sumaro.nav.NavigationEngine
import com.sumaro.nav.R
import java.util.Timer
import kotlin.concurrent.timerTask

class MainActivity : AppCompatActivity() {

    private lateinit var navEngine: NavigationEngine
    private var isRunning = false
    private var timer: Timer? = null

    private lateinit var textPos: TextView
    private lateinit var textHeading: TextView
    private lateinit var textMode: TextView
    private lateinit var textStatus: TextView
    private lateinit var btnToggle: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        textPos = findViewById(R.id.text_pos)
        textHeading = findViewById(R.id.text_heading)
        textMode = findViewById(R.id.text_mode)
        textStatus = findViewById(R.id.text_status)
        btnToggle = findViewById(R.id.btn_toggle)

        navEngine = NavigationEngine(this)
        
        navEngine.onStateUpdate = { state ->
            runOnUiThread {
                textPos.text = String.format("Pos: %.2f, %.2f", state.x, state.y)
                textHeading.text = String.format("Heading: %.2f deg", Math.toDegrees(state.heading))
                textMode.text = "Mode: \${state.mode}"
                textStatus.text = "GNSS: \${if (state.gnssAvailable) "AVAILABLE" else "LOST"}"
            }
        }

        btnToggle.setOnClickListener {
            if (isRunning) stopNav() else startNav()
        }

        checkPermissions()
    }

    private fun checkPermissions() {
        val permissions = arrayOf(
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION
        )
        if (!permissions.all { ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED }) {
            ActivityCompat.requestPermissions(this, permissions, 1)
        }
    }

    private fun startNav() {
        navEngine.start()
        isRunning = true
        btnToggle.text = "STOP"
        
        // Start check timer
        timer = Timer()
        timer?.schedule(timerTask {
            navEngine.checkStatus()
        }, 0, 1000)
    }

    private fun stopNav() {
        navEngine.stop()
        isRunning = false
        btnToggle.text = "START"
        timer?.cancel()
    }
    
    override fun onDestroy() {
        super.onDestroy()
        stopNav()
    }
}

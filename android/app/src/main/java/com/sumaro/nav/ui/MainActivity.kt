package com.sumaro.nav.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.*
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.sumaro.nav.ui.screens.CalibrationScreen
import com.sumaro.nav.ui.screens.GNSSStatusScreen
import com.sumaro.nav.ui.screens.HomeScreen
import com.sumaro.nav.ui.screens.NavigationMapScreen
import com.sumaro.nav.ui.theme.SUMAROTheme
import com.sumaro.nav.ui.viewmodel.NavigationViewModel

class MainActivity : ComponentActivity() {

    private val locationPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { _ -> }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            locationPermissionLauncher.launch(
                arrayOf(
                    Manifest.permission.ACCESS_FINE_LOCATION,
                    Manifest.permission.ACCESS_COARSE_LOCATION
                )
            )
        }

        setContent {
            SUMAROTheme {
                val navController = rememberNavController()
                val viewModel: NavigationViewModel = viewModel()

                NavHost(navController = navController, startDestination = "home") {
                    composable("home") {
                        HomeScreen(
                            viewModel = viewModel,
                            onNavigateToMap = { navController.navigate("map") },
                            onNavigateToGnss = { navController.navigate("gnss") },
                            onNavigateToCalibration = { navController.navigate("calibration") }
                        )
                    }
                    composable("map") {
                        NavigationMapScreen(
                            viewModel = viewModel,
                            onBack = { navController.popBackStack() }
                        )
                    }
                    composable("gnss") {
                        GNSSStatusScreen(
                            viewModel = viewModel,
                            onBack = { navController.popBackStack() }
                        )
                    }
                    composable("calibration") {
                        CalibrationScreen(
                            viewModel = viewModel,
                            onBack = { navController.popBackStack() }
                        )
                    }
                }
            }
        }
    }
}

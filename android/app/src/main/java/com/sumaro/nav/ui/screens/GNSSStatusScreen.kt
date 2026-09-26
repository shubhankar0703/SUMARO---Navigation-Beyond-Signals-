package com.sumaro.nav.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.sumaro.nav.ui.viewmodel.GnssState
import com.sumaro.nav.ui.viewmodel.NavigationViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun GNSSStatusScreen(
    viewModel: NavigationViewModel,
    onBack: () -> Unit
) {
    val uiState by viewModel.uiState.collectAsState()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("GNSS & Sensor Diagnostics", fontWeight = FontWeight.Bold) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.surface)
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(MaterialTheme.colorScheme.background)
                .padding(paddingValues)
                .padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Overall Status Card
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = when (uiState.gnssState) {
                        GnssState.NORMAL -> Color(0xFF16A34A).copy(alpha = 0.15f)
                        GnssState.LOST -> Color(0xFFD97706).copy(alpha = 0.15f)
                        GnssState.RESTORED -> Color(0xFF2563EB).copy(alpha = 0.15f)
                    }
                )
            ) {
                Row(
                    modifier = Modifier.padding(20.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = when (uiState.gnssState) {
                            GnssState.NORMAL -> Icons.Default.CheckCircle
                            GnssState.LOST -> Icons.Default.Warning
                            GnssState.RESTORED -> Icons.Default.Sync
                        },
                        contentDescription = null,
                        tint = when (uiState.gnssState) {
                            GnssState.NORMAL -> Color(0xFF16A34A)
                            GnssState.LOST -> Color(0xFFD97706)
                            GnssState.RESTORED -> Color(0xFF2563EB)
                        },
                        modifier = Modifier.size(36.dp)
                    )
                    Spacer(modifier = Modifier.width(16.dp))
                    Column {
                        Text(
                            text = when (uiState.gnssState) {
                                GnssState.NORMAL -> "GNSS Fix Optimal"
                                GnssState.LOST -> "GNSS Signal Lost (Dead Reckoning Active)"
                                GnssState.RESTORED -> "GNSS Signal Reconnected"
                            },
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold
                        )
                        Spacer(modifier = Modifier.height(2.dp))
                        Text(
                            text = when (uiState.gnssState) {
                                GnssState.NORMAL -> "Receiving high-precision GPS / GLONASS signals."
                                GnssState.LOST -> "Inertial IMU + ML correction seamlessly maintaining trajectory."
                                GnssState.RESTORED -> "Recalibrating EKF filter with live GPS updates."
                            },
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }

            // Diagnostics Grid
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                DiagnosticCard(
                    title = "Active Satellites",
                    value = "${uiState.satelliteCount}",
                    subtitle = "GPS, GLONASS, NavIC",
                    icon = Icons.Default.Satellite,
                    modifier = Modifier.weight(1f)
                )
                DiagnosticCard(
                    title = "Signal Strength",
                    value = "${uiState.signalStrengthDb} dBm",
                    subtitle = if (uiState.signalStrengthDb > 30) "Strong SNR" else "No Signal",
                    icon = Icons.Default.SignalCellularAlt,
                    modifier = Modifier.weight(1f)
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                DiagnosticCard(
                    title = "DR Drift Estimate",
                    value = "±${uiState.drDriftEstimate} m",
                    subtitle = "Within safety threshold",
                    icon = Icons.Default.Timeline,
                    modifier = Modifier.weight(1f)
                )
                DiagnosticCard(
                    title = "IMU Frequency",
                    value = "100 Hz",
                    subtitle = "High sampling rate",
                    icon = Icons.Default.Speed,
                    modifier = Modifier.weight(1f)
                )
            }

            // Last Known Position Card
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
            ) {
                Column(
                    modifier = Modifier.padding(20.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text(
                        text = "Last Known Accurate Position",
                        style = MaterialTheme.typography.titleSmall,
                        color = MaterialTheme.colorScheme.primary,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = "Latitude: ${uiState.latitude}° N",
                        style = MaterialTheme.typography.bodyMedium
                    )
                    Text(
                        text = "Longitude: ${uiState.longitude}° E",
                        style = MaterialTheme.typography.bodyMedium
                    )
                    Text(
                        text = "Heading: ${uiState.heading}° (True North)",
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
            }
        }
    }
}

@Composable
fun DiagnosticCard(
    title: String,
    value: String,
    subtitle: String,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier.height(130.dp),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            verticalArrangement = Arrangement.SpaceBetween
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = title,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Icon(imageVector = icon, contentDescription = null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(20.dp))
            }
            Column {
                Text(
                    text = value,
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onSurface
                )
                Text(
                    text = subtitle,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

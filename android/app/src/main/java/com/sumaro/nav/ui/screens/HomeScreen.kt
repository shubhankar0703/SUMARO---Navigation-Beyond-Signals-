package com.sumaro.nav.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.sumaro.nav.ui.viewmodel.AppLanguage
import com.sumaro.nav.ui.viewmodel.NavigationViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(
    viewModel: NavigationViewModel,
    onNavigateToMap: () -> Unit,
    onNavigateToGnss: () -> Unit,
    onNavigateToCalibration: () -> Unit
) {
    val uiState by viewModel.uiState.collectAsState()
    var showLangMenu by remember { mutableStateOf(false) }
    var showOfflineDialog by remember { mutableStateOf(false) }

    val startNavStr = when(uiState.currentLanguage) { AppLanguage.ENGLISH -> "Start Navigation"; AppLanguage.HINDI -> "नेविगेशन शुरू करें"; AppLanguage.MARATHI -> "नेव्हिगेशन सुरू करा" }
    val offlineMapsStr = when(uiState.currentLanguage) { AppLanguage.ENGLISH -> "Offline Maps"; AppLanguage.HINDI -> "ऑफ़लाइन मानचित्र"; AppLanguage.MARATHI -> "ऑफलाइन नकाशे" }
    val gnssStatusStr = when(uiState.currentLanguage) { AppLanguage.ENGLISH -> "GNSS Status"; AppLanguage.HINDI -> "जीएनएसएस स्थिति"; AppLanguage.MARATHI -> "जीएनएसएस स्थिती" }
    val calibrationStr = when(uiState.currentLanguage) { AppLanguage.ENGLISH -> "Vehicle Calibration"; AppLanguage.HINDI -> "वाहन अंशांकन"; AppLanguage.MARATHI -> "वाहन कॅलिब्रेशन" }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Default.Navigation,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.size(28.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = "SUMARO",
                            fontWeight = FontWeight.Bold,
                            fontSize = 22.sp,
                            letterSpacing = 1.5.sp
                        )
                    }
                },
                actions = {
                    Box {
                        TextButton(onClick = { showLangMenu = true }) {
                            Icon(Icons.Default.Language, contentDescription = null, modifier = Modifier.size(18.dp))
                            Spacer(modifier = Modifier.width(4.dp))
                            Text(uiState.currentLanguage.displayName, fontWeight = FontWeight.SemiBold)
                        }
                        DropdownMenu(
                            expanded = showLangMenu,
                            onDismissRequest = { showLangMenu = false }
                        ) {
                            AppLanguage.values().forEach { lang ->
                                DropdownMenuItem(
                                    text = { Text(lang.displayName) },
                                    onClick = {
                                        viewModel.setLanguage(lang)
                                        showLangMenu = false
                                    }
                                )
                            }
                        }
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface
                )
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(MaterialTheme.colorScheme.background)
                .padding(paddingValues)
                .padding(20.dp),
            verticalArrangement = Arrangement.SpaceBetween
        ) {
            // Header / Tagline
            Column {
                Text(
                    text = when (uiState.currentLanguage) {
                        AppLanguage.ENGLISH -> "Navigate Beyond Signals"
                        AppLanguage.HINDI -> "सिग्नल के पार नेविगेट करें"
                        AppLanguage.MARATHI -> "सिग्नलच्या पलीकडे नेव्हिगेट करा"
                    },
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.primary,
                    fontWeight = FontWeight.Medium
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = when (uiState.currentLanguage) {
                        AppLanguage.ENGLISH -> "Intelligent Dead Reckoning for tunnels, parking & urban canyons."
                        AppLanguage.HINDI -> "सुरंगों और घने शहरी क्षेत्रों के लिए बुद्धिमान डेड रेकनिंग।"
                        AppLanguage.MARATHI -> "बोगदे आणि दाट शहरी क्षेत्रांसाठी बुद्धिमान डेड रेकनिंग."
                    },
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            // 4 Large Pictorial Action Buttons
            Column(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    ActionCard(
                        title = startNavStr,
                        subtitle = "GPS + Dead Reckoning",
                        icon = Icons.Default.Navigation,
                        accentColor = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.weight(1f),
                        onClick = onNavigateToMap
                    )
                    ActionCard(
                        title = offlineMapsStr,
                        subtitle = "Downloaded (Pune)",
                        icon = Icons.Default.Map,
                        accentColor = MaterialTheme.colorScheme.secondary,
                        modifier = Modifier.weight(1f),
                        onClick = { showOfflineDialog = true }
                    )
                }
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    ActionCard(
                        title = gnssStatusStr,
                        subtitle = "${uiState.satelliteCount} Satellites",
                        icon = Icons.Default.Satellite,
                        accentColor = MaterialTheme.colorScheme.tertiary,
                        modifier = Modifier.weight(1f),
                        onClick = onNavigateToGnss
                    )
                    ActionCard(
                        title = calibrationStr,
                        subtitle = if (uiState.isCalibrationComplete) "Calibrated" else "Required",
                        icon = Icons.Default.Tune,
                        accentColor = if (uiState.isCalibrationComplete) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error,
                        modifier = Modifier.weight(1f),
                        onClick = onNavigateToCalibration
                    )
                }
            }

            // Footer info for riders
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
                shape = RoundedCornerShape(12.dp)
            ) {
                Row(
                    modifier = Modifier.padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(Icons.Default.Info, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                    Spacer(modifier = Modifier.width(12.dp))
                    Text(
                        text = "Two-Wheeler Mode active: Optimized for handlebar mounting & vibration filtering.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
    }

    if (showOfflineDialog) {
        AlertDialog(
            onDismissRequest = { showOfflineDialog = false },
            title = { Text("Offline Maps") },
            text = { Text("Offline map region (Pune & Mumbai) is fully downloaded and cached locally. Navigation will continue seamlessly offline.") },
            confirmButton = {
                TextButton(onClick = { showOfflineDialog = false }) {
                    Text("OK")
                }
            }
        )
    }
}

@Composable
fun ActionCard(
    title: String,
    subtitle: String,
    icon: ImageVector,
    accentColor: Color,
    modifier: Modifier = Modifier,
    onClick: () -> Unit
) {
    Card(
        modifier = modifier
            .height(130.dp)
            .clickable(onClick = onClick),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            verticalArrangement = Arrangement.SpaceBetween
        ) {
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .clip(RoundedCornerShape(10.dp))
                    .background(accentColor.copy(alpha = 0.15f)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                    tint = accentColor,
                    modifier = Modifier.size(24.dp)
                )
            }
            Column {
                Text(
                    text = title,
                    style = MaterialTheme.typography.titleMedium,
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

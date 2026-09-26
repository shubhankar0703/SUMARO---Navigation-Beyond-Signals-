package com.sumaro.nav.ui.screens

import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.sumaro.nav.ui.viewmodel.GnssState
import com.sumaro.nav.ui.viewmodel.NavTrackingMode
import com.sumaro.nav.ui.viewmodel.NavigationViewModel

@Composable
fun NavigationMapScreen(
    viewModel: NavigationViewModel,
    onBack: () -> Unit
) {
    val uiState by viewModel.uiState.collectAsState()

    // Pulsing animation for confidence ring when GNSS is lost
    val infiniteTransition = rememberInfiniteTransition(label = "confidence_pulse")
    val pulseScale by infiniteTransition.animateFloat(
        initialValue = 1f,
        targetValue = if (uiState.gnssState == GnssState.LOST) 1.5f else 1.0f,
        animationSpec = infiniteRepeatable(
            animation = tween(1000, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "pulse"
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
    ) {
        // Full screen map canvas representation (Road grid + route + vehicle marker + confidence ring)
        Canvas(modifier = Modifier.fillMaxSize()) {
            val width = size.width
            val height = size.height
            val centerX = width / 2f
            val centerY = height / 2f

            // Draw grid / map background
            val gridColor = Color.Gray.copy(alpha = 0.2f)
            val step = 100f
            var x = 0f
            while (x < width) {
                drawLine(gridColor, Offset(x, 0f), Offset(x, height), 1f)
                x += step
            }
            var y = 0f
            while (y < height) {
                drawLine(gridColor, Offset(0f, y), Offset(width, y), 1f)
                y += step
            }

            // Draw route line
            val routePath = Path().apply {
                moveTo(centerX - 100f, height)
                cubicTo(centerX - 100f, centerY + 200f, centerX + 150f, centerY + 100f, centerX, centerY)
                cubicTo(centerX - 150f, centerY - 100f, centerX + 100f, centerY - 300f, centerX + 100f, 0f)
            }
            drawPath(
                path = routePath,
                color = Color(0xFF0284C7),
                style = Stroke(width = 16f)
            )

            // Soft Confidence Ring (grows during GNSS loss)
            val baseRadius = uiState.confidenceRadius * 4f
            val ringRadius = baseRadius * pulseScale
            val ringColor = if (uiState.gnssState == GnssState.LOST) Color(0xFFD97706) else Color(0xFF0284C7)

            drawCircle(
                color = ringColor.copy(alpha = 0.25f),
                radius = ringRadius,
                center = Offset(centerX, centerY)
            )
            drawCircle(
                color = ringColor.copy(alpha = 0.5f),
                radius = ringRadius * 0.6f,
                center = Offset(centerX, centerY),
                style = Stroke(width = 3f)
            )
        }

        // Top Content: Status Banner & Mode Badge & Back Button
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Back Button
                IconButton(
                    onClick = onBack,
                    modifier = Modifier
                        .clip(CircleShape)
                        .background(MaterialTheme.colorScheme.surface)
                ) {
                    Icon(Icons.Default.ArrowBack, contentDescription = "Back", tint = MaterialTheme.colorScheme.onSurface)
                }

                // Mode Badge
                Surface(
                    shape = RoundedCornerShape(20.dp),
                    color = if (uiState.trackingMode == NavTrackingMode.GPS) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.secondary,
                    shadowElevation = 4.dp
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            imageVector = if (uiState.trackingMode == NavTrackingMode.GPS) Icons.Default.GpsFixed else Icons.Default.Explore,
                            contentDescription = null,
                            tint = Color.White,
                            modifier = Modifier.size(16.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = if (uiState.trackingMode == NavTrackingMode.GPS) "Mode: GPS" else "Mode: SUMARO DR",
                            color = Color.White,
                            fontWeight = FontWeight.Bold,
                            fontSize = 14.sp
                        )
                    }
                }
            }

            // Status Banner
            val bannerColor = when (uiState.gnssState) {
                GnssState.NORMAL -> Color(0xFF16A34A)
                GnssState.LOST -> Color(0xFFD97706)
                GnssState.RESTORED -> Color(0xFF2563EB)
            }
            val bannerText = when (uiState.gnssState) {
                GnssState.NORMAL -> "GNSS Active (${uiState.satelliteCount} Satellites)"
                GnssState.LOST -> "GNSS unavailable — continuing with SUMARO Dead Reckoning"
                GnssState.RESTORED -> "GNSS signal restored"
            }
            val bannerIcon = when (uiState.gnssState) {
                GnssState.NORMAL -> Icons.Default.CheckCircle
                GnssState.LOST -> Icons.Default.Warning
                GnssState.RESTORED -> Icons.Default.Sync
            }

            Surface(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                color = bannerColor,
                shadowElevation = 4.dp
            ) {
                Row(
                    modifier = Modifier.padding(14.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(imageVector = bannerIcon, contentDescription = null, tint = Color.White)
                    Spacer(modifier = Modifier.width(10.dp))
                    Text(
                        text = bannerText,
                        color = Color.White,
                        fontWeight = FontWeight.Bold,
                        fontSize = 13.sp,
                        modifier = Modifier.weight(1f)
                    )
                }
            }
        }

        // Bottom Card: Turn-by-Turn Instructions & Demo Simulators
        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Demo Simulation Controls (Great for hackathon judges)
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Button(
                    onClick = { viewModel.simulateGnssLoss() },
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFD97706)),
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(Icons.Default.CloudOff, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Simulate Tunnel (Loss)", fontSize = 12.sp)
                }
                Button(
                    onClick = { viewModel.simulateGnssRestore() },
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF16A34A)),
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(Icons.Default.CloudDone, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Simulate Exit (Restore)", fontSize = 12.sp)
                }
            }

            // Turn-by-Turn Card
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
            ) {
                Column(
                    modifier = Modifier.padding(20.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Box(
                            modifier = Modifier
                                .size(48.dp)
                                .clip(CircleShape)
                                .background(MaterialTheme.colorScheme.primary.copy(alpha = 0.15f)),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                imageVector = Icons.Default.TurnRight,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.size(28.dp)
                            )
                        }
                        Spacer(modifier = Modifier.width(16.dp))
                        Column {
                            Text(
                                text = uiState.currentInstruction,
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.Bold,
                                color = MaterialTheme.colorScheme.onSurface
                            )
                            Spacer(modifier = Modifier.height(2.dp))
                            Text(
                                text = uiState.nextInstruction,
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }

                    Divider(color = MaterialTheme.colorScheme.surfaceVariant)

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "ETA: 14 mins (4.2 km)",
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.SemiBold,
                            color = MaterialTheme.colorScheme.primary
                        )
                        Surface(
                            shape = RoundedCornerShape(8.dp),
                            color = MaterialTheme.colorScheme.surfaceVariant
                        ) {
                            Text(
                                text = "Drift: ±${uiState.drDriftEstimate}m",
                                modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }
            }
        }
    }
}

package com.mobileo.ui.screen

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.mobileo.service.ModelDownloadManager
import kotlin.math.roundToInt

/**
 * First-launch download gate screen.
 * Mirrors iOS ModelDownloadGateView → DownloadPermissionView → DownloadProgressView.
 */
@Composable
fun DownloadGateScreen(
    downloadManager: ModelDownloadManager,
    onDownloadComplete: () -> Unit
) {
    if (downloadManager.modelsReady) {
        onDownloadComplete()
        return
    }

    when (val state = downloadManager.state) {
        is ModelDownloadManager.State.Idle,
        is ModelDownloadManager.State.Paused -> {
            DownloadPermissionScreen(
                state = state,
                downloadManager = downloadManager
            )
        }
        is ModelDownloadManager.State.Downloading -> {
            DownloadProgressScreen(downloadManager = downloadManager)
        }
        is ModelDownloadManager.State.Completed -> {
            onDownloadComplete()
        }
        is ModelDownloadManager.State.Failed -> {
            DownloadErrorScreen(
                message = state.message,
                onRetry = { downloadManager.retry() }
            )
        }
    }
}

@Composable
private fun DownloadPermissionScreen(
    state: ModelDownloadManager.State,
    downloadManager: ModelDownloadManager
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text("Mobile-O", fontSize = 32.sp, fontWeight = FontWeight.Bold)
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            "Multimodal AI on your device",
            fontSize = 16.sp,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
        )
        Spacer(modifier = Modifier.height(48.dp))

        // Capabilities
        listOf(
            "Text Chat" to "Conversational AI",
            "Image Understanding" to "Analyze photos with AI",
            "Image Generation" to "Create images from text"
        ).forEach { (title, subtitle) ->
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .size(44.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(MaterialTheme.colorScheme.primaryContainer),
                    contentAlignment = Alignment.Center
                ) {
                    Text("✦", fontSize = 20.sp)
                }
                Spacer(modifier = Modifier.width(16.dp))
                Column {
                    Text(title, fontWeight = FontWeight.SemiBold)
                    Text(
                        subtitle,
                        fontSize = 13.sp,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(40.dp))

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Download Required", fontWeight = FontWeight.SemiBold)
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    "~3.8 GB of ONNX model files will be downloaded. Requires ~5 GB free space.",
                    fontSize = 13.sp,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                )
            }
        }

        Spacer(modifier = Modifier.height(24.dp))

        if (state is ModelDownloadManager.State.Paused) {
            Text(
                "Download paused — tap Resume to continue",
                fontSize = 13.sp,
                color = MaterialTheme.colorScheme.primary,
                textAlign = TextAlign.Center
            )
            Spacer(modifier = Modifier.height(12.dp))
        }

        Button(
            onClick = {
                if (state is ModelDownloadManager.State.Paused) {
                    downloadManager.resume()
                } else {
                    /* Caller triggers coroutine via LaunchedEffect observing state change */
                }
            },
            modifier = Modifier.fillMaxWidth().height(52.dp),
            shape = RoundedCornerShape(16.dp)
        ) {
            Text(
                if (state is ModelDownloadManager.State.Paused) "Resume Download" else "Download Models",
                fontSize = 16.sp,
                fontWeight = FontWeight.SemiBold
            )
        }
    }
}

@Composable
private fun DownloadProgressScreen(downloadManager: ModelDownloadManager) {
    val progress by animateFloatAsState(
        targetValue = downloadManager.overallProgress,
        label = "download_progress"
    )

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text("Downloading Models", fontSize = 22.sp, fontWeight = FontWeight.Bold)
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            "This may take a few minutes on Wi-Fi",
            fontSize = 14.sp,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
        )
        Spacer(modifier = Modifier.height(40.dp))

        LinearProgressIndicator(
            progress = { progress },
            modifier = Modifier.fillMaxWidth().height(8.dp).clip(RoundedCornerShape(4.dp))
        )
        Spacer(modifier = Modifier.height(12.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                "${(progress * 100).roundToInt()}%",
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.primary
            )
            if (downloadManager.etaSeconds > 0) {
                Text(
                    "ETA: ${formatEta(downloadManager.etaSeconds)}",
                    fontSize = 13.sp,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                )
            }
        }

        Spacer(modifier = Modifier.height(8.dp))
        Text(
            downloadManager.currentFileName.substringAfterLast("/"),
            fontSize = 12.sp,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f)
        )

        if (downloadManager.downloadSpeedBytesPerSec > 0) {
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                formatSpeed(downloadManager.downloadSpeedBytesPerSec),
                fontSize = 12.sp,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f)
            )
        }

        Spacer(modifier = Modifier.height(40.dp))

        OutlinedButton(
            onClick = { downloadManager.pause() },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Pause")
        }
    }
}

@Composable
private fun DownloadErrorScreen(message: String, onRetry: () -> Unit) {
    Column(
        modifier = Modifier.fillMaxSize().padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text("Download Failed", fontSize = 22.sp, fontWeight = FontWeight.Bold)
        Spacer(modifier = Modifier.height(12.dp))
        Text(
            message,
            textAlign = TextAlign.Center,
            color = MaterialTheme.colorScheme.error
        )
        Spacer(modifier = Modifier.height(32.dp))
        Button(onClick = onRetry) { Text("Retry") }
    }
}

private fun formatEta(seconds: Long): String {
    return when {
        seconds < 60 -> "${seconds}s"
        seconds < 3600 -> "${seconds / 60}m ${seconds % 60}s"
        else -> "${seconds / 3600}h ${(seconds % 3600) / 60}m"
    }
}

private fun formatSpeed(bytesPerSec: Long): String {
    return when {
        bytesPerSec < 1024 -> "${bytesPerSec} B/s"
        bytesPerSec < 1024 * 1024 -> "${bytesPerSec / 1024} KB/s"
        else -> "${"%.1f".format(bytesPerSec / (1024f * 1024f))} MB/s"
    }
}

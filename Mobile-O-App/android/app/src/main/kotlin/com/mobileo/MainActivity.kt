package com.mobileo

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.runtime.*
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.mobileo.service.ModelDownloadManager
import com.mobileo.ui.screen.CameraScreen
import com.mobileo.ui.screen.ConversationScreen
import com.mobileo.ui.screen.DownloadGateScreen
import com.mobileo.ui.screen.SettingsScreen
import com.mobileo.ui.theme.MobileOTheme
import com.mobileo.viewmodel.ChatViewModel
import com.mobileo.viewmodel.SettingsViewModel
import kotlinx.coroutines.launch

/**
 * Single-activity entry point.
 * Mirrors iOS MobileOApp.swift + ContentView.swift navigation logic.
 *
 * Navigation routes:
 *   "download"     — first-launch model download gate
 *   "conversation" — main chat + generation screen
 *   "settings"     — generation settings
 *   "camera"       — real-time camera understanding
 */
class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            MobileOTheme {
                MobileORoot()
            }
        }
    }
}

@Composable
fun MobileORoot() {
    val context = androidx.compose.ui.platform.LocalContext.current
    val scope = rememberCoroutineScope()

    // Shared ViewModels — survive navigation
    val chatViewModel: ChatViewModel = viewModel(factory = ChatViewModel.Factory(context))
    val settingsViewModel: SettingsViewModel = viewModel()

    // Download manager — lives at root scope
    val downloadManager = remember { ModelDownloadManager(context) }

    val navController = rememberNavController()

    // Determine start destination
    val startDest = if (downloadManager.modelsReady) "conversation" else "download"

    NavHost(
        navController = navController,
        startDestination = startDest
    ) {
        // --- Download gate (mirrors iOS ModelDownloadGateView) ---
        composable("download") {
            LaunchedEffect(Unit) {
                // Start download automatically if not paused or failed
                val state = downloadManager.state
                if (state == ModelDownloadManager.State.Idle) {
                    scope.launch { downloadManager.startDownload() }
                }
            }

            // Watch for completion
            LaunchedEffect(downloadManager.modelsReady) {
                if (downloadManager.modelsReady) {
                    chatViewModel.loadModels(downloadManager.modelsDirectory)
                    navController.navigate("conversation") {
                        popUpTo("download") { inclusive = true }
                    }
                }
            }

            DownloadGateScreen(
                downloadManager = downloadManager,
                onDownloadComplete = {
                    chatViewModel.loadModels(downloadManager.modelsDirectory)
                    navController.navigate("conversation") {
                        popUpTo("download") { inclusive = true }
                    }
                }
            )
        }

        // --- Main conversation screen (mirrors iOS ContentView) ---
        composable("conversation") {
            // Load models once when first entering this screen
            LaunchedEffect(Unit) {
                if (chatViewModel.modelsLoading && downloadManager.modelsReady) {
                    chatViewModel.loadModels(downloadManager.modelsDirectory)
                }
            }

            ConversationScreen(
                viewModel = chatViewModel,
                settingsViewModel = settingsViewModel,
                onNavigateToSettings = { navController.navigate("settings") },
                onNavigateToCamera = { navController.navigate("camera") }
            )
        }

        // --- Settings screen ---
        composable("settings") {
            SettingsScreen(
                settingsViewModel = settingsViewModel,
                onBack = { navController.popBackStack() }
            )
        }

        // --- Camera screen ---
        composable("camera") {
            CameraScreen(
                understandingModel = null, // injected via chatViewModel when ready
                onBack = { navController.popBackStack() }
            )
        }
    }
}

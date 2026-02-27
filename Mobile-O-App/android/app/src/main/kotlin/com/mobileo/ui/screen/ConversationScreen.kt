package com.mobileo.ui.screen

import android.graphics.Bitmap
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.mobileo.data.ChatMessage
import com.mobileo.data.Conversation
import com.mobileo.data.MessageContent
import com.mobileo.data.Role
import com.mobileo.viewmodel.ChatViewModel
import com.mobileo.viewmodel.SettingsViewModel
import kotlinx.coroutines.launch

/**
 * Main conversation screen.
 * Mirrors iOS ContentView + ConversationView + InputBar + MessageBubbleView.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ConversationScreen(
    viewModel: ChatViewModel,
    settingsViewModel: SettingsViewModel,
    onNavigateToSettings: () -> Unit,
    onNavigateToCamera: () -> Unit
) {
    val context = LocalContext.current
    var prompt by remember { mutableStateOf("") }
    var selectedBitmap by remember { mutableStateOf<Bitmap?>(null) }
    val listState = rememberLazyListState()
    val scope = rememberCoroutineScope()

    val messages = viewModel.conversation.messages

    // Auto-scroll to bottom on new messages
    LaunchedEffect(messages.size) {
        if (messages.isNotEmpty()) {
            listState.animateScrollToItem(messages.size - 1)
        }
    }

    // Image picker
    val imagePicker = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        uri?.let {
            val stream = context.contentResolver.openInputStream(uri)
            val bmp = android.graphics.BitmapFactory.decodeStream(stream)
            selectedBitmap = bmp
        }
    }

    // Auto-clear image when switching to generate mode
    LaunchedEffect(prompt) {
        val lower = prompt.lowercase().trim()
        if ((lower == "generate" || lower.startsWith("generate ")) && selectedBitmap != null) {
            selectedBitmap = null
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Mobile-O", fontWeight = FontWeight.SemiBold) },
                actions = {
                    IconButton(onClick = onNavigateToCamera) {
                        Icon(Icons.Default.CameraAlt, contentDescription = "Camera")
                    }
                    if (messages.isNotEmpty() && !viewModel.isGenerating) {
                        IconButton(onClick = { viewModel.conversation.clear() }) {
                            Icon(Icons.Default.Delete, contentDescription = "Clear")
                        }
                    }
                    IconButton(onClick = onNavigateToSettings) {
                        Icon(Icons.Default.Settings, contentDescription = "Settings")
                    }
                }
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            // Chat area
            Box(modifier = Modifier.weight(1f)) {
                if (messages.isEmpty()) {
                    WelcomeContent(onSuggestionClick = { prompt = it })
                } else {
                    LazyColumn(
                        state = listState,
                        contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                        modifier = Modifier.fillMaxSize()
                    ) {
                        items(messages, key = { it.id }) { message ->
                            MessageBubble(message = message)
                        }
                        // Progress indicator during generation
                        if (viewModel.isGenerating) {
                            item {
                                val prog = viewModel.generationProgress
                                if (prog != null) {
                                    GeneratingImageBubble(step = prog.first, total = prog.second)
                                } else {
                                    ThinkingBubble()
                                }
                            }
                        }
                    }
                }

                // Loading overlay while models load
                if (viewModel.modelsLoading) {
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.8f)),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            CircularProgressIndicator()
                            Spacer(modifier = Modifier.height(12.dp))
                            Text("Loading models…", color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f))
                        }
                    }
                }
            }

            // Input bar
            InputBar(
                prompt = prompt,
                onPromptChange = { prompt = it },
                selectedBitmap = selectedBitmap,
                onClearImage = { selectedBitmap = null },
                isGenerating = viewModel.isGenerating,
                onPickImage = {
                    imagePicker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
                },
                onSend = {
                    val current = prompt.trim()
                    val currentImage = selectedBitmap
                    prompt = ""
                    selectedBitmap = null
                    viewModel.sendMessage(
                        prompt = current,
                        selectedImage = currentImage,
                        numSteps = settingsViewModel.numSteps,
                        guidanceScale = settingsViewModel.guidanceScale,
                        enableCFG = settingsViewModel.enableCFG,
                        seed = settingsViewModel.seed
                    )
                    scope.launch {
                        if (messages.isNotEmpty()) listState.animateScrollToItem(messages.size)
                    }
                },
                onCancel = { viewModel.cancelGeneration() }
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Message Bubble — mirrors iOS MessageBubbleView.swift
// ---------------------------------------------------------------------------

@Composable
fun MessageBubble(message: ChatMessage) {
    val isUser = message.role == Role.USER
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start
    ) {
        if (!isUser) {
            Box(
                modifier = Modifier
                    .size(32.dp)
                    .clip(CircleShape)
                    .background(MaterialTheme.colorScheme.primary),
                contentAlignment = Alignment.Center
            ) {
                Text("M", color = MaterialTheme.colorScheme.onPrimary, fontWeight = FontWeight.Bold)
            }
            Spacer(modifier = Modifier.width(8.dp))
        }

        Column(horizontalAlignment = if (isUser) Alignment.End else Alignment.Start) {
            when (val content = message.content) {
                is MessageContent.Text -> {
                    Surface(
                        shape = RoundedCornerShape(
                            topStart = if (isUser) 18.dp else 4.dp,
                            topEnd = if (isUser) 4.dp else 18.dp,
                            bottomStart = 18.dp,
                            bottomEnd = 18.dp
                        ),
                        color = if (isUser) MaterialTheme.colorScheme.primary
                        else MaterialTheme.colorScheme.surfaceVariant
                    ) {
                        Text(
                            text = content.text,
                            modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp),
                            color = if (isUser) MaterialTheme.colorScheme.onPrimary
                            else MaterialTheme.colorScheme.onSurface
                        )
                    }
                }
                is MessageContent.Image -> {
                    Image(
                        bitmap = content.bitmap.asImageBitmap(),
                        contentDescription = "Generated image",
                        modifier = Modifier
                            .size(256.dp)
                            .clip(RoundedCornerShape(18.dp)),
                        contentScale = ContentScale.Crop
                    )
                }
                is MessageContent.TextWithImage -> {
                    Column(horizontalAlignment = Alignment.End) {
                        Image(
                            bitmap = content.bitmap.asImageBitmap(),
                            contentDescription = "Attached image",
                            modifier = Modifier
                                .size(160.dp)
                                .clip(RoundedCornerShape(12.dp)),
                            contentScale = ContentScale.Crop
                        )
                        if (content.text.isNotBlank()) {
                            Spacer(modifier = Modifier.height(4.dp))
                            Surface(
                                shape = RoundedCornerShape(4.dp, 18.dp, 18.dp, 18.dp),
                                color = MaterialTheme.colorScheme.primary
                            ) {
                                Text(
                                    content.text,
                                    modifier = Modifier.padding(14.dp, 10.dp),
                                    color = MaterialTheme.colorScheme.onPrimary
                                )
                            }
                        }
                    }
                }
            }

            // Performance metrics
            message.metrics?.let { m ->
                if (m.totalTime > 0) {
                    Text(
                        "${"%.1f".format(m.totalTime)}s",
                        fontSize = 11.sp,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f),
                        modifier = Modifier.padding(top = 2.dp)
                    )
                }
            }
        }
    }
}

@Composable
fun GeneratingImageBubble(step: Int, total: Int) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(
            modifier = Modifier
                .size(32.dp)
                .clip(CircleShape)
                .background(MaterialTheme.colorScheme.primary),
            contentAlignment = Alignment.Center
        ) {
            Text("M", color = MaterialTheme.colorScheme.onPrimary, fontWeight = FontWeight.Bold)
        }
        Spacer(modifier = Modifier.width(8.dp))
        Surface(
            shape = RoundedCornerShape(4.dp, 18.dp, 18.dp, 18.dp),
            color = MaterialTheme.colorScheme.surfaceVariant
        ) {
            Column(modifier = Modifier.padding(14.dp, 10.dp)) {
                Text("Generating image… ($step/$total)", fontSize = 14.sp)
                Spacer(modifier = Modifier.height(8.dp))
                LinearProgressIndicator(
                    progress = { if (total > 0) step.toFloat() / total else 0f },
                    modifier = Modifier.width(160.dp).height(4.dp).clip(RoundedCornerShape(2.dp))
                )
            }
        }
    }
}

@Composable
fun ThinkingBubble() {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(
            modifier = Modifier
                .size(32.dp)
                .clip(CircleShape)
                .background(MaterialTheme.colorScheme.primary),
            contentAlignment = Alignment.Center
        ) {
            Text("M", color = MaterialTheme.colorScheme.onPrimary, fontWeight = FontWeight.Bold)
        }
        Spacer(modifier = Modifier.width(8.dp))
        Surface(
            shape = RoundedCornerShape(4.dp, 18.dp, 18.dp, 18.dp),
            color = MaterialTheme.colorScheme.surfaceVariant
        ) {
            Row(
                modifier = Modifier.padding(14.dp, 10.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                Spacer(modifier = Modifier.width(8.dp))
                Text("Thinking…", fontSize = 14.sp)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Input Bar — mirrors iOS InputBar.swift
// ---------------------------------------------------------------------------

@Composable
fun InputBar(
    prompt: String,
    onPromptChange: (String) -> Unit,
    selectedBitmap: Bitmap?,
    onClearImage: () -> Unit,
    isGenerating: Boolean,
    onPickImage: () -> Unit,
    onSend: () -> Unit,
    onCancel: () -> Unit
) {
    Surface(
        shadowElevation = 8.dp,
        color = MaterialTheme.colorScheme.surface
    ) {
        Column(modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)) {
            // Image preview
            selectedBitmap?.let { bmp ->
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Image(
                        bitmap = bmp.asImageBitmap(),
                        contentDescription = "Selected image",
                        modifier = Modifier
                            .size(60.dp)
                            .clip(RoundedCornerShape(8.dp)),
                        contentScale = ContentScale.Crop
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    IconButton(onClick = onClearImage) {
                        Icon(Icons.Default.Close, contentDescription = "Remove image")
                    }
                }
                Spacer(modifier = Modifier.height(4.dp))
            }

            Row(
                verticalAlignment = Alignment.Bottom,
                modifier = Modifier.fillMaxWidth()
            ) {
                // Image attach button
                if (!isGenerating) {
                    IconButton(onClick = onPickImage) {
                        Icon(
                            Icons.Default.Image,
                            contentDescription = "Attach image",
                            tint = MaterialTheme.colorScheme.primary
                        )
                    }
                }

                // Text field
                OutlinedTextField(
                    value = prompt,
                    onValueChange = onPromptChange,
                    modifier = Modifier.weight(1f),
                    placeholder = { Text("Message or \"generate …\"") },
                    maxLines = 5,
                    shape = RoundedCornerShape(24.dp),
                    keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                    keyboardActions = KeyboardActions(onSend = { if (!isGenerating) onSend() }),
                    enabled = !isGenerating
                )

                Spacer(modifier = Modifier.width(8.dp))

                // Send / Cancel button
                if (isGenerating) {
                    IconButton(
                        onClick = onCancel,
                        modifier = Modifier
                            .size(48.dp)
                            .clip(CircleShape)
                            .background(MaterialTheme.colorScheme.errorContainer)
                    ) {
                        Icon(Icons.Default.Stop, contentDescription = "Cancel", tint = MaterialTheme.colorScheme.error)
                    }
                } else {
                    IconButton(
                        onClick = onSend,
                        enabled = prompt.isNotBlank() || selectedBitmap != null,
                        modifier = Modifier
                            .size(48.dp)
                            .clip(CircleShape)
                            .background(
                                if (prompt.isNotBlank() || selectedBitmap != null)
                                    MaterialTheme.colorScheme.primary
                                else
                                    MaterialTheme.colorScheme.surfaceVariant
                            )
                    ) {
                        Icon(
                            Icons.Default.Send,
                            contentDescription = "Send",
                            tint = if (prompt.isNotBlank() || selectedBitmap != null)
                                MaterialTheme.colorScheme.onPrimary
                            else
                                MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f)
                        )
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Welcome screen — mirrors iOS WelcomeView.swift
// ---------------------------------------------------------------------------

@Composable
fun WelcomeContent(onSuggestionClick: (String) -> Unit) {
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
            "Your on-device multimodal AI",
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
        )
        Spacer(modifier = Modifier.height(40.dp))

        val suggestions = listOf(
            "generate A serene mountain landscape at sunset",
            "generate A cute robot exploring a forest",
            "What can you help me with?",
            "Tell me about quantum computing"
        )
        suggestions.forEach { suggestion ->
            Surface(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 4.dp)
                    .clickable { onSuggestionClick(suggestion) },
                shape = RoundedCornerShape(12.dp),
                color = MaterialTheme.colorScheme.surfaceVariant
            ) {
                Text(
                    suggestion,
                    modifier = Modifier.padding(16.dp),
                    fontSize = 14.sp
                )
            }
        }
    }
}

package com.mobileo.ui.screen

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.mobileo.viewmodel.SettingsViewModel

/**
 * Settings screen — mirrors iOS SettingsView.swift.
 * Controls: num steps, guidance scale, CFG toggle, seed, scheduler type.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    settingsViewModel: SettingsViewModel,
    onBack: () -> Unit
) {
    var seedText by remember { mutableStateOf(settingsViewModel.seedDisplayString()) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Generation Settings", fontWeight = FontWeight.SemiBold) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back")
                    }
                }
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(horizontal = 16.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Spacer(modifier = Modifier.height(8.dp))

            // Inference steps
            SettingsSliderCard(
                title = "Inference Steps",
                value = settingsViewModel.numSteps.toFloat(),
                valueRange = 5f..50f,
                steps = 44,
                displayValue = settingsViewModel.numSteps.toString(),
                onValueChange = { settingsViewModel.numSteps = it.toInt() }
            )

            // Guidance scale
            SettingsSliderCard(
                title = "Guidance Scale",
                value = settingsViewModel.guidanceScale,
                valueRange = 1.0f..10.0f,
                steps = 0,
                displayValue = "%.1f".format(settingsViewModel.guidanceScale),
                onValueChange = { settingsViewModel.guidanceScale = it }
            )

            // CFG toggle
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Column {
                        Text("Classifier-Free Guidance", fontWeight = FontWeight.Medium)
                        Text(
                            "Improves prompt adherence",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                        )
                    }
                    Switch(
                        checked = settingsViewModel.enableCFG,
                        onCheckedChange = { settingsViewModel.enableCFG = it }
                    )
                }
            }

            // Seed
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text("Seed", fontWeight = FontWeight.Medium)
                    Text(
                        "Leave empty for random",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    OutlinedTextField(
                        value = seedText,
                        onValueChange = {
                            seedText = it
                            settingsViewModel.setSeedFromString(it)
                        },
                        modifier = Modifier.fillMaxWidth(),
                        placeholder = { Text("Random") },
                        singleLine = true
                    )
                }
            }

            // Scheduler type
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text("Scheduler", fontWeight = FontWeight.Medium)
                    Spacer(modifier = Modifier.height(8.dp))
                    SettingsViewModel.SchedulerType.entries.forEach { type ->
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            RadioButton(
                                selected = settingsViewModel.schedulerType == type,
                                onClick = { settingsViewModel.schedulerType = type }
                            )
                            Column {
                                Text(type.label, fontWeight = FontWeight.Medium)
                                Text(
                                    type.description,
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                                )
                            }
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(32.dp))
        }
    }
}

@Composable
private fun SettingsSliderCard(
    title: String,
    value: Float,
    valueRange: ClosedFloatingPointRange<Float>,
    steps: Int,
    displayValue: String,
    onValueChange: (Float) -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(title, fontWeight = FontWeight.Medium)
                Text(
                    displayValue,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.primary
                )
            }
            Slider(
                value = value,
                onValueChange = onValueChange,
                valueRange = valueRange,
                steps = steps,
                modifier = Modifier.fillMaxWidth()
            )
        }
    }
}

package com.mobileo.viewmodel

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel

/**
 * Generation settings persisted across sessions.
 * Mirrors iOS SettingsViewModel.swift.
 */
class SettingsViewModel : ViewModel() {

    // Mirrors iOS SettingsViewModel defaults exactly
    var numSteps: Int by mutableStateOf(15)
    var guidanceScale: Float by mutableStateOf(1.3f)
    var enableCFG: Boolean by mutableStateOf(true)
    var seed: Long? by mutableStateOf(null)

    // Scheduler type (DPM-Solver++ recommended)
    var schedulerType: SchedulerType by mutableStateOf(SchedulerType.DPM_SOLVER)

    enum class SchedulerType(val label: String, val description: String) {
        DPM_SOLVER("DPM-Solver++", "Fast and high quality, 12–20 steps"),
        CUSTOM("Custom Scheduler", "Adams-Bashforth-3, 8–12 steps")
    }

    /** Parse seed string from UI input. Null means random. */
    fun setSeedFromString(value: String) {
        seed = value.trim().toLongOrNull()
    }

    fun seedDisplayString(): String = seed?.toString() ?: ""
}

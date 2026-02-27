package com.mobileo.ml

import android.content.Context
import android.util.Log
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * DPM-Solver++ multistep scheduler for flow-matching diffusion.
 *
 * Direct port of iOS DPMSolverScheduler.swift.
 * Supports first-order (Euler) and second-order multistep steps.
 * All arithmetic operates on FloatArray — no GPU required.
 *
 * Config is read from res/raw/scheduler_config.json (same file as iOS).
 */
class DPMSolverScheduler(val config: Config) {

    @Serializable
    data class Config(
        val num_train_timesteps: Int = 1000,
        val beta_start: Float = 0.0001f,
        val beta_end: Float = 0.02f,
        val beta_schedule: String = "linear",
        val solver_order: Int = 2,
        val prediction_type: String = "flow_prediction",
        val algorithm_type: String = "dpmsolver++",
        val use_flow_sigmas: Boolean = true,
        val flow_shift: Float = 3.0f,
        val lower_order_final: Boolean = true,
        val apply_flow_shift: Boolean = false,
        val lambda_min_clipped: Float = -1000000.0f,
        val timestep_spacing: String = "uniform_tau"
    )

    companion object {
        private const val TAG = "DPMSolverScheduler"

        fun fromJson(jsonString: String): DPMSolverScheduler {
            val json = Json { ignoreUnknownKeys = true }
            val config = json.decodeFromString<Config>(jsonString)
            return DPMSolverScheduler(config)
        }

        fun fromContext(context: Context): DPMSolverScheduler {
            val raw = context.resources.openRawResource(
                context.resources.getIdentifier("scheduler_config", "raw", context.packageName)
            ).bufferedReader().readText()
            return fromJson(raw)
        }
    }

    // Set by setTimesteps()
    var timesteps: FloatArray = FloatArray(0)
        private set
    private var sigmas: FloatArray = FloatArray(0)

    // Precomputed per-step coefficients (mirrors Swift precomputed arrays)
    private var precomputedDt: FloatArray = FloatArray(0)
    private var precomputedHalfR: FloatArray = FloatArray(0)

    // Step state
    private var stepIndex: Int = 0
    private val modelOutputHistory = ArrayDeque<FloatArray>() // rolling window of solver_order outputs

    // Reusable buffers — avoids allocation in hot denoising loop
    private var tempBuffer: FloatArray = FloatArray(0)
    private var resultBuffer: FloatArray = FloatArray(0)

    /**
     * Configure timesteps and precompute step coefficients.
     * Mirrors iOS DPMSolverScheduler.setTimesteps(numInferenceSteps:).
     */
    fun setTimesteps(numInferenceSteps: Int) {
        val numTrainTimesteps = config.num_train_timesteps

        // Linear spacing from T-1 down to 0 — mirrors Swift implementation
        timesteps = FloatArray(numInferenceSteps) { i ->
            val progress = i.toFloat() / (numInferenceSteps - 1).toFloat()
            (numTrainTimesteps - 1).toFloat() * (1.0f - progress)
        }

        // Compute flow sigmas: σ_i = t_i / num_train_timesteps
        val rawSigmas = FloatArray(numInferenceSteps) { i -> timesteps[i] / numTrainTimesteps.toFloat() }

        // Apply flow_shift if configured (config.apply_flow_shift = false for this model)
        sigmas = if (config.apply_flow_shift && config.flow_shift != 1.0f) {
            val shift = config.flow_shift
            FloatArray(numInferenceSteps) { i ->
                val s = rawSigmas[i]
                shift * s / (1.0f + (shift - 1.0f) * s)
            }
        } else {
            rawSigmas
        }
        // Append terminal sigma = 0.0
        sigmas = sigmas + floatArrayOf(0.0f)

        precomputeCoefficients(numInferenceSteps)

        stepIndex = 0
        modelOutputHistory.clear()

        Log.d(TAG, "setTimesteps($numInferenceSteps): t[0]=${timesteps[0]}, t[-1]=${timesteps.last()}, σ[0]=${sigmas[0]}")
    }

    /**
     * Precompute dt and halfR for each step.
     * Mirrors iOS precomputeStepCoefficients().
     */
    private fun precomputeCoefficients(numSteps: Int) {
        precomputedDt = FloatArray(numSteps) { i ->
            val sigmaCurr = sigmas[i]
            val sigmaNext = if (i < sigmas.size - 1) sigmas[i + 1] else 0.0f
            sigmaNext - sigmaCurr   // dt = σ_{i+1} - σ_i (negative, flowing to zero)
        }

        precomputedHalfR = FloatArray(numSteps) { i ->
            if (i == 0) {
                0.0f
            } else {
                val sigmaPrev = sigmas[i - 1]
                val sigmaCurr = sigmas[i]
                val sigmaNext = if (i < sigmas.size - 1) sigmas[i + 1] else 0.0f
                val h = sigmaCurr - sigmaPrev
                val hNext = sigmaNext - sigmaCurr
                val r = hNext / max(abs(h), 1e-8f)
                val rClamped = max(-5.0f, min(5.0f, r))
                0.5f * rClamped
            }
        }
    }

    /**
     * Perform one denoising step.
     * Mirrors iOS DPMSolverScheduler.step(modelOutput:timestep:sample:).
     *
     * @param noisePred  model's noise prediction [1,32,16,16] flattened to FloatArray
     * @param timestep   current timestep value (unused — uses internal stepIndex)
     * @param latents    current latent FloatArray
     * @return updated latent FloatArray (same size as [latents])
     */
    fun step(noisePred: FloatArray, timestep: Float, latents: FloatArray): FloatArray {
        val count = latents.size
        ensureBuffers(count)

        // Store in rolling history
        modelOutputHistory.addLast(noisePred.copyOf())
        if (modelOutputHistory.size > config.solver_order) {
            modelOutputHistory.removeFirst()
        }

        val tIndex = stepIndex
        val order = min(config.solver_order, modelOutputHistory.size)
        val useLowerOrderFinal = config.lower_order_final && (tIndex >= timesteps.size - 2)
        val actualOrder = if (useLowerOrderFinal) 1 else order

        val result = if (actualOrder == 1 || modelOutputHistory.size < 2) {
            firstOrderStep(latents, noisePred, tIndex, count)
        } else {
            secondOrderStep(latents, tIndex, count)
        }

        stepIndex++
        return result
    }

    /**
     * First-order Euler step: x_{t+1} = x_t + dt * v_t
     * Mirrors iOS firstOrderStep().
     */
    private fun firstOrderStep(sample: FloatArray, modelOutput: FloatArray, tIndex: Int, count: Int): FloatArray {
        val dt = precomputedDt[tIndex]
        val result = resultBuffer
        for (i in 0 until count) {
            result[i] = sample[i] + dt * modelOutput[i]
        }
        return result
    }

    /**
     * Second-order multistep: x_{t+1} = x_t + dt * (v_curr + 0.5*r*(v_curr - v_prev))
     * Mirrors iOS secondOrderStep().
     */
    private fun secondOrderStep(sample: FloatArray, tIndex: Int, count: Int): FloatArray {
        val curr = modelOutputHistory.last()
        val prev = modelOutputHistory[modelOutputHistory.size - 2]
        val dt = precomputedDt[tIndex]
        val halfR = precomputedHalfR[tIndex]
        val result = resultBuffer

        for (i in 0 until count) {
            val diff = curr[i] - prev[i]           // v_curr - v_prev
            val d = curr[i] + halfR * diff          // v_curr + 0.5*r*(v_curr - v_prev)
            result[i] = sample[i] + dt * d
        }
        return result
    }

    private fun ensureBuffers(count: Int) {
        if (tempBuffer.size != count) tempBuffer = FloatArray(count)
        if (resultBuffer.size != count) resultBuffer = FloatArray(count)
    }

    /** Reset state for a new generation run. */
    fun reset() {
        stepIndex = 0
        modelOutputHistory.clear()
    }
}

// FloatArray concatenation helper
private operator fun FloatArray.plus(other: FloatArray): FloatArray {
    val result = FloatArray(size + other.size)
    copyInto(result)
    other.copyInto(result, size)
    return result
}

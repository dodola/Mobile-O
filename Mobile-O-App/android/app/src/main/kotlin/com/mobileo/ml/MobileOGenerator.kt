package com.mobileo.ml

import ai.onnxruntime.OnnxTensor
import android.content.Context
import android.graphics.Bitmap
import android.os.PowerManager
import android.util.Log
import com.mobileo.util.VaePostProcessor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.PI
import kotlin.math.sqrt

/**
 * SANA text-to-image generation pipeline using ONNX Runtime.
 *
 * Direct port of iOS MobileOGenerator.swift.
 *
 * Pipeline:
 *   text → Tokenizer → LLM (ONNX) → hidden states
 *         → MobileConditioningConnector (ONNX) → encoder_hidden_states
 *         → DPM-Solver++ denoising loop (ONNX DiT) → latents
 *         → VAE decode (ONNX) → NDK normalize → Bitmap
 *
 * Thread safety: all inference runs on Dispatchers.Default (background).
 * UI updates should be collected on the main thread via StateFlow.
 */
class MobileOGenerator(
    private val context: Context,
    private val sessions: OnnxSessionManager,
    private val tokenizer: Tokenizer,
    private val scheduler: DPMSolverScheduler
) {
    companion object {
        private const val TAG = "MobileOGenerator"

        // Mirrors iOS ConditioningConnectorConfiguration + diffusion constants
        const val VAE_SCALING_FACTOR = 0.41407f
        const val LATENT_CHANNELS = 32
        const val LATENT_SIZE = 16          // 512px / 32 = 16
        const val LATENT_ELEMENTS = LATENT_CHANNELS * LATENT_SIZE * LATENT_SIZE  // 8192
        const val IMAGE_SIZE = 512
    }

    private val connector = MobileConditioningConnector(sessions)

    // Token cache — mirrors iOS tokenCache dict
    private val tokenCache = mutableMapOf<String, IntArray>()

    // Cached unconditional embeddings (CFG), keyed by seqLen
    private var cachedUncondEmbeddings: Pair<FloatArray, FloatArray>? = null
    private var cachedUncondSeqLen: Int = -1

    // Timing info — mirrors iOS TimingInfo
    data class TimingInfo(
        val tokenizationMs: Long,
        val llmMs: Long,
        val connectorMs: Long,
        val diffusionMs: Long,
        val vaeMs: Long,
        val totalMs: Long
    )

    data class GenerationParams(
        val prompt: String,
        val numSteps: Int = 15,
        val guidanceScale: Float = 1.3f,
        val enableCFG: Boolean = true,
        val seed: Long? = null,
        val onProgress: ((step: Int, total: Int) -> Unit)? = null
    )

    /**
     * Generate an image from a text prompt.
     * Mirrors iOS MobileOGenerator.generate(_:).
     *
     * Must be called from a coroutine; runs on Dispatchers.Default.
     * Respects coroutine cancellation at each denoising step.
     */
    suspend fun generate(params: GenerationParams): Pair<Bitmap, TimingInfo> =
        withContext(Dispatchers.Default) {
            val wakeLock = acquireWakeLock()
            try {
                performPipeline(params)
            } finally {
                wakeLock?.release()
            }
        }

    private suspend fun performPipeline(params: GenerationParams): Pair<Bitmap, TimingInfo> {
        val totalStart = System.currentTimeMillis()

        // --- 1. Tokenize ---
        val tokStart = System.currentTimeMillis()
        val tokens = encodeGenerationPrompt(params.prompt)
        val tokMs = System.currentTimeMillis() - tokStart

        // --- 2. LLM forward pass → all hidden states ---
        val llmStart = System.currentTimeMillis()
        val (hiddenStates, seqLen) = runLlm(tokens)
        val llmMs = System.currentTimeMillis() - llmStart

        // --- 3. Connector → encoder_hidden_states (conditional) ---
        val connStart = System.currentTimeMillis()
        val (condHidden, condMask) = connector.forward(hiddenStates, seqLen)

        // Build CFG unconditional embeddings (zero → connector), cached by seqLen
        val (uncondHidden, uncondMask) = if (params.enableCFG) {
            getCachedUncondEmbeddings(seqLen)
        } else {
            condHidden to condMask
        }
        val connMs = System.currentTimeMillis() - connStart

        // --- 4. Initialize random latents [1, 32, 16, 16] ---
        val latents = randomNormalLatents(params.seed)

        // --- 5. Denoising loop ---
        val diffStart = System.currentTimeMillis()
        scheduler.setTimesteps(params.numSteps)
        var currentLatents = latents

        for ((idx, t) in scheduler.timesteps.withIndex()) {
            // Respect coroutine cancellation
            if (!withContext(Dispatchers.Default) { isActive }) {
                throw kotlinx.coroutines.CancellationException("Generation cancelled")
            }

            params.onProgress?.invoke(idx + 1, params.numSteps)

            // Conditional noise prediction
            val condNoise = runDiT(currentLatents, t, condHidden, condMask)

            // CFG: noise_pred = uncond + scale * (cond - uncond)
            val noisePred = if (params.enableCFG) {
                val uncondNoise = runDiT(currentLatents, t, uncondHidden, uncondMask)
                applyCFG(uncondNoise, condNoise, params.guidanceScale)
            } else {
                condNoise
            }

            // Scheduler step
            currentLatents = scheduler.step(noisePred, t, currentLatents)
        }
        val diffMs = System.currentTimeMillis() - diffStart

        // --- 6. VAE decode: latents → image ---
        val vaeStart = System.currentTimeMillis()
        val bitmap = decodeLatents(currentLatents)
        val vaeMs = System.currentTimeMillis() - vaeStart

        val totalMs = System.currentTimeMillis() - totalStart
        val timing = TimingInfo(tokMs, llmMs, connMs, diffMs, vaeMs, totalMs)

        Log.i(TAG, "Generation complete: total=${totalMs}ms (tok=${tokMs}, llm=${llmMs}, conn=${connMs}, diff=${diffMs}, vae=${vaeMs})")
        return Pair(bitmap, timing)
    }

    // ---------------------------------------------------------------------------
    // Text encoding
    // ---------------------------------------------------------------------------

    /**
     * Tokenize generation prompt with ChatML template.
     * Mirrors iOS encodeTextInternal() with token cache.
     */
    private fun encodeGenerationPrompt(prompt: String): IntArray {
        val cacheKey = prompt
        tokenCache[cacheKey]?.let { return it }
        val tokens = tokenizer.encodeGenerationPrompt(prompt)
        tokenCache[cacheKey] = tokens
        return tokens
    }

    // ---------------------------------------------------------------------------
    // LLM inference
    // ---------------------------------------------------------------------------

    /**
     * Run llm.onnx and extract all hidden states.
     * Mirrors iOS FastVLM.getAllHiddenStates(tokens:).
     *
     * Returns (List<FloatArray [seqLen * hiddenDim]>, seqLen)
     * The list has numLayers+1 entries (embedding layer + N transformer layers).
     */
    private fun runLlm(tokens: IntArray): Pair<List<FloatArray>, Int> {
        val session = sessions.llmSession ?: error("LLM ONNX session not loaded")
        val seqLen = tokens.size

        val inputIdsTensor = sessions.makeInputIdsTensor(tokens)
        val maskTensor = sessions.makeAttentionMaskTensor(seqLen)

        val inputs = mapOf(
            "input_ids" to inputIdsTensor,
            "attention_mask" to maskTensor
        )

        val results = session.run(inputs)

        // Extract all hidden_state_N outputs
        // export_onnx.py names them: logits, hidden_state_0, hidden_state_1, ..., hidden_state_N
        val hiddenStates = mutableListOf<FloatArray>()
        var i = 1  // skip logits at index 0
        while (true) {
            val key = "hidden_state_${i - 1}"
            val tensor = try {
                results.get(key).orElse(null) as? OnnxTensor ?: break
            } catch (e: Exception) { break }

            val buf = tensor.floatBuffer
            val arr = FloatArray(buf.remaining())
            buf.get(arr)
            // Shape is [1, seqLen, hiddenDim] — store as [seqLen * hiddenDim]
            hiddenStates.add(arr)
            i++
        }

        inputIdsTensor.close()
        maskTensor.close()
        results.close()

        Log.d(TAG, "LLM: seqLen=$seqLen, numHiddenStates=${hiddenStates.size}")
        return Pair(hiddenStates, seqLen)
    }

    // ---------------------------------------------------------------------------
    // DiT inference
    // ---------------------------------------------------------------------------

    /**
     * Run one DiT denoising step.
     * Mirrors iOS transformer.predict(latent:timestep:encoderHiddenStates:encoderAttentionMask:).
     *
     * @param latents       FloatArray [32*16*16 = 8192]
     * @param timestep      current timestep float value
     * @param condHidden    FloatArray [1 * effectiveSeqLen * 2304]
     * @param condMask      FloatArray [1 * effectiveSeqLen]
     * @return noise_pred FloatArray [8192]
     */
    private fun runDiT(
        latents: FloatArray,
        timestep: Float,
        condHidden: FloatArray,
        condMask: FloatArray
    ): FloatArray {
        val session = sessions.transformerSession ?: error("DiT ONNX session not loaded")

        val effectiveSeqLen = condMask.size

        val latentTensor = sessions.makeFloatTensor(
            latents,
            longArrayOf(1L, LATENT_CHANNELS.toLong(), LATENT_SIZE.toLong(), LATENT_SIZE.toLong())
        )
        val tTensor = sessions.makeTimestepTensor(timestep)
        val hiddenTensor = sessions.makeFloatTensor(
            condHidden,
            longArrayOf(1L, effectiveSeqLen.toLong(), MobileConditioningConnector.OUTPUT_DIM.toLong())
        )
        val maskTensor = sessions.makeFloatTensor(
            condMask,
            longArrayOf(1L, effectiveSeqLen.toLong())
        )

        val inputs = mapOf(
            "latent" to latentTensor,
            "timestep" to tTensor,
            "encoder_hidden_states" to hiddenTensor,
            "encoder_attention_mask" to maskTensor
        )

        val results = session.run(inputs)
        val noisePredTensor = results[0].value as? OnnxTensor
            ?: (results.get("noise_pred").orElse(null) as? OnnxTensor)
            ?: error("DiT output 'noise_pred' not found")

        val buf = noisePredTensor.floatBuffer
        val noisePred = FloatArray(buf.remaining())
        buf.get(noisePred)

        latentTensor.close(); tTensor.close(); hiddenTensor.close(); maskTensor.close()
        results.close()

        return noisePred
    }

    // ---------------------------------------------------------------------------
    // VAE decode
    // ---------------------------------------------------------------------------

    /**
     * Scale latents and decode via VAE to a Bitmap.
     * Mirrors iOS decodeLatents() → normalizeVAEOutput() → Metal floatToRGBA.
     */
    private fun decodeLatents(latents: FloatArray): Bitmap {
        val session = sessions.vaeSession ?: error("VAE ONNX session not loaded")

        // Scale: latents * (1 / vaeScalingFactor) — mirrors iOS scaleAndConvertToFloat16
        val scaled = FloatArray(latents.size) { i -> latents[i] / VAE_SCALING_FACTOR }

        val latentTensor = sessions.makeFloatTensor(
            scaled,
            longArrayOf(1L, LATENT_CHANNELS.toLong(), LATENT_SIZE.toLong(), LATENT_SIZE.toLong())
        )

        val results = session.run(mapOf("latent" to latentTensor))
        val imageTensor = results[0].value as? OnnxTensor
            ?: (results.get("image").orElse(null) as? OnnxTensor)
            ?: error("VAE output 'image' not found")

        val buf = imageTensor.floatBuffer
        val imageData = FloatArray(buf.remaining())
        buf.get(imageData)
        // imageData shape [1, 3, 512, 512] — drop batch dim → [3*512*512]
        val chw = if (imageData.size > 3 * IMAGE_SIZE * IMAGE_SIZE) {
            imageData.copyOfRange(0, 3 * IMAGE_SIZE * IMAGE_SIZE)
        } else {
            imageData
        }

        latentTensor.close()
        results.close()

        // NDK: normalize [-1,1]→[0,1] + CHW→RGBA Bitmap
        return VaePostProcessor.floatCHWtoBitmap(chw, IMAGE_SIZE, IMAGE_SIZE)
    }

    // ---------------------------------------------------------------------------
    // CFG
    // ---------------------------------------------------------------------------

    /**
     * Apply classifier-free guidance.
     * Mirrors iOS applyCFG(unconditional:conditional:guidanceScale:).
     * noise_pred = uncond + scale * (cond - uncond)
     */
    private fun applyCFG(uncond: FloatArray, cond: FloatArray, scale: Float): FloatArray {
        val result = FloatArray(cond.size)
        for (i in cond.indices) {
            result[i] = uncond[i] + scale * (cond[i] - uncond[i])
        }
        return result
    }

    private fun getCachedUncondEmbeddings(seqLen: Int): Pair<FloatArray, FloatArray> {
        if (cachedUncondSeqLen == seqLen && cachedUncondEmbeddings != null) {
            return cachedUncondEmbeddings!!
        }
        val result = connector.buildZeroEmbeddings(seqLen)
        cachedUncondEmbeddings = result
        cachedUncondSeqLen = seqLen
        return result
    }

    // ---------------------------------------------------------------------------
    // Random latent initialization
    // ---------------------------------------------------------------------------

    /**
     * Initialize latents from normal distribution using Box-Muller transform.
     * Mirrors iOS fillRandomNormal() with optional seeded RNG (xorshift64).
     */
    private fun randomNormalLatents(seed: Long?): FloatArray {
        val count = LATENT_ELEMENTS
        val latents = FloatArray(count)
        val rng = if (seed != null) SeededRandom(seed) else null

        var i = 0
        while (i < count) {
            val u1 = if (rng != null) rng.nextFloat() else Math.random().toFloat()
            val u2 = if (rng != null) rng.nextFloat() else Math.random().toFloat()
            val mag = sqrt(-2.0f * ln(u1.coerceAtLeast(1e-10f)))
            val z0 = mag * cos(2.0f * PI.toFloat() * u2)
            val z1 = mag * kotlin.math.sin(2.0f * PI.toFloat() * u2)
            latents[i] = z0
            if (i + 1 < count) latents[i + 1] = z1
            i += 2
        }
        return latents
    }

    // ---------------------------------------------------------------------------
    // Wake lock (mirrors iOS ProcessInfo.beginActivity idleSystemSleepDisabled)
    // ---------------------------------------------------------------------------

    private fun acquireWakeLock(): PowerManager.WakeLock? {
        return try {
            val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
            pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "MobileO:Generation").apply {
                acquire(10 * 60 * 1000L)  // max 10 minutes
            }
        } catch (e: Exception) {
            Log.w(TAG, "Could not acquire wake lock: ${e.message}")
            null
        }
    }

    fun clearCache() {
        tokenCache.clear()
        cachedUncondEmbeddings = null
        cachedUncondSeqLen = -1
        scheduler.reset()
    }
}

// ---------------------------------------------------------------------------
// Seeded RNG — mirrors iOS RandomNumberGeneratorWithSeed (xorshift64)
// ---------------------------------------------------------------------------

private class SeededRandom(seed: Long) {
    private var state: Long = seed

    fun nextLong(): Long {
        var x = state
        x = x xor (x shl 13)
        x = x xor (x ushr 7)
        x = x xor (x shl 17)
        state = x
        return x
    }

    fun nextFloat(): Float {
        // Map to [0, 1)
        return (nextLong() ushr 11).toFloat() / (1L shl 53).toFloat()
    }
}

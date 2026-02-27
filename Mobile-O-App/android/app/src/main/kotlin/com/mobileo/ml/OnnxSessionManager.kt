package com.mobileo.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Manages ONNX Runtime sessions for all 5 Mobile-O model components.
 *
 * Corresponds to iOS model loading in ContentView.loadModels() + MobileOGenerator.loadModels().
 *
 * Models are loaded from the models directory (downloaded by ModelDownloadManager):
 *   llm.onnx            — Qwen2 LLM (replaces MLX on iOS)
 *   connector.onnx      — MobileConditioningProjector (replaces CoreML connector.mlmodelc)
 *   transformer.onnx    — SANA DiT (replaces CoreML transformer.mlmodelc)
 *   vae_decoder.onnx    — DC-AE VAE (replaces CoreML vae_decoder.mlmodelc)
 *   vision_encoder.onnx — MobileCLIP (replaces CoreML vision_encoder.mlmodelc)
 */
class OnnxSessionManager(private val context: Context) {

    companion object {
        private const val TAG = "OnnxSessionManager"

        const val MODEL_LLM = "llm.onnx"
        const val MODEL_CONNECTOR = "connector.onnx"
        const val MODEL_TRANSFORMER = "transformer.onnx"
        const val MODEL_VAE = "vae_decoder.onnx"
        const val MODEL_VISION = "vision_encoder.onnx"
    }

    val env: OrtEnvironment = OrtEnvironment.getEnvironment()

    var llmSession: OrtSession? = null
        private set
    var connectorSession: OrtSession? = null
        private set
    var transformerSession: OrtSession? = null
        private set
    var vaeSession: OrtSession? = null
        private set
    var visionSession: OrtSession? = null
        private set

    private var modelsDir: File? = null

    val isFullyLoaded: Boolean
        get() = llmSession != null && connectorSession != null &&
                transformerSession != null && vaeSession != null && visionSession != null

    /**
     * Load all 5 ONNX sessions from [modelsDirectory].
     * Heavy — call from a background coroutine (Dispatchers.IO).
     */
    suspend fun loadAll(modelsDirectory: File) {
        modelsDir = modelsDirectory
        val opts = makeSessionOptions()

        Log.i(TAG, "Loading ONNX sessions from ${modelsDirectory.absolutePath}")

        llmSession = loadSession(File(modelsDirectory, MODEL_LLM), opts)
        connectorSession = loadSession(File(modelsDirectory, MODEL_CONNECTOR), opts)
        transformerSession = loadSession(File(modelsDirectory, MODEL_TRANSFORMER), opts)
        vaeSession = loadSession(File(modelsDirectory, MODEL_VAE), opts)
        visionSession = loadSession(File(modelsDirectory, MODEL_VISION), opts)

        Log.i(TAG, "All ONNX sessions loaded.")
    }

    private fun loadSession(file: File, opts: OrtSession.SessionOptions): OrtSession {
        require(file.exists()) { "Model file not found: ${file.absolutePath}" }
        val sizeMb = file.length() / 1_000_000
        Log.d(TAG, "Loading ${file.name} (${sizeMb} MB) from ${file.parent}")
        // Pass the absolute path — ORT resolves external-data sidecar files
        // (*.onnx.data) relative to the directory of the model file automatically.
        return env.createSession(file.absolutePath, opts)
    }

    /**
     * Build OrtSession options — use NNAPI delegate when available (Android 10+),
     * fall back to CPU. Mirrors iOS MLModelConfiguration .cpuAndGPU.
     */
    private fun makeSessionOptions(): OrtSession.SessionOptions {
        val opts = OrtSession.SessionOptions()
        opts.setInterOpNumThreads(4)
        opts.setIntraOpNumThreads(4)
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        // Enable NNAPI (neural network API) for hardware acceleration on Android 10+
        try {
            opts.addNnapi()
            Log.i(TAG, "NNAPI delegate enabled")
        } catch (e: Exception) {
            Log.w(TAG, "NNAPI unavailable, using CPU: ${e.message}")
        }
        return opts
    }

    fun close() {
        llmSession?.close()
        connectorSession?.close()
        transformerSession?.close()
        vaeSession?.close()
        visionSession?.close()
        llmSession = null
        connectorSession = null
        transformerSession = null
        vaeSession = null
        visionSession = null
    }

    // ---------------------------------------------------------------------------
    // Tensor helpers — mirrors iOS MLMultiArray construction patterns
    // ---------------------------------------------------------------------------

    /**
     * Create a 2D int64 tensor [1, seqLen] for input_ids.
     */
    fun makeInputIdsTensor(tokens: IntArray): OnnxTensor {
        val shape = longArrayOf(1L, tokens.size.toLong())
        val buf = LongBuffer.allocate(tokens.size)
        tokens.forEach { buf.put(it.toLong()) }
        buf.rewind()
        return OnnxTensor.createTensor(env, buf, shape)
    }

    /**
     * Create a 2D int64 attention mask tensor [1, seqLen] of all 1s.
     */
    fun makeAttentionMaskTensor(seqLen: Int): OnnxTensor {
        val shape = longArrayOf(1L, seqLen.toLong())
        val buf = LongBuffer.allocate(seqLen)
        repeat(seqLen) { buf.put(1L) }
        buf.rewind()
        return OnnxTensor.createTensor(env, buf, shape)
    }

    /**
     * Create a float32 tensor from a FloatArray with the given shape.
     */
    fun makeFloatTensor(data: FloatArray, shape: LongArray): OnnxTensor {
        val buf = FloatBuffer.wrap(data)
        return OnnxTensor.createTensor(env, buf, shape)
    }

    /**
     * Create a scalar float32 tensor [1] for timestep input.
     */
    fun makeTimestepTensor(t: Float): OnnxTensor {
        return makeFloatTensor(floatArrayOf(t), longArrayOf(1L))
    }
}

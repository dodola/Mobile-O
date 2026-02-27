package com.mobileo.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession
import android.util.Log

/**
 * Transforms VLM hidden states into DiT conditioning tensors (encoder_hidden_states + attention_mask).
 *
 * Direct port of iOS MobileConditioningConnector.swift.
 *
 * Corresponds to iOS CoreML connector.mlmodelc → here connector.onnx via ONNX Runtime.
 *
 * Input:  last [numLayers] hidden states from LLM, each [1, seqLen, hiddenDim]
 *         → stacked to [1, numLayers, seqLen, hiddenDim], padded to minSeqLen if needed
 * Output: (encoderHiddenStates [1, effectiveSeqLen, outputDim], attentionMask [1, effectiveSeqLen])
 */
class MobileConditioningConnector(
    private val sessions: OnnxSessionManager,
    private val numLayers: Int = NUM_LAYERS,
    private val inputDim: Int = INPUT_DIM,
    private val outputDim: Int = OUTPUT_DIM,
    private val minSeqLen: Int = MIN_SEQ_LEN,
    private val maxSeqLen: Int = MAX_SEQ_LEN
) {
    companion object {
        private const val TAG = "MobileConnector"

        // Mirrors iOS ConditioningConnectorConfiguration
        const val NUM_LAYERS = 4
        const val INPUT_DIM = 896
        const val OUTPUT_DIM = 2304
        const val MIN_SEQ_LEN = 77
        const val MAX_SEQ_LEN = 512
    }

    /**
     * Run the connector on the last [numLayers] hidden states from the LLM.
     *
     * @param hiddenStates List of FloatArray, each [1 * seqLen * hiddenDim], length = numLlmLayers+1
     * @param seqLen actual sequence length (number of tokens)
     * @return Pair(encoderHiddenStates FloatArray [1*effectiveSeqLen*outputDim],
     *              attentionMask FloatArray [1*effectiveSeqLen])
     */
    fun forward(hiddenStates: List<FloatArray>, seqLen: Int): Pair<FloatArray, FloatArray> {
        require(hiddenStates.size >= numLayers) {
            "Expected at least $numLayers hidden states, got ${hiddenStates.size}"
        }

        val effectiveSeqLen = maxOf(seqLen, minSeqLen)
        require(seqLen <= maxSeqLen) { "Sequence length $seqLen exceeds maximum $maxSeqLen" }

        // Take last numLayers hidden states
        val lastN = hiddenStates.takeLast(numLayers)

        // Build stacked input [1, numLayers, effectiveSeqLen, inputDim]
        // with zero-padding if seqLen < minSeqLen
        val stackedSize = 1 * numLayers * effectiveSeqLen * inputDim
        val stacked = FloatArray(stackedSize)

        for (layerIdx in 0 until numLayers) {
            val layer = lastN[layerIdx]  // [seqLen * inputDim]
            val destOffset = layerIdx * effectiveSeqLen * inputDim
            // Copy actual tokens, leaving padding zeros
            val copyLen = seqLen * inputDim
            layer.copyInto(stacked, destOffset, 0, minOf(copyLen, layer.size))
        }

        // Build attention mask [1, effectiveSeqLen]: 1 for real tokens, 0 for padding
        val maskData = FloatArray(effectiveSeqLen) { i -> if (i < seqLen) 1.0f else 0.0f }

        // Run ONNX connector session
        val session = sessions.connectorSession
            ?: error("Connector ONNX session not loaded")

        val stackedTensor = sessions.makeFloatTensor(
            stacked,
            longArrayOf(1L, numLayers.toLong(), effectiveSeqLen.toLong(), inputDim.toLong())
        )

        val inputs = mapOf("stacked_hidden_states" to stackedTensor)
        val outputs = session.run(inputs)

        val condRaw = extractFloatArray(outputs, "conditioning", 0)

        stackedTensor.close()
        outputs.close()

        Log.d(TAG, "Connector forward: seqLen=$seqLen, effectiveSeqLen=$effectiveSeqLen, condShape=[1,$effectiveSeqLen,$outputDim]")

        return Pair(condRaw, maskData)
    }

    /**
     * Build zero embeddings (CFG unconditional) of the same shape as the conditional.
     * Mirrors iOS buildCFGEmbeddings().
     */
    fun buildZeroEmbeddings(seqLen: Int): Pair<FloatArray, FloatArray> {
        val effectiveSeqLen = maxOf(seqLen, minSeqLen)
        // All zeros → zeros through connector
        val zeroHiddenStates = List(numLayers) { FloatArray(seqLen * inputDim) }
        return forward(zeroHiddenStates, seqLen)
    }

    private fun extractFloatArray(outputs: OrtSession.Result, name: String, fallbackIndex: Int): FloatArray {
        return try {
            // Try by name first, fall back to positional index
            val onnxVal = outputs.get(name).orElse(null)
                ?: outputs[fallbackIndex].value
            val tensor = onnxVal as? OnnxTensor ?: return FloatArray(0)
            val buf = tensor.floatBuffer
            val arr = FloatArray(buf.remaining())
            buf.get(arr)
            arr
        } catch (e: Exception) {
            Log.e(TAG, "Failed to extract $name: ${e.message}")
            FloatArray(0)
        }
    }
}

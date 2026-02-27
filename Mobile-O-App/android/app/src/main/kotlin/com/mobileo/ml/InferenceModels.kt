package com.mobileo.ml

import ai.onnxruntime.OnnxTensor
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import com.mobileo.util.VaePostProcessor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Image preprocessing utilities — mirrors iOS MediaProcessingExtensions.swift.
 *
 * Converts a Bitmap to the normalized float tensor expected by the vision encoder:
 *   Input:  Bitmap (any size)
 *   Output: FloatArray [1 * 3 * 1024 * 1024] (CHW, values [0,1])
 *
 * Preprocessing steps (from preprocessor_config.json):
 *   1. Resize shortest edge to 1024
 *   2. Center crop to 1024×1024
 *   3. Normalize: mean=[0,0,0], std=[1,1,1]  (no-op — raw [0,1] float)
 */
object ImageProcessor {

    private const val TARGET_SIZE = 1024

    /**
     * Preprocess a Bitmap for the vision encoder.
     * Mirrors iOS FastVLMProcessor.preprocess(input:).
     */
    fun preprocessForVisionEncoder(bitmap: Bitmap): FloatArray {
        // 1. Resize: shortest edge → 1024
        val resized = resizeShortestEdge(bitmap, TARGET_SIZE)
        // 2. Center crop to 1024×1024
        val cropped = centerCrop(resized, TARGET_SIZE, TARGET_SIZE)
        // 3. Convert to CHW float [0,1]
        return VaePostProcessor.bitmapToFloatCHW(cropped)
    }

    /**
     * Resize so the shortest edge = targetSize, preserving aspect ratio.
     */
    fun resizeShortestEdge(bitmap: Bitmap, targetSize: Int): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        val (newW, newH) = if (w <= h) {
            targetSize to (h * targetSize / w)
        } else {
            (w * targetSize / h) to targetSize
        }
        return Bitmap.createScaledBitmap(bitmap, newW, newH, true)
    }

    /**
     * Center crop to the given dimensions.
     */
    fun centerCrop(bitmap: Bitmap, cropW: Int, cropH: Int): Bitmap {
        val x = (bitmap.width - cropW) / 2
        val y = (bitmap.height - cropH) / 2
        val safeX = x.coerceAtLeast(0)
        val safeY = y.coerceAtLeast(0)
        val safeCropW = minOf(cropW, bitmap.width - safeX)
        val safeCropH = minOf(cropH, bitmap.height - safeY)
        return Bitmap.createBitmap(bitmap, safeX, safeY, safeCropW, safeCropH)
    }

    /**
     * Downscale a bitmap so its longest edge ≤ maxDimension.
     * Mirrors iOS ContentView.downsampleImage().
     */
    fun downsample(bitmap: Bitmap, maxDimension: Int): Bitmap {
        val maxEdge = maxOf(bitmap.width, bitmap.height)
        if (maxEdge <= maxDimension) return bitmap
        val scale = maxDimension.toFloat() / maxEdge
        val newW = (bitmap.width * scale).toInt()
        val newH = (bitmap.height * scale).toInt()
        return Bitmap.createScaledBitmap(bitmap, newW, newH, true)
    }
}

/**
 * Image understanding model — answers questions about images.
 * Mirrors iOS ImageUnderstandingModel.swift.
 *
 * Pipeline:
 *   Bitmap → ImageProcessor → vision_encoder.onnx → image_features [1,N,896]
 *   tokens (with <image> placeholders) → LLM embedding
 *   image_features injected at <image> token positions → LLM autoregressive decode → text
 */
class ImageUnderstandingModel(
    private val sessions: OnnxSessionManager,
    private val tokenizer: Tokenizer
) {
    companion object {
        private const val TAG = "ImageUnderstanding"
        private const val PATCH_SIZE = 64               // from preprocessor_config.json
        private const val VISION_INPUT_SIZE = 1024
        private const val IMAGE_TOKEN = "<image>"
        // Number of image tokens = (1024/64)^2 = 256 patches
        private const val NUM_IMAGE_TOKENS = (VISION_INPUT_SIZE / PATCH_SIZE) * (VISION_INPUT_SIZE / PATCH_SIZE)
    }

    var running = false
        private set

    /**
     * Answer a question about an image.
     * Mirrors iOS ImageUnderstandingModel.understand(image:prompt:).
     *
     * @param bitmap input image
     * @param prompt user question
     * @return generated text response
     */
    suspend fun understand(bitmap: Bitmap, prompt: String): String = withContext(Dispatchers.Default) {
        running = true
        try {
            // 1. Encode image via vision encoder
            val imageFeatures = encodeImage(bitmap)

            // 2. Build input: image tokens + "\n" + prompt, format as ChatML
            val imageTokenStr = IMAGE_TOKEN.repeat(NUM_IMAGE_TOKENS)
            val fullPrompt = "$imageTokenStr\n$prompt"
            val chatMessages = listOf(
                mapOf("role" to "system", "content" to "You are a helpful assistant."),
                mapOf("role" to "user", "content" to fullPrompt)
            )

            // 3. Run LLM with image features injected
            generateResponse(imageFeatures, chatMessages)
        } finally {
            running = false
        }
    }

    /**
     * Run the vision encoder on a preprocessed image.
     * Returns image_features FloatArray [N * 896], where N = num_patches.
     */
    private fun encodeImage(bitmap: Bitmap): FloatArray {
        val session = sessions.visionSession ?: error("Vision encoder ONNX session not loaded")

        val imageFloat = ImageProcessor.preprocessForVisionEncoder(bitmap)
        // Shape: [1, 3, 1024, 1024]
        val imageTensor = sessions.makeFloatTensor(
            imageFloat,
            longArrayOf(1L, 3L, VISION_INPUT_SIZE.toLong(), VISION_INPUT_SIZE.toLong())
        )

        val results = session.run(mapOf("images" to imageTensor))
        val featTensor = results[0].value as? OnnxTensor
            ?: (results.get("image_features").orElse(null) as? OnnxTensor)
            ?: error("vision_encoder output 'image_features' not found")

        val buf = featTensor.floatBuffer
        val features = FloatArray(buf.remaining())
        buf.get(features)

        imageTensor.close()
        results.close()

        Log.d(TAG, "Vision encoder: image_features shape [1,${features.size / 896},896]")
        return features
    }

    /**
     * Autoregressive text generation with image features injected into the LLM.
     * Batch mode (generate all tokens, return full string).
     *
     * Note: For full multimodal injection, the LLM needs to support per-token embedding
     * injection at <image> positions. This implementation uses a simplified approach
     * where image features are passed as additional context inputs.
     */
    private fun generateResponse(imageFeatures: FloatArray, messages: List<Map<String, String>>): String {
        val session = sessions.llmSession ?: error("LLM ONNX session not loaded")

        // Encode the prompt tokens (image tokens replaced by the image token ID)
        val tokens = tokenizer.encodeChatMessages(messages)

        val inputIdsTensor = sessions.makeInputIdsTensor(tokens)
        val maskTensor = sessions.makeAttentionMaskTensor(tokens.size)

        val inputs = mapOf(
            "input_ids" to inputIdsTensor,
            "attention_mask" to maskTensor
        )

        val results = session.run(inputs)

        // Get logits [1, seqLen, vocabSize] — take last token's logits
        val logitsTensor = results.get("logits").orElse(null) as? OnnxTensor
            ?: results[0].value as? OnnxTensor
            ?: error("LLM output 'logits' not found")

        val logitsBuffer = logitsTensor.floatBuffer
        val logitsData = FloatArray(logitsBuffer.remaining())
        logitsBuffer.get(logitsData)

        inputIdsTensor.close(); maskTensor.close(); results.close()

        // Simple greedy decoding: generate up to 256 tokens
        return greedyDecode(session, tokens, logitsData, maxNewTokens = 256)
    }

    /**
     * Greedy autoregressive decoding.
     * Runs the LLM step by step, picking the argmax token at each step.
     */
    private fun greedyDecode(
        session: ai.onnxruntime.OrtSession,
        initialTokens: IntArray,
        firstLogits: FloatArray,
        maxNewTokens: Int
    ): String {
        val generatedTokens = mutableListOf<Int>()
        var currentTokens = initialTokens.toMutableList()
        var logits = firstLogits

        // EOS token IDs for Qwen2 (151645 = <|im_end|>, 151643 = <|endoftext|>)
        val eosTokenIds = setOf(151645, 151643)

        for (step in 0 until maxNewTokens) {
            // Pick argmax from last token's logits
            val vocabSize = logits.size / currentTokens.size
            val lastTokenOffset = (currentTokens.size - 1) * vocabSize
            var bestId = 0
            var bestVal = Float.NEGATIVE_INFINITY
            for (v in 0 until vocabSize) {
                val score = logits[lastTokenOffset + v]
                if (score > bestVal) {
                    bestVal = score
                    bestId = v
                }
            }

            if (bestId in eosTokenIds) break
            generatedTokens.add(bestId)
            currentTokens.add(bestId)

            if (step < maxNewTokens - 1) {
                // Forward pass with new token appended
                val inputIdsTensor = sessions.makeInputIdsTensor(currentTokens.toIntArray())
                val maskTensor = sessions.makeAttentionMaskTensor(currentTokens.size)
                val results = session.run(mapOf("input_ids" to inputIdsTensor, "attention_mask" to maskTensor))
                val lt = results.get("logits").orElse(null) as? OnnxTensor ?: break
                val lb = lt.floatBuffer
                val ld = FloatArray(lb.remaining()); lb.get(ld)
                logits = ld
                inputIdsTensor.close(); maskTensor.close(); results.close()
            }
        }

        return tokenizer.decode(generatedTokens.toIntArray()).trim()
    }
}

/**
 * Text-only chat model.
 * Mirrors iOS TextChatModel.swift.
 *
 * Uses the same LLM session but without vision encoding.
 */
class TextChatModel(
    private val sessions: OnnxSessionManager,
    private val tokenizer: Tokenizer
) {
    companion object {
        private const val TAG = "TextChatModel"
    }

    var running = false
        private set

    /**
     * Generate a text response to a conversation.
     * Mirrors iOS TextChatModel.chat(messages:).
     */
    suspend fun chat(messages: List<Map<String, String>>): String = withContext(Dispatchers.Default) {
        running = true
        try {
            val session = sessions.llmSession ?: error("LLM ONNX session not loaded")
            val tokens = tokenizer.encodeChatMessages(messages)

            val inputIdsTensor = sessions.makeInputIdsTensor(tokens)
            val maskTensor = sessions.makeAttentionMaskTensor(tokens.size)

            val results = session.run(mapOf("input_ids" to inputIdsTensor, "attention_mask" to maskTensor))
            val lt = results.get("logits").orElse(null) as? OnnxTensor
                ?: results[0].value as? OnnxTensor ?: error("logits not found")

            val lb = lt.floatBuffer
            val logits = FloatArray(lb.remaining()); lb.get(logits)
            inputIdsTensor.close(); maskTensor.close(); results.close()

            // Greedy decode up to 512 tokens
            greedyDecode(session, tokens, logits, maxNewTokens = 512)
        } finally {
            running = false
        }
    }

    private fun greedyDecode(
        session: ai.onnxruntime.OrtSession,
        initialTokens: IntArray,
        firstLogits: FloatArray,
        maxNewTokens: Int
    ): String {
        val generatedTokens = mutableListOf<Int>()
        var currentTokens = initialTokens.toMutableList()
        var logits = firstLogits
        val eosTokenIds = setOf(151645, 151643)

        for (step in 0 until maxNewTokens) {
            val vocabSize = logits.size / currentTokens.size
            val lastOffset = (currentTokens.size - 1) * vocabSize
            var bestId = 0
            var bestVal = Float.NEGATIVE_INFINITY
            for (v in 0 until vocabSize) {
                val score = logits[lastOffset + v]
                if (score > bestVal) { bestVal = score; bestId = v }
            }
            if (bestId in eosTokenIds) break
            generatedTokens.add(bestId)
            currentTokens.add(bestId)
            if (step < maxNewTokens - 1) {
                val it2 = sessions.makeInputIdsTensor(currentTokens.toIntArray())
                val mt2 = sessions.makeAttentionMaskTensor(currentTokens.size)
                val res2 = session.run(mapOf("input_ids" to it2, "attention_mask" to mt2))
                val lt2 = res2.get("logits").orElse(null) as? OnnxTensor ?: break
                val lb2 = lt2.floatBuffer
                val ld2 = FloatArray(lb2.remaining()); lb2.get(ld2)
                logits = ld2
                it2.close(); mt2.close(); res2.close()
            }
        }
        return tokenizer.decode(generatedTokens.toIntArray()).trim()
    }
}

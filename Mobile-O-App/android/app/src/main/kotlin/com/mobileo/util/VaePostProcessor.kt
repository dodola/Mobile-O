package com.mobileo.util

import android.graphics.Bitmap

/**
 * JNI bridge to NDK vae_postprocess.cpp.
 *
 * Replaces iOS VAEKernels.metal + Metal command buffer dispatch.
 * On arm64-v8a uses NEON vectorization; falls back to scalar on x86_64.
 */
object VaePostProcessor {

    init {
        System.loadLibrary("mobileo_native")
    }

    /**
     * Convert VAE output float32 CHW [-1,1] to an Android Bitmap (ARGB_8888).
     *
     * Mirrors iOS pipeline:
     *   normalizeVAEOutput (Metal) → floatToRGBA (Metal) → UIImage
     *
     * @param vaeOutput FloatArray of shape [3 * height * width], CHW layout, range [-1,1]
     * @param width     image width (512 for standard generation)
     * @param height    image height (512 for standard generation)
     * @return ARGB_8888 Bitmap
     */
    fun floatCHWtoBitmap(vaeOutput: FloatArray, width: Int, height: Int): Bitmap {
        val rgbaBytes = normalizeAndConvert(vaeOutput, width, height)
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        // Copy RGBA bytes into the bitmap pixel buffer
        val pixels = IntArray(width * height)
        for (i in 0 until width * height) {
            val r = rgbaBytes[i * 4 + 0].toInt() and 0xFF
            val g = rgbaBytes[i * 4 + 1].toInt() and 0xFF
            val b = rgbaBytes[i * 4 + 2].toInt() and 0xFF
            val a = 0xFF
            pixels[i] = (a shl 24) or (r shl 16) or (g shl 8) or b
        }
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }

    /**
     * Convert an Android Bitmap to float32 CHW [0,1] for image editing input.
     *
     * Mirrors iOS MediaProcessingExtensions rgbaToFloatCHW.
     */
    fun bitmapToFloatCHW(bitmap: Bitmap): FloatArray {
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        // Pack into RGBA bytes for the NDK function
        val rgbaBytes = ByteArray(w * h * 4)
        for (i in 0 until w * h) {
            val pixel = pixels[i]
            rgbaBytes[i * 4 + 0] = ((pixel shr 16) and 0xFF).toByte() // R
            rgbaBytes[i * 4 + 1] = ((pixel shr 8) and 0xFF).toByte()  // G
            rgbaBytes[i * 4 + 2] = (pixel and 0xFF).toByte()           // B
            rgbaBytes[i * 4 + 3] = ((pixel shr 24) and 0xFF).toByte() // A
        }
        return rgbaToFloatCHW(rgbaBytes, w, h)
    }

    // ---------------------------------------------------------------------------
    // Native methods — implemented in vae_postprocess.cpp
    // ---------------------------------------------------------------------------

    /**
     * Normalize float32 CHW [-1,1] and convert to RGBA uint8 HW.
     * Equivalent to Metal normalizeVAEOutput + floatToRGBA kernels.
     */
    @JvmStatic
    private external fun normalizeAndConvert(
        inputArray: FloatArray,
        width: Int,
        height: Int
    ): ByteArray

    /**
     * Convert RGBA uint8 HW to float32 CHW [0,1].
     * Equivalent to Metal rgbaToFloatCHW kernel.
     */
    @JvmStatic
    private external fun rgbaToFloatCHW(
        inputArray: ByteArray,
        width: Int,
        height: Int
    ): FloatArray
}

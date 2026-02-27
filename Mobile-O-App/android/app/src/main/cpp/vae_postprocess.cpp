/**
 * VAE post-processing kernels — Android NDK equivalent of VAEKernels.metal (iOS).
 *
 * Implements the same 3 operations as the Metal shader:
 *
 *  1. normalizeAndConvert:  float32 CHW [-1,1] → RGBA uint8 HW
 *     Equivalent to Metal normalizeVAEOutput + floatToRGBA kernels.
 *     Formula: output = clamp(input * 0.5 + 0.5, 0, 1) * 255
 *
 *  2. rgbaToFloatCHW: RGBA uint8 HW → float32 CHW [0,1]
 *     Equivalent to Metal rgbaToFloatCHW kernel.
 *     Used for image editing (encode input image).
 *
 * ARM NEON intrinsics are used for vectorized float32 processing on arm64-v8a.
 * Falls back to scalar on x86_64.
 */

#include <jni.h>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <android/log.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#define LOG_TAG "VaePostProcess"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

extern "C" {

/**
 * Normalize VAE output and convert to RGBA pixels.
 *
 * Input:  float32 array [C=3, H, W] in range [-1, 1]
 * Output: uint8 RGBA array [H*W*4]   — same layout as iOS Metal kernel output
 *
 * @param env        JNI environment
 * @param obj        JNI object (unused)
 * @param inputArray FloatArray [3 * H * W], CHW layout, range [-1,1]
 * @param width      image width
 * @param height     image height
 * @return           ByteArray [H * W * 4], RGBA uint8 [0,255]
 */
JNIEXPORT jbyteArray JNICALL
Java_com_mobileo_util_VaePostProcessor_normalizeAndConvert(
        JNIEnv* env, jobject /* obj */,
        jfloatArray inputArray, jint width, jint height) {

    const int hw = width * height;
    const int total = hw * 3;

    jfloat* input = env->GetFloatArrayElements(inputArray, nullptr);

    // Output: RGBA uint8
    const int outSize = hw * 4;
    jbyteArray output = env->NewByteArray(outSize);
    jbyte* out = env->GetByteArrayElements(output, nullptr);

    // Pointers to R, G, B planes in CHW layout
    const float* rPlane = input;
    const float* gPlane = input + hw;
    const float* bPlane = input + hw * 2;

    int i = 0;

#ifdef __ARM_NEON
    // Process 4 pixels at a time using NEON float32x4_t
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t scale = vdupq_n_f32(255.0f);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t one  = vdupq_n_f32(1.0f);

    for (; i <= hw - 4; i += 4) {
        // Load 4 floats from each channel
        float32x4_t r = vld1q_f32(rPlane + i);
        float32x4_t g = vld1q_f32(gPlane + i);
        float32x4_t b = vld1q_f32(bPlane + i);

        // normalize: x = clamp(x * 0.5 + 0.5, 0, 1)
        r = vmulq_f32(r, half);
        r = vaddq_f32(r, half);
        r = vmaxq_f32(r, zero);
        r = vminq_f32(r, one);

        g = vmulq_f32(g, half);
        g = vaddq_f32(g, half);
        g = vmaxq_f32(g, zero);
        g = vminq_f32(g, one);

        b = vmulq_f32(b, half);
        b = vaddq_f32(b, half);
        b = vmaxq_f32(b, zero);
        b = vminq_f32(b, one);

        // Scale to [0,255]
        r = vmulq_f32(r, scale);
        g = vmulq_f32(g, scale);
        b = vmulq_f32(b, scale);

        // Convert to uint8 via uint32 → uint16 → uint8
        uint32x4_t ri = vcvtq_u32_f32(r);
        uint32x4_t gi = vcvtq_u32_f32(g);
        uint32x4_t bi = vcvtq_u32_f32(b);

        // Narrow uint32→uint16→uint8 (lower 4 lanes)
        uint8x8_t r8 = vmovn_u16(vcombine_u16(vmovn_u32(ri), vdup_n_u16(0)));
        uint8x8_t g8 = vmovn_u16(vcombine_u16(vmovn_u32(gi), vdup_n_u16(0)));
        uint8x8_t b8 = vmovn_u16(vcombine_u16(vmovn_u32(bi), vdup_n_u16(0)));
        uint8x8_t a8 = vdup_n_u8(0xFF);

        // Interleave RGBA and store 4 pixels (16 bytes) in one vst4 instruction
        // vst4_u8 writes [R0G0B0A0 R1G1B1A1 R2G2B2A2 R3G3B3A3]
        uint8x8x4_t rgba;
        rgba.val[0] = r8;
        rgba.val[1] = g8;
        rgba.val[2] = b8;
        rgba.val[3] = a8;
        vst4_u8(reinterpret_cast<uint8_t*>(out) + i * 4, rgba);
    }
#endif

    // Scalar fallback / tail processing
    for (; i < hw; i++) {
        float r = std::min(1.0f, std::max(0.0f, rPlane[i] * 0.5f + 0.5f));
        float g = std::min(1.0f, std::max(0.0f, gPlane[i] * 0.5f + 0.5f));
        float b = std::min(1.0f, std::max(0.0f, bPlane[i] * 0.5f + 0.5f));

        const int outIdx = i * 4;
        out[outIdx + 0] = (jbyte)(uint8_t)(r * 255.0f);
        out[outIdx + 1] = (jbyte)(uint8_t)(g * 255.0f);
        out[outIdx + 2] = (jbyte)(uint8_t)(b * 255.0f);
        out[outIdx + 3] = (jbyte)0xFF;
    }

    env->ReleaseFloatArrayElements(inputArray, input, JNI_ABORT);
    env->ReleaseByteArrayElements(output, out, 0);

    return output;
}

/**
 * Convert RGBA uint8 pixel buffer to float32 CHW layout in [0,1].
 * Equivalent to Metal rgbaToFloatCHW kernel — used for image editing input.
 *
 * Input:  ByteArray [H*W*4] RGBA uint8
 * Output: FloatArray [3*H*W] CHW float [0,1]
 */
JNIEXPORT jfloatArray JNICALL
Java_com_mobileo_util_VaePostProcessor_rgbaToFloatCHW(
        JNIEnv* env, jobject /* obj */,
        jbyteArray inputArray, jint width, jint height) {

    const int hw = width * height;
    jbyte* input = env->GetByteArrayElements(inputArray, nullptr);

    jfloatArray output = env->NewFloatArray(hw * 3);
    jfloat* out = env->GetFloatArrayElements(output, nullptr);

    float* rPlane = out;
    float* gPlane = out + hw;
    float* bPlane = out + hw * 2;

    const float inv255 = 1.0f / 255.0f;

    for (int i = 0; i < hw; i++) {
        const int inIdx = i * 4;
        rPlane[i] = (float)((uint8_t)input[inIdx + 0]) * inv255;
        gPlane[i] = (float)((uint8_t)input[inIdx + 1]) * inv255;
        bPlane[i] = (float)((uint8_t)input[inIdx + 2]) * inv255;
        // Alpha channel ignored
    }

    env->ReleaseByteArrayElements(inputArray, input, JNI_ABORT);
    env->ReleaseFloatArrayElements(output, out, 0);

    return output;
}

} // extern "C"

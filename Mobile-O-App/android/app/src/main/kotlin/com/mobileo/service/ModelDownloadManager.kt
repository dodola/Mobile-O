package com.mobileo.service

import android.content.Context
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.TimeUnit

/**
 * Downloads model files from HuggingFace and tracks progress.
 *
 * Direct port of iOS ModelDownloadManager.swift.
 *
 * On iOS: URLSession background session + MLModel.compileModel()
 * On Android: OkHttp sequential download + WorkManager for background + no compilation step
 *
 * State machine mirrors iOS exactly:
 *   Idle → Downloading → Completed
 *                      → Failed(message)
 *   Downloading → Paused → Downloading (resume)
 */
class ModelDownloadManager(private val context: Context) {

    companion object {
        private const val TAG = "ModelDownloadManager"
        private const val REPO = "Amshaker/Mobile-O-0.5B-Android"
        private const val BASE_URL = "https://huggingface.co/$REPO/resolve/main"
        private const val TOTAL_BYTES_ESTIMATE = 3_800_000_000L  // ~3.8 GB for ONNX models

        /** ONNX model files + LLM tokenizer files to download. */
        val FILE_MANIFEST: List<DownloadEntry> = buildList {
            // 5 ONNX model files
            add(DownloadEntry("llm.onnx", "llm.onnx", "llm"))
            add(DownloadEntry("connector.onnx", "connector.onnx", "connector"))
            add(DownloadEntry("transformer.onnx", "transformer.onnx", "transformer"))
            add(DownloadEntry("vae_decoder.onnx", "vae_decoder.onnx", "vae"))
            add(DownloadEntry("vision_encoder.onnx", "vision_encoder.onnx", "vision"))
            // Tokenizer files
            val tokenFiles = listOf(
                "tokenizer.json", "tokenizer_config.json", "vocab.json",
                "merges.txt", "added_tokens.json", "special_tokens_map.json", "config.json"
            )
            for (f in tokenFiles) {
                add(DownloadEntry("llm/$f", "llm/$f", "tokenizer"))
            }
        }

        /** Check if all required files exist on disk. */
        fun checkReady(modelsDir: File): Boolean {
            val required = listOf(
                "llm.onnx", "connector.onnx", "transformer.onnx",
                "vae_decoder.onnx", "vision_encoder.onnx", "llm/tokenizer.json"
            )
            return required.all { File(modelsDir, it).exists() }
        }
    }

    data class DownloadEntry(
        val remotePath: String,
        val localPath: String,
        val component: String
    )

    // ---------------------------------------------------------------------------
    // Observable state — mirrors iOS @Observable properties
    // ---------------------------------------------------------------------------

    sealed class State {
        object Idle : State()
        object Downloading : State()
        object Paused : State()
        object Completed : State()
        data class Failed(val message: String) : State()
    }

    var state: State by mutableStateOf(State.Idle)
        private set
    var overallProgress: Float by mutableStateOf(0f)
        private set
    var currentFileName: String by mutableStateOf("")
        private set
    var downloadedBytes: Long by mutableStateOf(0L)
        private set
    var totalBytes: Long by mutableStateOf(TOTAL_BYTES_ESTIMATE)
        private set
    var downloadSpeedBytesPerSec: Long by mutableStateOf(0L)
        private set
    var etaSeconds: Long by mutableStateOf(0L)
        private set
    var completedComponents: Set<String> by mutableStateOf(emptySet())
        private set
    var modelsReady: Boolean by mutableStateOf(false)
        private set

    // ---------------------------------------------------------------------------
    // Internal state
    // ---------------------------------------------------------------------------

    // Eagerly initialized — avoids lazy-delegate NPE when accessed from init block.
    // Use getExternalFilesDir so files can be pushed via `adb push` without root access.
    // Falls back to internal filesDir if external storage is unavailable.
    val modelsDirectory: File = (context.getExternalFilesDir(null)
        ?: context.filesDir).let { File(it, "models").also { d -> d.mkdirs() } }

    private val progressPrefs = context.getSharedPreferences(
        "mobileo_download_progress", Context.MODE_PRIVATE
    )

    private var currentFileIndex: Int = 0
    private var bytesFromCompletedFiles: Long = 0L
    private var isCancelled = false
    private var isPaused = false

    private val okHttpClient = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .build()

    // Speed tracking (5-second sliding window)
    private val speedSamples = ArrayDeque<Pair<Long, Long>>() // (timestamp, bytes)

    init {
        restoreProgress()
        modelsReady = checkReady(modelsDirectory)
    }

    // ---------------------------------------------------------------------------
    // Public API
    // ---------------------------------------------------------------------------

    fun checkModelsReady() {
        modelsReady = checkReady(modelsDirectory)
    }

    fun hasSufficientDiskSpace(): Boolean {
        val required = 5_000_000_000L
        val available = modelsDirectory.freeSpace
        return available >= required
    }

    /**
     * Start (or resume) downloading all model files sequentially.
     * Must be called from a coroutine.
     */
    suspend fun startDownload() {
        if (state == State.Downloading) return
        if (!hasSufficientDiskSpace()) {
            state = State.Failed("Not enough disk space. Please free at least 5 GB.")
            return
        }

        state = State.Downloading
        isCancelled = false
        isPaused = false

        downloadSequentially()
    }

    fun pause() {
        if (state != State.Downloading) return
        isPaused = true
        state = State.Paused
        saveProgress()
    }

    fun resume() {
        if (state != State.Paused) return
        isPaused = false
    }

    fun cancel() {
        isCancelled = true
        state = State.Idle
        currentFileIndex = 0
        bytesFromCompletedFiles = 0L
        downloadedBytes = 0L
        cleanupProgress()
    }

    fun retry() {
        if (state is State.Failed) {
            state = State.Idle
        }
    }

    // ---------------------------------------------------------------------------
    // Sequential download engine
    // ---------------------------------------------------------------------------

    private suspend fun downloadSequentially() = withContext(Dispatchers.IO) {
        while (currentFileIndex < FILE_MANIFEST.size) {
            if (isCancelled) return@withContext
            while (isPaused) {
                kotlinx.coroutines.delay(500)
                if (isCancelled) return@withContext
            }

            val entry = FILE_MANIFEST[currentFileIndex]
            val destFile = File(modelsDirectory, entry.localPath)

            // Skip already-downloaded files
            if (destFile.exists() && destFile.length() > 0) {
                markComponentIfComplete(entry.component)
                currentFileIndex++
                saveProgress()
                continue
            }

            currentFileName = entry.localPath
            destFile.parentFile?.mkdirs()

            val success = downloadFile(
                url = "$BASE_URL/${entry.remotePath}",
                dest = destFile,
                onProgress = { written, total ->
                    val currBytes = bytesFromCompletedFiles + written
                    downloadedBytes = currBytes
                    totalBytes = if (total > 0) maxOf(totalBytes, bytesFromCompletedFiles + total) else totalBytes
                    overallProgress = downloadedBytes.toFloat() / totalBytes.toFloat()
                    updateSpeed(written)
                }
            )

            if (!success) {
                if (!isCancelled) {
                    state = State.Failed("Download failed for ${entry.localPath}")
                    saveProgress()
                }
                return@withContext
            }

            bytesFromCompletedFiles += destFile.length()
            markComponentIfComplete(entry.component)
            currentFileIndex++
            saveProgress()
        }

        // All files done
        if (!isCancelled) {
            overallProgress = 1f
            state = State.Completed
            modelsReady = true
            cleanupProgress()
            Log.i(TAG, "All models downloaded successfully")
        }
    }

    /**
     * Download a single file with progress callback.
     * Supports resuming via Range header if partial file exists.
     */
    private fun downloadFile(
        url: String,
        dest: File,
        onProgress: (written: Long, total: Long) -> Unit
    ): Boolean {
        return try {
            val partFile = File("${dest.absolutePath}.part")
            val existingBytes = if (partFile.exists()) partFile.length() else 0L

            val requestBuilder = Request.Builder().url(url)
            if (existingBytes > 0) {
                requestBuilder.header("Range", "bytes=$existingBytes-")
                Log.d(TAG, "Resuming ${dest.name} from byte $existingBytes")
            }

            val response = okHttpClient.newCall(requestBuilder.build()).execute()
            if (!response.isSuccessful && response.code != 206) {
                Log.e(TAG, "HTTP ${response.code} for $url")
                return false
            }

            val contentLength = response.header("Content-Length")?.toLongOrNull() ?: -1L
            val body = response.body ?: return false

            val mode = if (existingBytes > 0 && response.code == 206) "a" else "w"
            FileOutputStream(partFile, mode == "a").use { fos ->
                val buf = ByteArray(8192)
                var written = existingBytes
                var bytesRead: Int
                val stream = body.byteStream()

                while (stream.read(buf).also { bytesRead = it } != -1) {
                    if (isCancelled || isPaused) {
                        fos.flush()
                        return false
                    }
                    fos.write(buf, 0, bytesRead)
                    written += bytesRead
                    onProgress(written, if (contentLength > 0) existingBytes + contentLength else -1L)
                }
            }

            partFile.renameTo(dest)
            Log.d(TAG, "Downloaded ${dest.name} (${dest.length() / 1_000_000} MB)")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Download error for ${dest.name}: ${e.message}")
            false
        }
    }

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    private fun markComponentIfComplete(component: String) {
        val componentFiles = FILE_MANIFEST.filter { it.component == component }
        val allExist = componentFiles.all { File(modelsDirectory, it.localPath).exists() }
        if (allExist) {
            completedComponents = completedComponents + component
        }
    }

    private fun updateSpeed(totalBytesWritten: Long) {
        val now = System.currentTimeMillis()
        speedSamples.addLast(Pair(now, totalBytesWritten))
        val cutoff = now - 5000L
        while (speedSamples.isNotEmpty() && speedSamples.first().first < cutoff) {
            speedSamples.removeFirst()
        }
        if (speedSamples.size >= 2) {
            val first = speedSamples.first()
            val last = speedSamples.last()
            val elapsed = (last.first - first.first).coerceAtLeast(1L)
            val bytes = speedSamples.sumOf { it.second }
            downloadSpeedBytesPerSec = bytes * 1000L / elapsed
            val remaining = totalBytes - downloadedBytes
            etaSeconds = if (downloadSpeedBytesPerSec > 0) remaining / downloadSpeedBytesPerSec else 0L
        }
    }

    // ---------------------------------------------------------------------------
    // Progress persistence — mirrors iOS saveProgress()/restoreProgress()
    // ---------------------------------------------------------------------------

    private fun saveProgress() {
        progressPrefs.edit()
            .putInt("fileIndex", currentFileIndex)
            .putLong("bytesFromCompleted", bytesFromCompletedFiles)
            .apply()
    }

    private fun restoreProgress() {
        currentFileIndex = progressPrefs.getInt("fileIndex", 0)
        bytesFromCompletedFiles = progressPrefs.getLong("bytesFromCompleted", 0L)
        downloadedBytes = bytesFromCompletedFiles
        if (currentFileIndex > 0 && currentFileIndex < FILE_MANIFEST.size) {
            state = State.Paused
        }
    }

    private fun cleanupProgress() {
        progressPrefs.edit().clear().apply()
    }
}

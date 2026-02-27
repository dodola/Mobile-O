package com.mobileo.viewmodel

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.mobileo.data.Conversation
import com.mobileo.data.MessageContent
import com.mobileo.data.MessageMetrics
import com.mobileo.data.Role
import com.mobileo.ml.DPMSolverScheduler
import com.mobileo.ml.ImageUnderstandingModel
import com.mobileo.ml.MobileOGenerator
import com.mobileo.ml.OnnxSessionManager
import com.mobileo.ml.TextChatModel
import com.mobileo.ml.Tokenizer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Root ViewModel coordinating all three inference sub-models.
 * Mirrors iOS ChatViewModel.swift + ContentView model loading.
 *
 * Lifecycle:
 *   1. Created by MainActivity
 *   2. loadModels() called once with models directory
 *   3. sendMessage() dispatches to the correct sub-pipeline
 */
class ChatViewModel(private val context: Context) : ViewModel() {

    companion object {
        private const val TAG = "ChatViewModel"
    }

    // ---------------------------------------------------------------------------
    // Observable state (Compose reads these via State delegation)
    // ---------------------------------------------------------------------------

    var modelsLoading: Boolean by mutableStateOf(true)
        private set
    var modelLoadError: String? by mutableStateOf(null)
        private set
    var isGenerating: Boolean by mutableStateOf(false)
        private set
    var generationProgress: Pair<Int, Int>? by mutableStateOf(null)  // (step, total)

    val conversation = Conversation()

    // ---------------------------------------------------------------------------
    // Sub-models (mirrors iOS generationModel / understandingModel / textChatModel)
    // ---------------------------------------------------------------------------

    private var sessions: OnnxSessionManager? = null
    private var tokenizer: Tokenizer? = null
    private var generator: MobileOGenerator? = null
    private var understandingModel: ImageUnderstandingModel? = null
    private var textChatModel: TextChatModel? = null

    private var currentJob: Job? = null

    // ---------------------------------------------------------------------------
    // Model Loading
    // ---------------------------------------------------------------------------

    /**
     * Load all ONNX sessions, tokenizer and sub-models.
     * Mirrors iOS ContentView.loadModels().
     */
    fun loadModels(modelsDirectory: File) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                Log.i(TAG, "Loading models from ${modelsDirectory.absolutePath}")

                val sess = OnnxSessionManager(context)
                sess.loadAll(modelsDirectory)

                val tok = Tokenizer(context)
                tok.load(File(modelsDirectory, "llm"))

                val sched = DPMSolverScheduler.fromContext(context)
                val gen = MobileOGenerator(context, sess, tok, sched)
                val understanding = ImageUnderstandingModel(sess, tok)
                val chat = TextChatModel(sess, tok)

                withContext(Dispatchers.Main) {
                    sessions = sess
                    tokenizer = tok
                    generator = gen
                    understandingModel = understanding
                    textChatModel = chat
                    modelsLoading = false
                    Log.i(TAG, "All models ready")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load models: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    modelLoadError = e.message
                    modelsLoading = false
                }
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Send Message — routing mirrors iOS ChatViewModel.sendMessage()
    // ---------------------------------------------------------------------------

    /**
     * Route a user message to the appropriate sub-pipeline.
     * Mirrors iOS ChatViewModel.sendMessage(prompt:selectedImage:...).
     */
    fun sendMessage(
        prompt: String,
        selectedImage: Bitmap?,
        numSteps: Int,
        guidanceScale: Float,
        enableCFG: Boolean,
        seed: Long?
    ) {
        val trimmed = prompt.trim()
        val lower = trimmed.lowercase()
        val isGenerate = lower == "generate" || lower.startsWith("generate ")
        val hasImage = selectedImage != null

        if (trimmed.isEmpty() && !hasImage) return

        currentJob = viewModelScope.launch {
            when {
                isGenerate -> handleImageGeneration(trimmed, numSteps, guidanceScale, enableCFG, seed)
                hasImage && selectedImage != null -> handleImageUnderstanding(trimmed, selectedImage)
                else -> handleTextChat(trimmed)
            }
        }
    }

    fun cancelGeneration() {
        currentJob?.cancel()
        currentJob = null
        isGenerating = false
        generationProgress = null
        conversation.isGenerating = false
    }

    // ---------------------------------------------------------------------------
    // Image Generation — mirrors iOS handleImageGeneration()
    // ---------------------------------------------------------------------------

    private suspend fun handleImageGeneration(
        prompt: String,
        numSteps: Int,
        guidanceScale: Float,
        enableCFG: Boolean,
        seed: Long?
    ) {
        val gen = generator ?: return
        val cleanPrompt = stripKeyword("generate", prompt) ?: return

        withContext(Dispatchers.Main) {
            conversation.addMessage(Role.USER, MessageContent.Text(prompt))
            conversation.isGenerating = true
            isGenerating = true
            generationProgress = Pair(0, numSteps)
        }

        try {
            val params = MobileOGenerator.GenerationParams(
                prompt = cleanPrompt,
                numSteps = numSteps,
                guidanceScale = guidanceScale,
                enableCFG = enableCFG,
                seed = seed,
                onProgress = { step, total ->
                    viewModelScope.launch(Dispatchers.Main) {
                        generationProgress = Pair(step, total)
                    }
                }
            )

            val (bitmap, timing) = gen.generate(params)

            withContext(Dispatchers.Main) {
                val metrics = MessageMetrics(
                    totalTime = timing.totalMs / 1000.0
                )
                conversation.addMessage(Role.ASSISTANT, MessageContent.Image(bitmap), metrics)
                Log.i(TAG, "Image generated in ${timing.totalMs}ms")
            }
        } catch (e: kotlinx.coroutines.CancellationException) {
            withContext(Dispatchers.Main) {
                conversation.addMessage(Role.ASSISTANT, MessageContent.Text("Generation interrupted."))
            }
        } catch (e: Exception) {
            Log.e(TAG, "Image generation failed: ${e.message}", e)
            withContext(Dispatchers.Main) {
                conversation.addMessage(Role.ASSISTANT, MessageContent.Text("Error: ${e.message}"))
            }
        } finally {
            withContext(Dispatchers.Main) {
                isGenerating = false
                generationProgress = null
                conversation.isGenerating = false
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Image Understanding — mirrors iOS handleImageUnderstanding()
    // ---------------------------------------------------------------------------

    private suspend fun handleImageUnderstanding(prompt: String, image: Bitmap) {
        val model = understandingModel ?: return

        withContext(Dispatchers.Main) {
            conversation.addMessage(Role.USER, MessageContent.TextWithImage(prompt, image))
            conversation.isGenerating = true
            isGenerating = true
            val placeholderId = conversation.addMessage(Role.ASSISTANT, MessageContent.Text("…"))
        }

        val startMs = System.currentTimeMillis()
        try {
            val response = model.understand(image, prompt)
            val totalMs = System.currentTimeMillis() - startMs

            withContext(Dispatchers.Main) {
                conversation.updateLastMessage(response)
                // Patch metrics onto last message
                val msgs = conversation.messages
                if (msgs.isNotEmpty()) {
                    val last = msgs.last()
                    conversation.messages[msgs.lastIndex] = last.copy(
                        metrics = MessageMetrics(totalTime = totalMs / 1000.0)
                    )
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Understanding failed: ${e.message}", e)
            withContext(Dispatchers.Main) {
                conversation.updateLastMessage("Error: ${e.message}")
            }
        } finally {
            withContext(Dispatchers.Main) {
                isGenerating = false
                conversation.isGenerating = false
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Text Chat — mirrors iOS handleTextChat()
    // ---------------------------------------------------------------------------

    private suspend fun handleTextChat(prompt: String) {
        val model = textChatModel ?: return

        val messages = withContext(Dispatchers.Main) {
            conversation.addMessage(Role.USER, MessageContent.Text(prompt))
            conversation.isGenerating = true
            isGenerating = true
            conversation.addMessage(Role.ASSISTANT, MessageContent.Text("…"))
            conversation.getMessagesForModel().dropLast(1)  // drop placeholder
        }

        val startMs = System.currentTimeMillis()
        try {
            val response = model.chat(messages)
            val totalMs = System.currentTimeMillis() - startMs

            withContext(Dispatchers.Main) {
                conversation.updateLastMessage(response)
                val msgs = conversation.messages
                if (msgs.isNotEmpty()) {
                    val last = msgs.last()
                    conversation.messages[msgs.lastIndex] = last.copy(
                        metrics = MessageMetrics(totalTime = totalMs / 1000.0)
                    )
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Chat failed: ${e.message}", e)
            withContext(Dispatchers.Main) {
                conversation.updateLastMessage("Error: ${e.message}")
            }
        } finally {
            withContext(Dispatchers.Main) {
                isGenerating = false
                conversation.isGenerating = false
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    /**
     * Strip command keyword prefix. Mirrors iOS ChatViewModel.stripKeyword().
     * Returns null if no content remains after stripping.
     */
    private fun stripKeyword(keyword: String, prompt: String): String? {
        val trimmed = prompt.trim()
        val lower = trimmed.lowercase()
        return when {
            lower.startsWith("$keyword ") -> {
                val clean = trimmed.drop(keyword.length + 1).trim()
                clean.ifEmpty { null }
            }
            lower == keyword -> null
            else -> trimmed.ifEmpty { null }
        }
    }

    override fun onCleared() {
        super.onCleared()
        sessions?.close()
    }

    // ---------------------------------------------------------------------------
    // Factory
    // ---------------------------------------------------------------------------

    class Factory(private val context: Context) : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>): T =
            ChatViewModel(context.applicationContext) as T
    }
}

package com.mobileo.data

import androidx.compose.runtime.mutableStateListOf
import java.util.UUID

/**
 * Ordered conversation history. Mirrors iOS Conversation.swift.
 *
 * Uses Compose SnapshotStateList so the UI automatically recomposes on changes.
 */
class Conversation {

    val messages = mutableStateListOf<ChatMessage>()
    var isGenerating: Boolean = false

    val isEmpty: Boolean get() = messages.isEmpty()

    /**
     * Append a new message and return its ID.
     * Mirrors iOS `addMessage(role:content:totalTime:) -> UUID`.
     */
    fun addMessage(role: Role, content: MessageContent, metrics: MessageMetrics? = null): UUID {
        val msg = ChatMessage(role = role, content = content, metrics = metrics)
        messages.add(msg)
        return msg.id
    }

    /**
     * Replace the text content of the last assistant message.
     * Used for streaming/progressive text updates.
     * Mirrors iOS `updateLastMessage(text:)`.
     */
    fun updateLastMessage(text: String) {
        val lastIndex = messages.indexOfLast { it.role == Role.ASSISTANT }
        if (lastIndex < 0) return
        val old = messages[lastIndex]
        messages[lastIndex] = old.copy(content = MessageContent.Text(text))
    }

    /**
     * Attach performance metrics to a specific message by ID.
     * Mirrors iOS `updateMessageMetrics(id:timeToFirstToken:totalTime:tokensPerSecond:)`.
     */
    fun updateMessageMetrics(
        id: UUID,
        timeToFirstToken: Double,
        totalTime: Double,
        tokensPerSecond: Double
    ) {
        val idx = messages.indexOfFirst { it.id == id }
        if (idx < 0) return
        messages[idx] = messages[idx].copy(
            metrics = MessageMetrics(
                timeToFirstToken = timeToFirstToken,
                totalTime = totalTime,
                tokensPerSecond = tokensPerSecond
            )
        )
    }

    /**
     * Build the message history list for the LLM (system + user/assistant turns).
     * Mirrors iOS `getMessagesForModel()`.
     */
    fun getMessagesForModel(): List<Map<String, String>> {
        val result = mutableListOf<Map<String, String>>()
        result.add(mapOf("role" to "system", "content" to "You are a helpful assistant."))
        for (msg in messages) {
            val role = if (msg.role == Role.USER) "user" else "assistant"
            val text = msg.textContent ?: continue
            if (text.isNotBlank()) {
                result.add(mapOf("role" to role, "content" to text))
            }
        }
        return result
    }

    fun clear() {
        messages.clear()
        isGenerating = false
    }
}

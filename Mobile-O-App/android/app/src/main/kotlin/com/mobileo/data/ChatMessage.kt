package com.mobileo.data

import android.graphics.Bitmap
import java.util.UUID

/**
 * Role of a chat participant. Mirrors iOS SeparatorStyle / conversation role.
 */
enum class Role {
    USER, ASSISTANT
}

/**
 * Content variants for a chat message — mirrors iOS ChatMessage.Content enum.
 */
sealed class MessageContent {
    data class Text(val text: String) : MessageContent()
    data class Image(val bitmap: Bitmap) : MessageContent()
    data class TextWithImage(val text: String, val bitmap: Bitmap) : MessageContent()
}

/**
 * Timing/performance metrics attached to assistant messages.
 * Mirrors iOS ChatMessage's timing fields.
 */
data class MessageMetrics(
    val timeToFirstToken: Double = 0.0,  // seconds
    val totalTime: Double = 0.0,          // seconds
    val tokensPerSecond: Double = 0.0
)

/**
 * A single chat message. Mirrors iOS ChatMessage.swift.
 */
data class ChatMessage(
    val id: UUID = UUID.randomUUID(),
    val role: Role,
    val content: MessageContent,
    val metrics: MessageMetrics? = null
) {
    /** Convenience: true if this message contains an image result (generated). */
    val isGeneratedImage: Boolean
        get() = content is MessageContent.Image && role == Role.ASSISTANT

    /** Convenience: extract text string if content is text-based. */
    val textContent: String?
        get() = when (content) {
            is MessageContent.Text -> content.text
            is MessageContent.TextWithImage -> content.text
            else -> null
        }
}

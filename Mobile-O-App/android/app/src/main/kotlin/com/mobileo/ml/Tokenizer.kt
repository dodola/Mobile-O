package com.mobileo.ml

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.File

/**
 * BPE tokenizer for Qwen2 that reads tokenizer.json / vocab.json / merges.txt.
 *
 * Replaces HuggingFace swift-transformers `Tokenizer` on iOS.
 *
 * Supports:
 *  - Byte-pair encoding (BPE) with Qwen2's byte-level pre-tokenization
 *  - ChatML conversation formatting
 *  - Encoding / decoding
 *
 * The tokenizer files are loaded from the downloaded LLM directory:
 *   llm/tokenizer.json   — full HuggingFace tokenizer spec
 *   llm/vocab.json       — token → id mapping
 *   llm/merges.txt       — BPE merge rules
 */
class Tokenizer(private val context: Context) {

    companion object {
        private const val TAG = "Tokenizer"
        private const val MAX_TOKEN_LENGTH = 512

        // Special tokens (Qwen2 / ChatML)
        const val IM_START = "<|im_start|>"
        const val IM_END = "<|im_end|>"
        const val IM_START_LEGACY = "<im_start>"
    }

    private val vocab = mutableMapOf<String, Int>()          // token string → id
    private val idToToken = mutableMapOf<Int, String>()      // id → token string
    private val merges = mutableListOf<Pair<String, String>>() // BPE merge rules
    private val mergeRanks = mutableMapOf<Pair<String, String>, Int>()
    private val byteEncoder = buildByteEncoder()             // byte → unicode char
    private val byteDecoder = byteEncoder.entries.associate { (k, v) -> v to k }

    var isLoaded = false
        private set

    // Encode special tokens as added tokens (not split by BPE)
    private val specialTokens = mutableMapOf<String, Int>()

    /**
     * Load tokenizer files from the LLM directory.
     * Call from Dispatchers.IO.
     */
    fun load(llmDir: File) {
        loadVocab(File(llmDir, "vocab.json"))
        loadMerges(File(llmDir, "merges.txt"))
        loadSpecialTokens(File(llmDir, "tokenizer.json"))
        isLoaded = true
        Log.i(TAG, "Tokenizer loaded: ${vocab.size} tokens, ${merges.size} merges")
    }

    // ---------------------------------------------------------------------------
    // Public API — mirrors swift-transformers Tokenizer
    // ---------------------------------------------------------------------------

    /**
     * Encode a string to token IDs. Mirrors iOS `tokenizer.encode(text:)`.
     * Truncates to MAX_TOKEN_LENGTH.
     */
    fun encode(text: String): IntArray {
        val tokens = tokenize(text)
        val ids = tokens.mapNotNull { vocab[it] ?: specialTokens[it] }
        return ids.take(MAX_TOKEN_LENGTH).toIntArray()
    }

    /**
     * Decode token IDs back to a string. Mirrors iOS `tokenizer.decode(tokens:)`.
     */
    fun decode(ids: IntArray): String {
        val tokens = ids.toList().mapNotNull { id ->
            idToToken[id] ?: specialTokens.entries.find { e -> e.value == id }?.key
        }
        return decodeTokens(tokens)
    }

    /**
     * Format messages into ChatML template and encode.
     * Mirrors iOS MobileOGenerator.formatChatTemplate() + tokenizer.encode().
     *
     * For generation:
     *   <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
     *   <|im_start|>user\nPlease generate image based on the following caption: {prompt}<|im_end|>\n
     *   <|im_start|>assistant\n<im_start>
     */
    fun encodeGenerationPrompt(prompt: String): IntArray {
        val text = buildString {
            append("$IM_START system\nYou are a helpful assistant.$IM_END\n")
            append("$IM_START user\nPlease generate image based on the following caption: $prompt$IM_END\n")
            append("$IM_START assistant\n$IM_START_LEGACY")
        }
        return encode(text)
    }

    /**
     * Format a conversation history as ChatML and encode.
     * Used for text chat (mirrors iOS TextChatModel).
     */
    fun encodeChatMessages(messages: List<Map<String, String>>): IntArray {
        val text = buildString {
            for (msg in messages) {
                val role = msg["role"] ?: continue
                val content = msg["content"] ?: continue
                append("$IM_START $role\n$content$IM_END\n")
            }
            append("$IM_START assistant\n")
        }
        return encode(text)
    }

    // ---------------------------------------------------------------------------
    // Loading
    // ---------------------------------------------------------------------------

    private fun loadVocab(file: File) {
        if (!file.exists()) { Log.w(TAG, "vocab.json not found: ${file.path}"); return }
        val json = JSONObject(file.readText())
        val keys = json.keys()
        while (keys.hasNext()) {
            val token = keys.next()
            val id = json.getInt(token)
            vocab[token] = id
            idToToken[id] = token
        }
    }

    private fun loadMerges(file: File) {
        if (!file.exists()) { Log.w(TAG, "merges.txt not found: ${file.path}"); return }
        var rank = 0
        file.forEachLine { line ->
            if (line.startsWith("#") || line.isBlank()) return@forEachLine
            val parts = line.trim().split(" ")
            if (parts.size == 2) {
                val pair = Pair(parts[0], parts[1])
                merges.add(pair)
                mergeRanks[pair] = rank++
            }
        }
    }

    private fun loadSpecialTokens(file: File) {
        if (!file.exists()) return
        try {
            val json = JSONObject(file.readText())
            val addedTokens = json.optJSONArray("added_tokens") ?: return
            for (i in 0 until addedTokens.length()) {
                val obj = addedTokens.getJSONObject(i)
                val content = obj.getString("content")
                val id = obj.getInt("id")
                specialTokens[content] = id
                idToToken[id] = content
            }
        } catch (e: Exception) {
            Log.w(TAG, "Could not parse special tokens: ${e.message}")
        }
    }

    // ---------------------------------------------------------------------------
    // BPE tokenization
    // ---------------------------------------------------------------------------

    private fun tokenize(text: String): List<String> {
        if (text.isEmpty()) return emptyList()

        // Split on special tokens first
        val segments = splitOnSpecialTokens(text)
        val result = mutableListOf<String>()

        for ((segment, isSpecial) in segments) {
            if (isSpecial) {
                result.add(segment)
            } else {
                // Byte-level pre-tokenization: split on whitespace/punctuation boundaries
                val words = preTokenize(segment)
                for (word in words) {
                    result.addAll(bpeMerge(word))
                }
            }
        }
        return result
    }

    /**
     * Split text into (segment, isSpecialToken) pairs.
     */
    private fun splitOnSpecialTokens(text: String): List<Pair<String, Boolean>> {
        val sortedSpecials = specialTokens.keys.sortedByDescending { it.length }
        val result = mutableListOf<Pair<String, Boolean>>()
        var remaining = text

        while (remaining.isNotEmpty()) {
            val matched = sortedSpecials.firstOrNull { remaining.startsWith(it) }
            if (matched != null) {
                result.add(Pair(matched, true))
                remaining = remaining.substring(matched.length)
            } else {
                // Find next special token position
                val nextSpecialPos = sortedSpecials
                    .mapNotNull { s -> remaining.indexOf(s).takeIf { it > 0 } }
                    .minOrNull() ?: remaining.length
                result.add(Pair(remaining.substring(0, nextSpecialPos), false))
                remaining = remaining.substring(nextSpecialPos)
            }
        }
        return result
    }

    /**
     * Qwen2-style whitespace/word pre-tokenization.
     * Converts each character to byte-level representation.
     */
    private fun preTokenize(text: String): List<String> {
        // Simple approach: split by whitespace, prepend Ġ (GPT-2 style) for non-first words
        val words = mutableListOf<String>()
        val pattern = Regex("""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+""")
        for (match in pattern.findAll(text)) {
            val word = match.value
            val byteStr = word.toByteArray(Charsets.UTF_8).joinToString("") { byteEncoder[it.toInt() and 0xFF] ?: "?" }
            words.add(byteStr)
        }
        return words
    }

    /**
     * Apply BPE merges to a word (byte-level string).
     */
    private fun bpeMerge(word: String): List<String> {
        if (word.length <= 1) return listOf(word)

        // Start with individual characters
        var symbols = word.map { it.toString() }.toMutableList()

        while (symbols.size > 1) {
            // Find lowest-rank merge pair
            var bestRank = Int.MAX_VALUE
            var bestIdx = -1

            for (i in 0 until symbols.size - 1) {
                val pair = Pair(symbols[i], symbols[i + 1])
                val rank = mergeRanks[pair] ?: Int.MAX_VALUE
                if (rank < bestRank) {
                    bestRank = rank
                    bestIdx = i
                }
            }

            if (bestIdx < 0 || bestRank == Int.MAX_VALUE) break

            // Merge at bestIdx
            val merged = symbols[bestIdx] + symbols[bestIdx + 1]
            symbols[bestIdx] = merged
            symbols.removeAt(bestIdx + 1)
        }

        return symbols
    }

    private fun decodeTokens(tokens: List<String>): String {
        val sb = StringBuilder()
        for (token in tokens) {
            // Each char in token is a byte-encoded unicode char — decode back
            for (ch in token) {
                val byte = byteDecoder[ch.toString()]
                if (byte != null) {
                    sb.append(byte.toChar())
                } else {
                    sb.append(ch)
                }
            }
        }
        return try {
            sb.toString().toByteArray(Charsets.ISO_8859_1).toString(Charsets.UTF_8)
        } catch (e: Exception) {
            sb.toString()
        }
    }

    // ---------------------------------------------------------------------------
    // Byte encoder/decoder — GPT-2 style, used by Qwen2
    // Maps byte values 0-255 to unicode characters that are valid vocab entries
    // ---------------------------------------------------------------------------

    private fun buildByteEncoder(): Map<Int, String> {
        val bs = mutableListOf<Int>()
        // Printable ASCII: 33-126, 161-172, 174-255
        for (b in '!'.code..'~'.code) bs.add(b)
        for (b in '¡'.code..'¬'.code) bs.add(b)
        for (b in '®'.code..'ÿ'.code) bs.add(b)

        val cs = bs.map { it }.toMutableList()
        var n = 0
        for (b in 0..255) {
            if (b !in bs) {
                bs.add(b)
                cs.add(256 + n)
                n++
            }
        }
        return bs.zip(cs.map { it.toChar().toString() }).toMap()
    }
}

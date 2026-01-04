# Add RAG Documents Support to Gemini Provider

## Summary

This PR adds support for RAG (Retrieval-Augmented Generation) documents in the Gemini provider's `dynamic_context()` functionality. Previously, documents retrieved from vector stores were silently ignored, causing RAG agents to fail.

## Problem

When using `.dynamic_context()` with Gemini agents, the retrieved documents were never passed to the Gemini API. The `create_request_body()` function in the Gemini provider ignored the `documents` field from `CompletionRequest`, making it impossible to use RAG with Gemini.

## Solution

### Changes Made

1. **Document Injection**: Modified `create_request_body()` to check for documents and inject them as a user message at the beginning of chat history using the existing `normalized_documents()` helper method.

2. **Text Document Handling**: Added special handling for `DocumentMediaType::TXT` documents to convert them to plain text parts (`PartKind::Text`) instead of trying to send them as base64-encoded inline data.

3. **Backward Compatibility**: Preserved existing behavior for other document types (images, PDFs, etc.) that use `InlineData` or `FileData`.

### Code Changes

File: `rig/rig-core/src/providers/gemini/completion.rs`

- Lines ~189-195: Added document injection logic
- Lines ~773-838: Added TXT document to text conversion

## Testing

Tested with a real-world RAG application:
- **Application**: Obsidian vault note organizer with 175+ markdown notes
- **Vector Store**: In-memory vector store with Ollama embeddings (nomic-embed-text)
- **Query**: "Tell me about mum canada visa"
- **Result**: Successfully retrieved 10 relevant notes and Gemini correctly received and processed them

### Before Fix
```
=== NOTES IN CONTEXT ===
NO NOTES IN CONTEXT
=== END OF NOTES ===
```

### After Fix
```
=== NOTES IN CONTEXT ===
markdown-cheat-sheet.md
Open guard passing.md
Drills.md
eink_calendar.md
Submissions from the side.md
Post-parental leave sync up.md
LP SideNav migration Overview.md
Sub defence.md
next month spending.md
mum canada visa.md
=== END OF NOTES ===

The note "mum canada visa.md" contains the following information:
[... content from note ...]
```

## Impact

- ✅ Enables RAG functionality for Gemini provider
- ✅ No breaking changes
- ✅ Maintains backward compatibility
- ✅ Follows existing patterns (uses `normalized_documents()`)

## Additional Notes

The fix aligns Gemini's behavior with other providers like Cohere and Anthropic that already support documents in RAG scenarios.

import { JSONObject } from "./types";

// --- decodeReadableStream function ---
export async function* decodeReadableStream(
  stream: ReadableStream<unknown>,
): AsyncGenerator<RawStreamingChoice> {
  const reader = stream.getReader();

  try {
    while (true) {
      const { done, value } = await reader.read();

      if (done) break;
      const chunk = value as RawStreamingChoice;

      yield chunk;
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * A delta from a streamed completion response.
 * Typically, you may see it used like this:
 *
 * @example
 * const result = await decodeReadableStream<StreamingCompletionResponse>();
 *
 */
export type RawStreamingChoice = Text | ToolCallChoice | { usage: number };

export interface Text {
  text: string;
}
/**
 * A tool call choice.
 * This may be returned as a result of a streamed completion response.
 */
export interface ToolCallChoice {
  id: string;
  call_id?: string;
  function: ToolCallFunction;
}

export interface ToolCallFunction {
  name: string;
  arguments: JSONObject;
}

async function processStream(stream: ReadableStream<unknown>) {}

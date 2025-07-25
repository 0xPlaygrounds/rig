export {
  OpenAIAgent as Agent,
  OpenAIEmbeddingModel as EmbeddingModel,
  OpenAIResponsesCompletionModel as ResponsesCompletionModel,
  OpenAICompletionsCompletionModel as CompletionsCompletionModel,
  OpenAITranscriptionModel as TranscriptionModel,
  OpenAIImageGenerationModel as ImageGenerationModel,
} from "../generated/rig_wasm.js";

/**
 * The OpenAI Responses API final response (adapted from the original Rig return type).
 */
export interface ResponsesStreamingCompletionResponse {
  usage: ResponsesUsage;
}

export interface ResponsesUsage {
  /** Total number of input tokens */
  input_tokens: number;

  /** Optional details on input tokens (e.g., cached tokens) */
  input_tokens_details?: InputTokensDetails;

  /** Total number of output tokens */
  output_tokens: number;

  /** Detailed breakdown of output tokens (e.g., reasoning) */
  output_tokens_details: OutputTokensDetails;

  /** Sum of input + output tokens */
  total_tokens: number;
}

export interface InputTokensDetails {
  cached_tokens: number;
}

export interface OutputTokensDetails {
  reasoning_tokens: number;
}

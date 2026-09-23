//! Registered OpenAI-compatible dialects and their endpoint policies.
//!
//! ```
//! use rig_core::providers::openai::wire::by_name;
//! assert!(by_name("deepseek").is_some());
//! ```

use super::{
    AcceptedWidths, Auth, AuthAlternative, BodyRewrite, Dialect, DimensionsField, EmbeddingQuirks,
    ImageBody, ModelWidth, OutputCap, Quirks, RerankQuirks, ResponsesQuirks, Route, Routing,
    SpeechBody, SystemInstructionsPlacement, TranscriptionBody,
};

/// Azure reads its API version from the environment because every Azure
/// route carries it and no default addresses the right API.
pub(super) const AZURE_API_VERSION_ENV: &str = "AZURE_API_VERSION";

/// Default Azure API version when no environment override is read.
pub const AZURE_DEFAULT_API_VERSION: &str = "2024-10-21";

/// Azure versions its speech endpoint separately from the rest.
pub(super) const AZURE_AUDIO_API_VERSION_ENV: &str = "AZURE_AUDIO_API_VERSION";

/// Default Azure speech API version.
pub(super) const AZURE_DEFAULT_AUDIO_API_VERSION: &str = "2025-04-01-preview";

/// Official OpenAI: the Responses endpoint by default, Chat Completions
/// beside it.
pub const OPENAI: Dialect = Dialect {
    base_url_env: Some("OPENAI_BASE_URL"),
    request_id_header: Some("x-request-id"),
    quirks: Quirks {
        completion_route: Route::Responses,
        // OpenAI's reasoning families take `max_completion_tokens`; every
        // compatible gateway on this wire takes `max_tokens`, which is the
        // baseline.
        output_cap: OutputCap::OpenAiReasoningFamilies,
        ..Quirks::openai()
    },
    ..Dialect::gateway("openai", "https://api.openai.com/v1", "OPENAI_API_KEY")
};

/// Azure OpenAI: the deployment is in the URL, the API version is a query
/// parameter, and the credential is an `api-key` header.
pub const AZURE: Dialect = Dialect {
    base_url_env: Some("AZURE_ENDPOINT"),
    // Azure tokens require bearer authentication rather than the api-key header.
    alternate_auth: Some(AuthAlternative {
        api_key_env: "AZURE_TOKEN",
        auth: Auth::Bearer,
    }),
    quirks: Quirks {
        auth: Auth::ApiKeyHeader,
        routing: Routing::AzureDeployment,
        // Verifying a credential without deploying and spending tokens is
        // not offered.
        verify_path: "",
        transcription_path: "/audio/translations",
        embedding: EmbeddingQuirks {
            // The deployment is in the URL, so the body carries no model.
            sends_model_field: false,
            ..EmbeddingQuirks::openai()
        },
        ..Quirks::openai()
    },
    // The base URL is the account's own resource endpoint; there is no
    // shared host, so the dialect names none.
    ..Dialect::gateway("azure.openai", "", "AZURE_API_KEY")
};

/// DeepSeek.
pub const DEEPSEEK: Dialect = Dialect {
    quirks: Quirks {
        // DeepSeek accepts json_object through additional_params, not json_schema.
        supports_response_format: false,
        emits_complete_single_chunk_tool_calls: true,
        verify_path: "/user/balance",
        rewrite: BodyRewrite::DeepSeek,
        ..Quirks::openai()
    },
    ..Dialect::gateway("deepseek", "https://api.deepseek.com", "DEEPSEEK_API_KEY")
};

/// Groq.
pub const GROQ: Dialect = Dialect {
    request_id_header: Some("x-request-id"),
    quirks: Quirks {
        emits_complete_single_chunk_tool_calls: true,
        rewrite: BodyRewrite::GroqCompoundTools,
        ..Quirks::openai()
    },
    ..Dialect::gateway("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY")
};

/// Hyperbolic.
pub const HYPERBOLIC: Dialect = Dialect {
    quirks: Quirks {
        // Hyperbolic does not support tool calling, and its
        // structured-output support is unverified.
        supports_tools: false,
        supports_response_format: false,
        completion_path: "/v1/chat/completions",
        embeddings_path: "/v1/embeddings",
        models_path: "/v1/models",
        verify_path: "/models",
        image_generation_path: "/v1/image/generation",
        audio_generation_path: "/v1/audio/generation",
        image_body: ImageBody::Hyperbolic,
        speech_body: SpeechBody::Hyperbolic,
        rewrite: BodyRewrite::Hyperbolic,
        ..Quirks::openai()
    },
    // The bare host: the chat path carries its own `/v1`.
    ..Dialect::gateway(
        "hyperbolic",
        "https://api.hyperbolic.xyz",
        "HYPERBOLIC_API_KEY",
    )
};

/// Mira's gateway.
pub const MIRA: Dialect = Dialect {
    quirks: Quirks {
        // The gateway rejects tool parameters, OpenAI structured-output
        // parameters, and unknown parameters such as `stream_options`.
        supports_tools: false,
        supports_response_format: false,
        stream_include_usage: false,
        completion_path: "/v1/chat/completions",
        models_path: "/v1/models",
        verify_path: "/user-credits",
        // The gateway can answer with a bare JSON string.
        accepts_bare_string_reply: true,
        rewrite: BodyRewrite::Mira,
        ..Quirks::openai()
    },
    ..Dialect::gateway("mira", "https://api.mira.network", "MIRA_API_KEY")
};

/// Perplexity.
pub const PERPLEXITY: Dialect = Dialect {
    quirks: Quirks {
        supports_tools: false,
        supports_response_format: false,
        stream_include_usage: false,
        // No endpoint checks a credential without spending tokens.
        verify_path: "",
        rewrite: BodyRewrite::Perplexity,
        ..Quirks::openai()
    },
    ..Dialect::gateway(
        "perplexity",
        "https://api.perplexity.ai",
        "PERPLEXITY_API_KEY",
    )
};

/// Together AI.
pub const TOGETHER: Dialect = Dialect {
    quirks: Quirks {
        // Structured-output support is per model on Together, so the schema
        // is dropped with a warning rather than sent and rejected.
        supports_response_format: false,
        completion_path: "/v1/chat/completions",
        embeddings_path: "/v1/embeddings",
        models_path: "/v1/models",
        verify_path: "/models",
        embedding: EmbeddingQuirks {
            requires_usage: false,
            supports_encoding_format: false,
            supports_user: false,
            ..EmbeddingQuirks::openai()
        },
        ..Quirks::openai()
    },
    // The bare host: every path carries its own `/v1`.
    ..Dialect::gateway("together", "https://api.together.xyz", "TOGETHER_API_KEY")
};

/// Hugging Face's inference router.
pub const HUGGINGFACE: Dialect = Dialect {
    quirks: Quirks {
        supports_response_format: false,
        // Chat lives under the router's `/v1`; verification, transcription
        // and image generation are root-relative, so the prefix cannot live
        // in the base URL.
        completion_path: "/v1/chat/completions",
        verify_path: "/api/whoami-v2",
        // Transcription and image generation address `/{model}` at the
        // router root, not a fixed path under `/v1`.
        model_is_modality_path: true,
        // The router's image endpoint takes none of OpenAI's fields and
        // answers with the image bytes rather than a JSON envelope.
        image_body: ImageBody::HuggingFace,
        rewrite: BodyRewrite::HuggingFaceRouter,
        ..Quirks::openai()
    },
    ..Dialect::gateway(
        "huggingface",
        "https://router.huggingface.co",
        "HUGGINGFACE_API_KEY",
    )
};

/// A local `llama-server`.
pub const LLAMACPP: Dialect = Dialect {
    base_url_env: Some("LLAMACPP_API_BASE_URL"),
    quirks: Quirks {
        // A server started without `--api-key` rejects a request that
        // carries an `Authorization` header.
        auth: Auth::OptionalBearer,
        supports_image_tool_results: true,
        verify_path: "/props",
        // These management routes live at the server root, outside /v1.
        root_relative_routes: &[
            "/props",
            "/health",
            "/slots",
            "/metrics",
            "/tokenize",
            "/detokenize",
            "/apply-template",
            "/infill",
            "/lora-adapters",
        ],
        rewrite: BodyRewrite::LlamaCpp,
        rerank: RerankQuirks {
            // `llama-server` serves one rerank handler behind four aliases
            // (`/rerank`, `/reranking`, `/v1/rerank`, `/v1/reranking`); the
            // base URL already carries `/v1`.
            path: "/rerank",
            // This is a batching hint; the server documents no document-count cap.
            max_documents: 1024,
            sends_model_field: true,
        },
        embedding: EmbeddingQuirks {
            // `llama-server`'s embeddings handler reads no width field, so
            // sending one would leave `ndims()` describing vectors the
            // server never returned.
            dimensions: DimensionsField::Ignored,
            ..EmbeddingQuirks::openai()
        },
        ..Quirks::openai()
    },
    ..Dialect::gateway("llamacpp", "http://localhost:8080/v1", "LLAMACPP_API_KEY")
};

/// The width contract of Mistral's embedding models.
///
/// `mistral-embed` is fixed at 1024 and reads no width field: Mistral
/// answers any other value with an error rather than truncating, so a
/// request naming one is refused before it is built. Codestral Embed is
/// configurable up to 3072 and takes its width as `output_dimension`.
///
/// The dated aliases are listed beside their rolling names because a caller
/// pinning `mistral-embed-2312` gets the same model, and a model absent from
/// this table reports `ndims() == 0`.
const MISTRAL_EMBEDDING_WIDTHS: &[ModelWidth] = &[
    ModelWidth {
        model: crate::providers::mistral::embedding::MISTRAL_EMBED,
        default: Some(1_024),
        accepted: AcceptedWidths::Fixed,
    },
    ModelWidth {
        model: "mistral-embed-2312",
        default: Some(1_024),
        accepted: AcceptedWidths::Fixed,
    },
    ModelWidth {
        model: crate::providers::mistral::embedding::CODESTRAL_EMBED,
        // Configurable with no documented native width, so a handle that
        // names none reports 0 rather than inventing one.
        default: None,
        accepted: AcceptedWidths::Range {
            // Mistral documents only a ceiling. The floor is rig's own
            // "unknown" sentinel, which never reaches the wire.
            min: 0,
            max: 3_072,
            requirement: "to be at most 3072 for Codestral Embed",
        },
    },
    ModelWidth {
        model: "codestral-embed-2505",
        default: None,
        accepted: AcceptedWidths::Range {
            min: 0,
            max: 3_072,
            requirement: "to be at most 3072 for Codestral Embed",
        },
    },
];

/// Mistral.
pub const MISTRAL: Dialect = Dialect {
    request_id_header: Some("mistral-correlation-id"),
    quirks: Quirks {
        // Mistral rejects `stream_options` and reports usage on its final
        // chunk regardless.
        stream_include_usage: false,
        emits_complete_single_chunk_tool_calls: true,
        completion_path: "/v1/chat/completions",
        embeddings_path: "/v1/embeddings",
        models_path: "/v1/models",
        verify_path: "/v1/models",
        transcription_path: "/v1/audio/transcriptions",
        rewrite: BodyRewrite::Mistral,
        embedding: EmbeddingQuirks {
            max_documents: 256,
            supports_user: false,
            // Codestral Embed takes its width as `output_dimension`.
            dimensions: DimensionsField::OutputDimension,
            widths: MISTRAL_EMBEDDING_WIDTHS,
            ..EmbeddingQuirks::openai()
        },
        ..Quirks::openai()
    },
    // The bare host: every path carries its own `/v1`.
    ..Dialect::gateway("mistral", "https://api.mistral.ai", "MISTRAL_API_KEY")
};

/// OpenRouter.
pub const OPENROUTER: Dialect = Dialect {
    quirks: Quirks {
        stream_include_usage: false,
        verify_path: "/key",
        // A gateway forwards its upstream's own finish reason and its own
        // reasoning blobs.
        native_finish_reason: true,
        reasoning_details: true,
        upstream_reasoning_issuer: true,
        response_format_with_tools: true,
        accepts_file_ids: false,
        rewrite: BodyRewrite::OpenRouter,
        // Its speech-to-text route takes the audio base64 in a JSON body,
        // not a multipart upload.
        transcription_body: TranscriptionBody::InputAudioJson,
        embedding: EmbeddingQuirks {
            requires_usage: false,
            ..EmbeddingQuirks::openai()
        },
        // The Responses route accepts system instructions as input messages.
        responses: ResponsesQuirks {
            system_instructions: SystemInstructionsPlacement::InputSystemMessages,
            ..ResponsesQuirks::openai()
        },
        ..Quirks::openai()
    },
    ..Dialect::gateway(
        "openrouter",
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
    )
};

/// Venice.
pub const VENICE: Dialect = Dialect {
    base_url_env: Some("VENICE_BASE_URL"),
    quirks: Quirks {
        image_generation_path: "/image/generate",
        // Its own endpoint, so its own body and its own reply: `width`/
        // `height` rather than OpenAI's `size`, and `images` holding the
        // base64 payloads themselves.
        image_body: ImageBody::Venice,
        ..Quirks::openai()
    },
    ..Dialect::gateway("venice", "https://api.venice.ai/api/v1", "VENICE_API_KEY")
};

/// Doubleword embedding widths: 32 through 4096, defaulting to 4096.
/// Validate both bounds locally because out-of-range requests can be clamped or
/// inconsistently rejected by the service.
const DOUBLEWORD_EMBEDDING_WIDTHS: &[ModelWidth] = &[ModelWidth {
    model: crate::providers::doubleword::QWEN3_EMBEDDING_8B,
    default: Some(4_096),
    accepted: AcceptedWidths::Range {
        min: 32,
        max: 4_096,
        requirement: "to be between 32 and 4096",
    },
}];

/// Doubleword.
pub const DOUBLEWORD: Dialect = Dialect {
    base_url_env: Some("DOUBLEWORD_BASE_URL"),
    quirks: Quirks {
        embedding: EmbeddingQuirks {
            requires_usage: false,
            supports_encoding_format: false,
            supports_user: false,
            widths: DOUBLEWORD_EMBEDDING_WIDTHS,
            // Doubleword refuses a zero width itself, and stating it here
            // keeps the refusal ahead of a request that cannot succeed.
            refuse_zero_width: Some("to be greater than zero"),
            ..EmbeddingQuirks::openai()
        },
        ..Quirks::openai()
    },
    ..Dialect::gateway(
        "doubleword",
        "https://api.doubleword.ai/v1",
        "DOUBLEWORD_API_KEY",
    )
};

/// Z.AI's OpenAI-compatible half. Its Anthropic half is a separate wire.
pub const ZAI: Dialect = Dialect {
    base_url_env: Some("ZAI_API_BASE"),
    ..Dialect::gateway("zai", "https://api.z.ai/api/paas/v4", "ZAI_API_KEY")
};

/// Z.AI's coding endpoint: the same dialect at a different base URL.
pub const ZAI_CODING: Dialect = Dialect {
    base_url: "https://api.z.ai/api/coding/paas/v4",
    ..ZAI
};

/// MiniMax's OpenAI-compatible half (global).
pub const MINIMAX: Dialect = Dialect {
    base_url_env: Some("MINIMAX_API_BASE"),
    ..Dialect::gateway("minimax", "https://api.minimax.io/v1", "MINIMAX_API_KEY")
};

/// MiniMax's China endpoint.
pub const MINIMAX_CHINA: Dialect = Dialect {
    base_url: "https://api.minimaxi.com/v1",
    ..MINIMAX
};

/// Moonshot's OpenAI-compatible half (global).
pub const MOONSHOT: Dialect = Dialect {
    base_url_env: Some("MOONSHOT_API_BASE"),
    quirks: Quirks {
        // Moonshot rejects `json_schema` response formats.
        supports_response_format: false,
        rewrite: BodyRewrite::Moonshot,
        ..Quirks::openai()
    },
    ..Dialect::gateway("moonshot", "https://api.moonshot.ai/v1", "MOONSHOT_API_KEY")
};

/// Moonshot's China endpoint.
pub const MOONSHOT_CHINA: Dialect = Dialect {
    base_url: "https://api.moonshot.cn/v1",
    ..MOONSHOT
};

/// Xiaomi MiMo's OpenAI-compatible half.
pub const XIAOMIMIMO: Dialect = Dialect {
    base_url_env: Some("XIAOMI_MIMO_API_BASE"),
    ..Dialect::gateway(
        "xiaomimimo",
        "https://api.xiaomimimo.com/v1",
        "XIAOMI_MIMO_API_KEY",
    )
};

/// The dialect named `name`, or `None` when this build has no such provider.
pub fn by_name(name: &str) -> Option<&'static Dialect> {
    all().find(|dialect| dialect.name == name)
}

/// Every dialect this build knows, in declaration order.
pub fn all() -> impl Iterator<Item = &'static Dialect> {
    crate::providers::registry::OPENAI_DIALECTS.iter().copied()
}

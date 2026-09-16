//! One `const` per OpenAI-shaped provider.
//!
//! Each constant's fields are taken from that provider's
//! `OpenAICompatibleProvider` impl and its client, so routing the provider
//! through [`Chat`](super::Chat) sends the bytes it sends today. Where a
//! constant departs from [`Quirks::openai`], the field's comment names the
//! measurement or the provider error that forced it.
//!
//! `mistralrs` has no module on this branch, so it has no constant. The
//! OpenAI halves of the dual-dialect providers (Z.AI, MiniMax, Moonshot,
//! Xiaomi MiMo) are here; their Anthropic halves belong to the Anthropic
//! wire.

use super::{
    Auth, AuthAlternative, BodyRewrite, Dialect, DimensionsField, EmbeddingQuirks, ImageBody,
    OutputCap, Quirks, RerankQuirks, Routing, SpeechBody,
};

/// Azure reads its API version from the environment because every Azure
/// route carries it and no default addresses the right API.
pub(super) const AZURE_API_VERSION_ENV: &str = "AZURE_API_VERSION";

/// Azure versions its speech endpoint separately from the rest.
pub(super) const AZURE_AUDIO_API_VERSION_ENV: &str = "AZURE_AUDIO_API_VERSION";

/// The speech `api-version` the Azure client defaulted to.
pub(super) const AZURE_DEFAULT_AUDIO_API_VERSION: &str = "2025-04-01-preview";

/// Official OpenAI Chat Completions.
pub const OPENAI: Dialect = Dialect {
    name: "openai",
    base_url: "https://api.openai.com/v1",
    api_key_env: "OPENAI_API_KEY",
    base_url_env: Some("OPENAI_BASE_URL"),
    request_id_header: Some("x-request-id"),
    alternate_auth: None,
    quirks: Quirks::openai(),
};

/// Azure OpenAI: the deployment is in the URL, the API version is a query
/// parameter, and the credential is an `api-key` header.
pub const AZURE: Dialect = Dialect {
    name: "azure.openai",
    // The account's own resource endpoint; there is no shared host.
    base_url: "",
    api_key_env: "AZURE_API_KEY",
    base_url_env: Some("AZURE_ENDPOINT"),
    request_id_header: None,
    // `azure::Azure::from_env` accepted `AZURE_API_KEY` or `AZURE_TOKEN`,
    // and they are not two spellings of one credential: the key goes out as
    // `api-key`, the token as `Authorization: Bearer`.
    alternate_auth: Some(AuthAlternative {
        api_key_env: "AZURE_TOKEN",
        auth: Auth::Bearer,
    }),
    quirks: Quirks {
        auth: Auth::ApiKeyHeader,
        routing: Routing::AzureDeployment,
        // An Azure model handle is a deployment name chosen by the account
        // owner, so it carries no family information to classify: a capped
        // reasoning deployment still gets the provider's own `Unsupported
        // parameter` error, which is the honest outcome until Azure can be
        // given a signal that does not require guessing.
        output_cap: OutputCap::Legacy,
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
};

/// DeepSeek.
pub const DEEPSEEK: Dialect = Dialect {
    name: "deepseek",
    base_url: "https://api.deepseek.com",
    api_key_env: "DEEPSEEK_API_KEY",
    base_url_env: None,
    request_id_header: None,
    alternate_auth: None,
    quirks: Quirks {
        output_cap: OutputCap::Legacy,
        // DeepSeek accepts only `json_object` response formats, passed
        // through `additional_params` — not the `json_schema` mapping of
        // `output_schema`.
        supports_response_format: false,
        emits_complete_single_chunk_tool_calls: true,
        verify_path: "/user/balance",
        rewrite: BodyRewrite::DeepSeek,
        ..Quirks::openai()
    },
};

/// Groq.
pub const GROQ: Dialect = Dialect {
    name: "groq",
    base_url: "https://api.groq.com/openai/v1",
    api_key_env: "GROQ_API_KEY",
    base_url_env: None,
    request_id_header: Some("x-request-id"),
    alternate_auth: None,
    quirks: Quirks {
        output_cap: OutputCap::Legacy,
        emits_complete_single_chunk_tool_calls: true,
        rewrite: BodyRewrite::GroqCompoundTools,
        ..Quirks::openai()
    },
};

/// Hyperbolic.
pub const HYPERBOLIC: Dialect = Dialect {
    name: "hyperbolic",
    // The bare host: the chat path carries its own `/v1`.
    base_url: "https://api.hyperbolic.xyz",
    api_key_env: "HYPERBOLIC_API_KEY",
    base_url_env: None,
    request_id_header: None,
    alternate_auth: None,
    quirks: Quirks {
        output_cap: OutputCap::Legacy,
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
        rewrite: BodyRewrite::Hyperbolic,
        ..Quirks::openai()
    },
};

/// Mira's gateway.
pub const MIRA: Dialect = Dialect {
    name: "mira",
    base_url: "https://api.mira.network",
    api_key_env: "MIRA_API_KEY",
    base_url_env: None,
    request_id_header: None,
    alternate_auth: None,
    quirks: Quirks {
        output_cap: OutputCap::Legacy,
        // The gateway rejects tool parameters, OpenAI structured-output
        // parameters, and unknown parameters such as `stream_options`.
        supports_tools: false,
        supports_response_format: false,
        stream_include_usage: false,
        completion_path: "/v1/chat/completions",
        models_path: "/v1/models",
        verify_path: "/user-credits",
        rewrite: BodyRewrite::Mira,
        ..Quirks::openai()
    },
};

/// Perplexity.
pub const PERPLEXITY: Dialect = Dialect {
    name: "perplexity",
    base_url: "https://api.perplexity.ai",
    api_key_env: "PERPLEXITY_API_KEY",
    base_url_env: None,
    request_id_header: None,
    alternate_auth: None,
    quirks: Quirks {
        output_cap: OutputCap::Legacy,
        supports_tools: false,
        supports_response_format: false,
        stream_include_usage: false,
        // No endpoint checks a credential without spending tokens.
        verify_path: "",
        rewrite: BodyRewrite::Perplexity,
        ..Quirks::openai()
    },
};

/// Together AI.
pub const TOGETHER: Dialect = Dialect {
    name: "together",
    // The bare host: every path carries its own `/v1`.
    base_url: "https://api.together.xyz",
    api_key_env: "TOGETHER_API_KEY",
    base_url_env: None,
    request_id_header: None,
    alternate_auth: None,
    quirks: Quirks {
        output_cap: OutputCap::Legacy,
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
};

/// Hugging Face's inference router.
pub const HUGGINGFACE: Dialect = Dialect {
    name: "huggingface",
    base_url: "https://router.huggingface.co",
    api_key_env: "HUGGINGFACE_API_KEY",
    base_url_env: None,
    request_id_header: None,
    alternate_auth: None,
    quirks: Quirks {
        output_cap: OutputCap::Legacy,
        supports_response_format: false,
        // Chat lives under the router's `/v1`; verification, transcription
        // and image generation are root-relative, so the prefix cannot live
        // in the base URL.
        completion_path: "/v1/chat/completions",
        verify_path: "/api/whoami-v2",
        // Transcription and image generation address `/{model}` at the
        // router root, not a fixed path under `/v1`.
        model_is_modality_path: true,
        rewrite: BodyRewrite::HuggingFaceRouter,
        ..Quirks::openai()
    },
};

/// A local `llama-server`.
pub const LLAMACPP: Dialect = Dialect {
    name: "llamacpp",
    base_url: "http://localhost:8080/v1",
    api_key_env: "LLAMACPP_API_KEY",
    base_url_env: Some("LLAMACPP_API_BASE_URL"),
    request_id_header: None,
    alternate_auth: None,
    quirks: Quirks {
        // A server started without `--api-key` rejects a request that
        // carries an `Authorization` header.
        auth: Auth::OptionalBearer,
        output_cap: OutputCap::Legacy,
        // Measured on llama.cpp b1-6d05498 with Qwen3-VL-2B: an image handed
        // back through a tool reaches the model, 3/3, matching a control
        // that sends the same bytes in a `user` message.
        supports_image_tool_results: true,
        // `llama-server` serves its operational routes unversioned.
        verify_path: "/props",
        rewrite: BodyRewrite::LlamaCpp,
        rerank: RerankQuirks {
            // `llama-server` serves one rerank handler behind four aliases
            // (`/rerank`, `/reranking`, `/v1/rerank`, `/v1/reranking`); the
            // base URL already carries `/v1`.
            path: "/rerank",
            // It posts one task per document and waits for all of them,
            // bounded only by memory — there is no documented cap, so this
            // is the batching hint, matching the embeddings default rather
            // than a number the server enforces.
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
};

/// Mistral.
pub const MISTRAL: Dialect = Dialect {
    name: "mistral",
    // The bare host: every path carries its own `/v1`.
    base_url: "https://api.mistral.ai",
    api_key_env: "MISTRAL_API_KEY",
    base_url_env: None,
    request_id_header: Some("mistral-correlation-id"),
    alternate_auth: None,
    quirks: Quirks {
        output_cap: OutputCap::Legacy,
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
            ..EmbeddingQuirks::openai()
        },
        ..Quirks::openai()
    },
};

/// OpenRouter.
pub const OPENROUTER: Dialect = Dialect {
    name: "openrouter",
    base_url: "https://openrouter.ai/api/v1",
    api_key_env: "OPENROUTER_API_KEY",
    base_url_env: None,
    request_id_header: None,
    alternate_auth: None,
    quirks: Quirks {
        output_cap: OutputCap::Legacy,
        stream_include_usage: false,
        verify_path: "/key",
        // A gateway forwards its upstream's own finish reason and its own
        // reasoning blobs.
        native_finish_reason: true,
        reasoning_details: true,
        // Its message conversion refused a provider file id outright.
        accepts_file_ids: false,
        rewrite: BodyRewrite::OpenRouter,
        embedding: EmbeddingQuirks {
            requires_usage: false,
            ..EmbeddingQuirks::openai()
        },
        ..Quirks::openai()
    },
};

/// Venice.
pub const VENICE: Dialect = Dialect {
    name: "venice",
    base_url: "https://api.venice.ai/api/v1",
    api_key_env: "VENICE_API_KEY",
    base_url_env: Some("VENICE_BASE_URL"),
    request_id_header: None,
    alternate_auth: None,
    quirks: Quirks {
        output_cap: OutputCap::Legacy,
        image_generation_path: "/image/generate",
        ..Quirks::openai()
    },
};

/// Doubleword.
pub const DOUBLEWORD: Dialect = Dialect {
    name: "doubleword",
    base_url: "https://api.doubleword.ai/v1",
    api_key_env: "DOUBLEWORD_API_KEY",
    base_url_env: Some("DOUBLEWORD_BASE_URL"),
    request_id_header: None,
    alternate_auth: None,
    quirks: Quirks {
        output_cap: OutputCap::Legacy,
        embedding: EmbeddingQuirks {
            requires_usage: false,
            supports_encoding_format: false,
            supports_user: false,
            ..EmbeddingQuirks::openai()
        },
        ..Quirks::openai()
    },
};

/// Z.AI's OpenAI-compatible half. Its Anthropic half is a separate wire.
pub const ZAI: Dialect = Dialect {
    name: "zai",
    base_url: "https://api.z.ai/api/paas/v4",
    api_key_env: "ZAI_API_KEY",
    base_url_env: Some("ZAI_API_BASE"),
    request_id_header: None,
    alternate_auth: None,
    quirks: Quirks {
        output_cap: OutputCap::Legacy,
        ..Quirks::openai()
    },
};

/// Z.AI's coding endpoint: the same dialect at a different base URL.
pub const ZAI_CODING: Dialect = Dialect {
    base_url: "https://api.z.ai/api/coding/paas/v4",
    ..ZAI
};

/// MiniMax's OpenAI-compatible half (global).
pub const MINIMAX: Dialect = Dialect {
    name: "minimax",
    base_url: "https://api.minimax.io/v1",
    api_key_env: "MINIMAX_API_KEY",
    base_url_env: Some("MINIMAX_API_BASE"),
    request_id_header: None,
    alternate_auth: None,
    quirks: Quirks {
        output_cap: OutputCap::Legacy,
        ..Quirks::openai()
    },
};

/// MiniMax's China endpoint.
pub const MINIMAX_CHINA: Dialect = Dialect {
    base_url: "https://api.minimaxi.com/v1",
    ..MINIMAX
};

/// Moonshot's OpenAI-compatible half (global).
pub const MOONSHOT: Dialect = Dialect {
    name: "moonshot",
    base_url: "https://api.moonshot.ai/v1",
    api_key_env: "MOONSHOT_API_KEY",
    base_url_env: Some("MOONSHOT_API_BASE"),
    request_id_header: None,
    alternate_auth: None,
    quirks: Quirks {
        output_cap: OutputCap::Legacy,
        // Moonshot rejects `json_schema` response formats.
        supports_response_format: false,
        rewrite: BodyRewrite::Moonshot,
        ..Quirks::openai()
    },
};

/// Moonshot's China endpoint.
pub const MOONSHOT_CHINA: Dialect = Dialect {
    base_url: "https://api.moonshot.cn/v1",
    ..MOONSHOT
};

/// Xiaomi MiMo's OpenAI-compatible half.
pub const XIAOMIMIMO: Dialect = Dialect {
    name: "xiaomimimo",
    base_url: "https://api.xiaomimimo.com/v1",
    api_key_env: "XIAOMI_MIMO_API_KEY",
    base_url_env: Some("XIAOMI_MIMO_API_BASE"),
    request_id_header: None,
    alternate_auth: None,
    quirks: Quirks {
        output_cap: OutputCap::Legacy,
        ..Quirks::openai()
    },
};

/// xAI.
///
/// xAI's *completion* is a Responses wire (`/v1/responses`), not this one;
/// this dialect exists because its image and speech endpoints are
/// OpenAI-shaped and were served by the shared generic models. The
/// chat-completions path is xAI's own spelling, for a caller that wants it.
pub const XAI: Dialect = Dialect {
    name: "xai",
    // The bare host: every path carries its own `/v1`.
    base_url: "https://api.x.ai",
    api_key_env: "XAI_API_KEY",
    base_url_env: None,
    request_id_header: None,
    alternate_auth: None,
    quirks: Quirks {
        output_cap: OutputCap::Legacy,
        completion_path: "/v1/chat/completions",
        embeddings_path: "/v1/embeddings",
        models_path: "/v1/models",
        verify_path: "/v1/api-key",
        transcription_path: "/v1/audio/transcriptions",
        image_generation_path: "/v1/images/generations",
        // xAI spells its text-to-speech endpoint `/v1/tts` and takes a body
        // of its own.
        audio_generation_path: "/v1/tts",
        image_body: ImageBody::Xai,
        speech_body: SpeechBody::Xai,
        ..Quirks::openai()
    },
};

/// GitHub Copilot's chat-completions route.
///
/// Defined so the chat wire is complete; Copilot's own wire is an enum over
/// this route and the Responses route, and its session-token exchange and
/// editor headers live there.
pub const COPILOT_CHAT: Dialect = Dialect {
    name: "copilot",
    base_url: "https://api.githubcopilot.com",
    api_key_env: "GITHUB_COPILOT_API_KEY",
    base_url_env: Some("GITHUB_COPILOT_API_BASE"),
    request_id_header: Some("x-request-id"),
    alternate_auth: None,
    quirks: Quirks {
        output_cap: OutputCap::Legacy,
        // Copilot verifies through its token exchange, not a path.
        verify_path: "",
        embedding: EmbeddingQuirks {
            requires_usage: false,
            ..EmbeddingQuirks::openai()
        },
        ..Quirks::openai()
    },
};

/// Every dialect this build knows, for
/// [`Dialect`]'s [`Deserialize`](serde::Deserialize) lookup.
///
/// Keyed by [`Dialect::name`], so the regional and endpoint variants that
/// share a provider name are not listed: a stored wire keeps its base URL,
/// which is what distinguishes them.
const ALL: &[&Dialect] = &[
    &OPENAI,
    &AZURE,
    &DEEPSEEK,
    &GROQ,
    &HYPERBOLIC,
    &MIRA,
    &PERPLEXITY,
    &TOGETHER,
    &HUGGINGFACE,
    &LLAMACPP,
    &MISTRAL,
    &OPENROUTER,
    &VENICE,
    &DOUBLEWORD,
    &ZAI,
    &MINIMAX,
    &MOONSHOT,
    &XIAOMIMIMO,
    &XAI,
    &COPILOT_CHAT,
];

/// The dialect named `name`, or `None` when this build has no such provider.
pub fn by_name(name: &str) -> Option<&'static Dialect> {
    ALL.iter().copied().find(|dialect| dialect.name == name)
}

/// Every dialect this build knows, in declaration order.
pub fn all() -> impl Iterator<Item = &'static Dialect> {
    ALL.iter().copied()
}

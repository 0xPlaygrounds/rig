//! Z.AI API clients and Rig integrations.
//!
//! Z.AI exposes OpenAI-compatible APIs for both its general platform and
//! coding-focused platform, plus an Anthropic-compatible endpoint for tools
//! like Claude Code.
//!
//! # OpenAI-compatible example
//! ```no_run
//! use rig_core::client::CompletionClient;
//! use rig_core::providers::zai;
//!
//! let client = zai::Client::new("YOUR_API_KEY").expect("Failed to build client");
//! let glm_4_6 = client.completion_model(zai::GLM_4_6);
//! ```
//!
//! # Anthropic-compatible example
//! ```no_run
//! use rig_core::client::CompletionClient;
//! use rig_core::providers::zai;
//!
//! let client = zai::AnthropicClient::new("YOUR_API_KEY").expect("Failed to build client");
//! let glm_4_6 = client.completion_model(zai::GLM_4_6);
//! ```

use crate::client;
use crate::providers::internal::anthropic_compatible::{
    AnthropicBaseUrl, impl_dual_dialect_provider,
};

/// General-purpose OpenAI-compatible base URL.
pub const GENERAL_API_BASE_URL: &str = "https://api.z.ai/api/paas/v4";
/// Coding-focused OpenAI-compatible base URL.
pub const CODING_API_BASE_URL: &str = "https://api.z.ai/api/coding/paas/v4";
/// Anthropic-compatible base URL.
pub const ANTHROPIC_API_BASE_URL: &str = "https://api.z.ai/api/anthropic";

/// `glm-4.6`
pub const GLM_4_6: &str = "glm-4.6";
/// `glm-4.6-air`
///
/// **Not in Z.AI's catalog.** The 4.6 generation shipped only as `glm-4.6`;
/// the `-air`/`-x`/`-airx` variants belong to 4.5 and 4.7. This identifier
/// appears neither in the `model` enum of Z.AI's chat-completion reference nor
/// in the pricing table, so naming it can only fail.
#[deprecated(
    note = "Z.AI does not serve this model. `glm-4.6-air` is not in Z.AI's documented model list; \
    use `GLM_4_6`, or `GLM_4_5_AIR` for an Air-class model"
)]
pub const GLM_4_6_AIR: &str = "glm-4.6-air";
/// `glm-4.6-x`
///
/// **Not in Z.AI's catalog.** See [`GLM_4_6_AIR`]: the 4.6 generation shipped
/// only as `glm-4.6`.
#[deprecated(
    note = "Z.AI does not serve this model. `glm-4.6-x` is not in Z.AI's documented model list; \
    use `GLM_4_6`, or `GLM_4_5_AIRX` for the fastest 4.5 tier that has a constant"
)]
pub const GLM_4_6_X: &str = "glm-4.6-x";
/// `glm-4.5`
pub const GLM_4_5: &str = "glm-4.5";
/// `glm-4.5-air`
pub const GLM_4_5_AIR: &str = "glm-4.5-air";
/// `glm-4.5v`
pub const GLM_4_5V: &str = "glm-4.5v";
/// `glm-4.5-airx`
pub const GLM_4_5_AIRX: &str = "glm-4.5-airx";

impl_dual_dialect_provider!(
    ext = ZAiExt,
    builder = ZAiBuilder,
    anthropic_ext = ZAiAnthropicExt,
    anthropic_builder = ZAiAnthropicBuilder,
    client_input = client::BearerAuth,
    api_key_env = "ZAI_API_KEY",
    base_url = GENERAL_API_BASE_URL,
    base_url_env = "ZAI_API_BASE",
    anthropic_provider_name = "z.ai",
    anthropic_base_url = ANTHROPIC_API_BASE_URL,
    anthropic_base_url_env = "ZAI_ANTHROPIC_API_BASE",
);

client::impl_capabilities!(
    ZAiExt,
    completion = super::openai::completion::GenericCompletionModel<ZAiExt, H>,
);

impl super::openai::completion::OpenAICompatibleProvider for ZAiExt {
    const PROVIDER_NAME: &'static str = "zai";

    type StreamingUsage = super::openai::Usage;

    // Z.AI documents `response_format` as an object whose `type` is
    // `text | json_object`; there is no `json_schema` member and no
    // `json_schema` sibling object, and every structured-output example sends
    // the schema as prose in a system message instead. What Z.AI's API *does*
    // with an undocumented `json_schema` block is unrecorded — no key was
    // available to find out — so the claim here is only that rig was sending a
    // shape the provider does not document.
    //
    // The flag's second consumer is why the undocumented shape is not merely
    // untidy: it also becomes `composes_native_output_with_tools`, which tells
    // rig-agent the schema is natively guaranteed. Seven other providers answer
    // that the same way, for reasons ranging from an observed rejection
    // (moonshot, deepseek) to unverified or per-model support (hyperbolic,
    // together, huggingface, perplexity, mira) — the conservative answer either
    // way.
    //
    // Consequence to be clear about: `false` means the schema is dropped with a
    // warning, not re-enforced somewhere else. `OutputMode::Auto` then resolves
    // to `Tool` only when the run also has executable tools
    // (`rig-agent/src/agent/completion.rs:106-117`); `prompt_typed` pins
    // `Native` unconditionally, so a typed prompt on Z.AI is now unconstrained,
    // exactly as on every other provider with this flag `false`. Callers who
    // need enforcement should ask for `OutputMode::Prompted` or `Tool`.
    const SUPPORTS_RESPONSE_FORMAT: bool = false;

    type Response = super::openai::CompletionResponse;
}

const ANTHROPIC_BASE_URLS: AnthropicBaseUrl = AnthropicBaseUrl::new(
    &[
        (GENERAL_API_BASE_URL, ANTHROPIC_API_BASE_URL),
        (CODING_API_BASE_URL, ANTHROPIC_API_BASE_URL),
    ],
    &[
        "/api/paas/v4",
        "/api/paas/v4/",
        "/api/coding/paas/v4",
        "/api/coding/paas/v4/",
    ],
    "/api/anthropic",
);

impl<H> ClientBuilder<H> {
    pub fn general(self) -> Self {
        self.base_url(GENERAL_API_BASE_URL)
    }

    pub fn coding(self) -> Self {
        self.base_url(CODING_API_BASE_URL)
    }
}

impl<H> AnthropicClientBuilder<H> {
    pub fn general(self) -> Self {
        self.base_url(ANTHROPIC_API_BASE_URL)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ANTHROPIC_API_BASE_URL, ANTHROPIC_BASE_URLS, CODING_API_BASE_URL, GENERAL_API_BASE_URL,
        GLM_4_5, GLM_4_5_AIR, GLM_4_5_AIRX, GLM_4_5V, GLM_4_6, ZAiExt,
    };
    use crate::providers::openai::completion::{
        CompletionRequest as OpenAICompletionRequest, OpenAICompatibleProvider, OpenAIRequestParams,
    };

    /// Z.AI's documented `model` enum, transcribed from the chat-completion
    /// API reference and cross-checked against the pricing table (both read
    /// 2026-08-16):
    ///
    /// * <https://docs.z.ai/api-reference/llm/chat-completion>
    /// * <https://docs.z.ai/guides/overview/pricing>
    const DOCUMENTED_MODELS: &[&str] = &[
        "glm-5.2",
        "glm-5.1",
        "glm-5-turbo",
        "glm-5",
        "glm-4.7",
        "glm-4.7-flash",
        "glm-4.7-flashx",
        "glm-4.6",
        "glm-4.5",
        "glm-4.5-air",
        "glm-4.5-x",
        "glm-4.5-airx",
        "glm-4.5-flash",
        "glm-4-32b-0414-128k",
        "glm-5v-turbo",
        "glm-4.6v",
        "glm-4.6v-flash",
        "glm-4.6v-flashx",
        "glm-4.5v",
    ];

    /// Every model constant this module treats as live must name a model Z.AI
    /// documents, and the two it marks deprecated must stay absent from the
    /// catalog.
    ///
    /// Both lists are enumerated by hand because Rust cannot reflect over a
    /// module's constants or read a `#[deprecated]` attribute from a test: this
    /// pins the *catalog membership* that justifies each constant's status, not
    /// the attribute itself. Adding a constant without adding it here is the
    /// gap that leaves.
    ///
    /// A model handle that the API cannot resolve fails every call that names
    /// it, so a public constant is a promise the provider has to keep; this
    /// pins both halves of that promise against a transcription of Z.AI's own
    /// enum, which is the only place the set is defined.
    ///
    /// **Unit test rather than a cassette because no `ZAI_API_KEY` was
    /// available in the environment where this was written**, so the 400 that
    /// `glm-4.6-air` produces could not be recorded (`tests/README.md` asks a
    /// unit test of provider-facing behavior to say why it is not a cassette
    /// test). The wire half is already written and `#[ignore]`d as
    /// `general/unknown_model_constant_glm_4_6_air` and its `_glm_4_6_x`
    /// sibling in `tests/providers/zai/cassette/models.rs`; recording those
    /// turns this documentation-based claim into an observed one.
    #[test]
    // The retired constants are the subject of the second assertion.
    #[allow(deprecated)]
    fn model_constants_match_zais_documented_catalog() {
        for model in [GLM_4_6, GLM_4_5, GLM_4_5_AIR, GLM_4_5V, GLM_4_5_AIRX] {
            assert!(
                DOCUMENTED_MODELS.contains(&model),
                "{model} is exported as a usable Z.AI model handle but is not in Z.AI's \
                 documented model enum; correct the constant or deprecate it"
            );
        }

        for model in [super::GLM_4_6_AIR, super::GLM_4_6_X] {
            assert!(
                !DOCUMENTED_MODELS.contains(&model),
                "{model} is back in Z.AI's documented model enum; drop its #[deprecated] \
                 attribute rather than leaving a usable model marked retired"
            );
        }
    }

    /// An `output_schema` must not reach Z.AI as OpenAI's `json_schema`
    /// `response_format`.
    ///
    /// Z.AI documents `response_format` as an object whose `type` is
    /// `text | json_object` (<https://docs.z.ai/api-reference/llm/chat-completion>),
    /// and its structured-output guide
    /// (<https://docs.z.ai/guides/capabilities/struct-output>) shows only
    /// `{"type": "json_object"}`, never a `json_schema` member, a `strict`
    /// flag or a `schema` field. The shared OpenAI path emits that block
    /// whenever `SUPPORTS_RESPONSE_FORMAT` is true and the turn carries no
    /// unanswered tools, so this asserts the request boundary directly.
    ///
    /// **Unit test rather than a cassette because no `ZAI_API_KEY` was
    /// available in the environment where this was written**, so what Z.AI
    /// does with the block could not be recorded; the claim pinned here is the
    /// one that needs no recording — what rig *sends*. `tests/README.md` asks
    /// a unit test of provider-facing behavior to say why it is not a cassette
    /// test. The recorded half is written and `#[ignore]`d as
    /// `general/structured_output_native_blocking` in
    /// `tests/providers/zai/cassette/structured_output.rs`.
    #[test]
    fn output_schema_does_not_become_a_json_schema_response_format() {
        let request = crate::completion::CompletionRequestBuilder::new(
            crate::test_utils::MockCompletionModel::default(),
            "hello",
        )
        .output_schema(schemars::schema_for!(serde_json::Value))
        .build();

        let request = OpenAICompletionRequest::try_from(OpenAIRequestParams {
            model: GLM_4_6.to_string(),
            request,
            strict_tools: false,
            tool_result_array_content: false,
            supports_response_format: ZAiExt::SUPPORTS_RESPONSE_FORMAT,
            supports_tools: ZAiExt::SUPPORTS_TOOLS,
        })
        .expect("request should convert");

        let body = serde_json::to_value(request).expect("request should serialize");
        assert!(
            body.get("response_format").is_none(),
            "Z.AI accepts only text/json_object response formats; request body was {body}"
        );
    }

    /// The same flag also governs what rig promises the agent runtime, so pin
    /// that consequence separately: with the schema dropped, Z.AI must not
    /// claim it composes native structured output with tool calls, or
    /// `OutputMode::Auto` stays in the mode documented as the only guaranteed
    /// one while nothing constrains the model at all.
    ///
    /// **Unit test rather than a cassette because no `ZAI_API_KEY` was
    /// available in the environment where this was written** — and this half
    /// is not observable on the wire in any case: a capability is a statement
    /// rig makes to itself, so a recording could not witness it.
    #[test]
    fn zai_does_not_claim_native_output_composes_with_tools() {
        use crate::client::CompletionClient;
        use crate::completion::CompletionModel;

        let client = crate::providers::zai::Client::new("dummy-key").expect("Client::new()");
        let model = client.completion_model(GLM_4_6);

        assert!(
            !model.capabilities().composes_native_output_with_tools,
            "Z.AI cannot constrain output natively, so it must not suppress rig-agent's \
             tool-mode fallback"
        );
    }

    #[test]
    fn test_client_initialization() {
        let _client = crate::providers::zai::Client::new("dummy-key").expect("Client::new()");
        let _client_from_builder = crate::providers::zai::Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder()");
        let _anthropic_client = crate::providers::zai::AnthropicClient::new("dummy-key")
            .expect("AnthropicClient::new()");
        let _anthropic_client_from_builder = crate::providers::zai::AnthropicClient::builder()
            .api_key("dummy-key")
            .build()
            .expect("AnthropicClient::builder()");
    }

    #[test]
    fn normalize_openai_style_bases_to_anthropic_base() {
        assert_eq!(
            ANTHROPIC_BASE_URLS
                .normalize(GENERAL_API_BASE_URL)
                .as_deref(),
            Some(ANTHROPIC_API_BASE_URL)
        );
        assert_eq!(
            ANTHROPIC_BASE_URLS
                .normalize(CODING_API_BASE_URL)
                .as_deref(),
            Some(ANTHROPIC_API_BASE_URL)
        );
        assert_eq!(
            ANTHROPIC_BASE_URLS
                .normalize("https://proxy.example.com/api/paas/v4")
                .as_deref(),
            Some("https://proxy.example.com/api/anthropic")
        );
        assert_eq!(
            ANTHROPIC_BASE_URLS
                .normalize("https://proxy.example.com/api/coding/paas/v4")
                .as_deref(),
            Some("https://proxy.example.com/api/anthropic")
        );
    }

    #[test]
    fn normalize_preserves_existing_anthropic_base() {
        assert_eq!(
            ANTHROPIC_BASE_URLS
                .normalize("https://proxy.example.com/api/anthropic")
                .as_deref(),
            Some("https://proxy.example.com/api/anthropic")
        );
    }

    #[test]
    fn anthropic_primary_override_wins() {
        let override_url = ANTHROPIC_BASE_URLS.resolve(
            Some("https://primary.example.com/api/anthropic"),
            Some(GENERAL_API_BASE_URL),
        );

        assert_eq!(
            override_url.as_deref(),
            Some("https://primary.example.com/api/anthropic")
        );
    }
}

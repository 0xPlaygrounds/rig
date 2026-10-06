// P2 removes this gate.
#[cfg(any())]
mod harness_switch {
    //! One `GenerationOptions` value, applied to five providers by a harness
    //! that switches between them, encodes to each provider's own JSON. The
    //! request carries no provider JSON: no `additional_params`.
    //!
    //! Encode-only: what these bodies send is pinned here against the
    //! mapping table; a recording of each lands with the acceptance index.

    use rig::bedrock::completion::{ANTHROPIC_CLAUDE_SONNET_5, Converse};
    use rig::completion::{
        CacheRetention, CompletionRequest, Effort, GenerationOptions, OnUnsupported, ServiceTier,
    };
    use rig::providers::anthropic::completion::CLAUDE_OPUS_4_8;
    use rig::providers::anthropic::wire::AnthropicConfig;
    use rig::providers::gemini::GeminiConfig;
    use rig::providers::gemini::completion::{GEMINI_3_FLASH_PREVIEW, GenerateContent};
    use rig::providers::openai::GPT_5_5;
    use rig::providers::openai::responses_api::wire::Responses;
    use rig::providers::openai::wire::{Chat, OPENROUTER, OpenAIConfig};
    use rig::test_utils::TraceCapture;
    use rig_core::error::EncodeError;
    use rig_core::operation::Completion;
    use rig_core::wire::{Body, Encoded, Mode, Operation, Wire};
    use rig_test_support::cassette_models::AnthropicModels;
    use serde_json::Value;

    pub(super) const KEY: &str = "typed-options";
    const OPENROUTER_MODEL: &str = "anthropic/claude-sonnet-4.5";

    /// `request` as `wire` prepares it, the way the driver does before encoding.
    pub(super) fn prepared<W: Wire<Op = Completion>>(
        wire: &W,
        request: CompletionRequest,
    ) -> CompletionRequest {
        Completion::prepare(request, &wire.describe()).expect("the request prepares")
    }

    /// The JSON body `wire` sends for `request`.
    pub(super) fn encode<W: Wire<Op = Completion, Payload = Encoded>>(
        wire: &W,
        request: CompletionRequest,
    ) -> Result<Value, EncodeError> {
        let encoded = wire.encode(prepared(wire, request), Mode::Unary)?;
        let Body::Bytes(bytes) = encoded.request.body() else {
            panic!("a completion body is JSON, not multipart");
        };
        Ok(serde_json::from_slice(bytes).expect("the body is JSON"))
    }

    /// The string at `pointer` in `body`.
    pub(super) fn text<'a>(body: &'a Value, pointer: &str) -> Option<&'a str> {
        body.pointer(pointer).and_then(Value::as_str)
    }

    /// The last element of the array at `pointer` in `body`.
    fn last<'a>(body: &'a Value, pointer: &str) -> &'a Value {
        body.pointer(pointer)
            .and_then(Value::as_array)
            .and_then(|items| items.last())
            .unwrap_or_else(|| panic!("{pointer} is a non-empty array in {body}"))
    }

    /// What a harness asks every provider for.
    fn options(policy: OnUnsupported) -> GenerationOptions {
        GenerationOptions::default()
            .reasoning(Effort::High)
            .cache(CacheRetention::Long)
            .service_tier(ServiceTier::Default)
            .on_unsupported(policy)
    }

    fn request(policy: OnUnsupported) -> CompletionRequest {
        CompletionRequest::new("Summarize the support handbook in one sentence.")
            .preamble("You are a support agent.")
            .max_tokens(4096)
            .options(options(policy))
    }

    fn anthropic() -> rig::providers::anthropic::Messages {
        AnthropicModels::new(AnthropicConfig::new(KEY), crate::cassettes::local_http())
            .completion(CLAUDE_OPUS_4_8)
            .wire
    }

    fn openai_responses() -> Responses {
        Responses::new(OpenAIConfig::new(KEY), GPT_5_5)
    }

    fn gemini() -> GenerateContent {
        GenerateContent::new(GeminiConfig::new(KEY), GEMINI_3_FLASH_PREVIEW)
    }

    fn openrouter() -> Chat {
        Chat::new(OpenAIConfig::with_key(&OPENROUTER, KEY), OPENROUTER_MODEL)
    }

    fn bedrock(request: CompletionRequest) -> Result<Value, EncodeError> {
        let wire = Converse::new(ANTHROPIC_CLAUDE_SONNET_5);
        let request = prepared(&wire, request);
        wire.encode(request, Mode::Unary)
            .map(|request| request.body)
    }

    #[test]
    fn one_options_value_drives_five_wires() {
        let capture = TraceCapture::default();
        let (anthropic, responses, gemini, bedrock, openrouter) =
            tracing::subscriber::with_default(capture.subscriber(), || {
                let encode_one = |result: Result<Value, EncodeError>| {
                    result.expect("the request encodes under Ignore")
                };
                (
                    encode_one(encode(&anthropic(), request(OnUnsupported::Ignore))),
                    encode_one(encode(&openai_responses(), request(OnUnsupported::Ignore))),
                    encode_one(encode(&gemini(), request(OnUnsupported::Ignore))),
                    encode_one(bedrock(request(OnUnsupported::Ignore))),
                    encode_one(encode(&openrouter(), request(OnUnsupported::Ignore))),
                )
            });

        // Anthropic, class A3: adaptive thinking sent explicitly, effort in
        // `output_config`, the 1 h markers at the top level and on the last
        // system block, the standard tier only.
        assert_eq!(text(&anthropic, "/thinking/type"), Some("adaptive"));
        assert_eq!(text(&anthropic, "/output_config/effort"), Some("high"));
        assert_eq!(text(&anthropic, "/cache_control/type"), Some("ephemeral"));
        assert_eq!(text(&anthropic, "/cache_control/ttl"), Some("1h"));
        assert_eq!(
            text(last(&anthropic, "/system"), "/cache_control/ttl"),
            Some("1h")
        );
        assert_eq!(text(&anthropic, "/service_tier"), Some("standard_only"));

        // OpenAI Responses on GPT-5.5, which keeps a prompt cache for 24 hours.
        assert_eq!(text(&responses, "/reasoning/effort"), Some("high"));
        assert_eq!(text(&responses, "/prompt_cache_retention"), Some("24h"));
        assert_eq!(text(&responses, "/service_tier"), Some("default"));

        // Gemini: a level, the standard tier, and no request field for a
        // long cache, so the cache is skipped with a warning.
        assert_eq!(
            text(&gemini, "/generationConfig/thinkingConfig/thinkingLevel"),
            Some("high")
        );
        assert_eq!(text(&gemini, "/serviceTier"), Some("standard"));
        assert!(gemini.get("cachedContent").is_none(), "{gemini}");

        // Bedrock Converse on Claude Sonnet 5: reasoning in the model's own
        // fields, 1 h checkpoints after the system blocks and at the end of
        // the last message.
        assert_eq!(
            text(&bedrock, "/additionalModelRequestFields/thinking/type"),
            Some("adaptive")
        );
        assert_eq!(
            text(
                &bedrock,
                "/additionalModelRequestFields/output_config/effort"
            ),
            Some("high")
        );
        assert_eq!(text(&bedrock, "/serviceTier/type"), Some("default"));
        assert_eq!(
            text(last(&bedrock, "/system"), "/cachePoint/ttl"),
            Some("1h")
        );
        let last_message = last(&bedrock, "/messages");
        assert_eq!(
            text(last(last_message, "/content"), "/cachePoint/ttl"),
            Some("1h")
        );

        // OpenRouter's documented top-level marker, not a system-message one.
        assert_eq!(text(&openrouter, "/reasoning/effort"), Some("high"));
        assert_eq!(text(&openrouter, "/cache_control/type"), Some("ephemeral"));
        assert_eq!(text(&openrouter, "/cache_control/ttl"), Some("1h"));
        assert_eq!(text(&openrouter, "/service_tier"), Some("default"));

        let skipped: Vec<String> = capture
            .warnings()
            .into_iter()
            .filter(|warning| warning.contains("option=cache"))
            .collect();
        assert_eq!(skipped.len(), 1, "{skipped:?}");
        assert!(skipped[0].contains("provider=gcp.gemini"), "{skipped:?}");
    }

    #[test]
    fn long_cache_on_gemini_is_an_error_under_the_default_policy() {
        let error = encode(&gemini(), request(OnUnsupported::Error))
            .expect_err("Gemini has no request field for a long cache");
        let unsupported = error
            .unsupported_option()
            .expect("the error names the unsupported option");
        assert_eq!(unsupported.option, "cache");
        assert_eq!(unsupported.provider, "gcp.gemini");
        assert_eq!(unsupported.model, GEMINI_3_FLASH_PREVIEW);
    }
}

// P2 removes this gate.
#[cfg(any())]
mod no_silent_drop {
    //! A cache asked of a dialect that cannot cache on request is refused,
    //! or skipped with a warning under `Ignore`; it is never dropped
    //! without a word, as `Chat::with_prompt_caching` was.
    //!
    //! Encode-only: a refused option sends no request to record.

    use rig::completion::{CacheRetention, CompletionRequest, GenerationOptions, OnUnsupported};
    use rig::providers::cohere::{COMMAND_A_03_2025, CohereConfig};
    use rig::providers::deepseek::DEEPSEEK_FLASH;
    use rig::providers::openai::wire::{Chat, DEEPSEEK, OpenAIConfig};
    use rig::test_utils::TraceCapture;

    use super::harness_switch::{KEY, encode};

    fn request(cache: CacheRetention, policy: OnUnsupported) -> CompletionRequest {
        CompletionRequest::new("Reply with the single word: pong")
            .max_tokens(16)
            .options(
                GenerationOptions::default()
                    .cache(cache)
                    .on_unsupported(policy),
            )
    }

    /// Cohere's Compatibility route, the default `CohereChat` route.
    fn cohere() -> rig::providers::cohere::CohereChat {
        CohereConfig::new(KEY).completion(COMMAND_A_03_2025)
    }

    fn deepseek() -> Chat {
        Chat::new(OpenAIConfig::with_key(&DEEPSEEK, KEY), DEEPSEEK_FLASH)
    }

    #[test]
    fn caching_on_a_dialect_that_cannot_cache_is_refused() {
        let refusals = [
            (
                encode(
                    &cohere(),
                    request(CacheRetention::Short, OnUnsupported::Error),
                ),
                "cohere",
                COMMAND_A_03_2025,
            ),
            (
                encode(
                    &deepseek(),
                    request(CacheRetention::Long, OnUnsupported::Error),
                ),
                "deepseek",
                DEEPSEEK_FLASH,
            ),
        ];
        for (result, provider, model) in refusals {
            let error = result.expect_err("the dialect cannot honour the cache");
            let unsupported = error
                .unsupported_option()
                .expect("the error names the unsupported option");
            assert_eq!(unsupported.option, "cache");
            assert_eq!(unsupported.provider, provider);
            assert_eq!(unsupported.model, model);
            assert!(!unsupported.reason.is_empty());
        }
    }

    #[test]
    fn under_ignore_the_option_is_skipped_with_a_warning() {
        let capture = TraceCapture::default();
        let body = tracing::subscriber::with_default(capture.subscriber(), || {
            encode(
                &cohere(),
                request(CacheRetention::Short, OnUnsupported::Ignore),
            )
        })
        .expect("Ignore encodes the rest of the request");
        for key in [
            "cache_control",
            "prompt_cache_retention",
            "prompt_cache_options",
        ] {
            assert!(body.get(key).is_none(), "{key} reached the body: {body}");
        }
        let warnings = capture.warnings();
        let skipped: Vec<&String> = warnings
            .iter()
            .filter(|warning| {
                warning.contains("option=cache") && warning.contains("provider=cohere")
            })
            .collect();
        assert_eq!(skipped.len(), 1, "{warnings:?}");
    }
}

// P3 removes this gate.
#[cfg(any())]
mod catalog_validation {
    //! The catalog knows which reasoning a model takes, and `validate`
    //! refuses the rest before a request is sent.
    //!
    //! Catalog data, not provider traffic: nothing here reaches a wire.

    use rig::catalog::Catalog;
    use rig::completion::{Effort, GenerationOptions, Reasoning};
    use rig::providers::anthropic::completion::CLAUDE_HAIKU_4_5;
    use rig::providers::gemini::completion::GEMINI_3_8_FLASH;
    use rig::providers::registry::ProviderId;

    fn reasoning(reasoning: impl Into<Reasoning>) -> GenerationOptions {
        GenerationOptions::default().reasoning(reasoning)
    }

    #[test]
    fn validate_rejects_a_reasoning_level_the_model_lacks() {
        let catalog = Catalog::builtin();

        let gemini = catalog
            .get(
                ProviderId::resolve("gcp.gemini").expect("Gemini is registered"),
                GEMINI_3_8_FLASH,
            )
            .expect("Gemini 3.8 Flash has a catalog entry");
        let refused = gemini
            .validate(&reasoning(Effort::Minimal))
            .expect_err("Gemini 3.8 Flash has no minimal level");
        assert_eq!(refused.option, "reasoning");
        assert_eq!(refused.provider, "gcp.gemini");
        assert_eq!(refused.model, GEMINI_3_8_FLASH);
        assert!(gemini.validate(&reasoning(Effort::High)).is_ok());

        let haiku = catalog
            .resolve("anthropic/claude-haiku-4-5")
            .expect("the reference resolves");
        assert_eq!(haiku.id, CLAUDE_HAIKU_4_5);
        assert!(haiku.reasoning.levels.is_empty(), "{:?}", haiku.reasoning);
        let refused = haiku
            .validate(&reasoning(Effort::High))
            .expect_err("Claude Haiku 4.5 takes a budget, not an effort");
        assert_eq!(refused.option, "reasoning");
        assert_eq!(refused.provider, "anthropic");
        assert!(
            haiku
                .validate(&reasoning(Reasoning::Budget { tokens: 2048 }))
                .is_ok()
        );
        assert!(
            haiku
                .validate(&reasoning(Reasoning::Budget { tokens: 512 }))
                .is_err(),
            "a budget under 1024 tokens is refused"
        );
    }
}

// P4 removes this gate.
#[cfg(any())]
mod typed_extras_unary {
    //! Vendor reply fields read through `extras::<P>()` from recorded unary
    //! replies, with no indexing into `raw`.

    use rig::completion::{CompletionRequest, CompletionResponse};
    use rig::driver::Transport;
    use rig::providers::deepseek::DEEPSEEK_FLASH;
    use rig::providers::deepseek::extension::DeepSeek;
    use rig::providers::openai::wire::{Chat, DEEPSEEK, OPENROUTER, OpenAIConfig};
    use rig::providers::openrouter::extension::OpenRouter;
    use rig::test_utils::RecordingHttpClient;
    use rig_core::operation::Completion;
    use rig_core::wire::Wire;

    pub(super) const KEY: &str = "typed-options";
    pub(super) const OPENROUTER_UNARY: &str = "raw_capture_matrix/raw_round_trips_openrouter_type";
    pub(super) const DEEPSEEK_UNARY: &str = "portability_matrix/from_openai_responses";

    /// The body of recorded interaction `index` of `provider/scenario`.
    pub(super) fn recorded_body(provider: &str, scenario: &str, index: usize) -> String {
        crate::cassettes::recorded_statuses_and_bodies(provider, scenario)
            .into_iter()
            .nth(index)
            .map(|(_, body)| body)
            .unwrap_or_else(|| panic!("{provider}/{scenario} records interaction {index}"))
    }

    pub(super) fn prompt() -> CompletionRequest {
        CompletionRequest::new("Reply with the single word: pong").max_tokens(16)
    }

    /// The recorded OpenRouter model, on the Chat wire.
    pub(super) fn openrouter() -> Chat {
        Chat::new(
            OpenAIConfig::with_key(&OPENROUTER, KEY),
            "openai/gpt-4o-mini",
        )
    }

    pub(super) fn deepseek() -> Chat {
        Chat::new(OpenAIConfig::with_key(&DEEPSEEK, KEY), DEEPSEEK_FLASH)
    }

    /// `body`, a recorded unary reply, decoded by `wire` as a live call does.
    pub(super) async fn unary<W>(
        wire: W,
        body: String,
        request: CompletionRequest,
    ) -> CompletionResponse
    where
        W: Wire<Op = Completion>,
        RecordingHttpClient: Transport<W>,
    {
        rig::Model::new(wire, RecordingHttpClient::new(body))
            .call(request)
            .await
            .expect("the recorded reply decodes")
    }

    #[tokio::test]
    async fn openrouter_and_deepseek_extras_from_unary_recordings() {
        let openrouter_reply = unary(
            openrouter(),
            recorded_body("openrouter", OPENROUTER_UNARY, 0),
            prompt(),
        )
        .await;
        let extras = openrouter_reply
            .extras::<OpenRouter>()
            .expect("an OpenRouter reply has OpenRouter extras")
            .expect("the extras deserialize");
        assert_eq!(extras.provider.as_deref(), Some("Azure"));
        assert_eq!(extras.cost, Some(2.7e-6));
        assert_eq!(extras.native_finish_reason.as_deref(), Some("stop"));
        assert!(openrouter_reply.extras::<DeepSeek>().is_none());

        let deepseek_reply = unary(
            deepseek(),
            recorded_body("deepseek", DEEPSEEK_UNARY, 0),
            prompt(),
        )
        .await;
        let extras = deepseek_reply
            .extras::<DeepSeek>()
            .expect("a DeepSeek reply has DeepSeek extras")
            .expect("the extras deserialize");
        assert_eq!(extras.prompt_cache_hit_tokens, Some(256));
        assert_eq!(extras.prompt_cache_miss_tokens, Some(247));
        assert!(deepseek_reply.extras::<OpenRouter>().is_none());
    }
}

// P5 removes this gate.
#[cfg(any())]
mod typed_extras_streamed {
    //! The same extras from recorded streams: a streamed reply's `raw` is the
    //! unary document, so one `Extras` type reads both.

    use bytes::Bytes;
    use futures::StreamExt;
    use rig::completion::{CompletionRequest, CompletionResponse};
    use rig::driver::Transport;
    use rig::providers::deepseek::extension::DeepSeek;
    use rig::providers::openrouter::extension::OpenRouter;
    use rig::test_utils::MockStreamingClient;
    use rig_core::operation::Completion;
    use rig_core::wire::Wire;

    use super::typed_extras_unary::{
        OPENROUTER_UNARY, deepseek, openrouter, prompt, recorded_body, unary,
    };

    pub(super) const OPENROUTER_STREAMED: &str =
        "raw_stream_capture_matrix/stream_raw_exposes_terminal_cost_and_provider";
    const DEEPSEEK_STREAMED: &str = "prompt_caching/streaming_probe";

    /// `body`, a recorded event stream, folded by `wire` as a live stream is.
    pub(super) async fn streamed<W>(
        wire: W,
        body: String,
        request: CompletionRequest,
    ) -> CompletionResponse
    where
        W: Wire<Op = Completion>,
        MockStreamingClient: Transport<W>,
    {
        let model = rig::Model::new(
            wire,
            MockStreamingClient {
                sse_bytes: Bytes::from(body),
            },
        );
        let mut stream = model.stream(request).expect("the recorded stream opens");
        while let Some(item) = stream.next().await {
            item.expect("the recorded stream yields no error");
        }
        stream.finish().await.expect("the recorded stream ends")
    }

    #[tokio::test]
    async fn the_same_extras_from_streamed_recordings() {
        let from_stream = streamed(
            openrouter(),
            recorded_body("openrouter", OPENROUTER_STREAMED, 0),
            prompt(),
        )
        .await
        .extras::<OpenRouter>()
        .expect("an OpenRouter reply has OpenRouter extras")
        .expect("the extras deserialize");
        let from_body = unary(
            openrouter(),
            recorded_body("openrouter", OPENROUTER_UNARY, 0),
            prompt(),
        )
        .await
        .extras::<OpenRouter>()
        .expect("an OpenRouter reply has OpenRouter extras")
        .expect("the extras deserialize");
        assert_eq!(from_stream.provider.as_deref(), Some("Azure"));
        assert_eq!(from_stream.cost, Some(2.7e-6));
        assert_eq!(from_stream.native_finish_reason.as_deref(), Some("stop"));
        assert_eq!(from_stream, from_body);

        let deepseek_reply = streamed(
            deepseek(),
            recorded_body("deepseek", DEEPSEEK_STREAMED, 0),
            prompt(),
        )
        .await;
        let extras = deepseek_reply
            .extras::<DeepSeek>()
            .expect("a DeepSeek reply has DeepSeek extras")
            .expect("the extras deserialize");
        assert_eq!(extras.prompt_cache_hit_tokens, Some(4864));
        assert_eq!(extras.prompt_cache_miss_tokens, Some(202));
    }
}

// P6 removes this gate.
#[cfg(any())]
mod citations_and_cost {
    //! Citations and cost in core types: a recorded citation reply decodes
    //! with citations and replays its native citations unchanged; a reply's
    //! cost comes from the provider when it reports one and from the catalog
    //! otherwise.

    use rig::catalog::Catalog;
    use rig::completion::{CompletionRequest, Message, Usage};
    use rig::message::{AssistantContent, SourceLocation};
    use rig::providers::cohere::{COMMAND_A_03_2025, CohereConfig, NativeChat};
    use rig::providers::deepseek::DEEPSEEK_FLASH;
    use rig::providers::openai::wire::{Chat, DEEPSEEK, OpenAIConfig};
    use rig::providers::registry::ProviderId;

    use super::harness_switch::encode;
    use super::typed_extras_streamed::{OPENROUTER_STREAMED, streamed};
    use super::typed_extras_unary::{
        DEEPSEEK_UNARY, KEY, OPENROUTER_UNARY, deepseek, openrouter, prompt, recorded_body, unary,
    };

    const COHERE_CITATIONS: &str = "native/a_conversation_switching_routes_replays";
    const QUESTION: &str = "Which dock has beacon amber-73?";

    fn close(left: f64, right: f64) -> bool {
        (left - right).abs() <= 1e-12
    }

    #[tokio::test]
    async fn a_cohere_citation_reply_decodes_with_citations() {
        let wire = NativeChat::new(CohereConfig::new(KEY), COMMAND_A_03_2025);
        let reply = unary(
            wire.clone(),
            recorded_body("cohere", COHERE_CITATIONS, 0),
            CompletionRequest::new(QUESTION).max_tokens(64),
        )
        .await;
        let text = reply
            .choice
            .iter()
            .find_map(|block| match block {
                AssistantContent::Text(text) => Some(text),
                _ => None,
            })
            .expect("the reply has a text block");
        let [citation] = text.citations() else {
            panic!("one citation: {:?}", text.citations());
        };
        assert_eq!(text.cited(citation), Some("Dock Seven"));
        let [source] = citation.sources.as_slice() else {
            panic!("one source: {:?}", citation.sources);
        };
        assert!(
            matches!(
                &source.location,
                SourceLocation::Document { id: Some(id), .. } if id == "harbor-record-1"
            ),
            "{source:?}"
        );

        // The next request sends the provider's own citation back, as the
        // recorded continuation did.
        let mut next =
            CompletionRequest::new("Repeat the dock's name in capital letters.").max_tokens(64);
        next.chat_history.splice(
            0..0,
            [
                Message::user(QUESTION),
                reply.message().expect("the reply is an assistant turn"),
            ],
        );
        let body = encode(&wire, next).expect("the continuation encodes");
        let turns = crate::cassettes::recorded_json_turns("cohere", COHERE_CITATIONS);
        let (recorded, _) = turns.get(2).expect("the scenario records a third request");
        assert_eq!(
            body.pointer("/messages/1/citations"),
            recorded.pointer("/messages/1/citations")
        );
    }

    #[tokio::test]
    async fn reported_cost_wins() {
        for reply in [
            unary(
                openrouter(),
                recorded_body("openrouter", OPENROUTER_UNARY, 0),
                prompt(),
            )
            .await,
            streamed(
                openrouter(),
                recorded_body("openrouter", OPENROUTER_STREAMED, 0),
                prompt(),
            )
            .await,
        ] {
            let cost = reply.usage.cost.expect("OpenRouter reports its cost");
            assert!(close(cost.total, 2.7e-6), "{cost:?}");
        }
    }

    #[tokio::test]
    async fn catalog_cost_when_none_is_reported() {
        let reply = unary(
            deepseek(),
            recorded_body("deepseek", DEEPSEEK_UNARY, 0),
            prompt(),
        )
        .await;
        let pricing = Catalog::builtin()
            .get(
                ProviderId::resolve("deepseek").expect("DeepSeek is registered"),
                DEEPSEEK_FLASH,
            )
            .and_then(|spec| spec.pricing.as_ref())
            .expect("the catalog prices DeepSeek Flash");
        let Usage {
            input_tokens: Some(input),
            output_tokens: Some(output),
            cached_input_tokens,
            cache_creation_input_tokens,
            cost: Some(cost),
            ..
        } = reply.usage
        else {
            panic!(
                "DeepSeek reports tokens and the catalog prices them: {:?}",
                reply.usage
            );
        };
        let read = cached_input_tokens.unwrap_or(0);
        let written = cache_creation_input_tokens.unwrap_or(0);
        let per_token = |price: f64| price / 1_000_000.0;
        assert!(close(
            cost.input,
            (input - read - written) as f64 * per_token(pricing.input)
        ));
        assert!(close(
            cost.output,
            output as f64 * per_token(pricing.output)
        ));
        assert!(close(
            cost.cache_read,
            read as f64 * per_token(pricing.cache_read.unwrap_or(pricing.input))
        ));
        assert!(close(
            cost.total,
            cost.input + cost.output + cost.cache_read + cost.cache_write
        ));

        // A model the catalog does not list has no cost.
        let unlisted = unary(
            Chat::new(OpenAIConfig::with_key(&DEEPSEEK, KEY), "not-in-the-catalog"),
            recorded_body("deepseek", DEEPSEEK_UNARY, 0),
            prompt(),
        )
        .await;
        assert_eq!(unlisted.usage.cost, None);
    }
}

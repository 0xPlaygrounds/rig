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
        UnsupportedOption,
    };
    use rig::providers::anthropic::completion::CLAUDE_OPUS_4_8;
    use rig::providers::anthropic::wire::AnthropicConfig;
    use rig::providers::gemini::GeminiConfig;
    use rig::providers::gemini::completion::{GEMINI_3_FLASH_PREVIEW, GenerateContent};
    use rig::providers::openai::GPT_5_5;
    use rig::providers::openai::responses_api::wire::Responses;
    use rig::providers::openai::wire::{Chat, OPENROUTER, OpenAIConfig};
    use rig::test_utils::TraceCapture;
    use rig_core::error::ProviderError;
    use rig_core::operation::Completion;
    use rig_core::wire::{Body, Encoded, Mode, Operation, Wire};
    use rig_test_support::cassette_models::AnthropicModels;
    use serde_json::Value;

    pub(super) const KEY: &str = "typed-options";
    const OPENROUTER_MODEL: &str = "anthropic/claude-sonnet-4.5";

    /// `request` as `wire` prepares it, the way the driver does before
    /// encoding. A refused option fails here, before any wire encodes.
    pub(super) fn prepared<W: Wire<Op = Completion>>(
        wire: &W,
        request: CompletionRequest,
    ) -> Result<CompletionRequest, ProviderError> {
        Completion::prepare(request, &wire.describe())
    }

    /// The option an error refuses, if it is a refusal.
    pub(super) fn unsupported(error: &ProviderError) -> Option<&UnsupportedOption> {
        match error {
            ProviderError::UnsupportedOption(option) => Some(option),
            _ => None,
        }
    }

    /// The JSON body `wire` sends for `request`.
    pub(super) fn encode<W: Wire<Op = Completion, Payload = Encoded>>(
        wire: &W,
        request: CompletionRequest,
    ) -> Result<Value, ProviderError> {
        let encoded = wire.encode(prepared(wire, request)?, Mode::Unary)?;
        let Body::Bytes(bytes) = encoded.request.body() else {
            return Err(ProviderError::request(
                "a completion body is JSON, not multipart",
            ));
        };
        Ok(serde_json::from_slice(bytes)?)
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

    pub(super) fn anthropic() -> rig::providers::anthropic::Messages {
        AnthropicModels::new(AnthropicConfig::new(KEY), crate::cassettes::local_http())
            .completion(CLAUDE_OPUS_4_8)
            .wire
    }

    pub(super) fn openai_responses() -> Responses {
        Responses::new(OpenAIConfig::new(KEY), GPT_5_5)
    }

    pub(super) fn gemini() -> GenerateContent {
        GenerateContent::new(GeminiConfig::new(KEY), GEMINI_3_FLASH_PREVIEW)
    }

    pub(super) fn openrouter() -> Chat {
        Chat::new(OpenAIConfig::with_key(&OPENROUTER, KEY), OPENROUTER_MODEL)
    }

    pub(super) fn bedrock_wire() -> Converse {
        Converse::new(ANTHROPIC_CLAUDE_SONNET_5)
    }

    pub(super) fn bedrock(request: CompletionRequest) -> Result<Value, ProviderError> {
        let wire = bedrock_wire();
        let request = prepared(&wire, request)?;
        let encoded = wire.encode(request, Mode::Unary)?;
        Ok(serde_json::to_value(&encoded.body)?)
    }

    #[test]
    fn one_options_value_drives_five_wires() {
        let capture = TraceCapture::default();
        let (anthropic, responses, gemini, bedrock, openrouter) =
            tracing::subscriber::with_default(capture.subscriber(), || {
                let encode_one = |result: Result<Value, ProviderError>| {
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
        // `output_config`, one automatic 1 h marker at the top level and none
        // placed by hand, the standard tier only.
        assert_eq!(text(&anthropic, "/thinking/type"), Some("adaptive"));
        assert_eq!(text(&anthropic, "/output_config/effort"), Some("high"));
        assert_eq!(text(&anthropic, "/cache_control/type"), Some("ephemeral"));
        assert_eq!(text(&anthropic, "/cache_control/ttl"), Some("1h"));
        assert!(
            last(&anthropic, "/system").get("cache_control").is_none(),
            "{anthropic}"
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
        let unsupported = unsupported(&error).expect("the error names the unsupported option");
        assert_eq!(unsupported.option, "cache");
        assert_eq!(unsupported.provider, "gcp.gemini");
        assert_eq!(unsupported.model, GEMINI_3_FLASH_PREVIEW);
    }
}

mod no_silent_drop {
    //! A cache asked of a dialect that cannot cache on request is refused,
    //! or skipped with a warning under `Ignore`; it is never dropped
    //! without a word, as `Chat::with_prompt_caching` was.
    //!
    //! Encode-only: a refused option sends no request to record.

    use rig::completion::{CacheRetention, CompletionRequest, OnUnsupported};
    use rig::providers::cohere::{COMMAND_A_03_2025, CohereConfig};
    use rig::providers::deepseek::DEEPSEEK_FLASH;
    use rig::providers::openai::wire::{Chat, DEEPSEEK, OpenAIConfig};
    use rig::test_utils::TraceCapture;

    use super::harness_switch::{KEY, bedrock_wire, encode, prepared, unsupported};

    fn request(cache: CacheRetention, policy: OnUnsupported) -> CompletionRequest {
        CompletionRequest::new("Reply with the single word: pong")
            .max_tokens(16)
            .cache(cache)
            .on_unsupported(policy)
    }

    /// Cohere's Compatibility route, the default `CohereChat` route.
    pub(super) fn cohere() -> rig::providers::cohere::CohereChat {
        CohereConfig::new(KEY).completion(COMMAND_A_03_2025)
    }

    pub(super) fn deepseek() -> Chat {
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
            let unsupported = unsupported(&error).expect("the error names the unsupported option");
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

    /// Bedrock Converse builds an SDK request, not request bytes, so only
    /// its required `ReplayTarget::map_options` and the driver's `prepare`
    /// stand between a set option and a silent drop. `prepare` refuses the
    /// option before the wire's `encode` runs.
    #[test]
    fn the_driver_refuses_an_option_before_any_wire_encodes() {
        let seed = |policy| {
            CompletionRequest::new("Reply with the single word: pong")
                .max_tokens(16)
                .seed(7)
                .on_unsupported(policy)
        };
        let error = prepared(&bedrock_wire(), seed(OnUnsupported::Error))
            .expect_err("Bedrock Converse has no seed");
        let refused = unsupported(&error).expect("the error names the unsupported option");
        assert_eq!(refused.option, "seed");
        assert_eq!(refused.provider, "aws_bedrock");

        let capture = TraceCapture::default();
        let request = tracing::subscriber::with_default(capture.subscriber(), || {
            prepared(&bedrock_wire(), seed(OnUnsupported::Ignore))
        })
        .expect("Ignore prepares the rest of the request");
        assert_eq!(request.options.seed, None, "an ignored option is cleared");
        let warnings = capture.warnings();
        let skipped = warnings
            .iter()
            .filter(|warning| warning.contains("option=seed"))
            .count();
        assert_eq!(skipped, 1, "{warnings:?}");
    }
}

mod option_matrix {
    //! A golden of section 6: each option set alone on each wire gives
    //! exactly the body delta or the refusal its cell names. `omit` passes
    //! only where the cell says "omit", because any other cell expects a
    //! change or an error; a wire that answers `Mapping::Nothing` for a set
    //! option fails with a non-`UnsupportedOption` error.
    //!
    //! Encode-only. The rows cover every completion wire and dialect in
    //! rig-core and Bedrock, `InteractionResume` included; Vertex AI, gRPC
    //! and Candle pin their cells in their own crates, which this target
    //! does not build.

    use rig::completion::{
        CacheRetention, CompletionRequest, Effort, GenerationOptions, OnUnsupported, ServiceTier,
        Verbosity,
    };
    use rig::providers::anthropic;
    use rig::providers::chatgpt;
    use rig::providers::gemini::GeminiConfig;
    use rig::providers::gemini::completion::GEMINI_3_FLASH_PREVIEW;
    use rig::providers::gemini::interactions_api::{InteractionResume, Interactions};
    use rig::providers::openai::responses_api::wire::Responses;
    use rig::providers::openai::wire::{
        AZURE, Chat, DOUBLEWORD, Dialect, GROQ, HUGGINGFACE, HYPERBOLIC, LLAMACPP, MINIMAX, MIRA,
        MISTRAL, MOONSHOT, OLLAMA, OPENAI, OPENROUTER, OpenAIConfig, PERPLEXITY, TOGETHER, VENICE,
        XIAOMIMIMO, ZAI,
    };
    use rig_core::error::ProviderError;
    use rig_core::wire::{Body, Mode, Wire};
    use rig_test_support::cassette_models::AnthropicModels;
    use serde_json::{Value, json};

    use super::harness_switch::{
        KEY, anthropic, bedrock, encode, gemini, openai_responses, openrouter, prepared,
        unsupported,
    };
    use super::no_silent_drop::{cohere, deepseek};

    type Encode = Box<dyn Fn(CompletionRequest) -> Result<Value, ProviderError>>;

    /// What one cell of section 6 says the wire does.
    enum Cell {
        /// The body equals the baseline with this object deep-merged in.
        Merge(Value),
        /// The body equals the baseline with this value pushed onto the
        /// array at the pointer (a block-level cache marker).
        Push(&'static str, Value),
        /// The body equals the baseline.
        Omit,
        /// `UnsupportedOption` naming the field and the provider.
        Refuse,
    }
    use Cell::{Merge, Omit, Push, Refuse};

    /// The values each field is set to, in `cells` column order.
    fn one_field_each() -> Vec<(&'static str, GenerationOptions)> {
        let base = GenerationOptions::default().on_unsupported(OnUnsupported::Error);
        vec![
            ("reasoning", base.clone().reasoning(Effort::High)),
            ("cache", base.clone().cache(CacheRetention::Short)),
            (
                "service_tier",
                base.clone().service_tier(ServiceTier::Default),
            ),
            ("verbosity", base.clone().verbosity(Verbosity::Low)),
            (
                "parallel_tool_calls",
                base.clone().parallel_tool_calls(false),
            ),
            ("top_p", base.clone().top_p(0.5)),
            ("seed", base.clone().seed(7)),
            ("stop", base.stop(["END"])),
        ]
    }

    /// One row per wire, one cell per field of `one_field_each`, read from
    /// section 6 for the model each wire uses. The request has no tools and
    /// no preamble.
    fn cells() -> Vec<(&'static str, Encode, [Cell; 8])> {
        let mut rows: Vec<(&'static str, Encode, [Cell; 8])> = vec![
            (
                // 6.1, class A3 (`claude-opus-4-8`).
                "anthropic",
                Box::new(|r| encode(&anthropic(), r)),
                [
                    Merge(
                        json!({"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}),
                    ),
                    Merge(json!({"cache_control": {"type": "ephemeral"}})),
                    Merge(json!({"service_tier": "standard_only"})),
                    Refuse,
                    Omit,   // no tools to bind
                    Refuse, // A3
                    Refuse,
                    Merge(json!({"stop_sequences": ["END"]})),
                ],
            ),
            (
                // 6.3, `gpt-5.5`: 24 h cache only; `top_p` depends on an
                // effective effort that is unverified, so it is refused.
                "openai",
                Box::new(|r| encode(&openai_responses(), r)),
                [
                    Merge(
                        json!({"reasoning": {"effort": "high"}, "include": ["reasoning.encrypted_content"]}),
                    ),
                    Refuse,
                    Merge(json!({"service_tier": "default"})),
                    Merge(json!({"text": {"verbosity": "low"}})),
                    Merge(json!({"parallel_tool_calls": false})),
                    Refuse,
                    Refuse,
                    Refuse,
                ],
            ),
            (
                // 6.4 GC, `gemini-3-flash-preview`.
                "gcp.gemini",
                Box::new(|r| encode(&gemini(), r)),
                [
                    Merge(
                        json!({"generationConfig": {"thinkingConfig": {"thinkingLevel": "high"}}}),
                    ),
                    Omit,
                    Merge(json!({"serviceTier": "standard"})),
                    Refuse,
                    Refuse,
                    Merge(json!({"generationConfig": {"topP": 0.5}})),
                    Merge(json!({"generationConfig": {"seed": 7}})),
                    Merge(json!({"generationConfig": {"stopSequences": ["END"]}})),
                ],
            ),
            (
                // 6.5, Claude Sonnet 5 (class A4).
                "aws_bedrock",
                Box::new(bedrock),
                [
                    Merge(
                        json!({"additionalModelRequestFields": {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}}),
                    ),
                    Push(
                        "/messages/0/content",
                        json!({"cachePoint": {"type": "default"}}),
                    ),
                    Merge(json!({"serviceTier": {"type": "default"}})),
                    Refuse,
                    Refuse,
                    Refuse,
                    Refuse,
                    Merge(json!({"inferenceConfig": {"stopSequences": ["END"]}})),
                ],
            ),
            (
                // 6.2, an Anthropic upstream.
                "openrouter",
                Box::new(|r| encode(&openrouter(), r)),
                [
                    Merge(json!({"reasoning": {"effort": "high"}})),
                    Merge(json!({"cache_control": {"type": "ephemeral"}})),
                    Merge(json!({"service_tier": "default"})),
                    Merge(json!({"verbosity": "low"})),
                    Merge(json!({"parallel_tool_calls": false})),
                    Merge(json!({"top_p": 0.5})),
                    Merge(json!({"seed": 7})),
                    Merge(json!({"stop": ["END"]})),
                ],
            ),
            (
                // 6.2, the Compatibility route; Command A does not think.
                "cohere",
                Box::new(|r| encode(&cohere(), r)),
                [
                    Refuse,
                    Refuse,
                    Refuse,
                    Refuse,
                    Refuse,
                    Merge(json!({"top_p": 0.5})),
                    Merge(json!({"seed": 7})),
                    Merge(json!({"stop": ["END"]})),
                ],
            ),
            (
                // 6.2.
                "deepseek",
                Box::new(|r| encode(&deepseek(), r)),
                [
                    Merge(json!({"thinking": {"type": "enabled"}, "reasoning_effort": "high"})),
                    Omit,
                    Refuse,
                    Refuse,
                    Refuse,
                    Merge(json!({"top_p": 0.5})),
                    Refuse,
                    Merge(json!({"stop": ["END"]})),
                ],
            ),
        ];
        rows.extend(chat_dialects());
        rows.extend(responses_dialects());
        rows.extend(messages_dialects());
        rows.extend(other_wires());
        rows
    }

    fn chat(dialect: &'static Dialect, model: &'static str) -> Encode {
        Box::new(move |r| encode(&Chat::new(OpenAIConfig::with_key(dialect, KEY), model), r))
    }

    fn responses(dialect: &'static Dialect, model: &'static str) -> Encode {
        Box::new(move |r| {
            encode(
                &Responses::new(OpenAIConfig::with_key(dialect, KEY), model),
                r,
            )
        })
    }

    fn messages(dialect: &'static anthropic::Dialect, model: &'static str) -> Encode {
        Box::new(move |r| {
            let wire = AnthropicModels::new(
                anthropic::AnthropicConfig::with_key(dialect, KEY),
                crate::cassettes::local_http(),
            )
            .completion(model)
            .wire;
            encode(&wire, r)
        })
    }

    /// Every option refused, as a wire with no confirmed field answers.
    fn refused() -> [Cell; 8] {
        [
            Refuse, Refuse, Refuse, Refuse, Refuse, Refuse, Refuse, Refuse,
        ]
    }

    /// 6.2: the Chat Completions dialects the rows above leave out.
    fn chat_dialects() -> Vec<(&'static str, Encode, [Cell; 8])> {
        let openai_like = |reasoning: Cell, cache: Cell, tier: Cell, verbosity: Cell| {
            [
                reasoning,
                cache,
                tier,
                verbosity,
                Merge(json!({"parallel_tool_calls": false})),
                Merge(json!({"top_p": 0.5})),
                Merge(json!({"seed": 7})),
                Merge(json!({"stop": ["END"]})),
            ]
        };
        vec![
            (
                // GPT-5.2 samples at its default effort `none`; a reasoning
                // model takes no `stop`.
                "openai",
                chat(&OPENAI, "gpt-5.2"),
                [
                    Merge(json!({"reasoning_effort": "high"})),
                    Merge(json!({"prompt_cache_retention": "in_memory"})),
                    Merge(json!({"service_tier": "default"})),
                    Merge(json!({"verbosity": "low"})),
                    Merge(json!({"parallel_tool_calls": false})),
                    Merge(json!({"top_p": 0.5})),
                    Merge(json!({"seed": 7})),
                    Refuse,
                ],
            ),
            (
                // `gpt-5-mini` never samples.
                "openai",
                chat(&OPENAI, "gpt-5-mini"),
                [
                    Merge(json!({"reasoning_effort": "high"})),
                    Merge(json!({"prompt_cache_retention": "in_memory"})),
                    Merge(json!({"service_tier": "default"})),
                    Merge(json!({"verbosity": "low"})),
                    Merge(json!({"parallel_tool_calls": false})),
                    Refuse,
                    Merge(json!({"seed": 7})),
                    Refuse,
                ],
            ),
            (
                // A model that does not reason takes `stop`, and only
                // `medium` verbosity.
                "openai",
                chat(&OPENAI, "gpt-4.1-mini"),
                [
                    Refuse,
                    Merge(json!({"prompt_cache_retention": "in_memory"})),
                    Merge(json!({"service_tier": "default"})),
                    Refuse,
                    Merge(json!({"parallel_tool_calls": false})),
                    Merge(json!({"top_p": 0.5})),
                    Merge(json!({"seed": 7})),
                    Merge(json!({"stop": ["END"]})),
                ],
            ),
            (
                // rig's default `api-version` predates the newer fields.
                "azure.openai",
                chat(&AZURE, "prod-deployment"),
                openai_like(Refuse, Refuse, Refuse, Refuse),
            ),
            (
                "groq",
                chat(&GROQ, "qwen/qwen3.8-27b"),
                openai_like(
                    Merge(json!({"reasoning_effort": "high"})),
                    Omit,
                    Merge(json!({"service_tier": "on_demand"})),
                    Refuse,
                ),
            ),
            (
                "mistral",
                chat(&MISTRAL, "mistral-medium-latest"),
                [
                    Merge(json!({"reasoning_effort": "high"})),
                    Omit,
                    Merge(json!({"service_tier": "standard_only"})),
                    Refuse,
                    Merge(json!({"parallel_tool_calls": false})),
                    Merge(json!({"top_p": 0.5})),
                    Merge(json!({"random_seed": 7})),
                    Merge(json!({"stop": ["END"]})),
                ],
            ),
            (
                "xai",
                chat(&rig::providers::xai::DIALECT, "grok-4.7"),
                [
                    Merge(json!({"reasoning_effort": "high"})),
                    Omit,
                    Merge(json!({"service_tier": "default"})),
                    Refuse,
                    Merge(json!({"parallel_tool_calls": false})),
                    Merge(json!({"top_p": 0.5})),
                    Refuse,
                    Refuse,
                ],
            ),
            (
                "together",
                chat(&TOGETHER, "deepseek-ai/DeepSeek-R1"),
                [
                    Merge(json!({"reasoning": {"enabled": true}, "reasoning_effort": "high"})),
                    Omit,
                    Refuse,
                    Refuse,
                    Refuse,
                    Merge(json!({"top_p": 0.5})),
                    Merge(json!({"seed": 7})),
                    Merge(json!({"stop": ["END"]})),
                ],
            ),
            (
                "venice",
                chat(&VENICE, "venice-uncensored"),
                openai_like(
                    Merge(json!({"reasoning_effort": "high"})),
                    Merge(json!({"prompt_cache_retention": "default"})),
                    Refuse,
                    Refuse,
                ),
            ),
            (
                "moonshot",
                chat(&MOONSHOT, "kimi-k3"),
                [
                    Merge(json!({"reasoning_effort": "high"})),
                    Merge(json!({"prompt_cache_options": {"mode": "implicit", "ttl": "5m"}})),
                    Refuse,
                    Refuse,
                    Refuse,
                    Refuse,
                    Refuse,
                    Merge(json!({"stop": ["END"]})),
                ],
            ),
            (
                "zai",
                chat(&ZAI, "glm-5"),
                [
                    Merge(json!({"thinking": {"type": "enabled"}, "reasoning_effort": "high"})),
                    Omit,
                    Refuse,
                    Refuse,
                    Refuse,
                    Merge(json!({"top_p": 0.5})),
                    Refuse,
                    Merge(json!({"stop": ["END"]})),
                ],
            ),
            (
                "llamacpp",
                chat(&LLAMACPP, "qwen3"),
                openai_like(
                    Merge(json!({"reasoning_effort": "high"})),
                    Omit,
                    Refuse,
                    Refuse,
                ),
            ),
            (
                "ollama",
                chat(&OLLAMA, "qwen3:4b"),
                [
                    Merge(json!({"reasoning_effort": "high"})),
                    Omit,
                    Refuse,
                    Refuse,
                    Refuse,
                    Merge(json!({"top_p": 0.5})),
                    Merge(json!({"seed": 7})),
                    Merge(json!({"stop": ["END"]})),
                ],
            ),
            (
                "perplexity",
                chat(&PERPLEXITY, "sonar-deep-research"),
                [
                    Merge(json!({"reasoning_effort": "high"})),
                    Refuse,
                    Refuse,
                    Refuse,
                    Refuse,
                    Merge(json!({"top_p": 0.5})),
                    Refuse,
                    Merge(json!({"stop": ["END"]})),
                ],
            ),
            (
                // MiniMax M2 thinks always and takes no level.
                "minimax",
                chat(&MINIMAX, "MiniMax-M2.7"),
                [
                    Refuse,
                    Omit,
                    Refuse,
                    Refuse,
                    Refuse,
                    Merge(json!({"top_p": 0.5})),
                    Refuse,
                    Merge(json!({"stop": ["END"]})),
                ],
            ),
            (
                // MiMo fixes `top_p` while it thinks, which it does unless
                // the request turns thinking off.
                "xiaomimimo",
                chat(&XIAOMIMIMO, "mimo-v2.5"),
                [
                    Refuse,
                    Omit,
                    Refuse,
                    Refuse,
                    Refuse,
                    Refuse,
                    Refuse,
                    Merge(json!({"stop": ["END"]})),
                ],
            ),
            (
                "copilot",
                chat(&rig::providers::copilot::wire::DIALECT, "gpt-4.1"),
                refused(),
            ),
            (
                "huggingface",
                chat(&HUGGINGFACE, "meta-llama/Llama-3.3-70B-Instruct"),
                refused(),
            ),
            (
                "hyperbolic",
                chat(&HYPERBOLIC, "meta-llama/Llama-3.3-70B-Instruct"),
                refused(),
            ),
            (
                "doubleword",
                chat(&DOUBLEWORD, "Qwen/Qwen3-235B-A22B"),
                refused(),
            ),
            ("mira", chat(&MIRA, "claude-3.5-sonnet"), refused()),
        ]
    }

    /// 6.3: the Responses dialects the rows above leave out.
    fn responses_dialects() -> Vec<(&'static str, Encode, [Cell; 8])> {
        let reasoning = || {
            Merge(
                json!({"reasoning": {"effort": "high"}, "include": ["reasoning.encrypted_content"]}),
            )
        };
        vec![
            (
                "xai",
                responses(&rig::providers::xai::DIALECT, "grok-4.7"),
                [
                    reasoning(),
                    Omit,
                    Merge(json!({"service_tier": "default"})),
                    Refuse,
                    Merge(json!({"parallel_tool_calls": false})),
                    Merge(json!({"top_p": 0.5})),
                    Refuse,
                    Refuse,
                ],
            ),
            (
                "openrouter",
                responses(&OPENROUTER, "anthropic/claude-sonnet-4.5"),
                [
                    reasoning(),
                    Merge(json!({"cache_control": {"type": "ephemeral"}})),
                    Merge(json!({"service_tier": "default"})),
                    Merge(json!({"text": {"verbosity": "low"}})),
                    Merge(json!({"parallel_tool_calls": false})),
                    Merge(json!({"top_p": 0.5})),
                    Refuse,
                    Refuse,
                ],
            ),
            (
                // The backend gets the ciphertext asked for on every request.
                "chatgpt",
                responses(&chatgpt::DIALECT, chatgpt::GPT_5_4),
                [
                    Merge(json!({"reasoning": {"effort": "high"}})),
                    Refuse,
                    Refuse,
                    Merge(json!({"text": {"verbosity": "low"}})),
                    Merge(json!({"parallel_tool_calls": false})),
                    Refuse,
                    Refuse,
                    Refuse,
                ],
            ),
            (
                "copilot",
                responses(&rig::providers::copilot::wire::DIALECT, "gpt-5.3-codex"),
                [
                    reasoning(),
                    Refuse,
                    Refuse,
                    Refuse,
                    Merge(json!({"parallel_tool_calls": false})),
                    Merge(json!({"top_p": 0.5})),
                    Refuse,
                    Refuse,
                ],
            ),
        ]
    }

    /// 6.1: the Messages-format dialects.
    fn messages_dialects() -> Vec<(&'static str, Encode, [Cell; 8])> {
        vec![
            (
                "zai",
                messages(&anthropic::wire::ZAI, "glm-5"),
                [Refuse, Omit, Refuse, Refuse, Refuse, Refuse, Refuse, Refuse],
            ),
            (
                // A short cache places a block marker on the last message.
                "minimax",
                messages(&anthropic::wire::MINIMAX, "MiniMax-M2.7"),
                [
                    Refuse,
                    Merge(json!({"messages": [{"role": "user", "content": [{
                        "type": "text",
                        "text": "Reply with the single word: pong",
                        "cache_control": {"type": "ephemeral"},
                    }]}]})),
                    Merge(json!({"service_tier": "standard"})),
                    Refuse,
                    Refuse,
                    Merge(json!({"top_p": 0.5})),
                    Refuse,
                    Refuse,
                ],
            ),
            (
                "moonshot",
                messages(&anthropic::wire::MOONSHOT, "kimi-k3"),
                [Refuse, Omit, Refuse, Refuse, Refuse, Refuse, Refuse, Refuse],
            ),
            (
                "xiaomimimo",
                messages(&anthropic::wire::XIAOMIMIMO, "mimo-v2.5"),
                [
                    Refuse,
                    Omit,
                    Refuse,
                    Refuse,
                    Omit, // no tools to bind
                    Merge(json!({"top_p": 0.5})),
                    Refuse,
                    Merge(json!({"stop_sequences": ["END"]})),
                ],
            ),
        ]
    }

    /// 6.4 and 6.5: Interactions, a resumed interaction, Cohere's native API
    /// and Ollama's `/api/chat`.
    fn other_wires() -> Vec<(&'static str, Encode, [Cell; 8])> {
        vec![
            (
                "gcp.gemini",
                Box::new(|r| {
                    encode(
                        &Interactions::new(GeminiConfig::new(KEY), GEMINI_3_FLASH_PREVIEW),
                        r,
                    )
                }),
                [
                    Merge(json!({"generation_config": {"thinking_level": "high"}})),
                    Omit,
                    Merge(json!({"service_tier": "standard"})),
                    Refuse,
                    Refuse,
                    Refuse,
                    Merge(json!({"generation_config": {"seed": 7}})),
                    Merge(json!({"generation_config": {"stop_sequences": ["END"]}})),
                ],
            ),
            (
                // A resumed interaction sends no body; its baseline is `null`.
                "gcp.gemini",
                Box::new(|r| {
                    let wire = InteractionResume::new(GeminiConfig::new(KEY), "interaction-1");
                    let encoded = wire.encode(prepared(&wire, r)?, Mode::Unary)?;
                    match encoded.request.body() {
                        Body::Bytes(bytes) if bytes.is_empty() => Ok(Value::Null),
                        _ => Err(ProviderError::request(
                            "a resumed interaction sends no body",
                        )),
                    }
                }),
                refused(),
            ),
            (
                "cohere",
                Box::new(|r| {
                    encode(
                        &rig::providers::cohere::NativeChat::new(
                            rig::providers::cohere::CohereConfig::new(KEY),
                            rig::providers::cohere::COMMAND_A_03_2025,
                        ),
                        r,
                    )
                }),
                [
                    // Command A does not think.
                    Refuse,
                    Refuse,
                    Refuse,
                    Refuse,
                    Refuse,
                    Merge(json!({"p": 0.5})),
                    Merge(json!({"seed": 7})),
                    Merge(json!({"stop_sequences": ["END"]})),
                ],
            ),
            (
                "cohere",
                Box::new(|r| {
                    encode(
                        &rig::providers::cohere::NativeChat::new(
                            rig::providers::cohere::CohereConfig::new(KEY),
                            "command-a-plus-05-2026",
                        ),
                        r,
                    )
                }),
                [
                    Merge(json!({"thinking": {"type": "enabled"}})),
                    Refuse,
                    Refuse,
                    Refuse,
                    Refuse,
                    Merge(json!({"p": 0.5})),
                    Merge(json!({"seed": 7})),
                    Merge(json!({"stop_sequences": ["END"]})),
                ],
            ),
            (
                "ollama",
                Box::new(|r| {
                    encode(
                        &rig::providers::ollama::Chat::new(
                            rig::providers::ollama::OllamaConfig::new(),
                            "qwen3:4b",
                        ),
                        r,
                    )
                }),
                [
                    Merge(json!({"think": "high"})),
                    Refuse,
                    Refuse,
                    Refuse,
                    Refuse,
                    Merge(json!({"options": {"top_p": 0.5}})),
                    Merge(json!({"options": {"seed": 7}})),
                    Merge(json!({"options": {"stop": ["END"]}})),
                ],
            ),
        ]
    }

    fn request(options: GenerationOptions) -> CompletionRequest {
        CompletionRequest::new("Reply with the single word: pong")
            .max_tokens(16)
            .options(options)
    }

    /// `patch` deep-merged into `body`, objects key by key, anything else
    /// replacing, as `request_params` merges layers.
    fn merged(mut body: Value, patch: &Value) -> Value {
        match (&mut body, patch) {
            (Value::Object(target), Value::Object(patch)) => {
                for (key, value) in patch {
                    let slot = target.shift_remove(key).unwrap_or(Value::Null);
                    target.insert(key.clone(), merged(slot, value));
                }
                body
            }
            _ => patch.clone(),
        }
    }

    #[test]
    fn every_option_alone_gives_its_section_6_cell() {
        for (wire, encode, row) in cells() {
            let baseline = encode(request(GenerationOptions::default()))
                .unwrap_or_else(|error| panic!("{wire}: the baseline encodes: {error}"));
            for ((field, options), cell) in one_field_each().into_iter().zip(row) {
                let result = encode(request(options));
                let expected = match cell {
                    Merge(patch) => merged(baseline.clone(), &patch),
                    Push(pointer, value) => {
                        let mut body = baseline.clone();
                        body.pointer_mut(pointer)
                            .and_then(Value::as_array_mut)
                            .unwrap_or_else(|| panic!("{wire}: {pointer} is an array"))
                            .push(value);
                        body
                    }
                    Omit => baseline.clone(),
                    Refuse => {
                        let error = result.expect_err(&format!("{wire}: `{field}` is refused"));
                        let unsupported = unsupported(&error).unwrap_or_else(|| {
                            panic!("{wire}: `{field}` failed without naming an option: {error}")
                        });
                        assert_eq!(unsupported.option, field, "{wire}");
                        assert_eq!(unsupported.provider, wire, "{wire}");
                        assert!(!unsupported.reason.is_empty(), "{wire}: {field}");
                        continue;
                    }
                };
                let body =
                    result.unwrap_or_else(|error| panic!("{wire}: `{field}` encodes: {error}"));
                assert_eq!(body, expected, "{wire}: `{field}`");
            }
        }
    }
}

mod option_layers {
    //! An agent's options and a run's options merge field by field through
    //! `GenerationOptions::overlay`, the one function rig-agent calls: what
    //! the run sets wins, the rest keeps the agent's value.

    use rig::completion::{CacheRetention, Effort, GenerationOptions, OnUnsupported, Reasoning};

    #[test]
    fn a_run_field_beats_the_agent_field_and_leaves_the_rest() {
        let agent = GenerationOptions::default()
            .reasoning(Effort::High)
            .seed(7)
            .stop(["AGENT"])
            .on_unsupported(OnUnsupported::Ignore);
        let run = GenerationOptions::default()
            .cache(CacheRetention::Long)
            .seed(9)
            .stop(["RUN"]);

        let resolved = agent.clone().overlay(&run);
        assert_eq!(resolved.reasoning, Some(Reasoning::Effort(Effort::High)));
        assert_eq!(resolved.cache, Some(CacheRetention::Long));
        assert_eq!(resolved.seed, Some(9));
        assert_eq!(resolved.stop, vec!["RUN".to_owned()]);
        assert_eq!(resolved.on_unsupported, Some(OnUnsupported::Ignore));

        // A run that sets nothing leaves the agent's options as they are.
        assert_eq!(agent.clone().overlay(&GenerationOptions::default()), agent);
    }

    #[test]
    fn a_run_restores_error_over_the_agents_ignore() {
        let agent = GenerationOptions::default()
            .reasoning(Effort::High)
            .on_unsupported(OnUnsupported::Ignore);
        let run = GenerationOptions::default().on_unsupported(OnUnsupported::Error);
        let resolved = agent.overlay(&run);
        assert_eq!(resolved.unsupported_policy(), OnUnsupported::Error);
        assert_eq!(resolved.reasoning, Some(Reasoning::Effort(Effort::High)));
    }
}

mod precedence {
    //! The merge keeps three behaviours every wire has today: tools in
    //! `additional_params.tools` join rig's own tools rather than replacing
    //! them, a `null` in `additional_params` is a value that is sent, and
    //! the writes a wire makes after the merge read the merged body, so a
    //! raw key still drives them. Two wires change, as section 12.0 states:
    //! OpenAI Responses now sends a raw `null`, and Gemini Interactions
    //! refuses a raw `tools` that is not an array. The ChatGPT backend still
    //! gets no typed field it does not accept.

    use rig::completion::{CompletionRequest, ServiceTier, ToolDefinition, Verbosity};
    use rig::message::{ToolChoice, ToolName};
    use rig::providers::chatgpt;
    use rig::providers::gemini::GeminiConfig;
    use rig::providers::gemini::completion::GEMINI_3_FLASH_PREVIEW;
    use rig::providers::gemini::interactions_api::Interactions;
    use rig::providers::openai::responses_api::wire::Responses;
    use rig::providers::openai::wire::OpenAIConfig;
    use serde_json::{Value, json};

    use super::harness_switch::{
        KEY, anthropic, encode, openai_responses, openrouter, unsupported,
    };
    use super::no_silent_drop::deepseek;

    fn request() -> CompletionRequest {
        CompletionRequest::new("Reply with the single word: pong")
            .max_tokens(16)
            .top_p(0.5)
    }

    fn lookup() -> ToolDefinition {
        ToolDefinition::new(
            ToolName::try_from("lookup").expect("the name is not empty"),
            "Look a word up.",
            json!({"type": "object", "properties": {}}),
        )
    }

    /// The `name` of each entry of the body's `tools`, in order.
    fn tool_names(body: &Value) -> Vec<&str> {
        body.get("tools")
            .and_then(Value::as_array)
            .map(|tools| {
                tools
                    .iter()
                    .filter_map(|tool| {
                        tool.get("name")
                            .or_else(|| tool.pointer("/function/name"))
                            .and_then(Value::as_str)
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    #[test]
    fn raw_tools_are_appended_to_rig_tools() {
        // `claude-opus-4-8` (A3) refuses `top_p`, so this half sets no option.
        let server_tool = json!({"type": "web_search_20250305", "name": "web_search"});
        let sent = encode(
            &anthropic(),
            CompletionRequest::new("Reply with the single word: pong")
                .max_tokens(16)
                .tool(lookup())
                .additional_params(json!({"tools": [server_tool]})),
        )
        .expect("the request encodes");
        assert_eq!(tool_names(&sent), ["lookup", "web_search"], "{sent}");

        let function = json!({"type": "function", "function": {
            "name": "raw_lookup", "parameters": {"type": "object", "properties": {}},
        }});
        let sent = encode(
            &openrouter(),
            request()
                .tool(lookup())
                .additional_params(json!({"tools": [function]})),
        )
        .expect("the request encodes");
        assert_eq!(tool_names(&sent), ["lookup", "raw_lookup"], "{sent}");
    }

    #[test]
    fn null_is_sent() {
        let sent = encode(
            &openrouter(),
            request().additional_params(json!({"top_p": null, "user": null})),
        )
        .expect("the request encodes");
        assert_eq!(sent.get("top_p"), Some(&Value::Null), "{sent}");
        assert_eq!(sent.get("user"), Some(&Value::Null), "{sent}");

        // Responses skips a raw `null` today; it is now sent as on Chat.
        let sent = encode(
            &openai_responses(),
            CompletionRequest::new("Reply with the single word: pong")
                .max_tokens(16)
                .additional_params(json!({"user": null})),
        )
        .expect("the request encodes");
        assert_eq!(sent.get("user"), Some(&Value::Null), "{sent}");
    }

    #[test]
    fn interactions_refuses_raw_tools_that_are_not_an_array() {
        let wire = Interactions::new(GeminiConfig::new(KEY), GEMINI_3_FLASH_PREVIEW);
        let error = encode(
            &wire,
            CompletionRequest::new("Reply with the single word: pong")
                .additional_params(json!({"tools": {"type": "google_search"}})),
        )
        .expect_err("a raw tools object is not appended");
        assert!(unsupported(&error).is_none(), "not an option: {error}");
    }

    #[test]
    fn codex_still_omits_the_typed_fields_the_backend_refuses() {
        let wire = Responses::new(
            OpenAIConfig::with_key(&chatgpt::DIALECT, KEY),
            chatgpt::GPT_5_4,
        );
        let schema = schemars::json_schema!({
            "type": "object",
            "properties": {"word": {"type": "string"}},
        });
        let sent = encode(
            &wire,
            CompletionRequest::new("Reply with the single word: pong")
                .max_tokens(6000)
                .temperature(0.5)
                .output_schema(schema)
                .parallel_tool_calls(false)
                .service_tier(ServiceTier::Flex)
                .verbosity(Verbosity::Low)
                .additional_params(json!({"metadata": {"run": "typed-options"}})),
        )
        .expect("the request encodes");

        // Today's strip, kept for typed fields: the recorded ChatGPT bodies
        // do not move.
        for key in ["max_output_tokens", "temperature"] {
            assert!(sent.get(key).is_none(), "{key} reached ChatGPT: {sent}");
        }
        assert!(sent.pointer("/text/format").is_none(), "{sent}");
        assert_eq!(sent.get("store"), Some(&json!(false)), "{sent}");

        // Mapped options and raw keys are no longer stripped.
        assert_eq!(
            sent.get("parallel_tool_calls"),
            Some(&json!(false)),
            "{sent}"
        );
        assert_eq!(sent.get("service_tier"), Some(&json!("flex")), "{sent}");
        assert_eq!(
            sent.pointer("/text/verbosity"),
            Some(&json!("low")),
            "{sent}"
        );
        assert_eq!(
            sent.get("metadata"),
            Some(&json!({"run": "typed-options"})),
            "{sent}"
        );
    }

    #[test]
    fn post_merge_rewrites_read_raw_keys() {
        // A raw `reasoning` still asks Responses for the reasoning ciphertext.
        let sent = encode(
            &openai_responses(),
            CompletionRequest::new("Reply with the single word: pong")
                .max_tokens(16)
                .additional_params(json!({"reasoning": {"effort": "low"}})),
        )
        .expect("the request encodes");
        let include = sent.get("include").and_then(Value::as_array);
        assert!(
            include.is_some_and(|items| items.contains(&json!("reasoning.encrypted_content"))),
            "{sent}"
        );

        // A raw `thinking: disabled` still keeps DeepSeek's forced tool choice.
        let sent = encode(
            &deepseek(),
            CompletionRequest::new("Reply with the single word: pong")
                .max_tokens(16)
                .tool(lookup())
                .tool_choice(ToolChoice::Required)
                .additional_params(json!({"thinking": {"type": "disabled"}})),
        )
        .expect("the request encodes");
        assert_eq!(sent.get("tool_choice"), Some(&json!("required")), "{sent}");
    }
}

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

mod typed_extras_unary {
    //! Vendor reply fields read through `extras::<P>()` from recorded unary
    //! replies, with no indexing into `raw`.

    use rig::completion::{CompletionRequest, CompletionResponse};
    use rig::driver::Transport;
    use rig::providers::deepseek::DEEPSEEK_FLASH;
    use rig::providers::deepseek::extension::DeepSeekExt;
    use rig::providers::openai::wire::{Chat, DEEPSEEK, OPENROUTER, OpenAIConfig};
    use rig::providers::openrouter::extension::OpenRouterExt;
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
            .extras::<OpenRouterExt>()
            .expect("an OpenRouter reply has OpenRouter extras")
            .expect("the extras deserialize");
        assert_eq!(extras.provider.as_deref(), Some("Azure"));
        assert_eq!(extras.cost, Some(2.7e-6));
        assert_eq!(extras.native_finish_reason.as_deref(), Some("stop"));
        assert!(openrouter_reply.extras::<DeepSeekExt>().is_none());

        let deepseek_reply = unary(
            deepseek(),
            recorded_body("deepseek", DEEPSEEK_UNARY, 0),
            prompt(),
        )
        .await;
        let extras = deepseek_reply
            .extras::<DeepSeekExt>()
            .expect("a DeepSeek reply has DeepSeek extras")
            .expect("the extras deserialize");
        assert_eq!(extras.prompt_cache_hit_tokens, Some(256));
        assert_eq!(extras.prompt_cache_miss_tokens, Some(247));
        assert!(deepseek_reply.extras::<OpenRouterExt>().is_none());
    }
}

mod anthropic_extras_both_ways {
    //! Anthropic's extras from recorded pairs, one prompt answered unary and
    //! streamed: a streamed reply's `raw` is the unary `Message`, so one
    //! `Extras` type reads both.

    use bytes::Bytes;
    use futures::StreamExt;
    use rig::completion::{CompletionRequest, CompletionResponse};
    use rig::providers::anthropic::completion::{CLAUDE_HAIKU_4_5, CLAUDE_SONNET_4_6};
    use rig::providers::anthropic::extension::{AnthropicExt, AnthropicExtras, CacheCreation};
    use rig::providers::anthropic::wire::AnthropicConfig;
    use rig::test_utils::MockStreamingClient;
    use rig_test_support::cassette_models::AnthropicModels;

    use super::typed_extras_unary::{KEY, recorded_body, unary};

    fn messages(model: &str) -> rig::providers::anthropic::Messages {
        AnthropicModels::new(AnthropicConfig::new(KEY), crate::cassettes::local_http())
            .completion(model)
            .wire
    }

    /// `body`, a recorded event stream, folded by `wire` as a live stream is.
    async fn streamed(
        wire: rig::providers::anthropic::Messages,
        body: String,
    ) -> CompletionResponse {
        let model = rig::Model::new(
            wire,
            MockStreamingClient {
                sse_bytes: Bytes::from(body),
            },
        );
        let mut stream = model
            .stream(CompletionRequest::new("extras"))
            .expect("the recorded stream opens");
        while let Some(item) = stream.next().await {
            item.expect("the recorded stream yields no error");
        }
        stream.finish().await.expect("the recorded stream ends")
    }

    /// The extras of the unary and the streamed recording of one turn.
    async fn both_ways(
        model: &str,
        unary_scenario: &str,
        streamed_scenario: &str,
    ) -> [AnthropicExtras; 2] {
        let from_body = unary(
            messages(model),
            recorded_body("anthropic", unary_scenario, 0),
            CompletionRequest::new("extras"),
        )
        .await;
        let from_stream = streamed(
            messages(model),
            recorded_body("anthropic", streamed_scenario, 0),
        )
        .await;
        [from_body, from_stream].map(|reply| {
            reply
                .extras::<AnthropicExt>()
                .expect("an Anthropic reply has Anthropic extras")
                .expect("the extras deserialize")
        })
    }

    #[tokio::test]
    async fn the_same_anthropic_extras_from_both_recordings() {
        let [from_body, from_stream] = both_ways(
            CLAUDE_HAIKU_4_5,
            "raw_capture_matrix/raw_exposes_stop_sequence",
            "raw_stream_capture_matrix/raw_exposes_stop_sequence",
        )
        .await;
        assert_eq!(from_stream.stop_reason.as_deref(), Some("stop_sequence"));
        assert_eq!(from_stream.stop_sequence.as_deref(), Some("alpha"));
        assert_eq!(from_stream.service_tier.as_deref(), Some("standard"));
        assert_eq!(from_stream.inference_geo.as_deref(), Some("not_available"));
        assert_eq!(from_stream.cache_creation, Some(CacheCreation::default()));
        assert_eq!(from_stream, from_body);

        let [from_body, from_stream] = both_ways(
            CLAUDE_SONNET_4_6,
            "raw_capture_matrix/raw_exposes_thinking_block_and_signature",
            "raw_stream_capture_matrix/terminal_raw_round_trips_for_thinking_stream",
        )
        .await;
        assert_eq!(from_stream.stop_reason.as_deref(), Some("end_turn"));
        assert_eq!(from_stream.inference_geo.as_deref(), Some("global"));
        assert_eq!(from_stream, from_body);
    }
}

mod typed_extras_streamed {
    //! The same extras from recorded streams: a streamed reply's `raw` is the
    //! unary document, so one `Extras` type reads both.

    use bytes::Bytes;
    use futures::StreamExt;
    use rig::completion::{CompletionRequest, CompletionResponse};
    use rig::driver::Transport;
    use rig::providers::deepseek::extension::DeepSeekExt;
    use rig::providers::openrouter::extension::OpenRouterExt;
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
        .extras::<OpenRouterExt>()
        .expect("an OpenRouter reply has OpenRouter extras")
        .expect("the extras deserialize");
        let from_body = unary(
            openrouter(),
            recorded_body("openrouter", OPENROUTER_UNARY, 0),
            prompt(),
        )
        .await
        .extras::<OpenRouterExt>()
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
            .extras::<DeepSeekExt>()
            .expect("a DeepSeek reply has DeepSeek extras")
            .expect("the extras deserialize");
        assert_eq!(extras.prompt_cache_hit_tokens, Some(4864));
        assert_eq!(extras.prompt_cache_miss_tokens, Some(202));
    }
}

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
        assert!(cost.input.is_some_and(|part| close(
            part,
            (input - read - written) as f64 * per_token(pricing.input)
        )));
        assert!(
            cost.output
                .is_some_and(|part| close(part, output as f64 * per_token(pricing.output)))
        );
        assert!(cost.cache_read.is_some_and(|part| close(
            part,
            read as f64 * per_token(pricing.cache_read.unwrap_or(pricing.input))
        )));
        let parts = [cost.input, cost.output, cost.cache_read, cost.cache_write];
        assert!(close(cost.total, parts.into_iter().flatten().sum::<f64>()));
        assert!(
            parts.iter().all(Option::is_some),
            "a catalog cost knows every part"
        );

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

mod usage_cost_sum {
    //! Summing usage sums cost only when every summed turn has one: a turn
    //! with an unknown cost makes the total unknown rather than too low. The
    //! empty usage is the identity, so a fold from it keeps the cost.

    use rig::completion::Usage;
    use serde_json::json;

    fn usage(value: serde_json::Value) -> Usage {
        serde_json::from_value(value).expect("the usage deserializes")
    }

    fn priced(input: f64, output: f64) -> Usage {
        usage(json!({
            "input_tokens": 10,
            "output_tokens": 5,
            "cost": {
                "input": input,
                "output": output,
                "cache_read": 0.0,
                "cache_write": 0.0,
                "total": input + output,
            },
        }))
    }

    #[test]
    fn cost_sums_only_when_every_turn_has_one() {
        let both = priced(1.0, 2.0) + priced(0.5, 0.25);
        let cost = both.cost.expect("both turns are priced");
        assert_eq!(
            (cost.input, cost.output, cost.total),
            (Some(1.5), Some(2.25), 3.75)
        );
        assert_eq!(both.input_tokens, Some(20));

        let unpriced = usage(json!({"input_tokens": 10, "output_tokens": 5}));
        let mixed = priced(1.0, 2.0) + unpriced;
        assert_eq!(mixed.cost, None);
        assert_eq!(mixed.input_tokens, Some(20));

        let mut total = Usage::default();
        total += priced(1.0, 2.0);
        assert_eq!(total.cost.map(|cost| cost.total), Some(3.0));
    }
}

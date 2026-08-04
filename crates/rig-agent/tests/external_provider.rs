#![allow(clippy::panic_in_result_fn)]

use std::{
    collections::HashSet,
    convert::Infallible,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use rig_agent::{
    AgentBuilder,
    agent::hook::{CompletionCallAction, RequestPatch},
    client::BindCompletionExt,
    hooks::{HookDecision, HookEntry, HookEvent},
    provider::{
        self, ExternalCompletionProvider, ExternalCompletionProviderEntry, ExternalProviderConfig,
        ExternalProviderConfigError, ExternalProviderId, ExternalProviderRegistry,
        ExternalProviderRegistryError, OwnedProviderDescriptor, Runtime,
    },
    tool::PortableTool,
};
use rig_core::{
    OneOrMany,
    completion::{CompletionError, CompletionRequest, CompletionResponse, Usage},
    http_runtime::HttpRuntime,
    message::AssistantContent,
    streaming::{CompletionStream, RawStreamingChoice, StreamFinal},
};
use serde::Deserialize;

#[derive(Clone, Deserialize)]
struct ScriptConfig {
    answer: String,
}

#[derive(Clone, Copy)]
enum ScriptMode {
    Text,
    ToolThenText,
    StructuredWithTools,
}

struct ScriptedProvider {
    id: ExternalProviderId,
    calls: Arc<AtomicUsize>,
    saw_native_schema_with_tools: Arc<AtomicBool>,
    mode: ScriptMode,
}

impl ScriptedProvider {
    fn new(mode: ScriptMode) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            id: ExternalProviderId::new("dev.rig.tests/scripted@1")?,
            calls: Arc::new(AtomicUsize::new(0)),
            saw_native_schema_with_tools: Arc::new(AtomicBool::new(false)),
            mode,
        })
    }
}

impl ExternalCompletionProvider for ScriptedProvider {
    type Config = ScriptConfig;

    fn id(&self) -> ExternalProviderId {
        self.id.clone()
    }

    fn descriptor(&self) -> OwnedProviderDescriptor {
        OwnedProviderDescriptor::named("scripted-external")
            .with_tools(true)
            .with_response_format(true)
            .with_composes_native_output_with_tools(true)
    }

    fn validate_config(&self, config: &Self::Config) -> Result<(), ExternalProviderConfigError> {
        if config.answer.is_empty() {
            Err(ExternalProviderConfigError::rejected(
                "answer must not be empty",
            ))
        } else {
            Ok(())
        }
    }

    fn complete(
        &self,
        config: Self::Config,
        model: String,
        _runtime: HttpRuntime,
        request: CompletionRequest,
    ) -> impl Future<Output = Result<CompletionResponse, CompletionError>>
    + rig_core::wasm_compat::WasmCompatSend {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        let mode = self.mode;
        let saw_native_schema_with_tools = Arc::clone(&self.saw_native_schema_with_tools);
        async move {
            let choice = match (mode, call) {
                (ScriptMode::ToolThenText, 0) => OneOrMany::one(AssistantContent::tool_call(
                    "call-1",
                    "echo",
                    serde_json::json!({"value": "hello"}),
                )),
                (ScriptMode::StructuredWithTools, _) => {
                    saw_native_schema_with_tools.store(
                        request.output_schema.is_some() && !request.tools.is_empty(),
                        Ordering::SeqCst,
                    );
                    OneOrMany::one(AssistantContent::text(r#"{"answer":"structured"}"#))
                }
                _ => OneOrMany::one(AssistantContent::text(config.answer)),
            };
            Ok(CompletionResponse::new(choice, usage(2, 3), "scripted-external").with_model(model))
        }
    }

    async fn open_stream(
        &self,
        config: Self::Config,
        model: String,
        _runtime: HttpRuntime,
        _request: CompletionRequest,
    ) -> Result<CompletionStream, CompletionError> {
        let final_record = StreamFinal::new("scripted-external", usage(4, 5)).with_model(model);
        Ok(CompletionStream::from_stream(futures::stream::iter([
            Ok(RawStreamingChoice::Message(config.answer)),
            Ok(RawStreamingChoice::FinalResponse(final_record)),
        ])))
    }
}

#[derive(Deserialize)]
struct EchoArgs {
    value: String,
}

struct Echo;

struct DropSignal(Arc<AtomicBool>);

struct ClosureState {
    calls: AtomicUsize,
    secret: String,
}

static SEQUENCED_CONFIG_ID: AtomicUsize = AtomicUsize::new(0);

struct SequencedConfig {
    instance: usize,
}

impl<'de> Deserialize<'de> for SequencedConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let _settings = serde_json::Value::deserialize(deserializer)?;
        Ok(Self {
            instance: SEQUENCED_CONFIG_ID.fetch_add(1, Ordering::SeqCst),
        })
    }
}

struct SameInstanceProvider {
    id: ExternalProviderId,
    validated: Arc<Mutex<HashSet<usize>>>,
}

impl ExternalCompletionProvider for SameInstanceProvider {
    type Config = SequencedConfig;

    fn id(&self) -> ExternalProviderId {
        self.id.clone()
    }

    fn descriptor(&self) -> OwnedProviderDescriptor {
        OwnedProviderDescriptor::named("same-instance-external")
    }

    fn validate_config(&self, config: &Self::Config) -> Result<(), ExternalProviderConfigError> {
        self.validated
            .lock()
            .map_err(|_| ExternalProviderConfigError::rejected("validation state is unavailable"))?
            .insert(config.instance);
        Ok(())
    }

    fn complete(
        &self,
        config: Self::Config,
        _model: String,
        _runtime: HttpRuntime,
        _request: CompletionRequest,
    ) -> impl Future<Output = Result<CompletionResponse, CompletionError>>
    + rig_core::wasm_compat::WasmCompatSend {
        let was_validated = self
            .validated
            .lock()
            .map(|instances| instances.contains(&config.instance))
            .unwrap_or(false);
        std::future::ready(if was_validated {
            Ok(CompletionResponse::new(
                OneOrMany::one(AssistantContent::text("validated")),
                Usage::new(),
                "same-instance-external",
            ))
        } else {
            Err(CompletionError::ProviderError(
                "invocation received a different, unvalidated config instance".to_owned(),
            ))
        })
    }

    fn open_stream(
        &self,
        config: Self::Config,
        _model: String,
        _runtime: HttpRuntime,
        _request: CompletionRequest,
    ) -> impl Future<Output = Result<CompletionStream, CompletionError>>
    + rig_core::wasm_compat::WasmCompatSend {
        let was_validated = self
            .validated
            .lock()
            .map(|instances| instances.contains(&config.instance))
            .unwrap_or(false);
        std::future::ready(if was_validated {
            Ok(CompletionStream::from_stream(futures::stream::iter([Ok(
                RawStreamingChoice::FinalResponse(StreamFinal::new(
                    "same-instance-external",
                    Usage::new(),
                )),
            )])))
        } else {
            Err(CompletionError::ProviderError(
                "stream invocation received a different, unvalidated config instance".to_owned(),
            ))
        })
    }
}

impl Drop for DropSignal {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

impl PortableTool for Echo {
    const NAME: &'static str = "echo";
    type Args = EchoArgs;
    type Output = String;
    type Error = Infallible;

    fn description(&self) -> String {
        "Echo a value".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"]
        })
    }

    fn call(
        &self,
        arguments: Self::Args,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + rig_core::wasm_compat::WasmCompatSend
    {
        std::future::ready(Ok(arguments.value))
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema, PartialEq)]
struct StructuredAnswer {
    answer: String,
}

fn usage(input: u64, output: u64) -> Usage {
    Usage {
        input_tokens: input,
        output_tokens: output,
        total_tokens: input + output,
        ..Usage::new()
    }
}

fn config(
    id: ExternalProviderId,
    answer: &str,
) -> Result<ExternalProviderConfig, ExternalProviderConfigError> {
    ExternalProviderConfig::new(id, "script-model", serde_json::json!({"answer": answer}))
}

fn assert_missing_handler_error(
    error: &rig_agent::completion::PromptError,
    id: &ExternalProviderId,
) {
    assert!(
        matches!(
            error,
            rig_agent::completion::PromptError::CompletionError(CompletionError::RequestError(_))
        ),
        "unexpected missing-handler error: {error:?}"
    );
    assert!(error.to_string().contains(id.as_str()));
}

#[test]
fn config_round_trip_is_data_only_and_debug_redacts_settings()
-> Result<(), Box<dyn std::error::Error>> {
    let config = config(
        ExternalProviderId::new("dev.rig.tests/scripted@1")?,
        "top-secret-answer",
    )?;
    let json = serde_json::to_value(&config)?;
    let object = json
        .as_object()
        .ok_or("external provider config must serialize as an object")?;
    assert_eq!(object.len(), 3);
    assert!(object.contains_key("driver"));
    assert!(object.contains_key("model"));
    assert!(object.contains_key("settings"));
    assert!(!object.contains_key("descriptor"));

    let round_trip: ExternalProviderConfig = serde_json::from_value(json)?;
    assert_eq!(round_trip.driver, config.driver);
    assert_eq!(round_trip.model, config.model);
    let debug = format!("{round_trip:?}");
    assert!(!debug.contains("top-secret-answer"));
    assert!(debug.contains("<redacted>"));
    Ok(())
}

#[test]
fn exact_version_lookup_fails_closed_and_duplicate_registration_is_typed()
-> Result<(), Box<dyn std::error::Error>> {
    let provider = ScriptedProvider::new(ScriptMode::Text)?;
    let id = provider.id();
    let registry = ExternalProviderRegistry::new()
        .register(ExternalCompletionProviderEntry::from_provider(provider))?;

    let duplicate = registry
        .clone()
        .register(ExternalCompletionProviderEntry::new(
            id.clone(),
            OwnedProviderDescriptor::named("duplicate"),
            |_config, _runtime, _request| async {
                Err(CompletionError::ProviderError("unused".to_owned()))
            },
            |_config, _runtime, _request| async {
                Err(CompletionError::ProviderError("unused".to_owned()))
            },
        ));
    assert!(matches!(
        duplicate,
        Err(ExternalProviderRegistryError::Duplicate { driver }) if driver == id
    ));

    let runtime = Runtime::new().with_external_registry(registry);
    let wrong_version = ExternalProviderId::new("dev.rig.tests/scripted@2")?;
    let wrong_config: provider::ProviderConfig = config(wrong_version, "unused")?.into();
    let error = match runtime.validate_provider(&wrong_config) {
        Ok(()) => return Err("a different version unexpectedly resolved".into()),
        Err(error) => error,
    };
    assert!(error.to_string().contains("scripted@2"));

    let control_error =
        match serde_json::from_str::<ExternalProviderId>(r#""dev.rig.tests/scripted\u001b@1""#) {
            Ok(_) => return Err("control characters were accepted during serde".into()),
            Err(error) => error,
        };
    assert!(!control_error.to_string().contains('\u{1b}'));
    Ok(())
}

#[tokio::test]
async fn typed_provider_runs_through_direct_and_high_level_unary_and_streaming_paths()
-> Result<(), Box<dyn std::error::Error>> {
    let provider = ScriptedProvider::new(ScriptMode::Text)?;
    let id = provider.id();
    let runtime = Arc::new(
        Runtime::new()
            .with_external_provider(ExternalCompletionProviderEntry::from_provider(provider))?,
    );
    let config = config(id, "ordinary async futures")?;
    let serialized = serde_json::to_string(&config)?;
    let config: ExternalProviderConfig = serde_json::from_str(&serialized)?;

    let direct = provider::complete(
        &config.clone().into(),
        &runtime,
        CompletionRequest::from_prompt("direct"),
    )
    .await?;
    assert_eq!(direct.provider, "scripted-external");
    assert_eq!(direct.model.as_deref(), Some("script-model"));

    let bound = config
        .clone()
        .bind_completion(Arc::clone(&runtime))
        .completion(CompletionRequest::from_prompt("bound"))
        .await?;
    assert_eq!(bound.provider, "scripted-external");

    let mut direct_stream = provider::open_stream(
        &config.clone().into(),
        &runtime,
        CompletionRequest::from_prompt("direct stream"),
    )
    .await?;
    while direct_stream.next().await.is_some() {}
    let direct_stream = direct_stream.into_response()?;
    assert_eq!(direct_stream.provider, "scripted-external");
    assert_eq!(direct_stream.model.as_deref(), Some("script-model"));
    assert_eq!(direct_stream.usage, usage(4, 5));

    let agent = AgentBuilder::new(config)
        .runtime(Arc::clone(&runtime))
        .build();
    assert_eq!(agent.prompt("unary").await?, "ordinary async futures");

    let streamed = agent.stream_run("streaming").into_final_response().await?;
    assert_eq!(streamed.output, "ordinary async futures");
    assert_eq!(streamed.usage, usage(4, 5));
    Ok(())
}

#[tokio::test]
async fn typed_validation_fails_before_callback_invocation()
-> Result<(), Box<dyn std::error::Error>> {
    let provider = ScriptedProvider::new(ScriptMode::Text)?;
    let calls = Arc::clone(&provider.calls);
    let id = provider.id();
    let runtime = Runtime::new()
        .with_external_provider(ExternalCompletionProviderEntry::from_provider(provider))?;
    let settings_secret = "typed-settings-secret-must-stay-redacted";
    let invalid =
        ExternalProviderConfig::new(id.clone(), "script-model", settings_secret.to_owned())?;

    let error = match provider::complete(
        &invalid.into(),
        &runtime,
        CompletionRequest::from_prompt("must not run"),
    )
    .await
    {
        Ok(_) => return Err("invalid typed settings unexpectedly invoked the provider".into()),
        Err(error) => error,
    };
    assert!(matches!(error, CompletionError::RequestError(_)));
    assert!(!error.to_string().contains(settings_secret));
    assert!(!format!("{error:?}").contains(settings_secret));
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let rejected = ExternalProviderConfig::new(
        id.clone(),
        "script-model",
        serde_json::json!({"answer": ""}),
    )?;
    assert!(
        provider::complete(
            &rejected.into(),
            &runtime,
            CompletionRequest::from_prompt("must still not run"),
        )
        .await
        .is_err()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let empty_model: ExternalProviderConfig = serde_json::from_value(serde_json::json!({
        "driver": id,
        "model": "",
        "settings": {"answer": "valid"}
    }))?;
    assert!(
        provider::complete(
            &empty_model.into(),
            &runtime,
            CompletionRequest::from_prompt("must never run"),
        )
        .await
        .is_err()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    Ok(())
}

#[tokio::test]
async fn typed_erasure_validates_the_same_config_instance_that_callbacks_receive()
-> Result<(), Box<dyn std::error::Error>> {
    let id = ExternalProviderId::new("dev.rig.tests/same-instance@1")?;
    let validated = Arc::new(Mutex::new(HashSet::new()));
    let provider = SameInstanceProvider {
        id: id.clone(),
        validated: Arc::clone(&validated),
    };
    let runtime = Runtime::new()
        .with_external_provider(ExternalCompletionProviderEntry::from_provider(provider))?;
    let provider_config: provider::ProviderConfig =
        ExternalProviderConfig::new(id, "same-instance-model", serde_json::json!({}))?.into();

    provider::complete(
        &provider_config,
        &runtime,
        CompletionRequest::from_prompt("unary"),
    )
    .await?;
    provider::open_stream(
        &provider_config,
        &runtime,
        CompletionRequest::from_prompt("stream"),
    )
    .await?;
    assert_eq!(
        validated
            .lock()
            .map_err(|_| "validation state is unavailable")?
            .len(),
        2,
        "each fulfillment must validate exactly its one invocation instance"
    );
    Ok(())
}

#[tokio::test]
async fn tool_loop_and_hooks_use_the_existing_high_level_driver()
-> Result<(), Box<dyn std::error::Error>> {
    let provider = ScriptedProvider::new(ScriptMode::ToolThenText)?;
    let calls = Arc::clone(&provider.calls);
    let id = provider.id();
    let runtime = Arc::new(
        Runtime::new()
            .with_external_provider(ExternalCompletionProviderEntry::from_provider(provider))?,
    );
    let hook_calls = Arc::new(AtomicUsize::new(0));
    let hook_probe = Arc::clone(&hook_calls);
    let hook = HookEntry::sync("external-observer", move |_| {
        hook_probe.fetch_add(1, Ordering::SeqCst);
        HookDecision::Continue
    });
    let agent = AgentBuilder::new(config(id, "after tool")?)
        .runtime(runtime)
        .tool(Echo)
        .add_hook(hook)
        .default_max_turns(2)
        .build();

    assert_eq!(agent.prompt("use the echo tool").await?, "after tool");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(hook_calls.load(Ordering::SeqCst) > 0);
    Ok(())
}

#[tokio::test]
async fn handler_owned_capabilities_drive_structured_output_with_tools()
-> Result<(), Box<dyn std::error::Error>> {
    let provider = ScriptedProvider::new(ScriptMode::StructuredWithTools)?;
    let saw_native_schema_with_tools = Arc::clone(&provider.saw_native_schema_with_tools);
    let id = provider.id();
    let runtime = Arc::new(
        Runtime::new()
            .with_external_provider(ExternalCompletionProviderEntry::from_provider(provider))?,
    );
    let agent = AgentBuilder::new(config(id, "unused")?)
        .runtime(runtime)
        .tool(Echo)
        .output_schema::<StructuredAnswer>()
        .build();

    assert_eq!(
        agent.prompt("return structured output").await?,
        r#"{"answer":"structured"}"#
    );
    assert!(saw_native_schema_with_tools.load(Ordering::SeqCst));

    let typed = agent
        .prompt_typed::<StructuredAnswer>("return typed output")
        .await?;
    assert_eq!(
        typed,
        StructuredAnswer {
            answer: "structured".to_owned()
        }
    );

    let extracted = agent
        .extractor("extract typed output")
        .run::<StructuredAnswer>()
        .await?;
    assert_eq!(extracted, typed);
    Ok(())
}

#[tokio::test]
async fn with_state_closure_constructor_shares_state_without_boxing_at_call_site()
-> Result<(), Box<dyn std::error::Error>> {
    let id = ExternalProviderId::new("dev.rig.tests/closure@1")?;
    let entry = ExternalCompletionProviderEntry::with_state(
        id.clone(),
        OwnedProviderDescriptor::named("closure-external"),
        ClosureState {
            calls: AtomicUsize::new(0),
            secret: "captured-state-secret".to_owned(),
        },
        |state, config, _runtime, _request| async move {
            let _captured_state_is_available = state.secret.len();
            let call = state.calls.fetch_add(1, Ordering::SeqCst) + 1;
            Ok(CompletionResponse::new(
                OneOrMany::one(AssistantContent::text(format!("{}-{call}", config.model))),
                Usage::new(),
                "closure-external",
            ))
        },
        |_state, config, _runtime, _request| async move {
            Ok(CompletionStream::from_stream(futures::stream::iter([
                Ok(RawStreamingChoice::Message(config.model)),
                Ok(RawStreamingChoice::FinalResponse(StreamFinal::new(
                    "closure-external",
                    Usage::new(),
                ))),
            ])))
        },
    );
    assert!(!format!("{entry:?}").contains("captured-state-secret"));
    let runtime = Runtime::new().with_external_provider(entry)?;
    let config = ExternalProviderConfig::new(id, "closure-model", serde_json::json!({}))?;

    let first = provider::complete(
        &config.clone().into(),
        &runtime,
        CompletionRequest::from_prompt("first"),
    )
    .await?;
    let second = provider::complete(
        &config.into(),
        &runtime,
        CompletionRequest::from_prompt("second"),
    )
    .await?;
    assert!(matches!(
        first.choice.first_ref(),
        AssistantContent::Text(text) if text.text == "closure-model-1"
    ));
    assert!(matches!(
        second.choice.first_ref(),
        AssistantContent::Text(text) if text.text == "closure-model-2"
    ));
    Ok(())
}

#[tokio::test]
async fn hook_model_patches_reach_external_unary_and_streaming_callbacks()
-> Result<(), Box<dyn std::error::Error>> {
    let id = ExternalProviderId::new("dev.rig.tests/model-patch@1")?;
    let seen_models = Arc::new(Mutex::new(Vec::new()));
    let entry = ExternalCompletionProviderEntry::with_state(
        id.clone(),
        OwnedProviderDescriptor::named("model-patch-external"),
        Arc::clone(&seen_models),
        |seen_models, config, _runtime, _request| async move {
            seen_models
                .as_ref()
                .lock()
                .map_err(|_| CompletionError::ProviderError("model capture failed".to_owned()))?
                .push(config.model.clone());
            Ok(CompletionResponse::new(
                OneOrMany::one(AssistantContent::text(config.model.clone())),
                Usage::new(),
                "model-patch-external",
            )
            .with_model(config.model))
        },
        |seen_models, config, _runtime, _request| async move {
            seen_models
                .as_ref()
                .lock()
                .map_err(|_| CompletionError::ProviderError("model capture failed".to_owned()))?
                .push(config.model.clone());
            Ok(CompletionStream::from_stream(futures::stream::iter([
                Ok(RawStreamingChoice::Message(config.model.clone())),
                Ok(RawStreamingChoice::FinalResponse(
                    StreamFinal::new("model-patch-external", Usage::new()).with_model(config.model),
                )),
            ])))
        },
    );
    let runtime = Arc::new(Runtime::new().with_external_provider(entry)?);
    let hook_calls = Arc::new(AtomicUsize::new(0));
    let hook_probe = Arc::clone(&hook_calls);
    let hook = HookEntry::sync("external-model-patch", move |event| {
        let HookEvent::BeforeModelCall { .. } = event else {
            return HookDecision::Continue;
        };
        let model = if hook_probe.fetch_add(1, Ordering::SeqCst) == 0 {
            "hook-unary-model"
        } else {
            "hook-stream-model"
        };
        HookDecision::CompletionCall(CompletionCallAction::Patch(
            RequestPatch::new().model(model),
        ))
    });
    let agent = AgentBuilder::new(ExternalProviderConfig::new(
        id,
        "configured-model",
        serde_json::json!({}),
    )?)
    .runtime(runtime)
    .add_hook(hook)
    .build();

    assert_eq!(agent.prompt("unary").await?, "hook-unary-model");
    assert_eq!(
        agent
            .stream_run("streaming")
            .into_final_response()
            .await?
            .output,
        "hook-stream-model"
    );
    assert_eq!(
        *seen_models
            .lock()
            .map_err(|_| "model capture is unavailable")?,
        ["hook-unary-model", "hook-stream-model"]
    );
    Ok(())
}

#[tokio::test]
async fn provider_errors_are_preserved_across_both_dispatch_paths()
-> Result<(), Box<dyn std::error::Error>> {
    let id = ExternalProviderId::new("dev.rig.tests/errors@1")?;
    let entry = ExternalCompletionProviderEntry::new(
        id.clone(),
        OwnedProviderDescriptor::named("error-external"),
        |_config, _runtime, _request| async {
            Err(CompletionError::ProviderError("unary sentinel".to_owned()))
        },
        |_config, _runtime, _request| async {
            Err(CompletionError::ProviderError("stream sentinel".to_owned()))
        },
    );
    let runtime = Runtime::new().with_external_provider(entry)?;
    let provider_config: provider::ProviderConfig =
        ExternalProviderConfig::new(id, "error-model", serde_json::json!({}))?.into();

    let unary = provider::complete(
        &provider_config,
        &runtime,
        CompletionRequest::from_prompt("unary"),
    )
    .await;
    assert!(matches!(
        unary,
        Err(CompletionError::ProviderError(message)) if message == "unary sentinel"
    ));

    let streaming = provider::open_stream(
        &provider_config,
        &runtime,
        CompletionRequest::from_prompt("stream"),
    )
    .await;
    assert!(matches!(
        streaming,
        Err(CompletionError::ProviderError(message)) if message == "stream sentinel"
    ));
    Ok(())
}

#[tokio::test]
async fn runtime_clones_keep_frozen_registry_snapshots_and_agents_share_handlers_concurrently()
-> Result<(), Box<dyn std::error::Error>> {
    let provider = ScriptedProvider::new(ScriptMode::Text)?;
    let calls = Arc::clone(&provider.calls);
    let id = provider.id();
    let empty_runtime = Runtime::new();
    let old_snapshot = empty_runtime.clone();
    let runtime = Arc::new(
        empty_runtime
            .with_external_provider(ExternalCompletionProviderEntry::from_provider(provider))?,
    );
    let provider_config: provider::ProviderConfig = config(id, "concurrent")?.into();
    assert!(old_snapshot.validate_provider(&provider_config).is_err());
    runtime.validate_provider(&provider_config)?;

    let agent = AgentBuilder::new(provider_config).runtime(runtime).build();
    let (left, right) = tokio::join!(agent.prompt("left"), agent.prompt("right"));
    assert_eq!(left?, "concurrent");
    assert_eq!(right?, "concurrent");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    Ok(())
}

#[tokio::test]
async fn dropping_external_provider_future_cancels_the_callback()
-> Result<(), Box<dyn std::error::Error>> {
    let id = ExternalProviderId::new("dev.rig.tests/pending@1")?;
    let dropped = Arc::new(AtomicBool::new(false));
    let drop_probe = Arc::clone(&dropped);
    let entry = ExternalCompletionProviderEntry::new(
        id.clone(),
        OwnedProviderDescriptor::named("pending-external"),
        move |_config, _runtime, _request| {
            let signal = DropSignal(Arc::clone(&drop_probe));
            async move {
                let _signal = signal;
                futures::future::pending::<()>().await;
                Ok(CompletionResponse::new(
                    OneOrMany::one(AssistantContent::text("unreachable")),
                    Usage::new(),
                    "pending-external",
                ))
            }
        },
        |_config, _runtime, _request| async {
            Ok(CompletionStream::from_stream(futures::stream::empty()))
        },
    );
    let runtime = Runtime::new().with_external_provider(entry)?;
    let config = ExternalProviderConfig::new(id, "pending-model", serde_json::json!({}))?;
    let provider_config: provider::ProviderConfig = config.into();
    let mut pending = Box::pin(provider::complete(
        &provider_config,
        &runtime,
        CompletionRequest::from_prompt("wait"),
    ));

    assert!(matches!(
        futures::poll!(pending.as_mut()),
        std::task::Poll::Pending
    ));
    assert!(!dropped.load(Ordering::SeqCst));
    drop(pending);
    assert!(dropped.load(Ordering::SeqCst));
    Ok(())
}

#[tokio::test]
async fn missing_handler_descriptor_errors_remain_retryable_in_both_drivers()
-> Result<(), Box<dyn std::error::Error>> {
    let id = ExternalProviderId::new("dev.rig.tests/missing@1")?;
    let external = ExternalProviderConfig::new(
        id.clone(),
        "missing-model",
        serde_json::json!({"secret": "must-not-run"}),
    )?;
    let agent = AgentBuilder::new(external)
        .runtime(Arc::new(Runtime::new()))
        .default_max_turns(3)
        .build();

    let (mut session, _hooks, _executor) = agent.runner("blocking").into_session();
    let first = match session.advance().await {
        Ok(_) => return Err("the blocking session found an absent handler".into()),
        Err(error) => error,
    };
    assert_missing_handler_error(&first, &id);
    let retried = match session.advance().await {
        Ok(_) => return Err("the blocking retry found an absent handler".into()),
        Err(error) => error,
    };
    assert_missing_handler_error(&retried, &id);

    let mut stream = agent.runner("streaming").stream();
    let first = match stream.next_item().await {
        Some(Err(error)) => error,
        Some(Ok(_)) => return Err("the stream found an absent handler".into()),
        None => return Err("stream ended before reporting its missing handler".into()),
    };
    assert_missing_handler_error(&first, &id);
    let retried = match stream.next_item().await {
        Some(Err(error)) => error,
        Some(Ok(_)) => return Err("the stream retry found an absent handler".into()),
        None => return Err("stream ended instead of retrying its missing handler".into()),
    };
    assert_missing_handler_error(&retried, &id);
    Ok(())
}

#[tokio::test]
async fn external_callback_errors_reissue_with_the_turn_patch_in_both_drivers()
-> Result<(), Box<dyn std::error::Error>> {
    let id = ExternalProviderId::new("dev.rig.tests/retry@1")?;
    let unary_calls = Arc::new(AtomicUsize::new(0));
    let stream_calls = Arc::new(AtomicUsize::new(0));
    let unary_models = Arc::new(Mutex::new(Vec::new()));
    let stream_models = Arc::new(Mutex::new(Vec::new()));
    let unary_call_probe = Arc::clone(&unary_calls);
    let stream_call_probe = Arc::clone(&stream_calls);
    let unary_model_probe = Arc::clone(&unary_models);
    let stream_model_probe = Arc::clone(&stream_models);
    let entry = ExternalCompletionProviderEntry::new(
        id.clone(),
        OwnedProviderDescriptor::named("retry-external"),
        move |config, _runtime, _request| {
            let call = unary_call_probe.fetch_add(1, Ordering::SeqCst);
            let captured = unary_model_probe
                .lock()
                .map(|mut models| models.push(config.model.clone()))
                .is_ok();
            async move {
                if !captured {
                    return Err(CompletionError::ProviderError(
                        "unary model capture failed".to_owned(),
                    ));
                }
                if call == 0 {
                    return Err(CompletionError::ProviderError("retry unary".to_owned()));
                }
                Ok(CompletionResponse::new(
                    OneOrMany::one(AssistantContent::text(config.model)),
                    Usage::new(),
                    "retry-external",
                ))
            }
        },
        move |config, _runtime, _request| {
            let call = stream_call_probe.fetch_add(1, Ordering::SeqCst);
            let captured = stream_model_probe
                .lock()
                .map(|mut models| models.push(config.model.clone()))
                .is_ok();
            async move {
                if !captured {
                    return Err(CompletionError::ProviderError(
                        "stream model capture failed".to_owned(),
                    ));
                }
                if call == 0 {
                    return Err(CompletionError::ProviderError("retry stream".to_owned()));
                }
                Ok(CompletionStream::from_stream(futures::stream::iter([
                    Ok(RawStreamingChoice::Message(config.model)),
                    Ok(RawStreamingChoice::FinalResponse(StreamFinal::new(
                        "retry-external",
                        Usage::new(),
                    ))),
                ])))
            }
        },
    );
    let runtime = Arc::new(Runtime::new().with_external_provider(entry)?);
    let agent = AgentBuilder::new(ExternalProviderConfig::new(
        id,
        "configured-model",
        serde_json::json!({}),
    )?)
    .runtime(runtime)
    .default_max_turns(3)
    .build();

    let (mut session, _hooks, _executor) = agent.runner("blocking").into_session();
    session.patch_next_turn(RequestPatch::new().model("retry-unary-model"));
    assert!(matches!(
        session.advance().await,
        Err(rig_agent::completion::PromptError::CompletionError(
            CompletionError::ProviderError(message)
        )) if message == "retry unary"
    ));
    let blocking = session.advance().await?;
    assert!(matches!(
        blocking,
        rig_agent::session::SessionEvent::Done(response)
            if response.output == "retry-unary-model"
    ));

    let mut stream = agent.runner("streaming").stream();
    stream.patch_next_turn(RequestPatch::new().model("retry-stream-model"));
    assert!(matches!(
        stream.next_item().await,
        Some(Err(rig_agent::completion::PromptError::CompletionError(
            CompletionError::ProviderError(message)
        ))) if message == "retry stream"
    ));
    let streamed = loop {
        match stream.next_item().await {
            Some(Ok(rig_agent::stream::AgentStreamItem::Final(response))) => break response,
            Some(Ok(_)) => {}
            Some(Err(error)) => return Err(error.into()),
            None => return Err("stream ended before the retried response".into()),
        }
    };
    assert_eq!(streamed.output, "retry-stream-model");
    assert_eq!(unary_calls.load(Ordering::SeqCst), 2);
    assert_eq!(stream_calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        *unary_models
            .lock()
            .map_err(|_| "unary model capture is unavailable")?,
        ["retry-unary-model", "retry-unary-model"]
    );
    assert_eq!(
        *stream_models
            .lock()
            .map_err(|_| "stream model capture is unavailable")?,
        ["retry-stream-model", "retry-stream-model"]
    );
    Ok(())
}

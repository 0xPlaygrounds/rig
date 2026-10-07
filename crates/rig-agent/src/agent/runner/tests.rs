use std::sync::{Arc, Mutex};

use futures::StreamExt;
use serde_json::json;

use crate::{
    agent::{AgentBuilder, AgentHook, HookContext, OutcomeAction, OutcomeEvent},
    completion::Document,
    test_utils::{MockCompletionModel, MockStreamEvent, MockTurn},
    tool::{Tool, ToolContext, ToolErrorKind, ToolExecutionError},
};
use rig_core::message::ToolChoice;

struct MetadataFailingTool;

#[derive(serde::Serialize, serde::Deserialize)]
struct ResultMetadata(String);

impl rig_core::tool::ContextValue for ResultMetadata {
    const KEY: &'static str = "test.result_metadata";
}

impl Tool for MetadataFailingTool {
    const NAME: &'static str = "flaky_tool";
    type Error = rig::tool::ToolExecutionError;
    type Args = serde_json::Value;
    type Output = String;

    fn description(&self) -> String {
        "Fails after attaching result metadata".into()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({"type": "object", "properties": {}})
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, ToolExecutionError> {
        context.insert_result(ResultMetadata("shared-result-metadata".to_string()))?;
        Err(ToolExecutionError::timeout("raw timeout failure"))
    }
}

#[derive(Clone, Default)]
struct Results(Arc<Mutex<Vec<(ToolErrorKind, String, String)>>>);

impl AgentHook for Results {
    async fn on_outcome(&self, _ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        let Some(result) = event.tool_result() else {
            return OutcomeAction::proceed();
        };
        if let Some(error) = result.error() {
            self.0.lock().expect("results").push((
                error.kind(),
                result.output().render(),
                event
                    .tool_context()
                    .expect("tool outcome carries its context")
                    .result::<ResultMetadata>()
                    .expect("tool result metadata decodes")
                    .expect("tool result metadata")
                    .0,
            ));
        }
        OutcomeAction::rewrite_tool_result(&event, "rewritten for model")
    }
}

#[test]
fn agent_exposes_read_only_name_and_description() {
    let named = AgentBuilder::new(MockCompletionModel::text("done"))
        .name("researcher")
        .description("Finds evidence")
        .build();
    assert_eq!(named.name(), Some("researcher"));
    assert_eq!(named.description(), Some("Finds evidence"));

    let unnamed = AgentBuilder::new(MockCompletionModel::text("done")).build();
    assert_eq!(unnamed.name(), None);
    assert_eq!(unnamed.description(), None);
}

#[tokio::test]
async fn runner_applies_per_run_request_overrides() {
    let model = MockCompletionModel::text("done");
    AgentBuilder::new(model.clone())
        .preamble("baseline preamble")
        .context("baseline document")
        .temperature(0.1)
        .max_tokens(10)
        .additional_params(json!({"baseline": true}))
        .build()
        .prompt("go")
        .preamble("run preamble")
        .document(Document {
            id: "run-one".into(),
            text: "first run document".into(),
            additional_props: Default::default(),
        })
        .documents([Document {
            id: "run-two".into(),
            text: "second run document".into(),
            additional_props: Default::default(),
        }])
        .temperature(0.7)
        .max_tokens(42)
        .replace_additional_params(json!({"override": true}))
        .tool_choice(ToolChoice::None)
        .run()
        .await
        .expect("runner request should succeed");

    let requests = model.requests();
    let request = requests.first().expect("one request");
    assert!(request.chat_history.iter().any(
        |message| matches!(message, crate::completion::Message::System { content } if content == "run preamble")
    ));
    let documents = rig_core::test_utils::sent_documents(request);
    assert!(
        documents
            .iter()
            .any(|(_, text)| text == "baseline document")
    );
    assert!(documents.iter().any(|(id, _)| id == "run-one"));
    assert!(documents.iter().any(|(id, _)| id == "run-two"));
    assert_eq!(request.temperature, Some(0.7));
    assert_eq!(request.max_tokens, Some(42));
    assert_eq!(request.additional_params, Some(json!({"override": true})));
    assert_eq!(request.tool_choice, Some(ToolChoice::None));
}

#[tokio::test]
async fn runner_can_merge_additional_params_into_the_baseline() {
    let model = MockCompletionModel::text("done");
    AgentBuilder::new(model.clone())
        .additional_params(json!({"baseline": true, "winner": "baseline"}))
        .build()
        .prompt("go")
        .merge_additional_params(
            json!({"override": true, "winner": "runner"})
                .as_object()
                .expect("object")
                .clone(),
        )
        .run()
        .await
        .expect("runner request should succeed");

    assert_eq!(
        model
            .requests()
            .first()
            .expect("one request")
            .additional_params,
        Some(json!({"baseline": true, "override": true, "winner": "runner"}))
    );
}

#[tokio::test]
async fn the_agents_options_reach_the_request_and_a_runs_options_overlay_them() {
    use rig_core::completion::{CacheRetention, Effort, GenerationOptions, Reasoning};

    let model = MockCompletionModel::from_turns([MockTurn::text("one"), MockTurn::text("two")]);
    let agent = AgentBuilder::new(model.clone())
        .options(GenerationOptions::default().reasoning(Effort::High).seed(7))
        .build();
    agent
        .prompt("go")
        .run()
        .await
        .expect("the agent's request succeeds");
    agent
        .prompt("again")
        .options(
            GenerationOptions::default()
                .cache(CacheRetention::Long)
                .seed(9),
        )
        .run()
        .await
        .expect("the run's request succeeds");

    let requests = model.requests();
    let [first, second] = requests.as_slice() else {
        panic!("two requests: {requests:?}");
    };
    assert_eq!(
        first.options,
        GenerationOptions::default().reasoning(Effort::High).seed(7)
    );
    assert_eq!(
        second.options.reasoning,
        Some(Reasoning::Effort(Effort::High))
    );
    assert_eq!(second.options.cache, Some(CacheRetention::Long));
    assert_eq!(second.options.seed, Some(9));
}

/// A test-only provider extension: a shared section of free-form fields.
mod extension {
    use rig_core::completion::{ExtensionOptions, ProviderExtension, ReplyExtras};
    use rig_core::message::Api;
    use serde_json::{Map, Value};

    #[derive(Clone, Debug, serde::Serialize)]
    pub(super) struct Shared {
        #[serde(rename = "*")]
        pub(super) fields: Map<String, Value>,
    }

    /// Alpha's options; [`Beta`] stores the same type with `with::<Beta>`.
    impl ExtensionOptions for Shared {
        type Ext = Alpha;
    }

    pub(super) struct NoExtras;

    impl ReplyExtras for NoExtras {
        fn from_reply(_api: &Api, _raw: &Value) -> Result<Self, serde_json::Error> {
            Ok(Self)
        }
    }

    pub(super) struct Alpha;

    impl ProviderExtension for Alpha {
        const PROVIDER: &'static str = "alpha";
        type Options = Shared;
        type Extras = NoExtras;
    }

    pub(super) struct Beta;

    impl ProviderExtension for Beta {
        const PROVIDER: &'static str = "beta";
        type Options = Shared;
        type Extras = NoExtras;
    }

    pub(super) fn shared(fields: Value) -> Shared {
        Shared {
            fields: match fields {
                Value::Object(fields) => fields,
                _ => Map::new(),
            },
        }
    }
}

#[tokio::test]
async fn the_agents_provider_options_reach_the_request_and_a_run_replaces_an_entry() {
    use extension::{Alpha, Beta, shared};
    use rig_core::completion::ProviderOptions;

    let agent_options = ProviderOptions::new()
        .with::<Alpha>(&shared(json!({"top_k": 4})))
        .and_then(|options| options.with::<Beta>(&shared(json!({"min_p": 0.1}))))
        .expect("the agent's options serialize");
    let run_options = ProviderOptions::new()
        .with::<Alpha>(&shared(json!({"top_k": 8})))
        .expect("the run's options serialize");
    let model = MockCompletionModel::from_turns([MockTurn::text("one"), MockTurn::text("two")]);
    let agent = AgentBuilder::new(model.clone())
        .provider_options(agent_options.clone())
        .build();
    agent
        .prompt("go")
        .run()
        .await
        .expect("the agent's request succeeds");
    agent
        .prompt("again")
        .provider_options(run_options.clone())
        .run()
        .await
        .expect("the run's request succeeds");

    let requests = model.requests();
    let [first, second] = requests.as_slice() else {
        panic!("two requests: {requests:?}");
    };
    assert_eq!(first.provider_options, agent_options);
    assert_eq!(
        second.provider_options.get::<Alpha>(),
        run_options.get::<Alpha>()
    );
    assert_eq!(
        second.provider_options.get::<Beta>(),
        agent_options.get::<Beta>()
    );
}

#[tokio::test]
async fn provider_option_on_the_agent_and_the_run_equals_the_long_form() {
    use extension::{Alpha, Beta, shared};
    use rig_core::completion::ProviderOptions;

    let beta = ProviderOptions::new()
        .with::<Beta>(&shared(json!({"min_p": 0.1})))
        .expect("beta's options serialize");
    let requests = |short: bool| {
        let beta = beta.clone();
        async move {
            let model =
                MockCompletionModel::from_turns([MockTurn::text("one"), MockTurn::text("two")]);
            let builder = AgentBuilder::new(model.clone()).provider_options(beta.clone());
            let agent = match short {
                true => builder.provider_option(shared(json!({"top_k": 4}))),
                false => builder.provider_options(
                    beta.with::<Alpha>(&shared(json!({"top_k": 4})))
                        .expect("alpha's options serialize"),
                ),
            }
            .build();
            agent.prompt("go").run().await.expect("the agent's request");
            let run = agent.prompt("again");
            let run = match short {
                true => run.provider_option(shared(json!({"top_k": 8}))),
                false => run.provider_options(
                    ProviderOptions::new()
                        .with::<Alpha>(&shared(json!({"top_k": 8})))
                        .expect("the run's options serialize"),
                ),
            };
            run.run().await.expect("the run's request");
            model
                .requests()
                .into_iter()
                .map(|request| request.provider_options)
                .collect::<Vec<_>>()
        }
    };
    let short = requests(true).await;
    assert_eq!(short, requests(false).await);
    assert_eq!(short.len(), 2);
    assert!(short.iter().all(|options| options.contains::<Beta>()));
}

/// The generation options of the last request `model` received, once `run`
/// has run.
async fn sent_options(
    model: &MockCompletionModel,
    run: crate::agent::AgentRunner,
) -> rig_core::completion::GenerationOptions {
    run.run().await.expect("the run's request succeeds");
    model
        .requests()
        .pop()
        .map(|request| request.options)
        .unwrap_or_default()
}

#[tokio::test]
async fn generation_option_shortcuts_on_the_agent_and_the_run_equal_their_long_forms() {
    use rig_core::completion::{
        CacheRetention, Effort, GenerationOptions, OnUnsupported, ServiceTier, Verbosity,
    };

    let long = GenerationOptions::new()
        .reasoning(Effort::Low)
        .cache(CacheRetention::Short)
        .service_tier(ServiceTier::Flex)
        .verbosity(Verbosity::Low)
        .parallel_tool_calls(false)
        .top_p(0.5)
        .seed(3)
        .stop(["x"])
        .on_unsupported(OnUnsupported::Ignore);
    let model = MockCompletionModel::text("done");
    let agent = AgentBuilder::new(model.clone())
        .reasoning(Effort::Low)
        .cache(CacheRetention::Short)
        .service_tier(ServiceTier::Flex)
        .verbosity(Verbosity::Low)
        .parallel_tool_calls(false)
        .top_p(0.5)
        .seed(3)
        .stop(["x"])
        .on_unsupported(OnUnsupported::Ignore)
        .build();
    assert_eq!(sent_options(&model, agent.prompt("go")).await, long);

    let run = GenerationOptions::new()
        .reasoning(Effort::High)
        .cache(CacheRetention::Long)
        .service_tier(ServiceTier::Priority)
        .verbosity(Verbosity::High)
        .parallel_tool_calls(true)
        .top_p(0.9)
        .seed(9)
        .stop(["y", "z"])
        .on_unsupported(OnUnsupported::Error);
    let model = MockCompletionModel::from_turns([MockTurn::text("one"), MockTurn::text("two")]);
    let agent = AgentBuilder::new(model.clone())
        .options(long.clone())
        .build();
    let short = agent
        .prompt("go")
        .reasoning(Effort::High)
        .cache(CacheRetention::Long)
        .service_tier(ServiceTier::Priority)
        .verbosity(Verbosity::High)
        .parallel_tool_calls(true)
        .top_p(0.9)
        .seed(9)
        .stop(["y", "z"])
        .on_unsupported(OnUnsupported::Error);
    let short = sent_options(&model, short).await;
    let long = sent_options(&model, agent.prompt("go").options(run.clone())).await;
    assert_eq!(short, long);
    assert_eq!(short, run);
}

#[tokio::test]
async fn generation_option_calls_apply_in_order() {
    use rig_core::completion::{Effort, GenerationOptions};

    let shared = GenerationOptions::new().reasoning(Effort::High).seed(1);
    // On the agent `options` replaces every field.
    let model = MockCompletionModel::text("done");
    let agent = AgentBuilder::new(model.clone())
        .seed(7)
        .top_p(0.2)
        .options(shared.clone())
        .build();
    assert_eq!(sent_options(&model, agent.prompt("go")).await, shared);
    let model = MockCompletionModel::text("done");
    let agent = AgentBuilder::new(model.clone())
        .options(shared.clone())
        .seed(7)
        .build();
    assert_eq!(
        sent_options(&model, agent.prompt("go")).await,
        shared.clone().seed(7)
    );
    // On a run `options` overlays the fields it sets, over a shortcut
    // called before it; a shortcut called after sets its field on top.
    let model = MockCompletionModel::from_turns([MockTurn::text("one"), MockTurn::text("two")]);
    let agent = AgentBuilder::new(model.clone()).build();
    let before = agent
        .prompt("go")
        .seed(7)
        .top_p(0.2)
        .options(shared.clone());
    assert_eq!(
        sent_options(&model, before).await,
        shared.clone().top_p(0.2)
    );
    let after = agent.prompt("go").options(shared.clone()).seed(7);
    assert_eq!(sent_options(&model, after).await, shared.seed(7));
}

#[tokio::test]
async fn runner_can_clear_configured_request_defaults() {
    let model = MockCompletionModel::text("done");
    AgentBuilder::new(model.clone())
        .preamble("baseline")
        .temperature(0.1)
        .max_tokens(10)
        .additional_params(json!({"baseline": true}))
        .tool_choice(ToolChoice::Required)
        .build()
        .prompt("go")
        .without_preamble()
        .without_temperature()
        .without_max_tokens()
        .without_additional_params()
        .without_tool_choice()
        .run()
        .await
        .expect("runner request should succeed");

    let requests = model.requests();
    let request = requests.first().expect("one request");
    assert!(
        !request
            .chat_history
            .iter()
            .any(|message| matches!(message, crate::completion::Message::System { .. }))
    );
    assert_eq!(request.temperature, None);
    assert_eq!(request.max_tokens, None);
    assert_eq!(request.additional_params, None);
    assert_eq!(request.tool_choice, None);
}

#[tokio::test]
async fn blocking_and_streaming_preserve_raw_failure_while_rewriting_presentation() {
    let blocking = Results::default();
    let blocking_model = MockCompletionModel::from_turns([
        MockTurn::tool_call("tc1", "flaky_tool", json!({})),
        MockTurn::text("done"),
    ]);
    AgentBuilder::new(blocking_model.clone())
        .tool(MetadataFailingTool)
        .add_hook(blocking.clone())
        .build()
        .prompt("go")
        .max_turns(3)
        .run()
        .await
        .expect("blocking run");

    let streaming = Results::default();
    let streaming_model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call_name_delta("tc1", "flaky_tool"),
            MockStreamEvent::tool_call_arguments_delta("tc1", "{}"),
            MockStreamEvent::tool_call("tc1", "flaky_tool", json!({})),
            MockStreamEvent::final_response_with_total_tokens(0),
        ],
        vec![
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response_with_total_tokens(0),
        ],
    ]);
    let mut stream = AgentBuilder::new(streaming_model.clone())
        .tool(MetadataFailingTool)
        .add_hook(streaming.clone())
        .build()
        .prompt("go")
        .max_turns(3)
        .stream();
    while let Some(item) = stream.next().await {
        item.expect("stream item");
    }

    assert_eq!(*blocking.0.lock().unwrap(), *streaming.0.lock().unwrap());
    assert_eq!(
        *blocking.0.lock().unwrap(),
        vec![(
            ToolErrorKind::Timeout,
            "raw timeout failure".into(),
            "shared-result-metadata".into()
        )]
    );

    let blocking_history = serde_json::to_value(
        &blocking_model
            .requests()
            .get(1)
            .expect("second blocking request")
            .chat_history,
    )
    .unwrap();
    let streaming_history = serde_json::to_value(
        &streaming_model
            .requests()
            .get(1)
            .expect("second streaming request")
            .chat_history,
    )
    .unwrap();
    assert_eq!(blocking_history, streaming_history);
    let history = blocking_history.to_string();
    assert!(history.contains("rewritten for model"));
    assert!(!history.contains("raw timeout failure"));
}

/// A runner's content-telemetry opt-in overrides the agent's, for that run
/// only.
#[test]
fn a_runner_overrides_content_telemetry_for_its_run() {
    let agent = AgentBuilder::new(MockCompletionModel::text("ok")).build();
    let runner = agent.prompt("go").record_content_telemetry(true);
    assert!(runner.config.record_telemetry_content);
    assert!(!agent.config.record_telemetry_content);
    assert!(
        !runner
            .record_content_telemetry(false)
            .config
            .record_telemetry_content
    );
}

/// An agent with its model's catalog entry checks each run's options before
/// the run starts: a refused option fails it with nothing sent, unless the
/// options ignore what the model cannot take.
#[tokio::test]
async fn a_model_spec_checks_every_runs_options_before_it_starts() {
    use rig_core::catalog::Catalog;
    use rig_core::completion::{Effort, OnUnsupported, Reasoning};
    use rig_core::error::ProviderError;

    let haiku = Catalog::builtin()
        .resolve("anthropic/claude-haiku-4-5")
        .expect("listed")
        .clone();
    let model = MockCompletionModel::from_turns([MockTurn::text("one"), MockTurn::text("two")]);
    let agent = AgentBuilder::new(model.clone())
        .model_spec(haiku)
        .reasoning(Effort::High)
        .build();

    let refused = agent.prompt("go").run().await.expect_err("refused");
    let crate::completion::PromptError::Provider(ProviderError::UnsupportedOption(option)) =
        &refused
    else {
        panic!("an unsupported option: {refused:?}");
    };
    assert_eq!(option.option, "reasoning");
    assert_eq!(option.provider, "anthropic");
    let mut stream = agent.prompt("go").stream();
    assert!(matches!(
        stream.next().await,
        Some(Err(crate::completion::PromptError::Provider(
            ProviderError::UnsupportedOption(_)
        )))
    ));
    let typed = agent
        .prompt_typed::<Vec<String>>("go")
        .retries(2)
        .await
        .expect_err("refused");
    assert!(
        matches!(
            &typed,
            crate::completion::StructuredOutputError::Prompt(
                crate::completion::PromptError::Provider(ProviderError::UnsupportedOption(_))
            )
        ),
        "an unsupported option: {typed:?}"
    );
    assert!(model.requests().is_empty(), "nothing was sent");

    agent
        .prompt("go")
        .reasoning(Reasoning::Budget { tokens: 2048 })
        .run()
        .await
        .expect("a budget Haiku 4.5 takes");
    agent
        .prompt("again")
        .on_unsupported(OnUnsupported::Ignore)
        .run()
        .await
        .expect("ignored, with a warning");
    let requests = model.requests();
    let [budget, ignored] = requests.as_slice() else {
        panic!("two requests: {requests:?}");
    };
    assert_eq!(
        budget.options.reasoning,
        Some(Reasoning::Budget { tokens: 2048 })
    );
    assert_eq!(
        ignored.options.reasoning, None,
        "the refused effort was dropped"
    );
}

/// A selection hook that sends every call to the model labelled `opus`.
struct SelectOpus;

impl AgentHook for SelectOpus {
    fn on_model_select(
        &self,
        _ctx: &HookContext,
        _event: crate::agent::ModelSelection<'_>,
    ) -> crate::agent::ModelSelectionAction {
        crate::agent::ModelSelectionAction::select("opus")
    }
}

/// A run that switches models is checked against the model each call goes
/// to: the new model's catalog entry, found by the provider and model id it
/// was registered with, whether `using_model_value`, `using_model` or a
/// selection hook switched it. A model the catalog does not list is not
/// checked, and the run says so.
#[tokio::test]
async fn a_model_spec_checks_the_model_a_switched_run_calls() {
    use rig_core::catalog::Catalog;
    use rig_core::completion::{Effort, Reasoning};
    use rig_core::error::ProviderError;
    use rig_core::test_utils::MockScript;

    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
    let capture = crate::test_utils::TraceCapture::default();
    let _default = tracing::subscriber::set_default(capture.subscriber());
    let scripted = |provider: &str, id: &str| {
        let mut model = MockCompletionModel::from_turns([MockTurn::text("ok")]);
        model.wire = MockScript::new(provider).with_id(id);
        model
    };
    let haiku = Catalog::builtin()
        .resolve("anthropic/claude-haiku-4-5")
        .expect("listed")
        .clone();
    let own = MockCompletionModel::from_turns([MockTurn::text("unused")]);
    let opus = scripted("anthropic", "claude-opus-4-8");
    let agent = AgentBuilder::new(own.clone())
        .model_spec(haiku)
        .model_route("opus", opus.clone())
        .reasoning(Effort::High)
        .build();
    let refused = |error: &crate::completion::PromptError| {
        matches!(
            error,
            crate::completion::PromptError::Provider(ProviderError::UnsupportedOption(_))
        )
    };

    // Haiku 4.5, the agent's own model, takes no effort level.
    let error = agent.prompt("go").run().await.expect_err("refused");
    assert!(refused(&error), "{error:?}");

    // Opus 4.8 takes `high`, whichever way the run switched to it.
    let value = scripted("anthropic", "claude-opus-4-8");
    agent
        .prompt("go")
        .using_model_value(value.clone())
        .run()
        .await
        .expect("Opus 4.8 takes `high`");
    agent
        .prompt("go")
        .using_model("opus")
        .run()
        .await
        .expect("the route is Opus 4.8");
    let opus_by_hook = scripted("anthropic", "claude-opus-4-8");
    let hooked = AgentBuilder::new(own.clone())
        .model_spec(
            Catalog::builtin()
                .resolve("anthropic/claude-haiku-4-5")
                .expect("listed")
                .clone(),
        )
        .model_route("opus", opus_by_hook.clone())
        .add_hook(SelectOpus)
        .reasoning(Effort::High)
        .build();
    hooked
        .prompt("go")
        .run()
        .await
        .expect("the hook selects Opus 4.8");

    // A switch to a listed model that refuses the option fails that call.
    let error = agent
        .prompt("go")
        .using_model_value(scripted("anthropic", "claude-haiku-4-5"))
        .run()
        .await
        .expect_err("Haiku 4.5 by value takes no effort level");
    assert!(refused(&error), "{error:?}");

    // A dated snapshot id finds its model, as the encoders find it.
    let error = agent
        .prompt("go")
        .using_model_value(scripted("anthropic", "claude-haiku-4-5-20251001"))
        .run()
        .await
        .expect_err("a Haiku 4.5 snapshot takes no effort level");
    assert!(refused(&error), "{error:?}");

    // A model the catalog does not list is let through, with a warning.
    let unlisted = scripted("anthropic", "claude-unlisted-9");
    capture.clear();
    agent
        .prompt("go")
        .using_model_value(unlisted.clone())
        .run()
        .await
        .expect("not checked");
    assert!(
        capture.events().iter().any(|event| {
            event.level == tracing::Level::WARN && event.message().contains("not checked")
        }),
        "the run says the options were not checked"
    );

    for model in [&value, &opus, &opus_by_hook, &unlisted] {
        let requests = model.requests();
        let [request] = requests.as_slice() else {
            panic!("one request: {requests:?}");
        };
        assert_eq!(
            request.options.reasoning,
            Some(Reasoning::Effort(Effort::High))
        );
    }
    assert!(own.requests().is_empty(), "the refused call was not sent");
}

/// Under `OnUnsupported::Ignore` a cache retention the model refuses is
/// dropped as a refused effort is; under `Error` it fails the run. A route
/// registered from a model value that names no model id is not checked, and
/// a route registered as a handler is found by its label read as a catalog
/// reference.
#[tokio::test]
async fn a_model_spec_drops_a_refused_cache_and_reads_a_handler_routes_label() {
    use rig_core::catalog::Catalog;
    use rig_core::completion::{CacheRetention, Effort, OnUnsupported};
    use rig_core::error::ProviderError;
    use rig_core::serve::adapters::ModelAdapter;

    let mut short_only = Catalog::builtin()
        .resolve("anthropic/claude-haiku-4-5")
        .expect("listed")
        .clone();
    short_only.caching.retention = vec![CacheRetention::Short];
    let own = MockCompletionModel::from_turns([MockTurn::text("one")]);
    let unnamed = MockCompletionModel::from_turns([MockTurn::text("unnamed")]);
    let handled = MockCompletionModel::from_turns([MockTurn::text("unused")]);
    let haiku = "anthropic/claude-haiku-4-5";
    let agent = AgentBuilder::new(own.clone())
        .model_spec(short_only)
        .model_route("unnamed", unnamed.clone())
        .model_route_handler(haiku, ModelAdapter::new(haiku, handled.clone()))
        .build();
    let refused = |error: &crate::completion::PromptError, option: &str| {
        matches!(
            error,
            crate::completion::PromptError::Provider(ProviderError::UnsupportedOption(refused))
                if refused.option == option
        )
    };

    agent
        .prompt("go")
        .cache(CacheRetention::Long)
        .on_unsupported(OnUnsupported::Ignore)
        .run()
        .await
        .expect("ignored, with a warning");
    let requests = own.requests();
    let [ignored] = requests.as_slice() else {
        panic!("one request: {requests:?}");
    };
    assert_eq!(
        ignored.options.cache, None,
        "the refused retention was dropped"
    );

    let error = agent
        .prompt("go")
        .cache(CacheRetention::Long)
        .run()
        .await
        .expect_err("refused");
    assert!(refused(&error, "cache"), "{error:?}");

    agent
        .prompt("go")
        .using_model("unnamed")
        .reasoning(Effort::High)
        .run()
        .await
        .expect("a model value with no model id is not checked");
    assert_eq!(unnamed.requests().len(), 1);

    let error = agent
        .prompt("go")
        .using_model(haiku)
        .reasoning(Effort::High)
        .run()
        .await
        .expect_err("the label names Haiku 4.5, which takes no effort level");
    assert!(refused(&error, "reasoning"), "{error:?}");
    assert!(
        handled.requests().is_empty(),
        "the refused call was not sent"
    );
    assert_eq!(own.requests().len(), 1, "the refused run sent nothing");
}

use super::*;
use crate::test_utils::{MockAddTool, MockCompletionModel, MockSubtractTool, MockToolIndex};
use crate::tool::{ToolContext, ToolExecutionError};

#[derive(Clone)]
struct BuilderHook;

impl AgentHook for BuilderHook {}

/// A model without any `Clone` impl must pass through the builder's
/// erasure seam (`AgentBuilder::new` → the bus's `CompletionAdapter`
/// registered under the agent's model key). The bound is the test: a
/// regression is a compile error.
#[test]
fn builder_accepts_non_clone_model() {
    struct NonCloneModel;

    impl rig_core::completion::CompletionModel for NonCloneModel {
        fn completion(
            &self,
            _request: rig_core::completion::CompletionRequest,
        ) -> impl Future<
            Output = Result<
                rig_core::completion::CompletionResponse,
                rig_core::completion::CompletionError,
            >,
        > + rig_core::wasm_compat::WasmCompatSend {
            std::future::ready(Err(rig_core::completion::CompletionError::ProviderError(
                "compile-time probe".to_string(),
            )))
        }

        fn stream(
            &self,
            _request: rig_core::completion::CompletionRequest,
        ) -> impl Future<
            Output = Result<
                rig_core::streaming::StreamingCompletionResponse,
                rig_core::completion::CompletionError,
            >,
        > + rig_core::wasm_compat::WasmCompatSend {
            std::future::ready(Err(rig_core::completion::CompletionError::ProviderError(
                "compile-time probe".to_string(),
            )))
        }
    }

    let _ = || AgentBuilder::new(NonCloneModel);
}

#[test]
fn hook_can_be_set_after_tool_configuration() {
    let _agent = AgentBuilder::new(MockCompletionModel::text("ok"))
        .tool(MockAddTool)
        .add_hook(BuilderHook)
        .build();
}

struct NamedTool;

impl NamedTool {
    fn new() -> Self {
        Self
    }
}

impl Tool for NamedTool {
    const NAME: &'static str = "registered_named";
    type Error = rig::tool::ToolExecutionError;
    type Args = serde_json::Value;
    type Output = String;

    fn description(&self) -> String {
        "uses its canonical name".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {}})
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, ToolExecutionError> {
        Ok("ok".to_string())
    }
}

#[tokio::test]
async fn typed_tool_builder_paths_advertise_canonical_name() {
    for agent in [
        AgentBuilder::new(MockCompletionModel::text("ok"))
            .tool(NamedTool::new())
            .build(),
        AgentBuilder::new(MockCompletionModel::text("ok"))
            .tool(MockAddTool)
            .tool(NamedTool::new())
            .build(),
    ] {
        let definitions = agent.tool_server_handle.tool_defs(None).await.unwrap();
        assert!(
            definitions
                .iter()
                .any(|definition| definition.name == NamedTool::NAME),
            "the provider definitions dropped the canonical tool name"
        );

        let mut context = ToolContext::new();
        let result = agent
            .tool_server_handle
            .execute(NamedTool::NAME, "{}", &mut context)
            .await;
        assert!(result.is_success());
        assert_eq!(result.output().as_text(), Some("ok"));
    }
}

#[tokio::test]
async fn retrieved_tools_are_exposed_only_for_prompted_retrieval() {
    let retrieval_only = AgentBuilder::new(MockCompletionModel::text("ok"))
        .retrieved_tools(
            1,
            MockToolIndex::new(["add"]),
            ToolSet::from_tools(vec![MockAddTool]),
        )
        .build();
    assert!(
        retrieval_only
            .tool_server_handle
            .tool_defs(None)
            .await
            .unwrap()
            .is_empty()
    );

    let agent = AgentBuilder::new(MockCompletionModel::text("ok"))
        .tool(MockSubtractTool)
        .retrieved_tools(
            1,
            MockToolIndex::new(["add"]),
            ToolSet::from_tools(vec![MockAddTool]),
        )
        .build();

    let always = agent.tool_server_handle.tool_defs(None).await.unwrap();
    assert_eq!(
        always
            .iter()
            .map(|definition| definition.name.as_str())
            .collect::<Vec<_>>(),
        vec!["subtract"]
    );

    let with_retrieval = agent
        .tool_server_handle
        .tool_defs(Some("add two numbers".to_string()))
        .await
        .unwrap();
    assert_eq!(
        with_retrieval
            .iter()
            .map(|definition| definition.name.as_str())
            .collect::<Vec<_>>(),
        vec!["add", "subtract"]
    );
}

/// An agent over a host's bus holds no driver to tap: recording is the
/// host's, through its driver. Asking the builder for it is the host's
/// programming error, refused at build like a wrong-family host key —
/// never a debug assertion about generated keys, never a silent agent
/// that records nothing.
#[test]
#[should_panic(expected = "cannot record: the host records through its driver")]
fn recording_over_a_host_bus_is_refused_at_build() {
    let (dispatcher, registrar, _driver) = crate::bus::Bus::channel();
    let _agent = AgentBuilder::over_bus(
        dispatcher,
        registrar,
        "host",
        rig_core::effect::HandlerKey::from("model"),
    )
    .record_effects()
    .build();
}

mod conversation_without_memory {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tracing::Subscriber;
    use tracing_subscriber::layer::{Context, SubscriberExt};
    use tracing_subscriber::{Layer, Registry};

    /// Collects the message field of every event at warn level or above.
    struct Warnings(Arc<Mutex<Vec<String>>>);

    impl<S: Subscriber> Layer<S> for Warnings {
        fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
            if *event.metadata().level() > tracing::Level::WARN {
                return;
            }
            struct Message(String);
            impl tracing::field::Visit for Message {
                fn record_debug(
                    &mut self,
                    field: &tracing::field::Field,
                    value: &dyn std::fmt::Debug,
                ) {
                    if field.name() == "message" {
                        self.0 = format!("{value:?}");
                    }
                }
            }
            let mut message = Message(String::new());
            event.record(&mut message);
            if let Ok(mut warnings) = self.0.lock() {
                warnings.push(message.0);
            }
        }
    }

    /// A conversation id with no memory backend is not silently inert: the
    /// build warns once, naming the setter. With a backend it does not.
    #[tokio::test]
    async fn build_warns_when_conversation_has_no_memory_backend() {
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
        let warnings = Arc::new(Mutex::new(Vec::new()));
        let subscriber = Registry::default().with(Warnings(warnings.clone()));
        let _default = tracing::subscriber::set_default(subscriber);

        let _agent = AgentBuilder::new(MockCompletionModel::text("x"))
            .conversation("thread-1")
            .build();
        let seen = warnings.lock().expect("warnings").clone();
        assert_eq!(seen.len(), 1, "{seen:?}");
        assert!(seen[0].contains("AgentBuilder::conversation"), "{seen:?}");

        warnings.lock().expect("warnings").clear();
        let _agent = AgentBuilder::new(MockCompletionModel::text("x"))
            .memory(rig_core::memory::InMemoryConversationMemory::new())
            .conversation("thread-1")
            .build();
        assert!(warnings.lock().expect("warnings").is_empty());
    }
}

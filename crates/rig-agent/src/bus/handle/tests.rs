use std::time::Duration;

use serde_json::json;

use super::*;
use crate::bus::Bus;
use rig_core::{
    completion::Message,
    driver::{Exchange, Local, Model, Opened, Opening, Step, Transport},
    effect::{EffectFamily, HandlerKey, family},
    embeddings::{Embedding, EmbeddingResponse},
    error::ErrorKind,
    memory::InMemoryConversationMemory,
    serve::adapters::{MemoryAdapter, ModelAdapter, ToolAdapter},
    test_utils::{MockCompletionModel, MockStreamEvent, MockTurn},
    tool::{Tool, ToolExecutionError},
    wire::Capabilities,
};

async fn within<T>(future: impl Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(5), future)
        .await
        .expect("a dispatch never hangs")
}

struct Double;

#[derive(serde::Deserialize)]
struct DoubleArgs {
    x: i64,
}

impl Tool for Double {
    const NAME: &'static str = "double";
    type Args = DoubleArgs;
    type Output = i64;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "doubles".into()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({"type": "object"})
    }

    async fn call(&self, _context: &mut ToolContext, args: DoubleArgs) -> Result<i64, Self::Error> {
        Ok(args.x * 2)
    }
}

/// A two-dimension embedding runtime: each text embeds to its length and
/// 1.0.
#[derive(Clone)]
struct Tiny;

impl Tiny {
    fn model() -> Model<Local<rig_core::operation::Embedding>, Tiny> {
        Model::new(
            Local::new("tiny").with_capabilities(Capabilities::embedding(8, 2)),
            Tiny,
        )
    }
}

impl Transport<Local<rig_core::operation::Embedding>> for Tiny {
    fn send(
        &self,
        texts: Vec<String>,
        _exchange: Exchange,
    ) -> Opening<Step<rig_core::operation::Embedding>> {
        let response = EmbeddingResponse {
            provider: "tiny".into(),
            ..EmbeddingResponse::new(
                texts
                    .into_iter()
                    .map(|document| Embedding {
                        vec: vec![document.len() as f64, 1.0],
                        document,
                    })
                    .collect(),
            )
        };
        Opening::ready(Opened::new(futures::stream::iter([Ok(Step::End(
            response,
        ))])))
    }
}

fn bus() -> (
    Dispatcher,
    crate::bus::Registrar,
    tokio::task::JoinHandle<()>,
) {
    let (dispatcher, registrar, mut driver) = Bus::channel();
    driver
        .register(
            "model",
            ModelAdapter::new(
                "mock",
                MockCompletionModel::from_turns([MockTurn::text("unary"), MockTurn::text("again")]),
            ),
        )
        .expect("register");
    driver
        .register(
            "streamer",
            ModelAdapter::new(
                "mock-stream",
                MockCompletionModel::from_stream_turns([vec![
                    MockStreamEvent::text("str"),
                    MockStreamEvent::text("eamed"),
                    MockStreamEvent::final_response_with_total_tokens(4),
                ]]),
            ),
        )
        .expect("register");
    driver
        .register("double", ToolAdapter::new(Double))
        .expect("register");
    driver
        .register(
            "memory",
            MemoryAdapter::new(InMemoryConversationMemory::new()),
        )
        .expect("register");
    driver
        .register("embed", ModelAdapter::new("tiny", Tiny::model()))
        .expect("register");
    (dispatcher, registrar, tokio::spawn(driver))
}

#[tokio::test]
async fn binding_checks_the_family_typed_at_bind_time() {
    let (dispatcher, _registrar, _task) = bus();
    let model: ModelHandle = dispatcher
        .handle(&HandlerKey::from("model"))
        .expect("model");
    assert_eq!(model.descriptor().family.family(), EffectFamily::Completion);
    assert_eq!(model.label().as_str(), "mock");

    let report = dispatcher
        .handle::<family::Completion>(&HandlerKey::from("double"))
        .expect_err("a tool key is not a model");
    assert_eq!(report.kind, ErrorKind::HandlerUnavailable);
    assert!(
        report.message.contains("tool_call family"),
        "{}",
        report.message
    );

    let report = dispatcher
        .handle::<family::Tool>(&HandlerKey::from("nope"))
        .expect_err("unknown");
    assert_eq!(report.kind, ErrorKind::HandlerUnavailable);
    assert!(report.message.contains("`nope`"));
}

#[tokio::test]
async fn tool_memory_index_and_embed_handles_call_their_families() {
    let (dispatcher, _registrar, _task) = bus();
    let tool: ToolHandle = dispatcher
        .handle(&HandlerKey::from("double"))
        .expect("tool");
    assert_eq!(tool.name(), "double");
    let crate::bus::ToolAnswer { result, .. } =
        within(tool.call("double", r#"{"x": 21}"#, ToolContext::new()))
            .await
            .expect("called");
    assert_eq!(result.output().as_json(), Some(&json!(42)));

    let memory: MemoryHandle = dispatcher
        .handle(&HandlerKey::from("memory"))
        .expect("memory");
    let conversation = ConversationId::from("c");
    within(memory.append(conversation.clone(), vec![Message::user("x")]))
        .await
        .expect("appended");
    let loaded = within(memory.load(conversation.clone()))
        .await
        .expect("loaded");
    assert_eq!(loaded.len(), 1);
    within(memory.clear(conversation.clone()))
        .await
        .expect("cleared");
    assert!(
        within(memory.load(conversation))
            .await
            .expect("loaded")
            .is_empty()
    );

    let embed: EmbedHandle = dispatcher
        .handle(&HandlerKey::from("embed"))
        .expect("embed");
    assert_eq!(embed.ndims(), Some(2));
    assert_eq!(embed.max_documents(), Some(8));
    assert_eq!(embed.modality(), Some(EmbedModality::Text));
    let embedding = within(embed.embed_text("abc")).await.expect("embedded");
    assert_eq!(embedding.vec, vec![3.0, 1.0]);
    let report = within(embed.embed_images(vec![vec![1]]))
        .await
        .expect_err("text handler");
    assert_eq!(report.kind, ErrorKind::HandlerUnavailable);
}

#[tokio::test]
async fn handles_fail_closed_when_the_driver_is_gone() {
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    driver
        .register("double", ToolAdapter::new(Double))
        .expect("register");
    let tool: ToolHandle = dispatcher
        .handle(&HandlerKey::from("double"))
        .expect("tool");
    drop(driver);
    assert!(tool.is_closed());
    let report = within(tool.call("double", "{}", ToolContext::new()))
        .await
        .expect_err("closed");
    assert_eq!(report.kind, ErrorKind::BusClosed);
}

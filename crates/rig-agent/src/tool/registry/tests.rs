use std::{
    future::{Future, pending, poll_fn},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    task::Poll,
    time::Duration,
};

use super::*;
use rig_core::message::{ImageMediaType, ToolResultContent};
use rig_core::tool::{ContextValue, ToolErrorKind, ToolOutput};

#[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
struct Counter(u32);
impl ContextValue for Counter {
    const KEY: &'static str = "test.counter";
}

#[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
struct Note(String);
impl ContextValue for Note {
    const KEY: &'static str = "test.note";
}

fn rich_error_output(label: &str) -> ToolOutput {
    ToolOutput::content(vec![
        ToolResultContent::text(label),
        ToolResultContent::image_base64("base64data==", Some(ImageMediaType::PNG), None),
    ])
    .expect("fixture content is non-empty")
}

fn assert_rich_error_output(result: &ToolResult, label: &str) {
    let content = result.output().as_content();
    assert_eq!(content.len(), 2);
    assert!(matches!(
        content.first(),
        Some(ToolResultContent::Text(text)) if text.text == label
    ));
    assert!(matches!(content.last(), Some(ToolResultContent::Image(_))));
}

struct Echo;

impl Tool for Echo {
    const NAME: &'static str = "echo";
    type Error = rig::tool::ToolExecutionError;
    type Args = serde_json::Value;
    type Output = serde_json::Value;

    fn description(&self) -> String {
        "echo arguments".into()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object"})
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, ToolExecutionError> {
        if let Some(Counter(value)) = context.get::<Counter>()? {
            context.insert(Counter(value + 1))?;
        }
        context.insert_result(Note("result-metadata".to_string()))?;
        Ok(args)
    }
}

#[tokio::test]
async fn toolset_dispatch_snapshot_is_canonical_and_returns_result_metadata() {
    let mut set = ToolSet::default();
    set.add_tool(Echo);
    let definitions = set.tool_definitions();
    assert_eq!(definitions[0].name, "echo");

    let mut context = ToolContext::new();
    context.insert(Counter(7)).unwrap();
    let result = set.execute("echo", r#"{"value":1}"#, &mut context).await;
    assert!(result.is_success());
    assert_eq!(
        result.output(),
        &ToolOutput::json(serde_json::json!({"value": 1}))
    );
    assert_eq!(
        context.get::<Counter>().unwrap(),
        Some(Counter(7)),
        "tool-local inbound mutations must not change the caller's context"
    );
    assert_eq!(
        context.result::<Note>().unwrap(),
        Some(Note("result-metadata".to_string()))
    );
}

struct PendingTool(Arc<AtomicBool>);

impl Tool for PendingTool {
    const NAME: &'static str = "pending";
    type Error = rig::tool::ToolExecutionError;
    type Args = ();
    type Output = ();

    fn description(&self) -> String {
        "never completes".into()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object"})
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, ToolExecutionError> {
        context.insert_result(Note("unpublished".to_string()))?;
        self.0.store(true, Ordering::SeqCst);
        pending().await
    }
}

#[tokio::test]
async fn cancelled_toolset_dispatch_does_not_retain_stale_result_metadata() {
    let mut set = ToolSet::default();
    let started = Arc::new(AtomicBool::new(false));
    set.add_tool(PendingTool(started.clone()));
    let mut context = ToolContext::new();
    context.insert_result(Note("stale".to_string())).unwrap();

    let mut execution = Box::pin(set.execute(PendingTool::NAME, "null", &mut context));
    tokio::time::timeout(
        Duration::from_secs(1),
        poll_fn(|cx| {
            assert!(execution.as_mut().poll(cx).is_pending());
            started.load(Ordering::SeqCst).then_some(()).map_or_else(
                || {
                    cx.waker().wake_by_ref();
                    Poll::Pending
                },
                Poll::Ready,
            )
        }),
    )
    .await
    .expect("pending tool did not start");
    drop(execution);

    assert_eq!(context.result::<Note>().unwrap(), None);
}

#[tokio::test]
async fn framework_argument_errors_remain_actionable_to_the_model() {
    let mut set = ToolSet::default();
    set.add_tool(Echo);

    let result = set
        .execute("echo", "{not json", &mut ToolContext::new())
        .await;

    assert!(result.is_error_kind(ToolErrorKind::InvalidArgs));
    assert!(
        result
            .output()
            .as_text()
            .is_some_and(|message| message.starts_with("failed to parse tool arguments:"))
    );
    assert_eq!(
        result.output().as_text(),
        result.error().and_then(ToolExecutionError::model_feedback)
    );
}

struct ForeignErrorTool;

impl Tool for ForeignErrorTool {
    const NAME: &'static str = "foreign_error";
    type Error = std::io::Error;
    type Args = ();
    type Output = ();

    fn description(&self) -> String {
        "returns a foreign error type".into()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object"})
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Err(std::io::Error::other("operator-only detail"))
    }
}

#[tokio::test]
async fn typed_foreign_errors_normalize_only_at_dispatch() {
    let direct: std::io::Error = ForeignErrorTool
        .call(&mut ToolContext::new(), ())
        .await
        .expect_err("direct call should retain its typed error");
    assert_eq!(direct.to_string(), "operator-only detail");

    let mut set = ToolSet::default();
    set.add_tool(ForeignErrorTool);
    let result = set
        .execute(ForeignErrorTool::NAME, "null", &mut ToolContext::new())
        .await;
    let error = result.error().expect("dispatch should normalize the error");
    assert_eq!(error.kind(), ToolErrorKind::Other);
    assert_eq!(error.message(), "operator-only detail");
    assert_eq!(error.model_feedback(), Some("the tool failed"));
    assert!(error.is::<std::io::Error>());
}

#[derive(Debug, thiserror::Error)]
#[error("domain timeout")]
struct DomainTimeout;

struct ClassifiedErrorTool;

impl Tool for ClassifiedErrorTool {
    const NAME: &'static str = "classified_error";
    type Error = DomainTimeout;
    type Args = ();
    type Output = ();

    fn description(&self) -> String {
        "classifies a domain error".into()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object"})
    }

    fn map_error(&self, error: Self::Error) -> ToolExecutionError {
        ToolExecutionError::timeout("safe timeout feedback").with_source(error)
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Err(DomainTimeout)
    }
}

#[tokio::test]
async fn tools_can_classify_typed_errors_at_the_erased_boundary() {
    let mut set = ToolSet::default();
    set.add_tool(ClassifiedErrorTool);
    let result = set
        .execute(ClassifiedErrorTool::NAME, "null", &mut ToolContext::new())
        .await;
    let error = result.error().expect("dispatch should normalize the error");
    assert_eq!(error.kind(), ToolErrorKind::Timeout);
    assert_eq!(error.retryable(), Some(true));
    assert_eq!(error.model_feedback(), Some("safe timeout feedback"));
    assert!(error.is::<DomainTimeout>());
}

#[tokio::test]
async fn dynamic_tool_preserves_concrete_error() {
    #[derive(Debug, thiserror::Error)]
    #[error("boom")]
    struct Boom;

    let tool = DynamicTool::new(
        "dynamic",
        "fails",
        serde_json::json!({"type":"object"}),
        |_context, _args| {
            Box::pin(async { Err(ToolExecutionError::provider("upstream").with_source(Boom)) })
        },
    );
    let set = ToolSet::from_dynamic_tools(vec![tool]);
    let result = set.execute("dynamic", "{}", &mut ToolContext::new()).await;
    assert!(result.error().is_some_and(|error| error.is::<Boom>()));
}

struct DirectRichOutput;

impl Tool for DirectRichOutput {
    const NAME: &'static str = "direct_rich_output";
    type Error = rig::tool::ToolExecutionError;
    type Args = serde_json::Value;
    type Output = ToolResultContent;

    fn description(&self) -> String {
        "returns a direct rich-content value".into()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object"})
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, ToolExecutionError> {
        Ok(ToolResultContent::image_base64(
            "base64data==",
            Some(ImageMediaType::PNG),
            None,
        ))
    }
}

#[tokio::test]
async fn direct_rich_typed_output_is_not_serialized_as_json() {
    let mut set = ToolSet::default();
    set.add_tool(DirectRichOutput);

    let result = set
        .execute(DirectRichOutput::NAME, "{}", &mut ToolContext::new())
        .await;

    assert!(result.is_success());
    assert!(matches!(
        result.output().as_content().first(),
        Some(ToolResultContent::Image(_))
    ));
    assert_eq!(result.output().as_json(), None);
}

struct TypedRichError {
    refuse: bool,
}

impl Tool for TypedRichError {
    const NAME: &'static str = "typed_rich_error";
    type Error = rig::tool::ToolExecutionError;
    type Args = serde_json::Value;
    type Output = String;

    fn description(&self) -> String {
        "returns rich failure feedback".into()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object"})
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, ToolExecutionError> {
        let error = if self.refuse {
            ToolExecutionError::refused("typed refusal")
        } else {
            ToolExecutionError::provider("typed failure")
        };
        Err(error.with_model_output(rich_error_output("typed feedback")))
    }
}

#[tokio::test]
async fn typed_failures_and_refusals_preserve_rich_model_output() {
    for refuse in [false, true] {
        let mut set = ToolSet::default();
        set.add_tool(TypedRichError { refuse });

        let result = set
            .execute(TypedRichError::NAME, "{}", &mut ToolContext::new())
            .await;

        assert_eq!(result.is_refused(), refuse);
        assert_eq!(result.is_error(), !refuse);
        assert_rich_error_output(&result, "typed feedback");
    }
}

#[tokio::test]
async fn dynamic_failures_and_refusals_preserve_rich_model_output() {
    for refuse in [false, true] {
        let tool = DynamicTool::new(
            "dynamic_rich_error",
            "returns rich failure feedback",
            serde_json::json!({"type": "object"}),
            move |_context, _args| {
                Box::pin(async move {
                    let error = if refuse {
                        ToolExecutionError::refused("dynamic refusal")
                    } else {
                        ToolExecutionError::provider("dynamic failure")
                    };
                    Err(error.with_model_output(rich_error_output("dynamic feedback")))
                })
            },
        );
        let set = ToolSet::from_dynamic_tools(vec![tool]);

        let result = set
            .execute("dynamic_rich_error", "{}", &mut ToolContext::new())
            .await;

        assert_eq!(result.is_refused(), refuse);
        assert_eq!(result.is_error(), !refuse);
        assert_rich_error_output(&result, "dynamic feedback");
    }
}

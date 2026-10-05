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

/// A tool-family handler that is not a tool adapter: it answers without
/// publishing a context (a replayer for a record without output, a host's
/// own handler).
struct Silent;

impl rig_core::serve::Serve for Silent {
    type Family = family::Tool;

    fn descriptor(&self) -> rig_core::effect::HandlerDescriptor {
        rig_core::effect::HandlerDescriptor {
            key: tool_key("silent"),
            family: FamilyDescriptor::Tool {
                name: "silent".into(),
                description: "answers without publishing".into(),
                parameters: serde_json::json!({"type": "object"}),
                embedding: None,
            },
            layers: Vec::new(),
        }
    }

    async fn serve(
        &self,
        _kind: EffectKind,
        _dispatch: rig_core::serve::Dispatch,
    ) -> rig_core::serve::Reply {
        rig_core::serve::Reply::Outcome(Ok(Outcome::ToolResult {
            result: ToolResult::success(ToolOutput::text("silent")),
        }))
    }
}

/// A tool-family handler whose answer is a report, not a result.
struct Refusing;

impl rig_core::serve::Serve for Refusing {
    type Family = family::Tool;

    fn descriptor(&self) -> rig_core::effect::HandlerDescriptor {
        rig_core::effect::HandlerDescriptor {
            key: tool_key("refusing"),
            family: FamilyDescriptor::Tool {
                name: "refusing".into(),
                description: "refuses every call".into(),
                parameters: serde_json::json!({"type": "object"}),
                embedding: None,
            },
            layers: Vec::new(),
        }
    }

    async fn serve(
        &self,
        _kind: EffectKind,
        _dispatch: rig_core::serve::Dispatch,
    ) -> rig_core::serve::Reply {
        rig_core::serve::Reply::Outcome(Err(ErrorReport::new(ErrorKind::Other, "refused")))
    }
}

/// The inline registration path keeps the caller's inbound values on every
/// ending: a tool that mutates its inbound slot and publishes, a handler
/// that publishes nothing, a handler that refuses. Only result metadata
/// changes, and only to what the call published.
#[tokio::test]
async fn registered_tool_execute_keeps_inbound_values_on_every_ending() {
    let mut context = ToolContext::new();
    context.insert(Counter(7)).unwrap();
    context.insert_result(Note("stale".to_string())).unwrap();

    let echo = RegisteredTool::from_tool(Echo);
    let result = echo.execute(r#"{"value":1}"#.into(), &mut context).await;
    assert!(result.is_success());
    assert_eq!(
        context.get::<Counter>().unwrap(),
        Some(Counter(7)),
        "the tool's inbound mutation stays in the call"
    );
    assert_eq!(
        context.result::<Note>().unwrap(),
        Some(Note("result-metadata".to_string()))
    );

    let silent = RegisteredTool::from_handler(Silent).expect("a tool-family handler");
    let result = silent.execute("{}".into(), &mut context).await;
    assert!(result.is_success());
    assert_eq!(context.get::<Counter>().unwrap(), Some(Counter(7)));
    assert_eq!(
        context.result::<Note>().unwrap(),
        None,
        "a handler that published nothing leaves no result metadata"
    );

    context.insert_result(Note("stale".to_string())).unwrap();
    let refusing = RegisteredTool::from_handler(Refusing).expect("a tool-family handler");
    let result = refusing.execute("{}".into(), &mut context).await;
    assert_eq!(
        result.error().map(ToolExecutionError::kind),
        Some(ToolErrorKind::Other)
    );
    assert_eq!(context.get::<Counter>().unwrap(), Some(Counter(7)));
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

#[tokio::test]
async fn dynamic_failures_and_refusals_preserve_rich_model_output() {
    for refuse in [false, true] {
        let tool = DynamicTool::new(
            rig_core::message::ToolName::new("dynamic_rich_error").expect("tool name"),
            "returns rich failure feedback",
            serde_json::json!({"type": "object"}),
            move |_args| {
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

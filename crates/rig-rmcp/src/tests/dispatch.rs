#[allow(unused_imports)]
use crate::handler::MAX_CONCURRENT_REFRESHES;
#[allow(unused_imports)]
use crate::native::{McpArgumentError, bounded_best_effort_cancellation};
#[allow(unused_imports)]
use crate::*;
#[allow(unused_imports)]
use rig_agent::tool::{ToolContext, ToolResult};
#[allow(unused_imports)]
use rig_core::message::ImageMediaType;
#[allow(unused_imports)]
use rig_core::tool::ToolExecutionError;
#[allow(unused_imports)]
use rig_core::tool::ToolOutput;
use std::{
    future::pending,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use rmcp::model::*;
use rmcp::service::RequestContext;
use rmcp::{RoleServer, ServerHandler, ServiceExt};
use serde_json::json;
use tokio::{
    sync::{Notify, RwLock},
    task::JoinHandle,
};

use rig_agent::tool::{
    ToolErrorKind,
    server::{ToolServer, ToolServerHandle},
};
use rig_core::message::ToolResultContent as RigToolResultContent;

#[derive(Clone)]
enum Scenario {
    Success,
    StructuredSuccess,
    StructuredOnly,
    Hang,
    ServiceError,
    ToolReportedError,
    ImageToolReportedError,
}

#[derive(Clone)]
struct ScenarioServer {
    scenario: Scenario,
    seen: Arc<RwLock<Option<Meta>>>,
    cancelled: Arc<Notify>,
}

impl ServerHandler for ScenarioServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_protocol_version(ProtocolVersion::LATEST)
            .with_server_info(Implementation::new("rig-mcp-test", "0.1.0"))
    }

    async fn call_tool(
        &self,
        _request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        *self.seen.write().await = Some(context.meta.clone());
        match self.scenario {
            Scenario::Success => Ok(CallToolResult::success(vec![ContentBlock::text("ok")])),
            Scenario::StructuredSuccess => {
                let mut response = CallToolResult::success(vec![
                    ContentBlock::text("before"),
                    ContentBlock::image("aGVsbG8=", "image/png"),
                    ContentBlock::text("after"),
                ]);
                response.structured_content = Some(json!({
                    "answer": 42,
                    "source": "fixture"
                }));
                let mut meta = Meta::new();
                meta.0.insert("response-id".into(), json!("response-123"));
                response.meta = Some(meta);
                Ok(response)
            }
            Scenario::StructuredOnly => {
                let mut response = CallToolResult::structured(json!({"answer": 42}));
                response.content.clear();
                Ok(response)
            }
            Scenario::Hang => {
                context.ct.cancelled().await;
                self.cancelled.notify_one();
                Err(ErrorData::internal_error("fixture request cancelled", None))
            }
            Scenario::ServiceError => {
                Err(ErrorData::internal_error("fixture service failed", None))
            }
            Scenario::ToolReportedError => Ok(CallToolResult::error(vec![ContentBlock::text(
                "tool reported exact failure",
            )])),
            Scenario::ImageToolReportedError => {
                Ok(CallToolResult::error(vec![ContentBlock::image(
                    "ZXJyb3ItaW1hZ2U=",
                    "image/png",
                )]))
            }
        }
    }
}

struct Fixture {
    handle: ToolServerHandle,
    /// The adapter itself, for tests that assert MCP argument semantics
    /// directly (the registry path parses JSON in rig-agent first).
    tool: McpTool,
    seen: Arc<RwLock<Option<Meta>>>,
    cancelled: Arc<Notify>,
    _client: rmcp::service::RunningService<rmcp::service::RoleClient, ClientInfo>,
    server_task: JoinHandle<()>,
}

async fn fixture(scenario: Scenario, timeout: Option<Duration>) -> Fixture {
    let seen = Arc::new(RwLock::new(None));
    let cancelled = Arc::new(Notify::new());
    let (client_to_server, server_from_client) = tokio::io::duplex(8192);
    let (server_to_client, client_from_server) = tokio::io::duplex(8192);
    let server = ScenarioServer {
        scenario,
        seen: seen.clone(),
        cancelled: cancelled.clone(),
    };
    let server_task = tokio::spawn(async move {
        let running = server
            .serve((server_from_client, server_to_client))
            .await
            .expect("server start");
        running.waiting().await.expect("server error");
    });
    let client = ClientInfo::default()
        .serve((client_from_server, client_to_server))
        .await
        .expect("client connect");
    let definition = Tool::new(
        "fixture_tool".to_string(),
        "fixture".to_string(),
        Arc::new(serde_json::Map::new()),
    );
    let tool = McpTool::from_mcp_server(definition, client.peer().clone()).with_timeout(timeout);
    let handle = ToolServer::new()
        .portable_dynamic_tool(tool.clone().into())
        .run();
    Fixture {
        handle,
        tool,
        seen,
        cancelled,
        _client: client,
        server_task,
    }
}

async fn execute(fixture: &Fixture, args: &str, context: &mut ToolContext) -> ToolResult {
    tokio::time::timeout(
        Duration::from_secs(5),
        fixture.handle.execute("fixture_tool", args, context),
    )
    .await
    .expect("MCP dispatch exceeded the outer safety timeout")
}

#[tokio::test]
async fn best_effort_cancellation_drops_stalled_delivery_after_grace_period() {
    struct DropProbe(Arc<AtomicBool>);

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    let dropped = Arc::new(AtomicBool::new(false));
    let drop_probe = DropProbe(dropped.clone());
    let stalled = async move {
        let _drop_probe = drop_probe;
        pending::<Result<(), rmcp::ServiceError>>().await
    };

    tokio::time::timeout(
        Duration::from_secs(1),
        bounded_best_effort_cancellation(stalled, Duration::from_millis(10)),
    )
    .await
    .expect("best-effort cancellation exceeded its grace period");

    assert!(dropped.load(Ordering::SeqCst));
}

#[test]
fn model_presentation_preserves_unrepresentable_mcp_blocks_as_json() {
    let blocks = vec![
        ContentBlock::resource(ResourceContents::TextResourceContents {
            uri: "file:///reports/summary.txt".to_string(),
            mime_type: Some("text/plain".to_string()),
            text: "full report".to_string(),
            meta: None,
        }),
        ContentBlock::resource(ResourceContents::BlobResourceContents {
            uri: "file:///reports/raw.bin".to_string(),
            mime_type: Some("application/octet-stream".to_string()),
            blob: "AAEC".to_string(),
            meta: None,
        }),
        ContentBlock::audio("UklGRg==", "audio/wav"),
        ContentBlock::resource_link(
            Resource::new("file:///reports/linked.txt", "linked.txt").with_mime_type("text/plain"),
        ),
        ContentBlock::image("YXZpZg==", "image/avif"),
        ContentBlock::resource(ResourceContents::BlobResourceContents {
            uri: "file:///images/chart.avif".to_string(),
            mime_type: Some("image/avif".to_string()),
            blob: "YmxvYi1hdmlm".to_string(),
            meta: None,
        }),
    ];
    let expected = blocks
        .iter()
        .map(|block| {
            RigToolResultContent::json(
                serde_json::to_value(block).expect("MCP block is JSON serializable"),
            )
        })
        .collect::<Vec<_>>();

    let result = CallToolResult::success(blocks);
    let content = mcp_result_output(&result)
        .expect("MCP content mapping")
        .into_content()
        .into_iter()
        .collect::<Vec<_>>();

    assert_eq!(content, expected);
    assert!(matches!(
        &content[0],
        RigToolResultContent::Json { value }
            if value["resource"]["uri"] == "file:///reports/summary.txt"
                && value["resource"]["mimeType"] == "text/plain"
                && value["resource"]["text"] == "full report"
    ));
    assert!(matches!(
        &content[1],
        RigToolResultContent::Json { value }
            if value["resource"]["uri"] == "file:///reports/raw.bin"
                && value["resource"]["mimeType"] == "application/octet-stream"
                && value["resource"]["blob"] == "AAEC"
    ));
    assert!(matches!(
        &content[2],
        RigToolResultContent::Json { value }
            if value["mimeType"] == "audio/wav" && value["data"] == "UklGRg=="
    ));
    assert!(matches!(
        &content[4],
        RigToolResultContent::Json { value }
            if value["mimeType"] == "image/avif" && value["data"] == "YXZpZg=="
    ));
    assert!(matches!(
        &content[5],
        RigToolResultContent::Json { value }
            if value["resource"]["uri"] == "file:///images/chart.avif"
                && value["resource"]["mimeType"] == "image/avif"
                && value["resource"]["blob"] == "YmxvYi1hdmlm"
    ));
}

#[test]
fn image_resource_blob_maps_to_an_image_block() {
    let result = CallToolResult::success(vec![ContentBlock::resource(
        ResourceContents::BlobResourceContents {
            uri: "file:///images/chart.png".to_string(),
            mime_type: Some("image/png".to_string()),
            blob: "aW1hZ2U=".to_string(),
            meta: None,
        },
    )]);

    assert_eq!(
        mcp_result_output(&result).expect("MCP content mapping"),
        ToolOutput::one(RigToolResultContent::image_base64(
            "aW1hZ2U=",
            Some(ImageMediaType::PNG),
            None,
        ))
    );
}

#[test]
fn string_valued_structured_content_remains_json() {
    let mut result = CallToolResult::structured(json!("forty-two"));
    result.content.clear();

    assert_eq!(
        mcp_result_output(&result).expect("MCP content mapping"),
        ToolOutput::json(json!("forty-two"))
    );
}

#[test]
fn structured_constructors_replace_their_canonical_text_fallback() {
    let value = json!({"answer": 42});
    for result in [
        CallToolResult::structured(value.clone()),
        CallToolResult::structured_error(value.clone()),
    ] {
        assert_eq!(
            mcp_result_output(&result).expect("MCP structured output"),
            ToolOutput::json(value.clone())
        );
    }
}

#[test]
fn structured_content_is_kept_alongside_real_rich_blocks() {
    let value = json!({"answer": 42});
    let mut result = CallToolResult::structured(value.clone());
    result
        .content
        .push(ContentBlock::image("aW1hZ2U=", "image/png"));
    result
        .content
        .push(ContentBlock::text("human-readable note"));

    let mut expected = vec![RigToolResultContent::json(value)];
    expected.push(RigToolResultContent::image_base64(
        "aW1hZ2U=",
        Some(ImageMediaType::PNG),
        None,
    ));
    expected.push(RigToolResultContent::text("human-readable note"));
    assert_eq!(
        mcp_result_output(&result).expect("MCP structured rich output"),
        ToolOutput::content(expected).expect("fixture content is non-empty")
    );
}

#[tokio::test]
async fn canonical_dispatch_forwards_context_meta() {
    let fixture = fixture(Scenario::Success, Some(Duration::from_secs(1))).await;
    let mut meta = Meta::new();
    meta.0.insert("authorization".into(), json!("Bearer test"));
    let mut context = ToolContext::new();
    context.insert(meta);

    let result = execute(&fixture, "{}", &mut context).await;
    assert!(result.is_success());
    assert_eq!(
        fixture
            .seen
            .read()
            .await
            .as_ref()
            .expect("server observed metadata")
            .0
            .get("authorization"),
        Some(&json!("Bearer test"))
    );
    fixture.server_task.abort();
}

#[tokio::test]
async fn canonical_dispatch_classifies_timeout() {
    let fixture = fixture(Scenario::Hang, Some(Duration::from_millis(25))).await;
    let result = execute(&fixture, "{}", &mut ToolContext::new()).await;
    assert!(result.is_error_kind(ToolErrorKind::Timeout));
    assert_eq!(
        result.output().as_text(),
        Some("MCP tool 'fixture_tool' timed out after 25ms")
    );
    tokio::time::timeout(Duration::from_secs(1), fixture.cancelled.notified())
        .await
        .expect("the timed-out MCP request should be cancelled at the peer");
    fixture.server_task.abort();
}

#[tokio::test]
async fn canonical_dispatch_classifies_service_error_and_preserves_source() {
    let fixture = fixture(Scenario::ServiceError, Some(Duration::from_secs(1))).await;
    let result = execute(&fixture, "{}", &mut ToolContext::new()).await;
    let error = result.error().expect("structured MCP service error");
    assert_eq!(error.kind(), ToolErrorKind::Provider);
    assert!(error.is::<rmcp::ServiceError>());
    assert!(error.message().contains("fixture service failed"));
    let output = result.output().render();
    assert!(output.contains("MCP tool 'fixture_tool' request failed"));
    assert!(output.contains("fixture service failed"));
    fixture.server_task.abort();
}

#[tokio::test]
async fn canonical_dispatch_preserves_tool_reported_error_message() {
    let fixture = fixture(Scenario::ToolReportedError, Some(Duration::from_secs(1))).await;
    let result = execute(&fixture, "{}", &mut ToolContext::new()).await;
    assert!(result.is_error_kind(ToolErrorKind::Other));
    assert_eq!(
        result.output(),
        &ToolOutput::one(RigToolResultContent::text("tool reported exact failure"))
    );
    assert_eq!(
        result.error().map(ToolExecutionError::message),
        Some("MCP tool 'fixture_tool' reported an execution error")
    );
    fixture.server_task.abort();
}

#[tokio::test]
async fn canonical_dispatch_preserves_non_text_tool_error_content() {
    let fixture = fixture(
        Scenario::ImageToolReportedError,
        Some(Duration::from_secs(1)),
    )
    .await;
    let mut context = ToolContext::new();
    let result = execute(&fixture, "{}", &mut context).await;

    assert!(result.is_error_kind(ToolErrorKind::Other));
    assert_eq!(
        result.output(),
        &ToolOutput::one(RigToolResultContent::image_base64(
            "ZXJyb3ItaW1hZ2U=",
            Some(ImageMediaType::PNG),
            None,
        ))
    );
    let raw = context
        .result::<CallToolResult>()
        .expect("raw MCP error result metadata");
    assert_eq!(raw.is_error, Some(true));
    assert!(matches!(raw.content.as_slice(), [ContentBlock::Image(_)]));
    fixture.server_task.abort();
}

#[tokio::test]
async fn canonical_dispatch_preserves_ordered_content_and_response_metadata() {
    let fixture = fixture(Scenario::StructuredSuccess, Some(Duration::from_secs(1))).await;
    let mut context = ToolContext::new();
    let result = execute(&fixture, "{}", &mut context).await;

    let mut expected_content = vec![RigToolResultContent::json(json!({
        "answer": 42,
        "source": "fixture"
    }))];
    expected_content.push(RigToolResultContent::text("before"));
    expected_content.push(RigToolResultContent::image_base64(
        "aGVsbG8=",
        Some(ImageMediaType::PNG),
        None,
    ));
    expected_content.push(RigToolResultContent::text("after"));
    assert_eq!(
        result.output(),
        &ToolOutput::content(expected_content).expect("fixture content is non-empty")
    );

    let raw = context
        .result::<CallToolResult>()
        .expect("raw MCP result metadata");
    assert_eq!(raw.content.len(), 3);
    assert_eq!(
        raw.structured_content,
        Some(json!({"answer": 42, "source": "fixture"}))
    );
    assert_eq!(
        context.result::<serde_json::Value>(),
        Some(&json!({"answer": 42, "source": "fixture"}))
    );
    assert_eq!(
        context
            .result::<Meta>()
            .and_then(|meta| meta.0.get("response-id")),
        Some(&json!("response-123"))
    );
    fixture.server_task.abort();
}

#[tokio::test]
async fn canonical_dispatch_uses_structured_content_when_blocks_are_empty() {
    let fixture = fixture(Scenario::StructuredOnly, Some(Duration::from_secs(1))).await;
    let mut context = ToolContext::new();
    let result = execute(&fixture, "{}", &mut context).await;

    assert_eq!(result.output(), &ToolOutput::json(json!({"answer": 42})));
    assert_eq!(
        context.result::<serde_json::Value>(),
        Some(&json!({"answer": 42}))
    );
    fixture.server_task.abort();
}

#[tokio::test]
async fn adapter_classifies_invalid_json_and_preserves_source() {
    let fixture = fixture(Scenario::Success, Some(Duration::from_secs(1))).await;
    let error = fixture
        .tool
        .execute_mcp("{".to_string(), None)
        .await
        .expect_err("structured argument error");
    assert_eq!(error.kind(), ToolErrorKind::InvalidArgs);
    assert!(matches!(
        error.downcast_ref::<McpArgumentError>(),
        Some(McpArgumentError::Json(_))
    ));
    let message = error.to_string();
    assert!(message.contains("MCP tool 'fixture_tool' received invalid arguments"));
    assert!(message.contains("invalid JSON"));
    fixture.server_task.abort();
}

#[tokio::test]
async fn adapter_rejects_non_object_arguments() {
    let fixture = fixture(Scenario::Success, Some(Duration::from_secs(1))).await;
    for args in [r#"[1,2]"#, r#""text""#, "7", "true"] {
        let error = fixture
            .tool
            .execute_mcp(args.to_string(), None)
            .await
            .expect_err("non-object arguments must not become an argument-less MCP call");
        assert_eq!(error.kind(), ToolErrorKind::InvalidArgs, "{args}");
    }

    // Empty input and explicit null remain the documented no-argument forms.
    for args in ["", "null"] {
        fixture
            .tool
            .execute_mcp(args.to_string(), None)
            .await
            .unwrap_or_else(|error| panic!("{args:?} should remain a no-argument call: {error}"));
    }
    fixture.server_task.abort();
}

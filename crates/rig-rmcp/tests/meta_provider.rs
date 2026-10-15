#![cfg(not(target_family = "wasm"))]

use std::{
    error::Error,
    sync::{Arc, Mutex, MutexGuard},
    time::Duration,
};

use rig_agent::{
    AgentBuilder,
    agent::{AgentHook, DispatchAction, DispatchEvent, HookContext},
};
use rig_core::{
    effect::{Answered, EffectKind},
    serve::serve_inline,
    test_utils::{MockCompletionModel, MockTurn},
    tool::{ContextValue, DynamicTool, ToolContext, ToolErrorKind},
};
use rig_rmcp::{McpMeta, McpMetaProvider, McpTool, Meta};
use rmcp::{
    RoleServer, ServerHandler, ServiceExt,
    model::{
        CallToolRequestParams, CallToolResult, ClientInfo, ContentBlock, ErrorData, Implementation,
        JsonObject, ProtocolVersion, ServerCapabilities, ServerInfo, Tool,
    },
    service::{RequestContext, RunningService},
};
use serde_json::{Value, json};
use tokio::{sync::Barrier, task::JoinHandle};

type TestResult<T = ()> = Result<T, Box<dyn Error + Send + Sync>>;

fn check(condition: bool, message: &str) -> TestResult {
    if condition {
        Ok(())
    } else {
        Err(std::io::Error::other(message).into())
    }
}

fn required<T>(value: Option<T>, message: &str) -> TestResult<T> {
    value.ok_or_else(|| std::io::Error::other(message).into())
}

fn lock<T>(value: &Mutex<T>) -> MutexGuard<'_, T> {
    value
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn metadata(value: Value) -> Meta {
    let mut meta = Meta::new();
    meta.0.insert("test/auth".into(), value);
    meta
}

#[derive(Clone, Debug, PartialEq)]
struct Observed {
    name: String,
    arguments: Option<JsonObject>,
    meta: Meta,
}

#[derive(Clone)]
struct RecordingServer {
    seen: Arc<Mutex<Vec<Observed>>>,
    barrier: Option<Arc<Barrier>>,
}

impl ServerHandler for RecordingServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_protocol_version(ProtocolVersion::LATEST)
            .with_server_info(Implementation::new("meta-provider-test", "0.1.0"))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        let mut meta = context.meta;
        // rmcp adds its own progress token to every request independently of application metadata.
        meta.0.shift_remove("progressToken");
        lock(&self.seen).push(Observed {
            name: request.name.to_string(),
            arguments: request.arguments,
            meta,
        });
        if let Some(barrier) = &self.barrier {
            barrier.wait().await;
        }
        Ok(CallToolResult::success(vec![ContentBlock::text("ok")]))
    }
}

struct Fixture {
    tool: McpTool,
    seen: Arc<Mutex<Vec<Observed>>>,
    _client: RunningService<rmcp::RoleClient, ClientInfo>,
    server: JoinHandle<TestResult>,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.server.abort();
    }
}

async fn fixture(concurrent: bool) -> TestResult<Fixture> {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let server = RecordingServer {
        seen: seen.clone(),
        barrier: concurrent.then(|| Arc::new(Barrier::new(2))),
    };
    let (client_to_server, server_from_client) = tokio::io::duplex(8192);
    let (server_to_client, client_from_server) = tokio::io::duplex(8192);
    let server = tokio::spawn(async move {
        let running = server.serve((server_from_client, server_to_client)).await?;
        running.waiting().await?;
        Ok(())
    });
    let client = tokio::time::timeout(
        Duration::from_secs(5),
        ClientInfo::default().serve((client_from_server, client_to_server)),
    )
    .await??;
    let schema: JsonObject = serde_json::from_value(json!({
        "type": "object",
        "properties": {"account": {"type": "string"}}
    }))?;
    let definition = Tool::new("account_lookup", "Looks up an account", Arc::new(schema));
    let tool = McpTool::from_mcp_server(definition, client.peer().clone())
        .with_timeout(Duration::from_secs(2));
    Ok(Fixture {
        tool,
        seen,
        _client: client,
        server,
    })
}

#[derive(Clone, rig_core::serde::Serialize, rig_core::serde::Deserialize)]
#[serde(crate = "rig_core::serde", transparent)]
struct Credentials(String);

impl ContextValue for Credentials {
    const KEY: &'static str = "test.credentials";
}

#[derive(Clone, Debug, PartialEq)]
struct ProviderInput {
    name: String,
    arguments: Option<JsonObject>,
    credential: Option<String>,
}

#[derive(Clone, Default)]
struct DeriveMeta(Arc<Mutex<Vec<ProviderInput>>>);

impl McpMetaProvider for DeriveMeta {
    fn meta_for(
        &self,
        tool: &str,
        arguments: Option<&JsonObject>,
        context: &ToolContext,
    ) -> Option<Meta> {
        let credential = context
            .get::<Credentials>()
            .ok()
            .flatten()
            .map(|value| value.0);
        lock(&self.0).push(ProviderInput {
            name: tool.to_owned(),
            arguments: arguments.cloned(),
            credential: credential.clone(),
        });
        Some(metadata(
            json!({"tool": tool, "arguments": arguments, "credential": credential}),
        ))
    }
}

struct StaticMeta(Option<Meta>);

impl McpMetaProvider for StaticMeta {
    fn meta_for(
        &self,
        _tool: &str,
        _arguments: Option<&JsonObject>,
        _context: &ToolContext,
    ) -> Option<Meta> {
        self.0.clone()
    }
}

fn context(credential: &str) -> TestResult<ToolContext> {
    let mut context = ToolContext::new();
    context.insert(Credentials(credential.into()))?;
    Ok(context)
}

#[tokio::test]
async fn cloned_adapter_derives_meta_from_exact_wire_arguments_and_host_context() -> TestResult {
    let fixture = fixture(false).await?;
    let provider = DeriveMeta::default();
    let adapter = fixture.tool.clone().with_meta_provider(provider.clone());
    let original_schema = adapter.definition().schema_as_json_value();
    let dynamic: DynamicTool = adapter.clone().try_into()?;
    let fallback = metadata(json!("fallback-secret"));
    let mut context = context("host-secret")?;
    context.insert(McpMeta(fallback.clone()))?;
    let arguments = json!({"account": "alice", "nested": {"flags": [1, true, null]}});
    let output = dynamic
        .execute_with(&mut context, arguments.clone())
        .await?;
    check(
        output.as_text() == Some("ok"),
        "request metadata changed model output",
    )?;
    check(
        dynamic.definition().parameters == original_schema,
        "provider changed the tool schema",
    )?;
    check(
        context.require::<McpMeta>()?.0 == fallback,
        "provider mutated context fallback",
    )?;
    let calls = lock(&fixture.seen).clone();
    let call = required(calls.first(), "server did not receive a request")?;
    let inputs = lock(&provider.0).clone();
    let input = required(inputs.first(), "provider did not receive a request")?;
    check(
        calls.len() == 1 && inputs.len() == 1,
        "provider must run once per request",
    )?;
    check(
        input.name == "account_lookup" && call.name == input.name,
        "wrong tool name",
    )?;
    check(
        input.arguments == call.arguments,
        "provider arguments differ from wire arguments",
    )?;
    check(
        input.arguments.as_ref() == arguments.as_object(),
        "provider did not receive parsed arguments",
    )?;
    check(
        input.credential.as_deref() == Some("host-secret"),
        "provider lost host context",
    )?;
    check(
        call.meta
            == metadata(
                json!({"tool": "account_lookup", "arguments": arguments, "credential": "host-secret"}),
            ),
        "provider metadata did not override fallback",
    )?;
    check(
        !serde_json::to_string(&dynamic.definition())?.contains("host-secret"),
        "credential leaked into schema",
    )?;
    check(
        !output.render().contains("host-secret"),
        "credential leaked into model output",
    )?;
    Ok(())
}

#[tokio::test]
async fn provider_none_and_absent_provider_use_context_fallback_but_empty_meta_overrides()
-> TestResult {
    let fixture = fixture(false).await?;
    let fallback = metadata(json!("fallback"));
    for (adapter, expected) in [
        (fixture.tool.clone(), fallback.clone()),
        (
            fixture.tool.clone().with_meta_provider(StaticMeta(None)),
            fallback.clone(),
        ),
        (
            fixture
                .tool
                .clone()
                .with_meta_provider(StaticMeta(Some(Meta::new()))),
            Meta::new(),
        ),
    ] {
        let dynamic: DynamicTool = adapter.try_into()?;
        let mut context = ToolContext::new();
        context.insert(McpMeta(fallback.clone()))?;
        dynamic.execute_with(&mut context, json!({})).await?;
        check(
            lock(&fixture.seen)
                .last()
                .is_some_and(|call| call.meta == expected),
            "metadata precedence changed",
        )?;
    }
    let dynamic: DynamicTool = fixture.tool.clone().try_into()?;
    dynamic.execute(json!({})).await?;
    check(
        lock(&fixture.seen)
            .last()
            .is_some_and(|call| call.meta == Meta::new()),
        "empty context gained request metadata",
    )?;
    Ok(())
}

#[derive(rig_core::serde::Serialize, rig_core::serde::Deserialize)]
#[serde(crate = "rig_core::serde", transparent)]
struct MalformedFallback(String);

impl ContextValue for MalformedFallback {
    const KEY: &'static str = "rmcp.meta";
}

#[tokio::test]
async fn provider_override_does_not_decode_unused_context_fallback() -> TestResult {
    let fixture = fixture(false).await?;
    let chosen = metadata(json!("chosen"));
    let dynamic: DynamicTool = fixture
        .tool
        .clone()
        .with_meta_provider(StaticMeta(Some(chosen.clone())))
        .try_into()?;
    let mut context = ToolContext::new();
    context.insert(MalformedFallback("not an MCP metadata map".into()))?;
    dynamic.execute_with(&mut context, json!({})).await?;
    check(
        lock(&fixture.seen)
            .last()
            .is_some_and(|call| call.meta == chosen),
        "valid override was blocked by an unused fallback",
    )?;
    let fallback: DynamicTool = fixture
        .tool
        .clone()
        .with_meta_provider(StaticMeta(None))
        .try_into()?;
    check(
        fallback
            .execute_with(&mut context, json!({}))
            .await
            .is_err(),
        "invalid fallback stopped producing its decode error",
    )?;
    check(
        lock(&fixture.seen).len() == 1,
        "invalid fallback reached the server",
    )?;
    Ok(())
}

#[tokio::test]
async fn concurrent_calls_keep_derived_metadata_isolated() -> TestResult {
    let fixture = fixture(true).await?;
    let provider = DeriveMeta::default();
    let dynamic: DynamicTool = fixture
        .tool
        .clone()
        .with_meta_provider(provider.clone())
        .try_into()?;
    let mut first = context("first-secret")?;
    let mut second = context("second-secret")?;
    let (first_result, second_result) = tokio::time::timeout(Duration::from_secs(5), async {
        tokio::join!(
            dynamic.execute_with(&mut first, json!({"account": "first"})),
            dynamic.execute_with(&mut second, json!({"account": "second"})),
        )
    })
    .await?;
    first_result?;
    second_result?;
    let calls = lock(&fixture.seen).clone();
    check(
        calls.len() == 2 && lock(&provider.0).len() == 2,
        "concurrent calls were lost",
    )?;
    for (account, credential) in [("first", "first-secret"), ("second", "second-secret")] {
        let arguments = json!({"account": account});
        let call = required(
            calls
                .iter()
                .find(|call| call.arguments.as_ref() == arguments.as_object()),
            "missing concurrent call",
        )?;
        check(
            call.meta
                == metadata(
                    json!({"tool": "account_lookup", "arguments": arguments, "credential": credential}),
                ),
            "metadata crossed concurrent calls",
        )?;
    }
    check(
        first.require::<Credentials>()?.0 == "first-secret",
        "first context changed",
    )?;
    check(
        second.require::<Credentials>()?.0 == "second-secret",
        "second context changed",
    )?;
    Ok(())
}

#[tokio::test]
async fn invalid_dynamic_arguments_do_not_invoke_provider_or_server() -> TestResult {
    let fixture = fixture(false).await?;
    let provider = DeriveMeta::default();
    let dynamic: DynamicTool = fixture
        .tool
        .clone()
        .with_meta_provider(provider.clone())
        .try_into()?;
    for args in ["{", "[1,2]", "\"text\"", "7", "true"] {
        let answer = serve_inline(
            dynamic.handler(),
            EffectKind::ToolCall {
                name: dynamic.name().to_string(),
                args: args.into(),
            },
        )
        .await?;
        check(
            matches!(answer, Answered::ToolResult { result } if result.is_error_kind(ToolErrorKind::InvalidArgs)),
            "invalid arguments were not classified InvalidArgs",
        )?;
    }
    check(
        lock(&provider.0).is_empty(),
        "invalid arguments reached the provider",
    )?;
    check(
        lock(&fixture.seen).is_empty(),
        "invalid arguments reached the server",
    )?;
    Ok(())
}

#[tokio::test]
async fn null_and_empty_object_keep_distinct_argument_forms() -> TestResult {
    let fixture = fixture(false).await?;
    let provider = DeriveMeta::default();
    let dynamic: DynamicTool = fixture
        .tool
        .clone()
        .with_meta_provider(provider.clone())
        .try_into()?;
    dynamic.execute(Value::Null).await?;
    dynamic.execute(json!({})).await?;
    let calls = lock(&fixture.seen).clone();
    let inputs = lock(&provider.0).clone();
    check(
        calls.len() == 2 && inputs.len() == 2,
        "missing no-argument requests",
    )?;
    for (call, input) in calls.iter().zip(&inputs) {
        check(
            call.arguments == input.arguments,
            "no-argument representation changed on the wire",
        )?;
    }
    check(
        inputs
            .first()
            .is_some_and(|input| input.arguments.is_none()),
        "null did not become no arguments",
    )?;
    check(
        inputs
            .get(1)
            .is_some_and(|input| input.arguments.as_ref().is_some_and(JsonObject::is_empty)),
        "empty object lost its argument map",
    )?;
    Ok(())
}

#[tokio::test]
async fn direct_execute_mcp_uses_explicit_metadata_without_provider() -> TestResult {
    let fixture = fixture(false).await?;
    let provider = DeriveMeta::default();
    let adapter = fixture.tool.clone().with_meta_provider(provider.clone());
    let explicit = metadata(json!("explicit"));
    adapter
        .execute_mcp("{}".into(), Some(explicit.clone()))
        .await?;
    adapter.execute_mcp("null".into(), None).await?;
    let calls = lock(&fixture.seen).clone();
    check(
        calls.first().is_some_and(|call| call.meta == explicit),
        "explicit metadata was replaced",
    )?;
    check(
        calls
            .get(1)
            .is_some_and(|call| call.meta == Meta::new() && call.arguments.is_none()),
        "direct call synthesized provider metadata",
    )?;
    check(
        lock(&provider.0).is_empty(),
        "direct execute_mcp invoked the provider",
    )?;
    Ok(())
}

struct RewriteAccount;

impl AgentHook for RewriteAccount {
    async fn on_dispatch(
        &self,
        _context: &HookContext,
        event: DispatchEvent<'_>,
    ) -> DispatchAction {
        if event.tool_name() == Some("account_lookup") {
            DispatchAction::rewrite_tool_args(event.kind, json!({"account": "approved"}))
        } else {
            DispatchAction::Proceed
        }
    }
}

#[tokio::test]
async fn provider_uses_hook_rewritten_arguments() -> TestResult {
    let fixture = fixture(false).await?;
    let provider = DeriveMeta::default();
    let dynamic: DynamicTool = fixture
        .tool
        .clone()
        .with_meta_provider(provider.clone())
        .try_into()?;
    let model = MockCompletionModel::from_turns([
        MockTurn::tool_call(
            "lookup-1",
            "account_lookup",
            json!({"account": "unapproved"}),
        ),
        MockTurn::text("done"),
    ]);
    let agent = AgentBuilder::new(model)
        .dynamic_tool(dynamic)
        .add_hook(RewriteAccount)
        .build();
    let response = tokio::time::timeout(
        Duration::from_secs(5),
        agent
            .prompt("look up the account")
            .tool_context(context("run-secret")?)
            .max_turns(2)
            .run(),
    )
    .await??;
    check(response.output() == "done", "agent did not finish")?;
    let calls = lock(&fixture.seen).clone();
    let call = required(calls.first(), "agent did not call the MCP tool")?;
    let arguments = json!({"account": "approved"});
    check(
        call.arguments.as_ref() == arguments.as_object(),
        "hook rewrite did not reach MCP",
    )?;
    check(
        call.meta
            == metadata(
                json!({"tool": "account_lookup", "arguments": arguments, "credential": "run-secret"}),
            ),
        "provider used pre-rewrite arguments",
    )?;
    check(
        lock(&provider.0).len() == 1,
        "provider did not run exactly once",
    )?;
    Ok(())
}

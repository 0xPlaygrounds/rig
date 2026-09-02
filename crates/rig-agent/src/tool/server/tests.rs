use std::{
    future::{Future, pending, poll_fn},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    task::Poll,
    time::Duration,
};

use crate::{
    test_utils::{
        BarrierMockToolIndex, MockAddTool, MockBarrierTool, MockControlledTool, MockSubtractTool,
        MockToolIndex,
    },
    tool::{
        Tool, ToolContext, ToolEmbedding, ToolExecutionError, ToolSet,
        server::{ToolServer, ToolServerHandle},
    },
};

async fn execute_tool(
    handle: &ToolServerHandle,
    name: &str,
    args: &str,
) -> Result<String, ToolExecutionError> {
    execute_tool_with_context(handle, name, args, &mut ToolContext::new()).await
}

/// A portable tool whose liveness follows `live`, standing in for a remote
/// tool whose transport can disconnect.
fn liveness_gated_tool(name: &str, live: Arc<AtomicBool>) -> crate::tool::PortableDynamicTool {
    crate::tool::PortableDynamicTool::new(
        name,
        "gated",
        serde_json::json!({"type": "object"}),
        |_| Box::pin(async { Ok(crate::tool::ToolOutput::text("ok")) }),
    )
    .with_liveness(move || live.load(Ordering::SeqCst))
}

/// The sync snapshot and the async, prompt-less `tool_defs` read the
/// same always-exposed registry in the same order.
#[tokio::test]
async fn sync_snapshot_matches_async_prompt_less_read() {
    let handle = ToolServer::new()
        .tool(MockAddTool)
        .tool(MockSubtractTool)
        .run();

    let sync_defs = handle.static_tool_defs();
    let async_defs = handle.tool_defs(None).await.unwrap();
    assert_eq!(sync_defs.len(), 2);
    assert_eq!(sync_defs, async_defs);

    let snapshot = handle.snapshot();
    assert_eq!(snapshot.definitions(), sync_defs.as_slice());
    assert_eq!(
        snapshot.names().collect::<Vec<_>>(),
        vec!["add", "subtract"]
    );
    assert_eq!(snapshot.len(), 2);
    assert!(!snapshot.is_empty());
}

/// A tool whose remote backing disconnected is retired by every read path —
/// sync snapshot, sync definitions, forked toolset, and the async read.
#[tokio::test]
async fn retired_tools_are_absent_from_every_read_path() {
    let live = Arc::new(AtomicBool::new(true));
    let handle = ToolServer::new()
        .tool(MockAddTool)
        .portable_dynamic_tool(liveness_gated_tool("remote", live.clone()))
        .run();
    assert_eq!(
        handle.snapshot().names().collect::<Vec<_>>(),
        vec!["add", "remote"]
    );

    live.store(false, Ordering::SeqCst);

    assert_eq!(handle.snapshot().names().collect::<Vec<_>>(), vec!["add"]);
    assert_eq!(handle.static_tool_defs().len(), 1);
    assert!(!handle.toolset().contains("remote"));
    assert_eq!(handle.tool_defs(None).await.unwrap().len(), 1);
}

/// A snapshot pins implementations: it keeps executing the tool it was
/// taken with after the registry replaces or removes that name.
#[tokio::test]
async fn snapshot_executes_pinned_implementation() {
    let handle = ToolServer::new().tool(MockAddTool).run();
    let snapshot = handle.snapshot();
    handle.remove_tool("add");
    assert!(handle.snapshot().is_empty());

    let mut context = ToolContext::new();
    let result = snapshot
        .execute("add", r#"{"x": 2, "y": 3}"#, &mut context)
        .await;
    assert_eq!(result.output().render(), "5");
}

/// `toolset()` forks the registry: the fork shares implementations but
/// later changes on either side stay local.
#[tokio::test]
async fn toolset_forks_the_registry() {
    let handle = ToolServer::new().tool(MockAddTool).run();
    let mut fork = handle.toolset();
    assert!(fork.contains("add"));

    fork.add_tool(MockSubtractTool);
    assert!(!handle.snapshot().names().any(|name| name == "subtract"));

    handle.remove_tool("add");
    assert!(fork.contains("add"));

    // The fork builds a second, independent server with the same tools.
    let second = ToolServer::new().run();
    second.append_toolset(fork);
    assert_eq!(
        execute_tool(&second, "add", r#"{"x": 1, "y": 1}"#)
            .await
            .unwrap(),
        "2"
    );
}

/// The sync read needs no executor: a plain test reads definitions that
/// another thread registered, with no runtime in sight.
#[test]
fn static_tool_defs_reads_without_a_runtime() {
    let handle = ToolServer::new().run();
    assert!(handle.static_tool_defs().is_empty());

    let writer = handle.clone();
    std::thread::spawn(move || {
        writer.add_tool(MockAddTool);
        writer.add_tool(MockSubtractTool);
    })
    .join()
    .expect("registering thread");

    let names = handle
        .static_tool_defs()
        .into_iter()
        .map(|definition| definition.name)
        .collect::<Vec<_>>();
    assert_eq!(names, vec!["add", "subtract"]);
}

async fn execute_tool_with_context(
    handle: &ToolServerHandle,
    name: &str,
    args: &str,
    context: &mut ToolContext,
) -> Result<String, ToolExecutionError> {
    let result = handle.execute(name, args, context).await;
    match result.error() {
        Some(error) => Err(error.clone()),
        None => Ok(result.output().render()),
    }
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
        _context: &mut crate::tool::ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, crate::tool::ToolExecutionError> {
        Ok("ok".to_string())
    }
}

struct ReplacementTool {
    description: &'static str,
    output: &'static str,
}

impl Tool for ReplacementTool {
    const NAME: &'static str = "replacement";
    type Error = rig::tool::ToolExecutionError;
    type Args = serde_json::Value;
    type Output = String;

    fn description(&self) -> String {
        self.description.to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {}})
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, ToolExecutionError> {
        Ok(self.output.to_string())
    }
}

#[derive(Debug, thiserror::Error)]
#[error("init error")]
struct InitError;

impl ToolEmbedding for NamedTool {
    type InitError = InitError;
    type Context = ();
    type State = ();

    fn embedding_docs(&self) -> Vec<String> {
        vec!["named retrieved tool".to_string()]
    }

    fn context(&self) -> Self::Context {}

    fn init(_state: Self::State, _context: Self::Context) -> Result<Self, Self::InitError> {
        Ok(Self::new())
    }
}

#[tokio::test]
pub async fn test_toolserver() {
    let server = ToolServer::new();

    let handle = server.run();

    handle.add_tool(MockAddTool);
    let res = handle.tool_defs(None).await.unwrap();

    assert_eq!(res.len(), 1);

    let json_args_as_string = serde_json::to_string(&serde_json::json!({"x": 2, "y": 5})).unwrap();
    let res = execute_tool(&handle, "add", &json_args_as_string)
        .await
        .unwrap();
    assert_eq!(res, "7");

    handle.remove_tool("add");
    let res = handle.tool_defs(None).await.unwrap();

    assert_eq!(res.len(), 0);
}

#[tokio::test]
async fn definition_snapshot_pins_the_exact_tool_registration() {
    let handle = ToolServer::new()
        .tool(ReplacementTool {
            description: "first schema",
            output: "first implementation",
        })
        .run();
    let snapshot = handle.snapshot_tool_defs(None).await.unwrap();

    handle.add_tool(ReplacementTool {
        description: "second schema",
        output: "second implementation",
    });

    assert_eq!(snapshot.definitions()[0].description, "first schema");
    let dispatch = snapshot
        .dispatch(ReplacementTool::NAME, "{}", &ToolContext::new())
        .await;
    assert_eq!(dispatch.result.output().render(), "first implementation");

    let live = handle
        .dispatch(ReplacementTool::NAME, "{}", &ToolContext::new())
        .await;
    assert_eq!(live.result.output().render(), "second implementation");

    let next_snapshot = handle.snapshot_tool_defs(None).await.unwrap();
    assert_eq!(next_snapshot.definitions()[0].description, "second schema");
    let dispatch = next_snapshot
        .dispatch(ReplacementTool::NAME, "{}", &ToolContext::new())
        .await;
    assert_eq!(dispatch.result.output().render(), "second implementation");
}

#[tokio::test]
pub async fn test_toolserver_append_toolset_matches_add_tool() {
    let mut via_add_tool = {
        let handle = ToolServer::new().run();
        handle.add_tool(MockAddTool);
        handle.add_tool(MockSubtractTool);
        handle.tool_defs(None).await.unwrap()
    };
    via_add_tool.sort_by(|a, b| a.name.cmp(&b.name));

    let mut via_append_toolset = {
        let handle = ToolServer::new().run();
        let mut toolset = ToolSet::default();
        toolset.add_tool(MockAddTool);
        toolset.add_tool(MockSubtractTool);
        handle.append_toolset(toolset);
        handle.tool_defs(None).await.unwrap()
    };
    via_append_toolset.sort_by(|a, b| a.name.cmp(&b.name));

    assert_eq!(via_add_tool.len(), via_append_toolset.len());
    assert!(
        via_add_tool
            .iter()
            .zip(via_append_toolset.iter())
            .all(|(a, b)| a.name == b.name),
        "append_toolset must surface the same LLM-visible tools as add_tool",
    );
}

#[tokio::test]
pub async fn builder_tool_uses_canonical_static_name() {
    let handle = ToolServer::new().tool(NamedTool::new()).run();

    let defs = handle.tool_defs(None).await.unwrap();
    assert_eq!(defs.len(), 1);
    assert_eq!(defs[0].name, NamedTool::NAME);
}

#[tokio::test]
pub async fn handle_add_tool_uses_canonical_static_name() {
    let handle = ToolServer::new().run();
    handle.add_tool(NamedTool::new());

    let defs = handle.tool_defs(None).await.unwrap();
    assert_eq!(defs.len(), 1);
    assert_eq!(defs[0].name, NamedTool::NAME);
}

#[tokio::test]
pub async fn retrieval_resolves_canonical_key() {
    let mut toolset = ToolSet::default();
    toolset.add_retrieved_tool(NamedTool::new());
    let handle = ToolServer::new()
        .retrieved_tools(1, MockToolIndex::new([NamedTool::NAME]), toolset)
        .run();

    let defs = handle
        .tool_defs(Some("use the changing tool".to_string()))
        .await
        .unwrap();
    assert_eq!(defs.len(), 1);
    assert_eq!(defs[0].name, NamedTool::NAME);
}

#[tokio::test]
pub async fn get_tool_defs_preserves_static_registration_order() {
    let handle = ToolServer::new().run();
    handle.add_tool(MockSubtractTool);
    handle.add_tool(MockAddTool);

    let defs = handle.tool_defs(None).await.unwrap();
    assert_eq!(
        defs.iter().map(|def| def.name.as_str()).collect::<Vec<_>>(),
        vec!["subtract", "add"]
    );
}

#[tokio::test]
pub async fn get_tool_defs_dedupes_dynamic_and_static_overlap() {
    // One shared toolset backs both lists, so a dynamically retrieved
    // name that is also static must yield a single definition.
    let handle = ToolServer::new()
        .tool(MockAddTool)
        .retrieved_tools(1, MockToolIndex::new(["add"]), ToolSet::default())
        .run();

    let defs = handle
        .tool_defs(Some("add two numbers".to_string()))
        .await
        .unwrap();
    assert_eq!(
        defs.len(),
        1,
        "dynamic/static name overlap must not produce duplicate declarations: {:?}",
        defs.iter().map(|def| def.name.as_str()).collect::<Vec<_>>()
    );
    assert_eq!(defs[0].name, "add");
}

#[tokio::test]
async fn retrieval_registration_preserves_existing_always_exposure() {
    let handle = ToolServer::new()
        .tool(MockAddTool)
        .retrieved_tools(
            1,
            MockToolIndex::new(["add"]),
            ToolSet::from_tools(vec![MockAddTool]),
        )
        .run();

    let defs = handle.tool_defs(None).await.unwrap();
    assert_eq!(
        defs.iter()
            .map(|definition| definition.name.as_str())
            .collect::<Vec<_>>(),
        vec!["add"],
        "merging a retrieval implementation must not demote an always-exposed registration"
    );
}

#[tokio::test]
pub async fn duplicate_registration_advertises_one_definition() {
    let handle = ToolServer::new().tool(MockAddTool).run();
    handle.add_tool(MockAddTool);

    let mut toolset = ToolSet::default();
    toolset.add_tool(MockAddTool);
    handle.append_toolset(toolset);

    let defs = handle.tool_defs(None).await.unwrap();
    assert_eq!(
        defs.len(),
        1,
        "re-registering a name must not advertise duplicate declarations"
    );
    assert_eq!(defs[0].name, "add");
}

#[tokio::test]
pub async fn test_toolserver_retrieved_tools() {
    // Create a toolset with both tools
    let mut toolset = ToolSet::default();
    toolset.add_tool(MockAddTool);
    toolset.add_tool(MockSubtractTool);

    // Create a mock index that will return "subtract" as the dynamic tool
    let mock_index = MockToolIndex::new(["subtract"]);

    // Build server with static tool "add" and dynamic tools from the mock index
    let server = ToolServer::new().tool(MockAddTool).retrieved_tools(
        1,
        mock_index,
        ToolSet::from_tools(vec![MockSubtractTool]),
    );

    let handle = server.run();

    // Test with None prompt - should only return static tools
    let res = handle.tool_defs(None).await.unwrap();
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].name, "add");

    // Test with Some prompt - should return both static and dynamic tools
    let res = handle
        .tool_defs(Some("calculate difference".to_string()))
        .await
        .unwrap();
    assert_eq!(res.len(), 2);

    // Check that both tools are present (order may vary)
    let tool_names: Vec<&str> = res.iter().map(|t| t.name.as_str()).collect();
    assert!(tool_names.contains(&"add"));
    assert!(tool_names.contains(&"subtract"));
}

#[tokio::test]
pub async fn test_toolserver_retrieved_tools_missing_implementation() {
    // Create a mock index that returns a tool ID that doesn't exist in the toolset
    let mock_index = MockToolIndex::new(["nonexistent_tool"]);

    // Build server with only static tool, but dynamic index references missing tool
    let server =
        ToolServer::new()
            .tool(MockAddTool)
            .retrieved_tools(1, mock_index, ToolSet::default());

    let handle = server.run();

    // Test with Some prompt - should only return static tool since dynamic tool is missing
    let res = handle
        .tool_defs(Some("some query".to_string()))
        .await
        .unwrap();
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].name, "add");
}

#[tokio::test]
pub async fn test_toolserver_concurrent_tool_execution() {
    let num_calls = 3;
    let barrier = Arc::new(tokio::sync::Barrier::new(num_calls));

    let server = ToolServer::new().tool(MockBarrierTool::new(barrier.clone()));
    let handle = server.run();

    // Make concurrent calls
    let futures: Vec<_> = (0..num_calls)
        .map(|_| execute_tool(&handle, "barrier_tool", "{}"))
        .collect();

    // If execution is sequential, the first call will block at the barrier forever.
    // We use a 1-second timeout to fail fast instead of hanging the test runner.
    let result =
        tokio::time::timeout(Duration::from_secs(1), futures::future::join_all(futures)).await;

    assert!(
        result.is_ok(),
        "Tool execution deadlocked! Tools are executing sequentially instead of concurrently."
    );

    // All calls should succeed
    for res in result.unwrap() {
        assert!(res.is_ok(), "Tool call failed: {res:?}");
        assert_eq!(res.unwrap(), "done");
    }
}

#[tokio::test]
pub async fn test_toolserver_write_while_tool_running() {
    let started = Arc::new(tokio::sync::Notify::new());
    let allow_finish = Arc::new(tokio::sync::Notify::new());

    // Build server with the controlled tool that waits at a barrier during execution
    let tool = MockControlledTool::new(started.clone(), allow_finish.clone());

    let server = ToolServer::new().tool(tool);
    let handle = server.run();

    // Start tool call in background
    let handle_clone = handle.clone();
    let call_task =
        tokio::spawn(async move { execute_tool(&handle_clone, "controlled", "{}").await });

    // Wait until we are strictly inside `call()`
    started.notified().await;

    // Write to the state (add a tool) while the tool call is mid-execution.
    // If the read lock were incorrectly held across tool execution, this
    // sync call would block forever and the test harness would time out.
    handle.add_tool(MockAddTool);

    // Allow the background tool to finish and clean up
    allow_finish.notify_one();
    let call_result = call_task.await.unwrap();
    assert_eq!(call_result.unwrap(), "42");
}

#[tokio::test]
pub async fn test_toolserver_parallel_retrieval() {
    // We expect exactly 2 parallel searches to hit the barrier at the same time
    let barrier = Arc::new(tokio::sync::Barrier::new(2));

    let index1 = BarrierMockToolIndex::new(barrier.clone(), "add");
    let index2 = BarrierMockToolIndex::new(barrier.clone(), "subtract");

    // Put both tools in the toolset so they resolve correctly
    let mut toolset = ToolSet::default();
    toolset.add_tool(MockAddTool);
    toolset.add_tool(MockSubtractTool);

    let server = ToolServer::new()
        .retrieved_tools(1, index1, ToolSet::default())
        .retrieved_tools(1, index2, toolset);

    let handle = server.run();

    // This will trigger a search across both indices.
    // If fetched sequentially, the first index will wait at the barrier forever.
    let get_defs = tokio::time::timeout(
        std::time::Duration::from_secs(1),
        handle.tool_defs(Some("do math".to_string())),
    )
    .await;

    assert!(
        get_defs.is_ok(),
        "Dynamic tools were fetched sequentially! The first query deadlocked waiting for the second query to start."
    );

    let defs = get_defs.unwrap().unwrap();
    assert_eq!(defs.len(), 2);

    let tool_names: Vec<&str> = defs.iter().map(|t| t.name.as_str()).collect();
    assert!(tool_names.contains(&"add"));
    assert!(tool_names.contains(&"subtract"));
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SessionId(String);

#[derive(serde::Serialize, serde::Deserialize)]
struct Counter(usize);

#[derive(serde::Deserialize, serde::Serialize)]
struct ContextReader;

impl crate::tool::Tool for ContextReader {
    const NAME: &'static str = "context_reader";
    type Error = rig::tool::ToolExecutionError;
    type Args = serde_json::Value;
    type Output = String;

    fn description(&self) -> String {
        "Reads SessionId from context".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {}})
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, ToolExecutionError> {
        if let Some(Counter(value)) = context.get::<Counter>() {
            context.insert(Counter(value + 1))?;
            context.insert_result(value + 1)?;
        }
        Ok(context.get::<SessionId>().map_or_else(
            || "no session".to_string(),
            |session| format!("session:{}", session.0),
        ))
    }
}

#[tokio::test]
async fn context_reaches_the_single_execute_path() {
    let handle = ToolServer::new().tool(ContextReader).run();
    let mut context = ToolContext::new();
    context.insert(SessionId("abc-123".to_string())).unwrap();
    let result = execute_tool_with_context(&handle, "context_reader", "{}", &mut context)
        .await
        .unwrap();
    assert_eq!(result, "session:abc-123");
}

#[tokio::test]
async fn server_dispatch_snapshot_isolates_and_only_publishes_result_metadata() {
    let handle = ToolServer::new().tool(ContextReader).run();
    let mut context = ToolContext::new();
    context.insert(Counter(0)).unwrap();

    let result = execute_tool_with_context(&handle, "context_reader", "{}", &mut context)
        .await
        .unwrap();

    assert_eq!(result, "no session");
    assert_eq!(
        context.get::<Counter>().map(|value| value.0),
        Some(0),
        "tool-local inbound mutations must not change the caller's context"
    );
    assert_eq!(context.result::<usize>(), Some(1));
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
        context.insert_result("unpublished".to_string())?;
        self.0.store(true, Ordering::SeqCst);
        pending().await
    }
}

#[tokio::test]
async fn cancelled_server_dispatch_does_not_retain_stale_result_metadata() {
    let started = Arc::new(AtomicBool::new(false));
    let handle = ToolServer::new().tool(PendingTool(started.clone())).run();
    let mut context = ToolContext::new();
    context.insert_result("stale".to_string()).unwrap();

    let mut execution = Box::pin(handle.execute(PendingTool::NAME, "null", &mut context));
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

    assert!(context.result::<String>().is_none());
}

#[tokio::test]
async fn empty_tool_context_uses_default() {
    let handle = ToolServer::new().tool(ContextReader).run();
    let result = execute_tool(&handle, "context_reader", "{}").await.unwrap();

    assert_eq!(result, "no session");
}

#[tokio::test]
async fn tool_ignoring_context_still_works() {
    let handle = ToolServer::new().tool(MockAddTool).run();
    let mut context = ToolContext::new();
    context.insert(SessionId("ignored".to_string())).unwrap();
    let args = serde_json::to_string(&serde_json::json!({"x": 3, "y": 7})).unwrap();
    let result = execute_tool_with_context(&handle, "add", &args, &mut context)
        .await
        .unwrap();

    assert_eq!(result, "10");
}

#[tokio::test]
async fn execute_classifies_a_missing_tool_as_not_found() {
    let handle = ToolServer::new().tool(MockAddTool).run();
    let error = execute_tool(&handle, "does_not_exist", "{}")
        .await
        .unwrap_err();
    assert_eq!(error.kind(), crate::tool::ToolErrorKind::NotFound);
    assert!(
        error
            .model_feedback()
            .is_some_and(|feedback| feedback.contains("does_not_exist"))
    );
}

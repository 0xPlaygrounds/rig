use super::*;
use crate::tool::{DynamicTool, McpToolOrigin, ToolOrigin, ToolOutput, ToolSet};

fn dynamic(name: &str) -> DynamicTool {
    let reply = format!("{name}!");
    DynamicTool::new(
        rig_core::message::ToolName::new(name).expect("tool name"),
        format!("the {name} tool"),
        serde_json::json!({"type": "object"}),
        move |_| {
            let reply = reply.clone();
            Box::pin(async move { Ok(ToolOutput::text(reply)) })
        },
    )
}

/// `ToolSet::catalog()` advertises exactly the always-exposed tools, in
/// registration order, and dispatches by the advertised name.
#[tokio::test]
async fn catalog_matches_definitions_and_dispatches_by_name() {
    let mut set = ToolSet::default();
    set.add_dynamic_tool(dynamic("alpha"));
    set.add_dynamic_tool(dynamic("beta"));
    let mut retrieval_only = ToolSet::default();
    retrieval_only.add_dynamic_tool(dynamic("gamma"));
    set.add_retrievable_tools(retrieval_only);

    let catalog = set.catalog();
    assert_eq!(catalog.names().collect::<Vec<_>>(), ["alpha", "beta"]);
    assert_eq!(catalog.definitions().len(), 2);
    assert_eq!(
        catalog.definitions(),
        &set.tool_definitions()[..2],
        "always-exposed definitions are the set's, in order"
    );
    assert!(set.contains("gamma") && !catalog.contains("gamma"));

    let mut context = ToolContext::new();
    let result = catalog.execute("beta", "{}", &mut context).await;
    assert_eq!(result.output().as_text(), Some("beta!"));

    let missing = catalog.execute("gamma", "{}", &mut context).await;
    assert!(!missing.is_success(), "retrieval-only tools are not pinned");
}

/// An owned execution matches `execute` (result and published context) and
/// its future is `Send + 'static` — spawnable on any executor.
#[tokio::test]
async fn an_owned_execution_matches_the_borrowed_one_and_is_static() {
    fn assert_send_static<T: Send + 'static>(value: T) -> T {
        value
    }

    let mut set = ToolSet::default();
    set.add_dynamic_tool(dynamic("alpha"));
    let catalog = set.catalog();

    let mut context = ToolContext::new();
    let borrowed = catalog.execute("alpha", "{}", &mut context).await;

    // An owned execution — the catalog, arguments and context all moved into
    // the future — is `Send + 'static`, so a host can spawn it.
    let owned_catalog = catalog.clone();
    let (owned, owned_context) = assert_send_static(async move {
        let mut context = ToolContext::new();
        let result = owned_catalog.execute("alpha", "{}", &mut context).await;
        (result, context)
    })
    .await;
    assert_eq!(owned.output().as_text(), borrowed.output().as_text());
    assert_eq!(
        format!("{owned_context:?}"),
        format!("{context:?}"),
        "owned dispatch publishes the same context metadata"
    );
}

fn describe(entry: CatalogEntry<'_>) -> String {
    match entry {
        CatalogEntry::Registered(tool) => match tool.origin() {
            ToolOrigin::Native { type_name } => {
                format!(
                    "native {}",
                    type_name.rsplit("::").next().unwrap_or_default()
                )
            }
            ToolOrigin::Dynamic => format!("dynamic {}", tool.definition().name),
            ToolOrigin::Mcp(mcp) => format!(
                "mcp {}/{} as {}, source {:?}",
                mcp.server().unwrap_or_default(),
                mcp.tool(),
                tool.definition().name,
                mcp.source::<u32>(),
            ),
            other => format!("{other:?}"),
        },
        CatalogEntry::ProviderHosted(tool) => format!("provider {}", tool.kind),
    }
}

/// Every catalog entry says what kind of tool it is and where it came from:
/// registrations keep their origin, provider-hosted tools follow them. A
/// snapshot keeps the provider tools it was taken with.
#[test]
fn entries_record_each_origin_and_the_provider_hosted_tools() {
    let mcp = dynamic("search").with_origin(ToolOrigin::Mcp(
        McpToolOrigin::new("remote_search")
            .with_server("docs")
            .with_source(7_u32),
    ));
    let handle = crate::tool::server::ToolServer::new()
        .tool(crate::test_utils::MockAddTool)
        .dynamic_tool(dynamic("alpha"))
        .dynamic_tool(mcp)
        .provider_tool(ProviderToolDefinition::new("web_search"))
        .run();

    let catalog = handle.snapshot();
    assert_eq!(
        catalog.entries().map(describe).collect::<Vec<_>>(),
        [
            "native MockAddTool",
            "dynamic alpha",
            "mcp docs/remote_search as search, source Some(7)",
            "provider web_search",
        ]
    );
    assert_eq!(catalog.len(), 3, "provider tools are not dispatched here");

    handle.remove_provider_tool("web_search");
    handle.add_provider_tool(ProviderToolDefinition::new("code_interpreter"));
    let mut set = ToolSet::default();
    set.add_provider_tool(ProviderToolDefinition::new("file_search"));
    handle.add_tools(set);
    let kinds = |catalog: &ToolCatalog| {
        catalog
            .provider_tools()
            .iter()
            .map(|tool| tool.kind.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        kinds(&handle.snapshot()),
        ["code_interpreter", "file_search"]
    );
    assert_eq!(kinds(&catalog), ["web_search"]);
}

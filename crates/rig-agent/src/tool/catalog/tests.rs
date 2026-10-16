use crate::tool::{DynamicTool, ToolContext, ToolOutput, ToolSet};

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

    let beta = catalog.get("beta").expect("advertised");
    let result = beta.execute("{}".into(), &mut ToolContext::new()).await;
    assert_eq!(result.output().as_text(), Some("beta!"));
    assert!(
        catalog.get("gamma").is_none(),
        "retrieval-only tools are not pinned"
    );
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

    let alpha = catalog.get("alpha").expect("advertised");
    let mut context = ToolContext::new();
    let borrowed = alpha.execute("{}".into(), &mut context).await;

    // An owned execution — the registration, arguments and context all
    // moved into the future — is `Send + 'static`, so a host can spawn it.
    let owned_tool = alpha.clone();
    let (owned, owned_context) = assert_send_static(async move {
        let mut context = ToolContext::new();
        let result = owned_tool.execute("{}".into(), &mut context).await;
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

use super::*;
use crate::tool::{PortableDynamicTool, ToolOutput, ToolSet};

fn portable(name: &str) -> PortableDynamicTool {
    let reply = format!("{name}!");
    PortableDynamicTool::new(
        name,
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
    set.add_portable_dynamic_tool(portable("alpha"));
    set.add_portable_dynamic_tool(portable("beta"));
    let mut retrieval_only = ToolSet::default();
    retrieval_only.add_portable_dynamic_tool(portable("gamma"));
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

/// `execute_owned` matches `execute` (result and published context) and
/// its future is `Send + 'static` — spawnable on any executor.
#[tokio::test]
async fn execute_owned_matches_execute_and_is_static() {
    fn assert_send_static<T: Send + 'static>(value: T) -> T {
        value
    }

    let mut set = ToolSet::default();
    set.add_portable_dynamic_tool(portable("alpha"));
    let catalog = set.catalog();

    let mut context = ToolContext::new();
    let borrowed = catalog.execute("alpha", "{}", &mut context).await;

    let (owned, owned_context) = assert_send_static(catalog.clone().execute_owned(
        "alpha".to_string(),
        "{}".to_string(),
        ToolContext::new(),
    ))
    .await;
    assert_eq!(owned.output().as_text(), borrowed.output().as_text());
    assert_eq!(
        format!("{owned_context:?}"),
        format!("{context:?}"),
        "owned dispatch publishes the same context metadata"
    );
}

#[tokio::test]
async fn retain_names_narrows_definitions_and_dispatch() {
    let mut set = ToolSet::default();
    set.add_portable_dynamic_tool(portable("alpha"));
    set.add_portable_dynamic_tool(portable("beta"));
    let mut catalog = set.catalog();
    catalog.retain_names(&BTreeSet::from(["beta".to_string()]));
    assert_eq!(catalog.names().collect::<Vec<_>>(), ["beta"]);
    assert_eq!(catalog.take_definitions().len(), 1);
    assert!(catalog.definitions().is_empty());
    assert!(
        !catalog
            .execute("alpha", "{}", &mut ToolContext::new())
            .await
            .is_success()
    );
}

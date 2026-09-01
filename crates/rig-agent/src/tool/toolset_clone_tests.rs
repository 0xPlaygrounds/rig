use std::sync::Arc;

use super::{RegisteredTool, ToolSet};
use crate::test_utils::{MockAddTool, MockSubtractTool};

fn erased_ptr(set: &ToolSet, name: &str) -> *const () {
    match &set.get(name).expect("registered").clone() {
        RegisteredTool::Static(tool) => Arc::as_ptr(tool).cast(),
        RegisteredTool::Embedding(tool) => Arc::as_ptr(tool).cast(),
    }
}

/// A clone shares the tool implementations (pointer-equal `Arc`s) and is
/// independent for subsequent registration changes.
#[test]
fn clone_shares_implementations_and_diverges_on_mutation() {
    let mut original = ToolSet::default();
    original.add_tool(MockAddTool);

    let mut clone = original.clone();
    assert_eq!(erased_ptr(&original, "add"), erased_ptr(&clone, "add"));
    assert_eq!(original.tool_definitions(), clone.tool_definitions());

    clone.add_tool(MockSubtractTool);
    assert!(clone.contains("subtract"));
    assert!(!original.contains("subtract"));

    original.delete_tool("add");
    assert!(clone.contains("add"));
}

use super::{RegisteredTool, ToolSet};
use crate::test_utils::{MockAddTool, MockSubtractTool};

fn registered(set: &ToolSet, name: &str) -> RegisteredTool {
    set.get(name).expect("registered").clone()
}

/// A clone shares the tool implementations (pointer-equal erased handlers)
/// and is independent for subsequent registration changes.
#[test]
fn clone_shares_implementations_and_diverges_on_mutation() {
    let mut original = ToolSet::default();
    original.add_tool(MockAddTool);

    let mut clone = original.clone();
    assert!(
        registered(&original, "add")
            .handler()
            .ptr_eq(registered(&clone, "add").handler())
    );
    assert_eq!(original.tool_definitions(), clone.tool_definitions());

    clone.add_tool(MockSubtractTool);
    assert!(clone.contains("subtract"));
    assert!(!original.contains("subtract"));

    original.delete_tool("add");
    assert!(clone.contains("add"));
}

#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

//! Test that `#[rig_tool]` propagates the function's visibility to the generated
//! structs and static. A `pub` function should produce a `pub` tool struct that
//! is accessible from outside the defining module.

mod tools {
    use rig_derive::rig_tool;

    #[rig_tool(
        description = "A public tool for testing visibility",
        params(x = "A number")
    )]
    pub async fn public_adder(x: i32) -> Result<i32, rig::tool::ToolError> {
        Ok(x + 1)
    }

    // Private function — struct should NOT be accessible outside this module.
    #[allow(dead_code)]
    #[rig_tool(
        description = "A private tool for testing visibility",
        params(x = "A number")
    )]
    async fn private_adder(x: i32) -> Result<i32, rig::tool::ToolError> {
        Ok(x + 1)
    }

    /// Verify that a private tool is accessible within its defining module.
    pub async fn use_private_tool() -> i32 {
        use rig::tool::Tool;
        let tool = PrivateAdder;
        tool.call(PrivateAdderParameters { x: 99 }).await.unwrap()
    }
}

#[tokio::test]
async fn test_pub_tool_accessible_from_outside_module() {
    use rig::tool::Tool;

    // PublicAdder and its parameters struct are accessible outside the `tools` module.
    let tool = tools::PublicAdder;
    let def = tool.definition(String::default()).await;
    assert_eq!(def.name, "public_adder");
    assert_eq!(def.description, "A public tool for testing visibility");

    let result = tool
        .call(tools::PublicAdderParameters { x: 41 })
        .await
        .unwrap();
    assert_eq!(result, serde_json::json!(42));
}

#[test]
fn test_pub_static_accessible_from_outside_module() {
    // The generated static should also be pub.
    let _ = tools::PUBLIC_ADDER;
}

#[tokio::test]
async fn test_private_tool_accessible_within_module() {
    // PrivateAdder is only accessible inside `tools`, but we can call through a pub wrapper.
    assert_eq!(tools::use_private_tool().await, 100);
}

use rig::tool::Tool;
use rig_derive::rig_tool;

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub enum SortOrder {
    #[serde(rename = "asc")]
    Ascending,
    #[serde(rename = "desc")]
    Descending,
}

/// List items with sorting and filtering
#[rig_tool]
fn list_items(
    /// Field to sort by
    sort_by: String,
    /// Sort direction
    order: SortOrder,
    /// Filter tags
    tags: Vec<String>,
    /// Maximum number of results
    limit: Option<i32>,
) -> Result<Vec<String>, rig::tool::ToolError> {
    let direction = match order {
        SortOrder::Ascending => "ascending",
        SortOrder::Descending => "descending",
    };
    Ok(vec![format!(
        "sorted by {sort_by} {direction}, tags={tags:?}, limit={limit:?}"
    )])
}

#[tokio::main]
async fn main() {
    let def = ListItems.definition(String::default()).await;
    println!(
        "Tool definition:\n{}",
        serde_json::to_string_pretty(&def).unwrap()
    );
}

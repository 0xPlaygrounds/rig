//! The system prompt, put together for each model call from three parts:
//! the agent's own [`SystemPrompt`](super::agent::SystemPrompt), the
//! [`ToolRules`] of the tools it is offered, and the app's
//! [`PromptSection`]s, such as the project's instructions and the
//! environment. Nothing in it depends on the turn, so it stays the same
//! from call to call and the provider's prompt cache keeps it; a section
//! that changes rarely sorts before one that changes more often.

use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;

/// A part of every agent's system prompt, on an entity of its own. A plugin
/// spawns one and changes its `text` when what it describes changes (with
/// `set_if_neq`, so an unchanged section is not marked changed); an empty
/// `text` leaves the section out. Sections are sent in `order`, then
/// `tag`, each between `<tag>` and `</tag>`.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Clone, Debug, Default, PartialEq)]
pub struct PromptSection {
    /// Where the section goes: lower first. See the `ORDER_*` constants.
    pub order: i32,
    /// The tag around the text, such as `project_instructions`.
    pub tag: String,
    /// The text.
    pub text: String,
}

impl PromptSection {
    /// The order of instructions that hold for every project, such as a
    /// plugin's own rules.
    pub const ORDER_RULES: i32 = 100;
    /// The order of the project's instructions (`AGENTS.md`).
    pub const ORDER_PROJECT: i32 = 200;
    /// The order of facts about the machine and the day, which change most
    /// often and so come last.
    pub const ORDER_ENVIRONMENT: i32 = 300;

    /// A section of `text` between `<tag>` and `</tag>`, at `order`.
    pub fn new(order: i32, tag: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            order,
            tag: tag.into(),
            text: text.into(),
        }
    }
}

/// How a tool should be used, on the tool's entity: lines added to the
/// system prompt of every agent the tool is offered to. Registered with
/// [`AppToolsExt::add_tool_with`](super::tools::AppToolsExt::add_tool_with).
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Clone, Debug, Default)]
pub struct ToolRules(pub Vec<String>);

/// The system prompt of an agent whose own prompt is `role`, offered tools
/// with `rules` (in the order the tools are sent), with the app's
/// `sections`. A rule two tools share is said once.
pub(crate) fn system_prompt<'a>(
    role: &str,
    rules: impl IntoIterator<Item = &'a ToolRules>,
    sections: impl IntoIterator<Item = &'a PromptSection>,
) -> String {
    let mut lines: Vec<&str> = Vec::new();
    for rule in rules.into_iter().flat_map(|rules| rules.0.iter()) {
        let rule = rule.trim();
        if !rule.is_empty() && !lines.contains(&rule) {
            lines.push(rule);
        }
    }
    let mut sections: Vec<&PromptSection> = sections
        .into_iter()
        .filter(|section| !section.text.trim().is_empty())
        .collect();
    sections.sort_by(|a, b| a.order.cmp(&b.order).then_with(|| a.tag.cmp(&b.tag)));

    let mut prompt = role.trim().to_owned();
    if !lines.is_empty() {
        let rules: Vec<String> = lines.iter().map(|line| format!("- {line}")).collect();
        push_tagged(&mut prompt, "tool_rules", &rules.join("\n"));
    }
    for section in sections {
        push_tagged(&mut prompt, &section.tag, section.text.trim());
    }
    prompt
}

/// Appends `text` between `<tag>` and `</tag>`, after a blank line.
fn push_tagged(prompt: &mut String, tag: &str, text: &str) {
    if !prompt.is_empty() {
        prompt.push_str("\n\n");
    }
    prompt.push_str(&format!("<{tag}>\n{text}\n</{tag}>"));
}

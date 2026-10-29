//! Deferred tools with tool search (corpus spec 24).
//!
//! Tools named by the `[tool_search]` settings (name globs, or every tool a
//! plugin crate added, such as an MCP client's) are left out of each
//! request until the agent activates them: `tool_search {query}` names the
//! five deferred tools that match its keywords best and activates them for
//! the calling agent, which is offered them from its next request on. A
//! deferred tool called before it was activated still runs; its result
//! says it is now active. What each agent activated is saved with the
//! session; a new agent, a subagent too, starts with none. `/tools` counts
//! the active and deferred tools, and the system prompt says how many are
//! deferred.

use std::cmp::Reverse;

use bevy_ecs::system::SystemParam;
use glob::Pattern;
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::{Section, ToolDefinition};
use rig_harness::rig_core::effect::Answered;
use rig_harness::rig_core::message::ToolResultContent;
use rig_harness::rig_core::tool::ToolOutput;
use schemars::JsonSchema;
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// The search tool's name.
pub const TOOL_SEARCH: &str = "tool_search";

/// How many tools a search names at most.
const TOP: usize = 5;

/// Which tools are deferred, offered only once an agent activated them:
/// the `[tool_search]` table of the settings, such as
/// `tools = ["jira_*"]` and `plugins = ["rig-mcp-client"]`.
#[derive(Resource, Deserialize, Clone, Debug, Default)]
#[serde(default, deny_unknown_fields)]
pub struct DeferredTools {
    /// Name globs, such as `jira_*` or `*`; an invalid one matches nothing.
    pub tools: Vec<String>,
    /// Plugin crates every tool of which is deferred, such as an MCP
    /// client's: its tools' [`ProvidedBy`] names it.
    pub plugins: Vec<String>,
}

/// The deferred tools the agent activated, by name. Saved with the
/// session.
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Saved, Default)]
pub struct ActivatedTools(pub Vec<String>);

impl ActivatedTools {
    fn contains(&self, name: &str) -> bool {
        self.0.iter().any(|activated| activated == name)
    }
}

/// The plugin's stage systems, for others to order against: in Prepare,
/// the one that takes the deferred tools out of a request (after the
/// default prompt went in, before compaction measures it); in Review, the one that notes a guessed call's
/// activation at the end of its result.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ToolSearchSystems {
    Offer,
    Activate,
}

#[derive(Default)]
pub struct ToolSearchPlugin;

impl Plugin for ToolSearchPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<DeferredTools>("tool_search")
            .register_required_components::<Agent, ActivatedTools>()
            .add_open_tool(
                TOOL_SEARCH,
                "Find tools that are not offered yet by keywords over their names and \
                 descriptions, and activate the best matches: they are offered from your next \
                 step on. Some tools are deferred: not listed until found. When no listed tool \
                 fits, call it with a few keywords before giving up.",
                ToolOptions {
                    footprint: Some(Touches::Reads(Scope::Anything)),
                },
                search,
            )
            .add_command("tools", "Count the active and deferred tools", tools)
            .configure_sets(
                AgentPass,
                (
                    ToolSearchSystems::Offer.in_set(PrepareSystems::Edit),
                    ToolSearchSystems::Activate.in_set(Stage::Review),
                ),
            )
            .add_systems(
                AgentPass,
                (
                    offer.in_set(ToolSearchSystems::Offer),
                    activate_on_call.in_set(ToolSearchSystems::Activate),
                ),
            );
    }
}

/// The arguments of `tool_search`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Search {
    /// Keywords, such as `jira issue`.
    query: String,
}

/// The registered tools, and which are deferred.
#[derive(SystemParam)]
struct Registry<'w, 's> {
    config: Res<'w, DeferredTools>,
    tools: Query<'w, 's, (&'static ToolDef, Option<&'static ProvidedBy>)>,
    plugins: Query<'w, 's, &'static PluginSource>,
}

impl Registry<'_, '_> {
    /// Whether the tool `def`, added by the plugin `by`, is deferred.
    fn deferred_def(&self, def: &ToolDefinition, by: Option<&ProvidedBy>) -> bool {
        let name = def.name.as_str();
        if name == TOOL_SEARCH {
            return false;
        }
        let from_plugin = by
            .and_then(|by| self.plugins.get(by.0).ok())
            .is_some_and(|source| self.config.plugins.contains(&source.package));
        let glob = |glob: &String| Pattern::new(glob).is_ok_and(|glob| glob.matches(name));
        from_plugin || self.config.tools.iter().any(glob)
    }

    /// The deferred tools `access` lets the agent call.
    fn deferred<'a>(&'a self, access: &'a ToolAccess) -> impl Iterator<Item = &'a ToolDefinition> {
        self.tools
            .iter()
            .filter(|(def, by)| {
                access.allows(def.0.name.as_str()) && self.deferred_def(&def.0, *by)
            })
            .map(|(def, _)| &*def.0)
    }

    /// Whether `name` is a registered deferred tool.
    fn is_deferred(&self, name: &str) -> bool {
        self.tools
            .iter()
            .find(|(def, _)| def.0.name.as_str() == name)
            .is_some_and(|(def, by)| self.deferred_def(&def.0, by))
    }
}

/// Leaves the deferred tools the agent has not activated out of each turn's
/// request, and says in the system prompt how many tools are deferred.
fn offer(
    mut calls: Query<
        (&mut ModelRequest, &ForAgent),
        (
            Added<Admitted>,
            ModelCalls,
            Or<(With<TurnCall>, With<SideCall>)>,
        ),
    >,
    registry: Registry,
    agents: Query<(&ToolAccess, &ActivatedTools)>,
) {
    for (mut call, &ForAgent(agent)) in &mut calls {
        let Ok((access, activated)) = agents.get(agent) else {
            continue;
        };
        call.request_mut().tools.retain(|tool| {
            let name = tool.name.as_str();
            activated.contains(name) || !registry.is_deferred(name)
        });
        let deferred = registry.deferred(access).count();
        let searchable = call
            .request
            .tools
            .iter()
            .any(|tool| tool.name == TOOL_SEARCH);
        if deferred > 0
            && searchable
            && let Some(prompt) = &mut call.prompt
        {
            // The total, not what is left, so the prompt stays cached.
            prompt.sections.push(Section::new(
                "tool_search",
                format!(
                    "{deferred} more tools are deferred: `tool_search` finds and activates them."
                ),
            ));
        }
    }
}

/// `tool_search`: names the deferred tools that match the query best and
/// activates them for the calling agent.
fn search(
    called: On<ToolCalled<Search>>,
    registry: Registry,
    mut agents: Query<(&ToolAccess, &mut ActivatedTools)>,
    mut decisions: Query<&mut Decision, ToolCalls>,
) {
    let Ok((access, mut activated)) = agents.get_mut(called.agent) else {
        return;
    };
    let terms: Vec<String> = words(&called.args.query).collect();
    let mut found: Vec<(usize, &ToolDefinition)> = registry
        .deferred(access)
        .map(|def| (score(def, &terms), def))
        .filter(|(score, _)| *score > 0)
        .collect();
    found.sort_by_key(|(score, def)| (Reverse(*score), def.name.as_str()));
    found.truncate(TOP);
    let text = if found.is_empty() {
        format!("No deferred tool matches `{}`.", called.args.query)
    } else {
        let mut lines: Vec<String> = found
            .iter()
            .map(|(_, def)| {
                let summary = def.description.lines().next().unwrap_or_default();
                format!("- {}: {summary}", def.name.as_str())
            })
            .collect();
        lines.push("These tools are now active: call them directly.".to_owned());
        lines.join("\n")
    };
    for (_, def) in &found {
        let name = def.name.as_str();
        // Only a new name changes what is saved.
        if !activated.contains(name) {
            activated.0.push(name.to_owned());
        }
    }
    if let Ok(mut decision) = decisions.get_mut(called.call) {
        decision.decide(
            Decider::plugin("tool-search"),
            ToolResult::success(ToolOutput::text(text)),
        );
    }
}

/// The lowercase words of `text`.
fn words(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
}

/// How well `def` matches the query `terms`: a name word counts most, then
/// a part of the name, then a word of the description.
fn score(def: &ToolDefinition, terms: &[String]) -> usize {
    let name = def.name.as_str().to_lowercase();
    let name_words: Vec<String> = words(&name).collect();
    let description: Vec<String> = words(&def.description).collect();
    terms
        .iter()
        .map(|term| {
            if name_words.contains(term) {
                3
            } else if name.contains(term.as_str()) {
                2
            } else if description.contains(term) {
                1
            } else {
                0
            }
        })
        .sum()
}

/// A deferred tool the model called before activating it runs anyway (a
/// tool left out of a request still runs if the agent's `ToolAccess`
/// allows it): before its result is committed, this activates the tool and
/// says so at the end of the result.
fn activate_on_call(
    mut calls: Query<(&ToolRequest, &ForAgent, &mut Outcome), (ReadyToSettle, ToolCalls)>,
    registry: Registry,
    mut agents: Query<(&ToolAccess, &mut ActivatedTools)>,
) {
    // Once per call: the tool is active once noted, which this checks.
    for (request, &ForAgent(agent), mut outcome) in &mut calls {
        let name = request.call.function.name.as_str();
        let Ok(Answered::ToolResult { result }) = &outcome.0 else {
            continue;
        };
        let Ok((access, mut activated)) = agents.get_mut(agent) else {
            continue;
        };
        if !access.allows(name) || activated.contains(name) || !registry.is_deferred(name) {
            continue;
        }
        activated.0.push(name.to_owned());
        let mut content = result.output().as_content().to_vec();
        content.push(ToolResultContent::text(format!(
            "(`{name}` was deferred and not offered yet; it ran and is now active.)"
        )));
        if let Ok(output) = ToolOutput::content(content) {
            *outcome = Outcome::from(result.clone().with_output(output));
        }
    }
}

/// `/tools`: how many tools the agent is offered, how many are deferred,
/// and which it activated.
fn tools(
    In(args): In<CommandArgs>,
    registry: Registry,
    agents: Query<(&ToolAccess, &ActivatedTools)>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((access, activated)) = agents.get(args.agent) else {
        return;
    };
    let usable = registry
        .tools
        .iter()
        .filter(|(def, _)| access.allows(def.0.name.as_str()))
        .count();
    let deferred = registry
        .deferred(access)
        .filter(|def| !activated.contains(def.name.as_str()))
        .count();
    let active = usable.saturating_sub(deferred);
    let mut text = format!("Tools: {active} active, {deferred} deferred.");
    if !activated.0.is_empty() {
        text.push_str(&format!(" Activated: {}.", activated.0.join(", ")));
    }
    notices.write(Notice::info(args.agent, text));
}

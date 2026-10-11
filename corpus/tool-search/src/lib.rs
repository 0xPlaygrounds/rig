//! Deferred tools with tool search (corpus spec 24).
//!
//! Tools named by [`DeferredTools`] (name globs, or every tool a plugin
//! crate added, such as an MCP client's) are left out of each request until
//! the agent activates them: `tool_search {query}` names the five deferred
//! tools that match its keywords best and activates them for the calling
//! agent, which is offered them from its next request on. A deferred tool
//! called before it was activated still runs, activates it, and the agent
//! is told so. What each agent activated is saved with the session; a new
//! agent, a subagent too, starts with none. `/tools` counts the active and
//! deferred tools, and the system prompt says how many are deferred.

use std::cmp::Reverse;

use bevy_ecs::system::SystemParam;
use glob::Pattern;
use rig_core::completion::ToolDefinition;
use rig_harness::harness_protocol::Home;
use rig_harness::prelude::*;
use schemars::JsonSchema;
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// The search tool's name.
pub const TOOL_SEARCH: &str = "tool_search";

/// How many tools a search names at most.
const TOP: usize = 5;

/// Which tools are deferred: offered only once an agent activated them.
/// Inserted before the plugin is added, it is kept; otherwise it is read
/// from `RIG_HOME/tool-search.json`, such as
/// `{"tools": ["jira_*"], "plugins": ["rig-mcp"]}`.
#[derive(Resource, Clone, Debug, Default)]
pub struct DeferredTools {
    /// Name globs, such as `jira_*` or `*`.
    pub names: Vec<Pattern>,
    /// Plugin crates every tool of which is deferred, such as an MCP
    /// client's: its tools' [`ProvidedBy`] names it.
    pub plugins: Vec<String>,
}

/// `tool-search.json` as written.
#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct ConfigFile {
    tools: Vec<String>,
    plugins: Vec<String>,
}

impl DeferredTools {
    /// The tools whose names match one of `globs`; an invalid glob is
    /// skipped.
    pub fn matching<'a>(globs: impl IntoIterator<Item = &'a str>) -> Self {
        Self {
            names: globs
                .into_iter()
                .filter_map(|glob| Pattern::new(glob).ok())
                .collect(),
            plugins: Vec::new(),
        }
    }

    /// `RIG_HOME/tool-search.json`, or nothing deferred.
    fn from_home() -> Self {
        let path = Home::from_env().root().join("tool-search.json");
        let file: ConfigFile = std::fs::read_to_string(path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default();
        Self {
            plugins: file.plugins,
            ..Self::matching(file.tools.iter().map(String::as_str))
        }
    }
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

    /// Activates `name`, unless it is already.
    fn activate(&mut self, name: &str) {
        if !self.contains(name) {
            self.0.push(name.to_owned());
        }
    }
}

#[derive(Default)]
pub struct ToolSearchPlugin;

impl Plugin for ToolSearchPlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<DeferredTools>() {
            app.insert_resource(DeferredTools::from_home());
        }
        app.register_required_components::<Agent, ActivatedTools>()
            .add_open_tool(
                TOOL_SEARCH,
                "Find tools that are not offered yet by keywords over their names and \
                 descriptions, and activate the best matches: they are offered from your next \
                 step on.",
                ToolOptions {
                    rules: &[
                        "Some tools are deferred: not listed until found. When no listed tool \
                         fits, call `tool_search` with a few keywords before giving up.",
                    ],
                    footprint: Some(Footprint::ReadOnly),
                },
                search,
            )
            .add_command("tools", "Count the active and deferred tools", tools)
            .add_observer(offer)
            .add_observer(activate_on_call);
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
            .is_some_and(|source| self.config.plugins.contains(&source.krate));
        from_plugin || self.config.names.iter().any(|glob| glob.matches(name))
    }

    /// Every deferred tool.
    fn deferred(&self) -> impl Iterator<Item = &ToolDefinition> {
        self.tools
            .iter()
            .filter(|(def, by)| self.deferred_def(&def.0, *by))
            .map(|(def, _)| &def.0)
    }

    /// Whether `name` is a registered deferred tool.
    fn is_deferred(&self, name: &str) -> bool {
        self.tools
            .iter()
            .find(|(def, _)| def.0.name.as_str() == name)
            .is_some_and(|(def, by)| self.deferred_def(&def.0, by))
    }
}

/// Leaves the deferred tools the agent has not activated out of each
/// request, and says in its system prompt how many tools are deferred.
fn offer(
    mut prepare: On<PrepareRequest>,
    registry: Registry,
    agents: Query<(&ToolAccess, &ActivatedTools)>,
) {
    let Ok((access, activated)) = agents.get(prepare.agent) else {
        return;
    };
    let prepare = prepare.event_mut();
    prepare.tools.retain(|tool| {
        let name = tool.name.as_str();
        activated.contains(name) || !registry.is_deferred(name)
    });
    let deferred = registry
        .deferred()
        .filter(|def| access.allows(def.name.as_str()))
        .count();
    let searchable = prepare.tools.iter().any(|tool| tool.name == TOOL_SEARCH);
    if deferred > 0 && searchable {
        // The total, not what is left, so the prompt stays cached.
        prepare.preamble.push_str(&format!(
            "\n\n{deferred} more tools are deferred: `tool_search` finds and activates them."
        ));
    }
}

/// `tool_search`: names the deferred tools that match the query best and
/// activates them for the calling agent.
fn search(
    called: On<ToolCalled<Search>>,
    registry: Registry,
    mut agents: Query<(&ToolAccess, &mut ActivatedTools)>,
    mut commands: Commands,
) {
    let terms: Vec<String> = words(&called.args.query).collect();
    let Ok((access, mut activated)) = agents.get_mut(called.agent) else {
        return;
    };
    let mut found: Vec<(usize, &ToolDefinition)> = registry
        .deferred()
        .filter(|def| access.allows(def.name.as_str()))
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
    // Only a new name changes what is saved.
    for (_, def) in &found {
        if !activated.contains(def.name.as_str()) {
            activated.activate(def.name.as_str());
        }
    }
    commands
        .entity(called.call)
        .insert(ToolOutput(ToolResult::success(text.into())));
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

/// A deferred tool the model called before activating it runs anyway (the
/// kernel runs any tool the agent's `ToolAccess` allows); this activates
/// it and tells the agent with its next request.
fn activate_on_call(
    call: On<Add<CallOf>>,
    calls: Query<(&ToolCallRun, &CallOf)>,
    turns: Query<&TurnOf>,
    registry: Registry,
    mut agents: Query<(&ToolAccess, &mut ActivatedTools)>,
    mut commands: Commands,
) {
    let Ok((run, CallOf(turn))) = calls.get(call.entity) else {
        return;
    };
    let Ok(&TurnOf(agent)) = turns.get(*turn) else {
        return;
    };
    let name = run.call.function.name.as_str();
    let Ok((access, mut activated)) = agents.get_mut(agent) else {
        return;
    };
    if !access.allows(name) || activated.contains(name) || !registry.is_deferred(name) {
        return;
    }
    activated.activate(name);
    let note = format!(
        "(`{name}` was deferred and not offered yet; the call ran and `{name}` is now active.)"
    );
    commands.trigger(
        Deliver::new(agent, note, DeliveryMode::Note).with_origin(Origin::plugin("tool-search")),
    );
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
    let (mut active, mut deferred) = (0, 0);
    for (def, by) in &registry.tools {
        let name = def.0.name.as_str();
        if !access.allows(name) {
            continue;
        }
        if registry.deferred_def(&def.0, by) && !activated.contains(name) {
            deferred += 1;
        } else {
            active += 1;
        }
    }
    let mut text = format!("Tools: {active} active, {deferred} deferred.");
    if !activated.0.is_empty() {
        text.push_str(&format!(" Activated: {}.", activated.0.join(", ")));
    }
    notices.write(Notice::info(args.agent, text));
}

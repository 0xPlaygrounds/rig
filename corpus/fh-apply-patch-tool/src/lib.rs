//! The `apply_patch` tool for models trained on the multi-file
//! `*** Begin Patch` format, offered in place of `edit` and `write` to the
//! models `RIG_HOME/patch.toml` names.
//!
//! Each turn's request is fitted to the model it goes to, in
//! [`PrepareSystems::Edit`]: a model whose id matches a pattern of
//! [`PatchModels`] is offered `apply_patch` and not `edit` or `write`, any
//! other the opposite, so switching model switches the tools from the next
//! request on. A call of `apply_patch` is checked whole first, against the
//! files under its agent's `WorkDir` (else the project root), which no path
//! may leave: a hunk whose context is not found, a file to update or delete
//! that is missing, or a file to add that exists, and nothing changes. A
//! patch that applies is carried out as one `write` call per file it adds,
//! changes or moves, so every other plugin's rule on `write` (approvals,
//! protected paths, formatting) applies to each file; if any of them is
//! refused or fails, the files written are put back, and the deletes run
//! only once every write succeeded.

use rig_core::effect::EffectId;
use rig_core::message::{ToolFunction, ToolName};
use rig_core::tool::ToolOutput;
use rig_harness::prelude::*;
use schemars::JsonSchema;
use serde::Deserialize;

mod patch;

use patch::{FileWrite, Plan, check};

/// The tool's name.
pub const APPLY_PATCH: &str = "apply_patch";

/// The tools a patch model is not offered.
const EDITING: [&str; 2] = ["edit", "write"];

/// `RIG_HOME/patch.toml`: `models = ["gpt-5*", "codex-*"]`, glob patterns
/// matched against a model's id (`gpt-5-codex`) and its reference
/// (`openai/gpt-5-codex`). The patterns above when the file is missing.
#[derive(Resource, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct PatchModels {
    /// The patterns of the models offered `apply_patch`.
    pub models: Vec<String>,
}

impl Default for PatchModels {
    fn default() -> Self {
        Self {
            models: vec!["gpt-5*".to_owned(), "codex-*".to_owned()],
        }
    }
}

impl PatchModels {
    /// Whether the model of catalog id `id` and reference `reference` is
    /// offered `apply_patch`.
    pub fn takes_patches(&self, id: &str, reference: &str) -> bool {
        self.models.iter().any(|pattern| {
            glob::Pattern::new(pattern)
                .is_ok_and(|pattern| pattern.matches(id) || pattern.matches(reference))
        })
    }
}

/// The plugin's Prepare system, which edits the tools of each turn's
/// request, for other plugins that edit them or route the request to
/// order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ApplyPatchSystems;

/// Adds `apply_patch` and offers it, or `edit` and `write`, by model.
#[derive(Default)]
pub struct ApplyPatchPlugin;

impl Plugin for ApplyPatchPlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<PatchModels>() {
            let file = RigHome::of(app.world()).root().join("patch.toml");
            let models = std::fs::read_to_string(&file)
                .ok()
                .and_then(|text| toml::from_str(&text).ok())
                .unwrap_or_default();
            app.insert_resource::<PatchModels>(models);
        }
        app.init_resource::<ProjectRoot>()
            .configure_sets(AgentPass, ApplyPatchSystems.in_set(PrepareSystems::Edit))
            .add_systems(AgentPass, offer_by_model.in_set(ApplyPatchSystems))
            .add_open_tool(
                APPLY_PATCH,
                DESCRIPTION,
                ToolOptions {
                    rules: &[
                        "Use `apply_patch` for every file change: add, update, move and delete \
                         files in one patch.",
                    ],
                    footprint: None,
                },
                start_patch,
            )
            .add_observer(checked)
            .add_observer(part_settled)
            .add_observer(follow_successor::<PatchPart>);
    }
}

/// What the model reads about the tool.
const DESCRIPTION: &str = "Apply a multi-file patch. The input is `*** Begin Patch`, then any \
number of `*** Add File: <path>` (every line `+`), `*** Delete File: <path>`, or `*** Update \
File: <path>` (optionally followed by `*** Move to: <new path>`) with hunks: `@@` or `@@ <text \
of a line before the hunk>`, then lines starting with ` ` (context), `-` (removed) or `+` \
(added), then `*** End Patch`. Hunks are found by their context lines, not line numbers. The \
whole patch applies or nothing changes.";

/// The arguments of `apply_patch`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PatchArgs {
    /// The whole patch, from `*** Begin Patch` to `*** End Patch`.
    pub input: String,
}

/// Offers each turn's request `apply_patch` or `edit` and `write`, by the
/// model it goes to: its own connection, else its agent's.
fn offer_by_model(
    mut calls: Query<(&mut ModelRequest, &ForAgent), (Added<Admitted>, With<TurnCall>, ModelCalls)>,
    agents: Query<&Connection>,
    models: Res<PatchModels>,
) {
    for (mut call, &ForAgent(agent)) in &mut calls {
        let connection = call.connection.as_ref().or_else(|| agents.get(agent).ok());
        let Some(spec) = connection.map(|connection| connection.spec.clone()) else {
            continue;
        };
        let offered = |name: &str| {
            call.request
                .tools
                .iter()
                .any(|tool| tool.name.as_str() == name)
        };
        let patches = models.takes_patches(&spec.id, &spec.reference());
        if patches && offered(APPLY_PATCH) {
            call.request
                .tools
                .retain(|tool| !EDITING.contains(&tool.name.as_str()));
        } else if !patches {
            call.request
                .tools
                .retain(|tool| tool.name.as_str() != APPLY_PATCH);
        }
    }
}

/// On the entity that checks a patch off the main thread: the call it is
/// for.
#[derive(Component)]
struct Checking {
    call: Entity,
    agent: Entity,
    effect: EffectId,
    id: String,
}

/// What checking a patch found: the plan, or why it does not apply.
type Checked = Result<Plan, String>;

/// On an `apply_patch` call carried out: the plan and what each of its
/// `write` calls answered, by index.
#[derive(Component)]
struct Patching {
    plan: Plan,
    parts: Vec<Option<ToolResult>>,
}

/// On a `write` call made for the `apply_patch` call `call`: the plan's
/// write `index`.
#[derive(Component, Clone, Copy)]
struct PatchPart {
    call: Entity,
    index: usize,
}

/// Checks each patch the model sends, off the main thread, in its agent's
/// directory.
fn start_patch(
    called: On<ToolCalled<PatchArgs>>,
    dirs: Query<&WorkDir>,
    root: Res<ProjectRoot>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let base = dirs
        .get(called.agent)
        .map_or_else(|_| root.0.clone(), |WorkDir(dir)| dir.clone());
    let input = called.args.input.clone();
    let checking = Checking {
        call: called.call,
        agent: called.agent,
        effect: called.effect,
        id: called.tool_call.id.to_string(),
    };
    let work = async move {
        blocking(move || Ok(check(&input, &base)))
            .await
            .unwrap_or_else(|error| Err(error.to_string()))
    };
    commands.spawn((checking, Running::spawn(pool(), &wake, work)));
}

/// Fails a patch that does not apply, finishes one that only deletes,
/// and runs each write of any other as a `write` call of its own.
fn checked(
    done: On<Insert<Done<Checked>>>,
    mut checks: Query<(&Checking, &mut Done<Checked>)>,
    turns: Query<&CallOf>,
    mut calls: Query<&mut Decision, ToolCalls>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let Ok((checking, mut result)) = checks.get_mut(done.entity) else {
        return;
    };
    let result = std::mem::replace(&mut result.0, Err(String::new()));
    let call = checking.call;
    commands.entity(done.entity).despawn();
    let plan = match result {
        Ok(plan) => plan,
        Err(why) => {
            if let Ok(mut decision) = calls.get_mut(call) {
                decision.decide(ToolResult::failed(ToolExecutionError::invalid_args(why)));
            }
            return;
        }
    };
    if plan.writes.is_empty() {
        finish(&mut commands, call, plan, Vec::new(), &wake);
        return;
    }
    let turn = turns.get(call).ok().map(|&CallOf(turn)| turn);
    let Ok(write) = ToolName::new("write") else {
        return;
    };
    for (index, FileWrite { path, after, .. }) in plan.writes.iter().enumerate() {
        let args = serde_json::json!({ "path": path.to_string_lossy(), "content": after });
        let request = ToolRequest {
            call: ToolCall::from_wire(
                format!("{}-{index}", checking.id),
                ToolFunction::new(write.clone(), args),
            ),
        };
        let mut part = commands.spawn((
            request,
            Effect::child_of(Some(checking.effect)),
            ForAgent(checking.agent),
            PatchPart { call, index },
        ));
        if let Some(turn) = turn {
            part.insert(CallOf(turn));
        }
    }
    let parts = vec![None; plan.writes.len()];
    commands.entity(call).insert(Patching { plan, parts });
}

/// Keeps what each `write` of a patch answered, and finishes the patch
/// once all did.
fn part_settled(
    settled: On<Add<Settled>>,
    parts: Query<(&PatchPart, &Outcome), Without<SupersededBy>>,
    mut patches: Query<&mut Patching>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let Ok((&PatchPart { call, index }, outcome)) = parts.get(settled.entity) else {
        return;
    };
    let Ok(mut patching) = patches.get_mut(call) else {
        return;
    };
    if let Some(slot) = patching.parts.get_mut(index) {
        *slot = Some(outcome.tool_result());
    }
    if patching.parts.iter().all(Option::is_some) {
        let parts = patching.parts.drain(..).flatten().collect();
        let plan = std::mem::replace(
            &mut patching.plan,
            Plan {
                writes: Vec::new(),
                deletes: Vec::new(),
                summary: Vec::new(),
            },
        );
        commands.entity(call).remove::<Patching>();
        finish(&mut commands, call, plan, parts, &wake);
    }
}

/// Answers the `apply_patch` call `call`, off the main thread: once every
/// write in `parts` succeeded, deletes what the plan deletes and lists
/// each file; else puts back each file written and says why.
fn finish(commands: &mut Commands, call: Entity, plan: Plan, parts: Vec<ToolResult>, wake: &Wake) {
    let work = async move {
        blocking(move || Ok(complete(plan, &parts)))
            .await
            .unwrap_or_else(ToolResult::failed)
    };
    commands
        .entity(call)
        .insert(Running::answering(pool(), wake, work));
}

/// The result of a patch whose writes answered `parts`.
fn complete(plan: Plan, parts: &[ToolResult]) -> ToolResult {
    let failed: Vec<String> = plan
        .writes
        .iter()
        .zip(parts)
        .filter(|(_, result)| !result.is_success())
        .map(|(write, result)| format!("- {}: {}", write.path.display(), result.output().render()))
        .collect();
    let deleted: Result<(), String> = if failed.is_empty() {
        plan.deletes.iter().try_for_each(|path| {
            std::fs::remove_file(path).map_err(|error| format!("- {}: {error}", path.display()))
        })
    } else {
        Ok(())
    };
    if failed.is_empty()
        && let Ok(()) = deleted
    {
        let mut text = plan.summary.join("\n");
        for (write, result) in plan.writes.iter().zip(parts) {
            let output = result.output().render();
            if let Some((_, notes)) = output.split_once('\n') {
                text.push_str(&format!("\n\n{}:{notes}", write.path.display()));
            }
        }
        return ToolResult::success(ToolOutput::text(text));
    }
    let written = plan.writes.iter().zip(parts);
    let put_back = written
        .filter(|(_, result)| result.is_success())
        .map(|(write, _)| restore(write));
    let not_restored: Vec<String> = put_back.filter_map(Result::err).collect();
    let mut text = "The patch was not applied; the files it wrote were put back.".to_owned();
    for line in failed.into_iter().chain(deleted.err()).chain(not_restored) {
        text.push_str(&format!("\n\n{line}"));
    }
    ToolResult::failed(ToolExecutionError::other(text))
}

/// Puts `write`'s file back as it was before the patch.
fn restore(write: &FileWrite) -> Result<(), String> {
    let restored = match &write.before {
        Some(before) => rig_tools::fs::write_atomic(&write.path, before.as_bytes()),
        None => std::fs::remove_file(&write.path),
    };
    restored.map_err(|error| format!("- {} could not be put back: {error}", write.path.display()))
}

/// The pool a patch is checked and finished on.
fn pool() -> &'static bevy_tasks::TaskPool {
    bevy_tasks::IoTaskPool::get_or_init(bevy_tasks::TaskPool::default)
}

#[cfg(test)]
mod tests;

//! The `apply_patch` tool for models trained on the multi-file
//! `*** Begin Patch` format, offered in place of `edit` and `write` to the
//! models `RIG_HOME/patch.toml` names.
//!
//! Each turn's request is fitted to the model it goes to, in
//! [`PrepareSystems::Edit`]: a model whose id matches a pattern of
//! [`PatchModels`] is offered `apply_patch` and not `edit` or `write`, any
//! other the opposite, so switching model switches the tools from the next
//! request on. A call of `apply_patch` is checked whole first, against the
//! app's workspace, its paths placed as the file tools place theirs: a
//! hunk whose context is not found, a file to update or delete that is
//! missing, or a file to add that exists, and nothing changes. A patch
//! that applies is carried out as one `write` call per file it adds,
//! changes or moves to, and one `delete` call per file it deletes or moves
//! from, so every other plugin's rule on those (approvals, protected paths,
//! formatting) applies to each file. They are gated as a group, in
//! [`GateSystems::After`]: they start together once every one is past its
//! gates, or none does. Register rig-tools' `Delete` beside it
//! ([`AppPerCallExt::add_tool_per_call`]) for patches that delete or move
//! files.

use crate::early_tool_start::EarlySystems;
use corpus_shared::permission_modes::PermissionSystems;
use rig_core::effect::EffectId;
use rig_core::message::{CallId, LocalCallId, ToolFunction, ToolName};
use rig_core::tool::ToolOutput;
use rig_harness::prelude::*;
use schemars::JsonSchema;
use serde::Deserialize;

mod patch;

use patch::{Plan, check};

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

/// The plugin's stage systems, for other plugins to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ApplyPatchSystems {
    /// Edits the tools of each turn's request, by the model it goes to, in
    /// [`PrepareSystems::Edit`], after permission modes took theirs out:
    /// order a router that picks the model before it.
    Offer,
    /// Holds, releases or refuses a patch's calls together, in
    /// [`GateSystems::After`], before early tool starts answer the calls
    /// let through.
    Group,
}

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
            .init_resource::<ProjectWorkspace>()
            .configure_sets(
                AgentPass,
                (
                    ApplyPatchSystems::Offer
                        .in_set(PrepareSystems::Edit)
                        .after(PermissionSystems::Offer),
                    ApplyPatchSystems::Group
                        .in_set(GateSystems::After)
                        .before(EarlySystems),
                ),
            )
            .add_systems(
                AgentPass,
                (
                    offer_by_model.in_set(ApplyPatchSystems::Offer),
                    gate_together.in_set(ApplyPatchSystems::Group),
                ),
            )
            .add_open_tool(
                APPLY_PATCH,
                DESCRIPTION,
                ToolOptions::default(),
                start_patch,
            )
            .add_observer(checked)
            .add_observer(part_settled)
            .add_observer(follow_successor::<PatchPart>);
    }
}

/// What the model reads about the tool.
const DESCRIPTION: &str = "Apply a multi-file patch; use it for every file change. The input is `*** Begin Patch`, then any \
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
            call.request_mut()
                .tools
                .retain(|tool| !EDITING.contains(&tool.name.as_str()));
        } else if !patches {
            call.request_mut()
                .tools
                .retain(|tool| tool.name.as_str() != APPLY_PATCH);
        }
    }
}

/// Who decides for this plugin.
const NAME: &str = "fh-apply-patch-tool";

/// On the entity that checks a patch off the main thread: the call it is
/// for.
#[derive(Component)]
struct Checking {
    call: Entity,
    agent: Entity,
    effect: EffectId,
}

/// What checking a patch found: the plan, or why it does not apply.
type Checked = Result<Plan, String>;

/// On an `apply_patch` call carried out: its member calls, one `write` per
/// file written and one `delete` per file deleted, gated together, and
/// what each answered.
#[derive(Component)]
struct Patching {
    summary: Vec<String>,
    /// Each member's call and what it does, as `write a.txt`.
    members: Vec<(Entity, String)>,
    /// This plugin's hold on each member, while the others are at their
    /// gates.
    holds: Vec<Option<Entity>>,
    /// Whether every member is past its gates: released together, or
    /// refused together.
    gated: bool,
    /// Whether a gate refused a member, so none ran.
    refused: bool,
    /// The members refused here, as another was refused.
    withheld: Vec<bool>,
    results: Vec<Option<ToolResult>>,
}

/// On a member call of the `apply_patch` call `call`: its `index`.
#[derive(Component, Clone, Copy)]
struct PatchPart {
    call: Entity,
    index: usize,
}

/// Checks each patch the model sends, off the main thread, in its agent's
/// place in the workspace: its `WorkDir`, which no path may leave, else the
/// project root, as the file tools place their paths.
fn start_patch(
    called: On<ToolCalled<PatchArgs>>,
    dirs: Query<&WorkDir>,
    (root, workspace): (Res<ProjectRoot>, Res<ProjectWorkspace>),
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let dir = dirs.get(called.agent).ok();
    let (base, confine) = dir.map_or_else(|| (root.0.clone(), false), |dir| (dir.0.clone(), true));
    let (input, workspace) = (called.args.input.clone(), workspace.0.clone());
    let checking = Checking {
        call: called.call,
        agent: called.agent,
        effect: called.effect,
    };
    let work = async move {
        blocking(move || Ok(check(&input, workspace.as_ref(), &base, confine)))
            .await
            .unwrap_or_else(|error| Err(error.to_string()))
    };
    commands.spawn((checking, Running::spawn(pool(), &wake, work)));
}

/// Fails a patch that does not apply, and spawns a member call per file of
/// one that does, for the gates to judge each as its own.
fn checked(
    done: On<Insert<Done<Checked>>>,
    mut checks: Query<(&Checking, &mut Done<Checked>)>,
    turns: Query<&CallOf>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
) {
    let Ok((checking, mut result)) = checks.get_mut(done.entity) else {
        return;
    };
    let result = std::mem::replace(&mut result.0, Err(String::new()));
    let (call, agent) = (checking.call, checking.agent);
    commands.entity(done.entity).despawn();
    let Ok(mut decision) = calls.get_mut(call) else {
        return;
    };
    let plan = match result {
        Ok(plan) => plan,
        Err(why) => {
            let error = ToolExecutionError::invalid_args(why);
            decision.decide(Decider::plugin(NAME), ToolResult::failed(error));
            return;
        }
    };
    let writes = plan.writes.into_iter().map(|(path, content)| {
        let args = serde_json::json!({ "path": path, "content": content });
        (format!("write {path}"), "write", args)
    });
    let deletes = plan.deletes.into_iter().map(|path| {
        let args = serde_json::json!({ "path": path });
        (format!("delete {path}"), "delete", args)
    });
    let turn = turns.get(call).ok().map(|&CallOf(turn)| turn);
    let mut members = Vec::new();
    for (index, (label, tool, args)) in writes.chain(deletes).enumerate() {
        let Ok(name) = ToolName::new(tool) else {
            continue;
        };
        let id = CallId::Local(LocalCallId::new());
        let request = ToolRequest {
            call: ToolCall::new(id, ToolFunction::new(name, args)),
        };
        let mut member = commands.spawn((
            request,
            Effect::with_parent_id(Some(checking.effect)),
            ForAgent(agent),
            PatchPart { call, index },
        ));
        if let Some(turn) = turn {
            member.insert(CallOf(turn));
        }
        members.push((member.id(), label));
    }
    if members.is_empty() {
        let summary = ToolOutput::text("The patch changes no file.");
        decision.decide(Decider::plugin(NAME), ToolResult::success(summary));
        return;
    }
    commands.entity(call).insert(Patching {
        summary: plan.summary,
        holds: vec![None; members.len()],
        withheld: vec![false; members.len()],
        results: vec![None; members.len()],
        members,
        gated: false,
        refused: false,
    });
}

/// After every gate, keeps a patch's members together: while any is held
/// or waits, holds the others about to start; once a gate or the kernel
/// refused one, refuses the rest and closes the questions still open about
/// them; once all are past their gates, releases them, so they start in one
/// pass.
fn gate_together(
    mut patches: Query<&mut Patching>,
    questions: Query<(), (With<Ask>, Without<Outcome>)>,
    mut members: Query<
        (
            &mut Decision,
            Option<&Outcome>,
            Option<&Holds>,
            Has<Waiting>,
        ),
        ToolCalls,
    >,
    mut commands: Commands,
) {
    for mut patching in &mut patches {
        if patching.gated {
            continue;
        }
        let patching = &mut *patching;
        let mut refused = false;
        let mut past = true;
        for ((member, _), ours) in patching.members.iter().zip(&patching.holds) {
            let Ok((decision, outcome, holds, waiting)) = members.get(*member) else {
                continue;
            };
            let failed = |outcome: &Outcome| outcome.0.is_err();
            refused |= decision.get().is_some_and(failed) || outcome.is_some_and(failed);
            let held = holds.is_some_and(|holds| holds.iter().any(|hold| Some(hold) != *ours));
            past &= !held && !waiting;
        }
        if refused {
            patching.refused = true;
            let each = patching.members.iter().zip(&mut patching.withheld);
            for ((member, label), withheld) in each {
                if let Ok((mut decision, None, holds, _)) = members.get_mut(*member) {
                    let why = format!("{label}: not run, as another file of the patch was refused");
                    *withheld = decision.decide(Decider::plugin(NAME), Outcome::refused(why));
                    let asking = holds.into_iter().flat_map(Holds::iter);
                    for question in asking.filter(|hold| questions.contains(*hold)) {
                        commands.entity(question).try_despawn();
                    }
                }
            }
        }
        if refused || past {
            patching.gated = true;
            for hold in patching.holds.iter_mut().filter_map(Option::take) {
                commands.entity(hold).despawn();
            }
            continue;
        }
        // Only a member about to start: one held by another plugin keeps
        // only that hold, so its release moves the loop on.
        for ((member, _), ours) in patching.members.iter().zip(&mut patching.holds) {
            let free = members
                .get(*member)
                .is_ok_and(|(decision, outcome, holds, waiting)| {
                    outcome.is_none() && holds.is_none() && !waiting && !decision.is_decided()
                });
            if free {
                let why = Name::new("waiting for the other files of the patch");
                *ours = Some(commands.spawn((HoldOf(*member), why)).id());
            }
        }
    }
}

/// Keeps what each member of a patch answered, and answers the patch once
/// all did.
fn part_settled(
    settled: On<Add<Settled>>,
    parts: Query<(&PatchPart, &Outcome), Without<SupersededBy>>,
    mut patches: Query<(&mut Patching, &mut Decision), ToolCalls>,
    mut commands: Commands,
) {
    let Ok((&PatchPart { call, index }, outcome)) = parts.get(settled.entity) else {
        return;
    };
    let Ok((mut patching, mut decision)) = patches.get_mut(call) else {
        return;
    };
    if let Some(slot) = patching.results.get_mut(index) {
        *slot = Some(outcome.tool_result());
    }
    if patching.results.iter().all(Option::is_some) {
        commands.entity(call).remove::<Patching>();
        decision.decide(Decider::plugin(NAME), answer(&patching));
    }
}

/// The answer of a patch whose members all answered: each file with its
/// counts, and what a member said beyond its first line (another plugin's
/// note, such as a formatter's); else what failed.
fn answer(patching: &Patching) -> ToolResult {
    let answered = patching
        .members
        .iter()
        .zip(patching.results.iter().flatten());
    let mut failed = answered
        .clone()
        .zip(&patching.withheld)
        .filter(|((_, result), _)| !result.is_success())
        .peekable();
    if failed.peek().is_none() {
        let mut text = patching.summary.join("\n");
        for ((_, label), result) in answered {
            if let Some((_, notes)) = result.output().render().split_once('\n') {
                text.push_str(&format!("\n\n{label}:{notes}"));
            }
        }
        return ToolResult::success(ToolOutput::text(text));
    }
    let mut text = if patching.refused {
        "The patch was not applied; no file changed.".to_owned()
    } else {
        "The patch was applied only in part: every other file changed.".to_owned()
    };
    for (((_, label), result), _) in failed.filter(|(_, withheld)| !**withheld) {
        let output = result.output().render();
        text.push_str(&format!("\n\n- {label}: {output}"));
    }
    ToolResult::failed(ToolExecutionError::other(text))
}

/// The pool a patch is checked on.
fn pool() -> &'static bevy_tasks::TaskPool {
    bevy_tasks::IoTaskPool::get_or_init(bevy_tasks::TaskPool::default)
}

#[cfg(test)]
mod tests;

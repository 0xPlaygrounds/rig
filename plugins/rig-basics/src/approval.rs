//! The user's approval of an effect before it starts, asked once however many
//! plugins want it. A plugin adds its reason with [`NeedsApproval::add`] in
//! Prepare or Gate; after every gate, [`ApprovalSystems`] holds the effect
//! with one [`Ask`] listing every reason. [`ALLOW`] makes it [`Approved`];
//! [`DENY`], or a text alone, refuses it with the user's note; with nobody to
//! answer ([`Answerers`]) it is refused at once.
//!
//! ```
//! use rig_basics::approval::{Approved, NeedsApproval};
//! use rig_harness::prelude::*;
//!
//! /// Wants the user to approve every `shell` call.
//! fn guard(
//!     calls: Query<(Entity, &ToolRequest), (ReadyToStart, ToolCalls, Without<Approved>)>,
//!     mut commands: Commands,
//! ) {
//!     for (call, request) in &calls {
//!         if request.call.function.name == "shell" {
//!             commands.entity(call).queue(NeedsApproval::add("guard", "It runs a command."));
//!         }
//!     }
//! }
//!
//! let mut app = App::new();
//! app.add_systems(AgentPass, guard.in_set(Stage::Gate));
//! ```

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_core::effect::Decider;
use rig_ecs::stage::{
    AgentPass, Decision, EffectRequest, ForAgent, GateSystems, HoldOf, Outcome, ReadyToStart,
};
use rig_ecs::tools::ToolRequest;

use crate::preview::PreviewSystems;
use rig_harness::ask::{Answer, Answerers, Ask};

/// The choice that lets an effect go on.
pub const ALLOW: &str = "allow";
/// The choice that refuses it; offered first, as the default answer.
pub const DENY: &str = "deny";

/// On an effect: plugins want the user to approve it before it starts, for
/// these reasons, in the order given. Add one with [`Self::add`], so
/// several plugins' reasons make one question.
#[derive(Component, Clone, Debug, Default, PartialEq, Eq)]
pub struct NeedsApproval {
    /// Why, by whom.
    pub reasons: Vec<ApprovalReason>,
}

/// Why a plugin wants an effect approved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApprovalReason {
    /// The plugin, by name, such as `confirm-destructive`.
    pub by: String,
    /// What the user reads, such as "It deletes files.".
    pub text: String,
}

impl NeedsApproval {
    /// Adds the reason `text` of the plugin `by` to the effect's, unless
    /// it has that one:
    /// `commands.entity(call).queue(NeedsApproval::add("guard", "…"))`.
    pub fn add(by: impl Into<String>, text: impl Into<String>) -> impl EntityCommand {
        let reason = ApprovalReason {
            by: by.into(),
            text: text.into(),
        };
        move |mut entity: EntityWorldMut| match entity.get_mut::<Self>() {
            Some(mut needs) if !needs.reasons.contains(&reason) => needs.reasons.push(reason),
            Some(_) => {}
            None => {
                entity.insert(Self {
                    reasons: vec![reason],
                });
            }
        }
    }
}

/// On an effect the user approved: it is never asked about again.
#[derive(Component, Clone, Copy, Debug, Default)]
#[component(storage = "SparseSet")]
pub struct Approved;

/// The harness's asking, in Gate after every gate and preview and before the
/// kernel's fallback:
/// `.after(GateSystems::After).after(PreviewSystems).before(GateSystems::Fallback)`.
///
/// A plugin that gathers approvals, such as one asking about a reply's calls
/// together, runs `.before(ApprovalSystems)` and holds or decides them; the
/// harness asks only about what is undecided and unheld then. To batch a
/// reply's later calls with its earlier ones, it marks them in Prepare
/// (`Added<Admitted>`): a call waiting on an earlier one (`Waiting`) is not
/// [`ReadyToStart`] there, so Gate never sees it.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ApprovalSystems;

/// On an [`Ask`] about approving the effect it holds.
#[derive(Component)]
struct Approving;

/// Asks the user to approve each effect that [`NeedsApproval`]. A plugin
/// that asks adds it, unless `app.is_plugin_added::<ApprovalPlugin>()`.
pub struct ApprovalPlugin;

impl Plugin for ApprovalPlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(
            AgentPass,
            ApprovalSystems
                .after(GateSystems::After)
                .after(PreviewSystems)
                .before(GateSystems::Fallback),
        )
        .add_systems(AgentPass, (answered, ask).chain().in_set(ApprovalSystems));
    }
}

/// Asks about each undecided, unheld effect that needs approval, holding
/// it, or refuses it with nobody to ask.
fn ask(
    mut effects: Query<
        (
            Entity,
            &NeedsApproval,
            Option<&ForAgent>,
            Option<&ToolRequest>,
            Option<&EffectRequest>,
            &mut Decision,
        ),
        (ReadyToStart, Without<Approved>, Without<Ask>),
    >,
    answerers: Answerers,
    mut commands: Commands,
) {
    for (effect, needs, agent, tool, request, mut decision) in &mut effects {
        if decision.is_decided() || needs.reasons.is_empty() {
            continue;
        }
        let agent = agent.map(|&ForAgent(agent)| agent);
        let reasons: Vec<&str> = needs
            .reasons
            .iter()
            .map(|reason| reason.text.as_str())
            .collect();
        if !answerers.can_answer(agent) {
            let by = needs.reasons.first().map(|reason| reason.by.clone());
            decision.decide(
                Decider::plugin(by.unwrap_or_default()),
                Outcome::refused(format!(
                    "Needs the user's approval, and nobody can answer: {}",
                    reasons.join(" ")
                )),
            );
            continue;
        }
        let (title, mut body) = match (tool, request) {
            (Some(ToolRequest { call }), _) => (
                format!("Allow `{}`?", call.function.name),
                format!(
                    "{}\n\n",
                    serde_json::to_string(&call.function.arguments).unwrap_or_default()
                ),
            ),
            (None, Some(request)) => (format!("Allow `{}`?", request.kind.name()), String::new()),
            (None, None) => ("Allow this?".to_owned(), String::new()),
        };
        body.push_str(&reasons.join("\n"));
        body.push_str("\nType a reason to deny it with.");
        let ask = Ask::new(title)
            .body(body)
            .choice(DENY, "Deny")
            .choice(ALLOW, "Allow")
            .free_text();
        let mut asking = commands.spawn((ask, Approving, HoldOf(effect)));
        if let Some(agent) = agent {
            asking.insert(ForAgent(agent));
        }
    }
}

/// Carries out each answer: the effect is approved, or refused with the
/// reason typed, and the question lets go of it.
fn answered(
    asks: Query<(Entity, &Outcome, &HoldOf), (With<Approving>, Added<Outcome>)>,
    mut effects: Query<&mut Decision, Without<Ask>>,
    mut commands: Commands,
) {
    for (ask, outcome, &HoldOf(effect)) in &asks {
        match Answer::of(outcome) {
            Some(answer) if answer.is(ALLOW) => {
                commands.entity(effect).insert(Approved);
            }
            answer => {
                let reason = answer.and_then(|answer| answer.text);
                let why = match reason.as_deref().map(str::trim) {
                    Some(reason) if !reason.is_empty() => format!("The user denied this: {reason}"),
                    _ => "The user denied this.".to_owned(),
                };
                if let Ok(mut decision) = effects.get_mut(effect) {
                    decision.decide(Decider::Person, Outcome::refused(why));
                }
            }
        }
        commands.entity(ask).remove::<HoldOf>();
    }
}

#[cfg(test)]
mod tests;

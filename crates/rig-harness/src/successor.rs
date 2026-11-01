//! Following an effect to its successor. Any plugin may take an effect
//! over by spawning a successor it is `SupersededBy`, such as a watchdog
//! sending a stalled call again ([`resend`]): the old effect then settles
//! as superseded, with no outcome applied. The plugin that owns the effect,
//! and waits for its outcome by a marker of its own, follows it with the
//! kernel's [`follow_successor`](rig_ecs::stage::follow_successor), so the
//! successor's outcome is the one it takes.
//!
//! ```no_run
//! use rig_harness::prelude::*;
//!
//! /// On the plugin's own model call, whose outcome it takes.
//! #[derive(Component)]
//! struct Review;
//!
//! fn build(app: &mut App) {
//!     app.add_observer(follow_successor::<Review>);
//! }
//!
//! /// Sends every call that ran a minute again, whoever owns it.
//! fn unstall(calls: Query<(Entity, &Started), (ModelCalls, Without<Outcome>)>, mut commands: Commands) {
//!     for (call, started) in &calls {
//!         if started.at.elapsed().is_ok_and(|ran| ran > Duration::from_secs(60)) {
//!             commands.queue(resend(call));
//!         }
//!     }
//! }
//! ```

use bevy_ecs::prelude::*;
use rig_ecs::agent::CallOf;
use rig_ecs::stage::{Effect, EffectRequest, ForAgent, Settled, SupersededBy};
use rig_ecs::tools::ToolRequest;
use rig_ecs::turn::{ModelRequest, SideCall, TurnCall};

use crate::load::ProvidedBy;

/// Sends `effect` again: spawns a successor that supersedes it, with its
/// request, unless it settled or was superseded already. A [`TurnCall`]
/// or [`SideCall`] is built anew by the kernel. A plugin's own
/// [`ModelRequest`], [`EffectRequest`] or [`ToolRequest`] is copied as it
/// is now, with its Prepare edits (a model call's connection is routed and
/// adapted again in Prepare, as `Adapted` is not copied), and its
/// [`Effect`] parent and purpose, `CallOf`, `ForAgent`, `ChildOf`,
/// `ProvidedBy` and `Name`; the kernel gives it the call's `InReply`. The
/// owner's markers follow by
/// [`follow_successor`](rig_ecs::stage::follow_successor):
/// `commands.queue(resend(call))`.
pub fn resend(effect: Entity) -> impl Command {
    move |world: &mut World| {
        let Ok(old) = world.get_entity(effect) else {
            return;
        };
        let Some(was) = old.get::<Effect>() else {
            return;
        };
        if old.contains::<Settled>() || old.contains::<SupersededBy>() {
            return;
        }
        let mut again = Effect::with_parent_id(was.parent);
        again.purpose.clone_from(&was.purpose);
        let turn_call = old.contains::<TurnCall>();
        let build_from = old.get::<SideCall>().copied();
        let copied = if turn_call || build_from.is_some() {
            (None, None, None)
        } else {
            (
                old.get::<ModelRequest>().cloned(),
                old.get::<EffectRequest>().cloned(),
                old.get::<ToolRequest>().cloned(),
            )
        };
        let call_of = old.get::<CallOf>().map(|&CallOf(turn)| CallOf(turn));
        let agent = old
            .get::<ForAgent>()
            .map(|&ForAgent(agent)| ForAgent(agent));
        let child_of = old.get::<ChildOf>().map(|of| ChildOf(of.parent()));
        let provided = old
            .get::<ProvidedBy>()
            .map(|&ProvidedBy(by)| ProvidedBy(by));
        let name = old.get::<Name>().cloned();
        let mut successor = world.spawn(again);
        successor.add_related::<SupersededBy>(&[effect]);
        if turn_call {
            successor.insert(TurnCall);
        }
        let (model, request, tool) = copied;
        insert(&mut successor, build_from);
        insert(&mut successor, model);
        insert(&mut successor, request);
        insert(&mut successor, tool);
        insert(&mut successor, call_of);
        insert(&mut successor, agent);
        insert(&mut successor, child_of);
        insert(&mut successor, provided);
        insert(&mut successor, name);
    }
}

/// Inserts `component` on `entity`, if there is one.
fn insert(entity: &mut EntityWorldMut, component: Option<impl Component>) {
    if let Some(component) = component {
        entity.insert(component);
    }
}

#[cfg(test)]
mod tests;

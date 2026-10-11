//! Following an effect to its successor. Any plugin may take an effect
//! over by spawning a successor that `Supersedes` it, such as a watchdog
//! sending a stalled call again ([`resend`]): the old effect then settles
//! as superseded, with no outcome applied. The plugin that owns the effect,
//! and waits for its outcome by a marker of its own, follows it with
//! [`follow_successor`], so the successor's outcome is the one it takes.
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
use rig_ecs::stage::{Effect, EffectRequest, ForAgent, HoldOf, Settled, SupersededBy, Supersedes};
use rig_ecs::tools::ToolRequest;
use rig_ecs::turn::{BuildFrom, ModelRequest, TurnCall};

use crate::load::ProvidedBy;

/// When an effect carrying `M` is superseded, moves `M`, and the hold the
/// effect keeps (`HoldOf`), to the successor, so the owner keeps its work.
/// The successor's hold is inserted before the old one is removed, so the
/// held effect is never unheld in between:
/// `app.add_observer(follow_successor::<Summarizing>)`.
pub fn follow_successor<M: Component>(
    superseded: On<Add<SupersededBy>>,
    owned: Query<(&SupersededBy, Option<&HoldOf>), With<M>>,
    mut commands: Commands,
) {
    let old = superseded.entity;
    let Ok((by, hold)) = owned.get(old) else {
        return;
    };
    let new = by.successor();
    let held = hold.map(|&HoldOf(held)| held);
    commands.queue(move |world: &mut World| {
        let Ok(mut to) = world.get_entity_mut(new) else {
            return;
        };
        if let Some(held) = held {
            to.insert(HoldOf(held));
        }
        let Ok(mut from) = world.get_entity_mut(old) else {
            return;
        };
        let marker = from.take::<M>();
        if held.is_some() {
            from.remove::<HoldOf>();
        }
        if let (Some(marker), Ok(mut to)) = (marker, world.get_entity_mut(new)) {
            to.insert(marker);
        }
    });
}

/// Sends `effect` again: spawns a successor that supersedes it, with its
/// request, unless it settled or was superseded already. A [`TurnCall`]
/// or [`BuildFrom`] is built anew by the kernel. A plugin's own
/// [`ModelRequest`], [`EffectRequest`] or [`ToolRequest`] is copied as it
/// is now, with its Prepare edits (a model call's adapters, apart in
/// `Adapters`, are pushed again in Prepare), and its [`Effect`] parent and
/// purpose, `CallOf`, `ForAgent`, `ChildOf`, `ProvidedBy` and `Name`. The
/// owner's markers follow by [`follow_successor`]: `commands.queue(resend(call))`.
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
        let mut again = Effect::child_of(was.parent);
        again.purpose.clone_from(&was.purpose);
        let turn_call = old.contains::<TurnCall>();
        let build_from = old.get::<BuildFrom>().copied();
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
        let mut successor = world.spawn((again, Supersedes(effect)));
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

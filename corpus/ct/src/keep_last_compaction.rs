//! A compaction that drops the oldest turns, in place of the default.
//!
//! It replaces `rig_basics::compaction::CompactionPlugin`: disable that
//! entry in `plugins.toml` and add this one. When a turn's request would
//! leave less of the model's window free than `[compaction]`'s reserve,
//! every turn but the last [`KEEP`] is taken out of the conversation, with
//! no summary and no model call; `/compact` does the same at once. A turn is
//! a message the user typed and everything after it. It is written only
//! against `rig_harness`'s public API: nothing of the default compaction
//! is used.

use rig_harness::prelude::*;
use rig_harness::rig_core::conversation::Conversation as Chat;
use rig_harness::rig_memory::{CompactionPolicy, estimate_tokens};
use serde::{Deserialize, Deserializer};

/// Who this plugin is, in the conversation's changes.
pub const NAME: &str = "keep-last-compaction";

/// The turns kept: the user's last messages and everything after them.
pub const KEEP: usize = 2;

/// Drops all but the last [`KEEP`] turns near the model's window and on
/// `/compact`.
#[derive(Default)]
pub struct KeepLastCompactionPlugin;

impl Plugin for KeepLastCompactionPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<Policy>("compaction")
            .add_command(
                "compact",
                "Drop all but the last two turns to free context",
                compact,
            )
            .add_systems(AgentPass, near_the_window.in_set(PrepareSystems::Fit));
    }
}

/// `[compaction]` of `settings.toml`, read as the default compaction reads
/// it, so its reserve keeps meaning the same.
#[derive(Resource, Default)]
struct Policy(CompactionPolicy);

impl<'de> Deserialize<'de> for Policy {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        CompactionPolicy::deserialize(deserializer).map(Policy)
    }
}

/// Drops the oldest turns of a turn's request, and of its agent's
/// conversation, when the request as it will be sent nears the window.
fn near_the_window(
    mut calls: Query<(&mut ModelRequest, &Adapted, &ForAgent), (Added<Admitted>, With<TurnCall>)>,
    mut agents: Query<(&mut Conversation, &LastUsage)>,
    policy: Res<Policy>,
) {
    for (mut model, adapted, &ForAgent(agent)) in &mut calls {
        let Ok((mut conversation, last)) = agents.get_mut(agent) else {
            continue;
        };
        let (connection, request) = (&adapted.0, adapted.sent(&model));
        let used = last
            .context()
            .unwrap_or(0)
            .max(estimate_tokens(&request.chat_history));
        if !policy.0.over_threshold(used, &connection.spec) {
            continue;
        }
        let Some(kept) = drop_oldest(&mut conversation) else {
            continue;
        };
        if let Some(history) = model.history.as_mut()
            && let Some(at) = history.index_of(kept)
        {
            history.splice(..at, [], Author::plugin(NAME));
        }
    }
}

/// `/compact`: drops the oldest turns now.
fn compact(
    In(args): In<CommandArgs>,
    mut agents: Query<&mut Conversation>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(mut conversation) = agents.get_mut(args.agent) else {
        return;
    };
    let text = match drop_oldest(&mut conversation) {
        Some(_) => format!("Kept the last {KEEP} turns."),
        None => "Nothing to drop.".to_owned(),
    };
    notices.write(Notice::info(args.agent, text));
}

/// Takes every turn but the last [`KEEP`] out of `conversation`, and the
/// id of the first message kept; `None` when there were no more.
fn drop_oldest(conversation: &mut Chat) -> Option<MessageId> {
    let typed: Vec<usize> = conversation
        .entries()
        .filter(|entry| entry.is_users_own())
        .map(|entry| entry.index)
        .collect();
    let from = *typed.get(typed.len().checked_sub(KEEP)?)?;
    let kept = conversation.id(from)?;
    if from == 0 {
        return None;
    }
    conversation.splice(..from, [], Author::plugin(NAME));
    Some(kept)
}

#[cfg(test)]
mod tests;

//! Version 2 of the worklist: version 1's commands and `/work doing <n>`;
//! each agent's checklist saved as items with an id, a status and the turn
//! they were added in, at version 2 of [`KEY`](crate::KEY).

use rig_harness::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{DONE, KEY, Newer, Shows, Unreadable, build_common, index, refusal, verb};

/// Version 2 of the worklist plugin.
#[derive(Default)]
pub struct WorklistV2Plugin;

/// Where an item stands.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    /// Not started.
    #[default]
    Open,
    /// Being worked on.
    Doing,
    /// Finished.
    Done,
}

/// One item of a checklist.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Item {
    /// Its id, unique in its list.
    pub id: u32,
    /// What it says.
    pub text: String,
    /// Where it stands.
    pub status: Status,
    /// The user turn it was added in; `None` when unknown, as for an item
    /// version 1 saved.
    pub added_turn: Option<u32>,
}

/// An agent's checklist; none on an agent that never had one.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[reflect(opaque, Component, Follows, Default, Clone, Serialize, Deserialize)]
pub struct Worklist(pub Vec<Item>);

impl Shows for Worklist {
    fn lines(&self) -> Vec<String> {
        let numbered = self.0.iter().zip(1..);
        numbered
            .map(|(item, n)| {
                let mark = match item.status {
                    Status::Open => "[ ]",
                    Status::Doing => "[~]",
                    Status::Done => "[x]",
                };
                let turn = item
                    .added_turn
                    .map_or_else(|| "?".to_owned(), |turn| turn.to_string());
                format!("{n}. {mark} {} (turn {turn})", item.text)
            })
            .collect()
    }
}

/// The steps reading older saved checklists: version 1's strings.
const UPGRADES: &[Upgrade] = &[from_v1];

/// Version 1's `["a", "[x] b"]` as version 2's items: `b` done, every
/// turn unknown.
pub fn from_v1(old: Value) -> Result<Value, String> {
    let strings: Vec<String> = serde_json::from_value(old).map_err(|e| e.to_string())?;
    let items: Vec<Item> = strings
        .into_iter()
        .zip(1..)
        .map(|(saved, id)| {
            let (text, status) = match saved.strip_prefix(DONE) {
                Some(text) => (text.to_owned(), Status::Done),
                None => (saved, Status::Open),
            };
            Item {
                id,
                text,
                status,
                added_turn: None,
            }
        })
        .collect();
    serde_json::to_value(items).map_err(|e| e.to_string())
}

impl Plugin for WorklistV2Plugin {
    fn build(&self, app: &mut App) {
        app.register_type::<Worklist>()
            .saved_upgrades::<Worklist>(UPGRADES);
        let _ = app.saved_as::<Worklist>(KEY);
        app.add_command(
            "work",
            "The agent's checklist: /work add <text>, /work doing <n>, /work done <n>, /work reset",
            work,
        );
        build_common::<Worklist>(app);
    }
}

/// `/work add <text>`, `/work doing <n>`, `/work done <n>`, `/work reset`.
fn work(
    In(args): In<CommandArgs>,
    agents: Query<(
        Option<&Worklist>,
        &Conversation,
        Has<Newer>,
        Has<Unreadable>,
    )>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let Ok((list, conversation, newer, unreadable)) = agents.get(agent) else {
        return;
    };
    let (verb, rest) = verb(&args.args);
    if let Some(why) = refusal(verb, newer, unreadable) {
        notices.write(Notice::error(agent, why));
        return;
    }
    let mut items = list.map(|list| list.0.clone()).unwrap_or_default();
    let status = match verb {
        "doing" => Some(Status::Doing),
        "done" => Some(Status::Done),
        _ => None,
    };
    match (verb, status) {
        ("add", _) if !rest.is_empty() => {
            let id = items
                .iter()
                .map(|item| item.id)
                .max()
                .unwrap_or(0)
                .saturating_add(1);
            let turns = conversation
                .entries()
                .filter(|entry| entry.is_users_own())
                .count();
            items.push(Item {
                id,
                text: rest.to_owned(),
                status: Status::Open,
                added_turn: u32::try_from(turns).ok(),
            });
        }
        (_, Some(status)) => {
            let Some(item) = index(rest, items.len()).and_then(|at| items.get_mut(at)) else {
                notices.write(Notice::error(agent, format!("worklist: no item {rest}")));
                return;
            };
            item.status = status;
        }
        ("reset", _) => {
            // The kernel never writes over a followed value it set aside
            // in this run, so after a reset of an unreadable list the new
            // one is shown but not saved.
            items.clear();
            commands.entity(agent).remove::<Unreadable>();
        }
        _ => {
            let usage =
                "worklist: /work add <text>, /work doing <n>, /work done <n> or /work reset";
            notices.write(Notice::error(agent, usage));
            return;
        }
    }
    commands.entity(agent).insert(Worklist(items));
}

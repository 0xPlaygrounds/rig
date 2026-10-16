//! Version 1 of the worklist: `/work add <text>`, `/work done <n>`, a
//! panel; each agent's checklist saved as a list of strings.

use rig_harness::prelude::*;
use serde::{Deserialize, Serialize};

use crate::plugin_state_upgrade::{
    DONE, KEY, Newer, Shows, Unreadable, build_common, index, refusal, verb,
};

/// Version 1 of the worklist plugin.
#[derive(Default)]
pub struct WorklistV1Plugin;

/// An agent's checklist: one string per item, a done one led by `[x] `;
/// none on an agent that never had one.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[reflect(opaque, Component, Follows, Default, Clone, Serialize, Deserialize)]
pub struct Worklist(pub Vec<String>);

impl Shows for Worklist {
    fn lines(&self) -> Vec<String> {
        let numbered = self.0.iter().zip(1..);
        numbered.map(|(item, n)| format!("{n}. {item}")).collect()
    }
}

impl Plugin for WorklistV1Plugin {
    fn build(&self, app: &mut App) {
        app.register_type::<Worklist>();
        let _ = app.saved_as::<Worklist>(KEY);
        app.add_command(
            "work",
            "The agent's checklist: /work add <text>, /work done <n>",
            work,
        );
        build_common::<Worklist>(app);
    }
}

/// `/work add <text>`, `/work done <n>`, `/work reset`.
fn work(
    In(args): In<CommandArgs>,
    agents: Query<(Option<&Worklist>, Has<Newer>, Has<Unreadable>)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let Ok((list, newer, unreadable)) = agents.get(agent) else {
        return;
    };
    let (verb, rest) = verb(&args.args);
    if let Some(why) = refusal(verb, newer, unreadable) {
        notices.write(Notice::error(agent, why));
        return;
    }
    let mut items = list.map(|list| list.0.clone()).unwrap_or_default();
    match verb {
        "add" if !rest.is_empty() => items.push(rest.to_owned()),
        "done" => {
            let Some(item) = index(rest, items.len()).and_then(|at| items.get_mut(at)) else {
                notices.write(Notice::error(agent, format!("worklist: no item {rest}")));
                return;
            };
            if !item.starts_with(DONE) {
                *item = format!("{DONE}{item}");
            }
        }
        "reset" => {
            // The kernel never writes over a followed value it set aside
            // in this run, so after a reset of an unreadable list the new
            // one is shown but not saved.
            items.clear();
            commands.entity(agent).remove::<Unreadable>();
        }
        _ => {
            let usage = "worklist: /work add <text>, /work done <n> or /work reset";
            notices.write(Notice::error(agent, usage));
            return;
        }
    }
    commands.entity(agent).insert(Worklist(items));
}

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::SystemTime;

use rig_core::test_utils::MockCompletionModel;
use rig_core::tool::{FileChange, FilesChanged, ToolContext};
use rig_harness::clock::ClockPlugin;
use rig_harness_test_support::{
    Ended, MemoryStore, ToolOutputs, app_on, calls, log_ended, log_tool_outputs, reply, run_until,
    scripted_models,
};
use serde::Deserialize;

use super::*;

/// What `slow_write` was let do, and the texts of the calls it ran, in
/// the order they started.
#[derive(Default)]
struct Desk {
    released: HashSet<String>,
    ran: Vec<String>,
}

/// The desk, shared by the test and the tool's threads.
#[derive(Clone, Default)]
struct Shared(Arc<(Mutex<Desk>, Condvar)>);

impl Shared {
    /// Lets the calls that write `text` finish.
    fn release(&self, text: &str) {
        let (desk, changed) = &*self.0;
        let mut desk = desk.lock().unwrap_or_else(PoisonError::into_inner);
        desk.released.insert(text.to_owned());
        changed.notify_all();
    }

    /// Counts a call writing `text`, and waits until it is released.
    fn run(&self, text: &str) {
        let (desk, changed) = &*self.0;
        let mut desk = desk.lock().unwrap_or_else(PoisonError::into_inner);
        desk.ran.push(text.to_owned());
        while !desk.released.contains(text) {
            desk = changed.wait(desk).unwrap_or_else(PoisonError::into_inner);
        }
    }

    fn ran(&self) -> Vec<String> {
        let (desk, _) = &*self.0;
        let desk = desk.lock().unwrap_or_else(PoisonError::into_inner);
        desk.ran.clone()
    }
}

/// A `write` that blocks until the test releases its text.
struct SlowWrite {
    dir: PathBuf,
    desk: Shared,
}

#[derive(Deserialize)]
struct SlowArgs {
    path: String,
    text: String,
}

impl Tool for SlowWrite {
    const NAME: &'static str = "slow_write";
    type Args = SlowArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Write a file, slowly.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": { "path": { "type": "string" }, "text": { "type": "string" } },
            "required": ["path", "text"],
        })
    }

    fn footprint(&self, args: &SlowArgs) -> Touches {
        Touches::Writes(Scope::Paths(vec![self.dir.join(&args.path)]))
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: SlowArgs,
    ) -> Result<String, ToolExecutionError> {
        let (desk, file) = (self.desk.clone(), self.dir.join(&args.path));
        let change = blocking(move || {
            desk.run(&args.text);
            let before = std::fs::read_to_string(&file).ok();
            std::fs::write(&file, &args.text).map_err(ToolExecutionError::from_error)?;
            let (path, added) = (file.display().to_string(), args.text.lines().count());
            Ok(match before {
                None => FileChange::created(path, added),
                Some(before) => FileChange::modified(path, added, before.lines().count()),
            })
        })
        .await?;
        context.insert_result(FilesChanged(vec![change]))?;
        Ok("written".to_owned())
    }
}

/// The hold that keeps the agent's call waiting, and why.
fn wait_of(world: &mut World, agent: Entity) -> Option<(Entity, String)> {
    let mut holds = world.query::<(Entity, &HoldOf, &Name)>();
    let mut calls = world.query::<&ForAgent>();
    let holds: Vec<_> = holds
        .iter(world)
        .map(|(hold, &HoldOf(call), name)| (hold, call, name.to_string()))
        .collect();
    holds
        .into_iter()
        .find(|&(_, call, _)| calls.get(world, call).is_ok_and(|by| by.0 == agent))
        .map(|(hold, _, name)| (hold, name))
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(FileLeasePlugin);
}

#[test]
fn writes_to_one_file_wait_for_its_lease_and_its_owner_hears_of_the_change() {
    let dir = std::env::current_exe()
        .unwrap_or_default()
        .with_file_name("file-lease-project");
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).ok();
    let dir = dir.canonicalize().unwrap_or(dir);
    let write = |id: &str, path: &str, text: &str| {
        calls(
            id,
            "slow_write",
            serde_json::json!({ "path": path, "text": text }),
        )
    };
    let a = MockCompletionModel::from_stream_turns([
        write("a1", "x.txt", "A\n"),
        write("a2", "y.txt", "Y\n"),
        reply("A done"),
    ]);
    let b = MockCompletionModel::from_stream_turns([
        write("b1", "x.txt", "B1\nB2\nB3\n"),
        reply("B done"),
    ]);
    let c = MockCompletionModel::from_stream_turns([write("c1", "x.txt", "C\n"), reply("C")]);
    let d = MockCompletionModel::from_stream_turns([write("d1", "x.txt", "D\n"), reply("D")]);
    let desk = Shared::default();
    // Only the lease holds B back; C and D would write at once if they ran.
    for text in ["B1\nB2\nB3\n", "C\n", "D\n"] {
        desk.release(text);
    }

    let (mut app, wakes) = app_on(&MemoryStore::default());
    let settings = "[file_lease]\ntools = [\"slow_write\"]\n";
    app.insert_resource(scripted_models([
        ("lease/a", &a),
        ("lease/b", &b),
        ("lease/c", &c),
        ("lease/d", &d),
    ]))
    .insert_resource(Settings::from_toml(settings))
    .insert_resource(ProjectRoot(dir.clone()))
    .add_plugins((AgentPlugin, SessionLogPlugin, ClockPlugin, FileLeasePlugin))
    .add_tool(SlowWrite {
        dir: dir.clone(),
        desk: desk.clone(),
    })
    .init_resource::<ToolOutputs>()
    .add_observer(log_tool_outputs)
    .init_resource::<Ended>()
    .add_observer(log_ended);
    let mut agent = |name: &str| {
        let choice = ModelChoice(format!("lease/{}", name.to_lowercase()));
        app.world_mut()
            .spawn((Agent, Name::new(name.to_owned()), choice))
            .id()
    };
    let [agent_a, agent_b, agent_c, agent_d] = ["A", "B", "C", "D"].map(&mut agent);
    app.finish();
    app.update();

    let ask = |app: &mut App, agent: Entity| {
        let message = Deliver::new(agent, "Write x.txt.", DeliveryMode::Steer);
        app.world_mut().trigger(message);
    };
    let ended = |count: usize| move |world: &World| world.resource::<Ended>().0.len() == count;
    ask(&mut app, agent_a);
    run_until(&mut app, &wakes, |_| desk.ran() == ["A\n"]);
    let mut waits = Vec::new();
    for waiter in [agent_b, agent_c, agent_d] {
        ask(&mut app, waiter);
        let mut wait = None;
        for _ in 0..50 {
            app.update();
            wait = wait_of(app.world_mut(), waiter);
            if wait.is_some() {
                break;
            }
        }
        let (hold, why) = wait.unzip();
        assert_eq!(why.as_deref(), Some("waiting for A"));
        waits.push(hold);
    }
    // A holds x.txt: B's call is pending, neither run nor answered.
    let read = |dir: &PathBuf| std::fs::read_to_string(dir.join("x.txt")).unwrap_or_default();
    assert_eq!(desk.ran(), ["A\n"]);
    assert!(!app.world().resource::<ToolOutputs>().0.contains_key("b1"));
    assert!(!read(&dir).contains('B'));

    // C's turn is interrupted while it waits: its call leaves the queue.
    app.world_mut().trigger(StopTurn::new(agent_c));
    run_until(&mut app, &wakes, ended(1));
    assert!(wait_of(app.world_mut(), agent_c).is_none());

    // D's wait runs out: its call is refused.
    if let Some(Some(hold)) = waits.get(2) {
        app.world_mut()
            .entity_mut(*hold)
            .insert(WakeAt(SystemTime::UNIX_EPOCH));
    }
    run_until(&mut app, &wakes, ended(2));
    let refused = app.world().resource::<ToolOutputs>().0.get("d1").cloned();
    assert!(
        refused.is_some_and(|why| why.contains("x.txt is busy (held by A's call)")),
        "{:?}",
        app.world().resource::<ToolOutputs>().0
    );

    // A is done with x.txt: B writes it, after A.
    desk.release("A\n");
    run_until(&mut app, &wakes, |world| {
        world.resource::<ToolOutputs>().0.contains_key("b1")
    });
    assert_eq!(read(&dir), "B1\nB2\nB3\n");
    desk.release("Y\n");
    run_until(&mut app, &wakes, ended(4));

    // A's write of y.txt holds no lease of x.txt, so it may start before
    // or after B's: only the writes to x.txt are ordered.
    let mut ran = desk.ran();
    let y = ran.iter().position(|text| text == "Y\n");
    ran.retain(|text| text != "Y\n");
    assert!(y.is_some_and(|y| y > 0), "A wrote y.txt after x.txt");
    assert_eq!(ran, ["A\n", "B1\nB2\nB3\n"], "neither C nor D wrote");
    let outputs = &app.world().resource::<ToolOutputs>().0;
    assert_eq!(outputs.get("b1").map(String::as_str), Some("written"));
    let third = format!("{:?}", a.requests().get(2));
    assert!(
        third.contains("agent B changed x.txt (+3 −1) while you were working on it"),
        "{third}"
    );
    let second = format!("{:?}", a.requests().get(1));
    assert!(!second.contains("agent B changed"), "{second}");
}

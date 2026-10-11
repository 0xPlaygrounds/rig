use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex, PoisonError};

use rig_cassette::journal::MemoryStore;
use rig_core::test_utils::MockCompletionModel;
use rig_core::tool::{FileChange, FilesChanged, ToolContext};
use rig_harness_test_support::{
    Ended, ToolOutputs, app_on, calls, log_ended, log_tool_outputs, reply, run_until,
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

/// The agent's call of `slow_write` waits for the lease, named for whom.
fn waiting_for(world: &mut World, agent: Entity) -> Option<String> {
    let mut waits = world.query::<(&EffectRequest, &ForAgent, &Name, Has<HoldOf>)>();
    waits
        .iter(world)
        .find(|(request, by, _, held)| request.kind.name() == WAIT && by.0 == agent && *held)
        .map(|(.., name, _)| name.as_str().to_owned())
}

#[test]
fn writes_to_one_file_wait_for_its_lease_and_its_owner_hears_of_the_change() {
    let dir = std::env::current_exe()
        .unwrap_or_default()
        .with_file_name("file-lease-project");
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).ok();
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
    let desk = Shared::default();
    // Only the lease holds B back; C would write at once if it ran.
    desk.release("B1\nB2\nB3\n");
    desk.release("C\n");

    let (mut app, wakes) = app_on(&MemoryStore::default());
    let settings = "[file_lease]\ntools = [\"slow_write\"]\n";
    app.insert_resource(scripted_models([
        ("lease/a", &a),
        ("lease/b", &b),
        ("lease/c", &c),
    ]))
    .insert_resource(Settings::from_toml(settings))
    .insert_resource(ProjectRoot(dir.clone()))
    .add_plugins((AgentPlugin, JournalPlugin, FileLeasePlugin))
    .add_tool(SlowWrite {
        dir: dir.clone(),
        desk: desk.clone(),
    })
    .init_resource::<ToolOutputs>()
    .add_observer(log_tool_outputs)
    .init_resource::<Ended>()
    .add_observer(log_ended);
    let mut agent = |name: &str, model: &str| {
        let choice = ModelChoice(model.to_owned());
        app.world_mut()
            .spawn((Agent, Name::new(name.to_owned()), choice))
            .id()
    };
    let (agent_a, agent_b, agent_c) = (
        agent("A", "lease/a"),
        agent("B", "lease/b"),
        agent("C", "lease/c"),
    );
    app.finish();
    app.update();

    let ask = |app: &mut App, agent: Entity| {
        let message = Deliver::new(agent, "Write x.txt.", DeliveryMode::Steer);
        app.world_mut().trigger(message);
    };
    ask(&mut app, agent_a);
    run_until(&mut app, &wakes, |_| desk.ran() == ["A\n"]);
    for waiter in [agent_b, agent_c] {
        ask(&mut app, waiter);
        let mut waits = None;
        for _ in 0..50 {
            app.update();
            waits = waiting_for(app.world_mut(), waiter);
            if waits.is_some() {
                break;
            }
        }
        assert_eq!(waits.as_deref(), Some("waiting for A"));
    }
    // A holds x.txt: B's call is pending, neither run nor answered.
    let read = |dir: &PathBuf| std::fs::read_to_string(dir.join("x.txt")).unwrap_or_default();
    assert_eq!(desk.ran(), ["A\n"]);
    assert!(!app.world().resource::<ToolOutputs>().0.contains_key("b1"));
    assert!(!read(&dir).contains('B'));

    // C's turn is interrupted while it waits: its call leaves the queue.
    app.world_mut().trigger(Interrupt::new(agent_c));
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 1
    });
    assert!(waiting_for(app.world_mut(), agent_c).is_none());

    // A is done with x.txt: B writes it, after A.
    desk.release("A\n");
    run_until(&mut app, &wakes, |world| {
        world.resource::<ToolOutputs>().0.contains_key("b1")
    });
    assert_eq!(read(&dir), "B1\nB2\nB3\n");
    desk.release("Y\n");
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 3
    });

    assert_eq!(desk.ran(), ["A\n", "B1\nB2\nB3\n", "Y\n"], "C never wrote");
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

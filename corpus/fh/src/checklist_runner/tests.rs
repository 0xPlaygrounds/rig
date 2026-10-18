use std::collections::HashMap;
use std::path::PathBuf;

use rig_core::catalog::{Catalog, Connector};
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::MemoryStore;
use serde_json::json;

use super::*;
use rig_harness_test_support::{AnswerAsks, Answers, app_on, calls, connect, reply, run_until};

/// Each item's scripted model, by its title.
#[derive(Resource, Default)]
struct Scripts(HashMap<String, MockCompletionModel>);

/// Each item started, with where every item of its run stood then.
#[derive(Resource, Default)]
struct Started(Vec<(usize, Vec<ItemState>)>);

/// The plan file, and whether the user's line was added to it yet.
#[derive(Resource)]
struct PlanFile {
    path: PathBuf,
    edited: bool,
}

const USER_LINE: &str = "A line the user added while item 1 ran.";

/// Answers each item's agent by its item's script, keeps where the run
/// stood, and, while item 1 runs, adds a line to the end of the plan as
/// the user would in an editor.
fn on_item(
    added: On<Add<PlanItemAgent>>,
    agents: Query<&PlanItemAgent>,
    runs: Query<&PlanRun>,
    scripts: Res<Scripts>,
    mut started: ResMut<Started>,
    mut file: ResMut<PlanFile>,
    mut commands: Commands,
) {
    let Ok(item) = agents.get(added.entity) else {
        return;
    };
    let Ok(run) = runs.get(item.run) else {
        return;
    };
    let states = run.items.iter().map(|item| item.state.clone()).collect();
    started.0.push((item.index, states));
    let title = run.items.get(item.index).map(|item| item.title.as_str());
    let model = title.and_then(|title| scripts.0.get(title));
    if let Some(connection) = model.and_then(connect) {
        commands.entity(added.entity).insert(connection);
    }
    if item.index == 0 && !file.edited {
        file.edited = true;
        let text = std::fs::read_to_string(&file.path).unwrap_or_default();
        std::fs::write(&file.path, format!("{text}{USER_LINE}\n")).ok();
    }
}

/// The run of the plan, once it finished and its agents are gone.
fn finished(world: &World, starts: usize) -> bool {
    let mut entities = world.iter_entities();
    let agents = entities.any(|entity| entity.contains::<PlanItemAgent>());
    let run = world
        .iter_entities()
        .find_map(|entity| entity.get::<PlanRun>());
    !agents && run.is_some_and(|run| run.finished) && world.resource::<Started>().0.len() >= starts
}

#[test]
fn items_run_in_order_fail_block_tick_the_file_and_a_rerun_carries_on() {
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")), PathBuf::from);
    let dir = target.join(format!("corpus-fh-checklist-runner-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok();
    let path = dir.join("PLAN.md");
    let plan = "# Migration\n\nWe move the store from files to sqlite.\n\n\
                - [ ] one\n  - keep the old reader\n- [ ] two (parallel)\n\
                - [ ] three (after: 1, 2)\n- [ ] four\n";
    std::fs::write(&path, plan).ok();

    let script = |turns: Vec<Vec<_>>| MockCompletionModel::from_stream_turns(turns);
    let fail = calls("f1", FAIL_ITEM, json!({ "reason": "no access" }));
    let mut scripts = Scripts::default();
    scripts
        .0
        .insert("one".into(), script(vec![reply("did one")]));
    scripts
        .0
        .insert("two".into(), script(vec![fail, reply("cannot")]));
    scripts
        .0
        .insert("four".into(), script(vec![reply("did four")]));
    let one = scripts.0.get("one").cloned();

    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(Models(Connector::new(Catalog::default())))
        .insert_resource(scripts)
        .insert_resource(PlanFile {
            path: path.clone(),
            edited: false,
        })
        .init_resource::<Started>()
        .init_resource::<WorkerOutbox>()
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            ChecklistRunnerPlugin,
            AnswerAsks,
        ))
        .add_observer(on_item);
    let main = script(vec![reply("noted"), reply("noted again")]);
    let mut owner = app.world_mut().spawn(Agent);
    if let Some(connection) = connect(&main) {
        let choice = ModelChoice(connection.spec.reference());
        owner.insert((connection, choice));
    }
    let owner = owner.id();
    app.finish();
    app.update();

    let line = format!("run-plan {} --jobs 2", path.display());
    app.world_mut()
        .trigger(RunCommand::new(owner, line.clone()));
    run_until(&mut app, &wakes, |world| {
        finished(world, 3) && main.request_count() >= 1
    });

    // The views were told where the run ended.
    let outbox = app.world().resource::<WorkerOutbox>();
    let last = outbox
        .events
        .iter()
        .rev()
        .find(|event| event.kind == PlanRuns::KIND);
    let runs = last.and_then(|event| serde_json::from_value::<PlanRuns>(event.data.clone()).ok());
    let run = runs.as_ref().and_then(|runs| runs.runs.first());
    assert_eq!(
        run.map(|run| run.title.as_str()),
        Some("PLAN.md (finished)")
    );
    let items = run.map(|run| run.items.join("\n")).unwrap_or_default();
    assert!(items.contains("1. one\n   done"), "{items}");

    let first: Vec<(usize, Vec<ItemState>)> = app.world().resource::<Started>().0.clone();
    let order: Vec<usize> = first.iter().map(|(index, _)| *index).collect();
    let file = std::fs::read_to_string(&path).unwrap_or_default();
    let item_one_asked = format!("{:?}", one.map(|one| one.requests()));
    let told = format!(
        "{:?}",
        app.world()
            .get::<Conversation>(owner)
            .map(|conversation| conversation.messages())
    );

    // Rerun after making 2 succeed: the user retries it when asked.
    let mut scripts = app.world_mut().resource_mut::<Scripts>();
    scripts
        .0
        .insert("two".into(), script(vec![reply("did two")]));
    scripts
        .0
        .insert("three".into(), script(vec![reply("did three")]));
    app.world_mut().resource_mut::<Started>().0.clear();
    *app.world_mut().resource_mut::<Answers>() = Answers::new([Answer::choice("yes")]);
    app.world_mut().trigger(RunCommand::new(owner, line));
    run_until(&mut app, &wakes, |world| {
        finished(world, 2) && main.request_count() >= 2
    });

    let mut rerun: Vec<usize> = app
        .world()
        .resource::<Started>()
        .0
        .iter()
        .map(|(index, _)| *index)
        .collect();
    rerun.sort_unstable();
    let after = std::fs::read_to_string(&path).unwrap_or_default();
    std::fs::remove_dir_all(&dir).ok();

    // 1 and 2 started together; 3 never did; 4 started once 3 was blocked.
    assert_eq!(order.len(), 3, "{order:?}");
    assert!(
        order.starts_with(&[0, 1]) || order.starts_with(&[1, 0]),
        "{order:?}"
    );
    let second = first
        .get(1)
        .map(|(_, states)| states.clone())
        .unwrap_or_default();
    assert!(
        second.iter().filter(|state| state.active()).count() == 2,
        "{second:?}"
    );
    let at_four = first.iter().find(|(index, _)| *index == 3);
    let three_then = at_four.and_then(|(_, states)| states.get(2));
    assert!(
        matches!(three_then, Some(ItemState::Blocked(_))),
        "{first:?}"
    );
    assert!(
        item_one_asked.contains("We move the store"),
        "{item_one_asked}"
    );
    assert!(
        item_one_asked.contains("keep the old reader"),
        "{item_one_asked}"
    );

    assert!(file.contains("- [x] one\n  > did one\n"), "{file}");
    assert!(
        file.contains("- [!] two (parallel)\n  > failed: no access\n"),
        "{file}"
    );
    assert!(
        file.contains("- [ ] three (after: 1, 2)\n- [x] four\n  > did four\n"),
        "{file}"
    );
    assert!(file.ends_with(&format!("{USER_LINE}\n")), "{file}");
    assert!(
        told.contains("Failed") && told.contains("no access"),
        "{told}"
    );
    assert!(told.contains("Blocked") && told.contains("three"), "{told}");

    // Only 2 and 3 ran again, and every box is ticked.
    assert_eq!(rerun, [1, 2]);
    assert!(
        after.contains("- [x] two (parallel)\n  > did two\n"),
        "{after}"
    );
    assert!(
        after.contains("- [x] three (after: 1, 2)\n  > did three\n"),
        "{after}"
    );
    assert!(!after.contains("no access"), "{after}");
    assert!(after.contains(USER_LINE), "{after}");
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ChecklistRunnerPlugin);
}

/// The view half shows a panel per run the agent half published, and
/// drops one for a run that is gone.
#[cfg(feature = "tui")]
#[test]
fn the_view_has_a_panel_per_run() {
    use rig_harness::view::mirror;
    use rig_harness::view::protocol::{EventRecord, Seq, SessionId};

    let mut view = App::new();
    view.add_plugins((rig_tui::TuiPlugin, ChecklistRunnerView));
    view.finish();
    let run = |title: &str| RunShown {
        title: title.to_owned(),
        items: vec!["1. one\n   waiting".to_owned()],
    };
    let record = |seq, runs: Vec<RunShown>| EventRecord {
        session: SessionId("s-1".into()),
        seq: Seq(seq),
        at_ms: 0,
        source: None,
        kind: PlanRuns::KIND.to_owned(),
        data: serde_json::to_value(PlanRuns { runs }).unwrap_or_default(),
    };
    let panels = |view: &mut App| {
        let world = view.world_mut();
        let mut panels = world.query_filtered::<(), With<rig_tui::TuiPanel>>();
        panels.iter(world).count()
    };
    // The screen's own panels, before any run's.
    let screen = panels(&mut view);
    mirror::apply(
        view.world_mut(),
        [record(
            1,
            vec![run("a.md (running)"), run("b.md (running)")],
        )],
    );
    view.update();
    view.update();
    assert_eq!(panels(&mut view), screen + 2);
    mirror::apply(view.world_mut(), [record(2, vec![run("a.md (finished)")])]);
    view.update();
    view.update();
    assert_eq!(panels(&mut view), screen + 1);
}

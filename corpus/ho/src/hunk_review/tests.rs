use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use rig_basics::coding_tools::EditTool;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::pick::PickPlugin;
use rig_harness_test_support::{
    Collected, Ended, MemoryStore, app_on, calls, collect, connect, first_agent, log_ended, reply,
    run_until, scratch_home,
};
use serde_json::json;

use super::*;

/// The pickers open.
fn pickers(app: &mut App) -> Vec<Entity> {
    let mut pickers = app.world_mut().query_filtered::<Entity, With<Picker>>();
    pickers.iter(app.world()).collect()
}

/// An app whose agent's turn edited line 1 and line 3 of a 3-line `a.txt`
/// in two `edit` calls and ended, with a front that answers questions;
/// then the model answers `Noted.`.
fn edited(name: &str) -> (App, Receiver<()>, Entity, MockCompletionModel, PathBuf) {
    let dir = scratch_home(name).root().to_owned();
    std::fs::create_dir_all(&dir).ok();
    let file = dir.join("a.txt");
    std::fs::write(&file, "one\ntwo\nthree\n").ok();
    let path = file.display().to_string();
    let edit = |id: &str, old: &str, new: &str| {
        let edits = json!([{ "old_text": old, "new_text": new }]);
        calls(id, "edit", json!({ "path": path, "edits": edits }))
    };
    let model = MockCompletionModel::from_stream_turns([
        edit("e1", "one", "ONE"),
        edit("e3", "three", "THREE"),
        reply("Done."),
        reply("Noted."),
    ]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.add_plugins((
        AgentPlugin,
        SessionLogPlugin,
        PickPlugin,
        EditTool,
        HunkReviewPlugin,
    ))
    .init_resource::<Ended>()
    .add_observer(log_ended);
    collect::<Notice>(&mut app);
    app.world_mut().spawn(Answerer::default());
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(&model) {
        world.entity_mut(agent).insert(connection);
    }
    world.trigger(Deliver::new(
        agent,
        "Shout lines 1 and 3.",
        DeliveryMode::Steer,
    ));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    (app, wakes, agent, model, file)
}

/// The decisions on the agent's hunks, in order.
fn decisions(world: &World, agent: Entity) -> Vec<Option<Decision>> {
    let review = world.get::<HunkReview>(agent);
    review.map_or_else(Vec::new, |review| {
        numbered(review)
            .map(|(.., hunk)| hunk.decision.clone())
            .collect()
    })
}

#[test]
fn a_reverted_hunk_is_put_back_and_the_agent_told_why() {
    let (mut app, wakes, agent, model, file) = edited("hunk-review");
    assert_eq!(decisions(app.world(), agent), vec![None, None]);
    assert_eq!(pickers(&mut app).len(), 1);

    for line in [
        "review keep 1",
        "review revert 2 keep the header",
        "review done",
    ] {
        app.world_mut().trigger(RunCommand::new(agent, line));
        app.update();
    }
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 2
    });

    let text = std::fs::read_to_string(&file).unwrap_or_default();
    assert_eq!(text, "ONE\ntwo\nthree\n");
    let requests = model.requests();
    let last = format!("{:?}", requests.last());
    assert!(last.contains("reverted"), "{last}");
    assert!(last.contains("keep the header"), "{last}");
}

#[test]
fn a_hunk_changed_since_the_turn_is_a_conflict() {
    let (mut app, _wakes, agent, _model, file) = edited("hunk-review-conflict");
    std::fs::write(&file, "ONE\ntwo\nthree, by the test\n").ok();
    app.world_mut()
        .trigger(RunCommand::new(agent, "review revert 2"));
    app.update();

    let text = std::fs::read_to_string(&file).unwrap_or_default();
    assert_eq!(text, "ONE\ntwo\nthree, by the test\n");
    assert_eq!(
        decisions(app.world(), agent),
        vec![None, Some(Decision::Conflict)]
    );
    let notices = &app.world().resource::<Collected<Notice>>().0;
    assert!(
        notices
            .iter()
            .any(|notice| notice.text.contains("changed since the turn ended"))
    );

    // Esc defers the review; `/review` reopens it.
    for picker in pickers(&mut app) {
        app.world_mut().trigger(Pick::person(picker, None));
    }
    app.update();
    assert!(pickers(&mut app).is_empty());
    app.world_mut().trigger(RunCommand::new(agent, "review"));
    app.update();
    assert_eq!(pickers(&mut app).len(), 1);
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(HunkReviewPlugin);
}

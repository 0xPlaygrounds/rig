use std::path::PathBuf;

use rig_cassette::journal::MemoryStore;
use rig_core::test_utils::MockCompletionModel;
use serde_json::json;

use super::*;
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};

/// The turns that ended.
#[derive(Resource, Default)]
struct Ended(usize);

/// A skill directory beside the test binary (never the system's
/// temporary directory): `release`, with a notes file its body links to,
/// and a skill whose frontmatter is broken.
fn skills() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let root = exe
        .parent()?
        .join(format!("corpus-skills-{}", std::process::id()));
    let release = root.join("release");
    let broken = root.join("broken");
    std::fs::create_dir_all(&release).ok()?;
    std::fs::create_dir_all(&broken).ok()?;
    let skill = "---\nname: release\ndescription: How to cut a release\n---\n\
                 Run ./release.sh, after reading [the notes](notes.md).\n";
    std::fs::write(release.join("SKILL.md"), skill).ok()?;
    std::fs::write(release.join("notes.md"), "Tag it first.").ok()?;
    std::fs::write(broken.join("SKILL.md"), "no frontmatter here").ok()?;
    Some(root)
}

#[test]
fn a_skill_is_listed_then_loaded_once_per_turn() {
    let root = skills();
    assert!(root.is_some(), "the skill directory is written");
    let Some(root) = root else {
        return;
    };
    let model = MockCompletionModel::from_stream_turns([
        calls("1", LOAD_SKILL, json!({ "name": "release" })),
        calls("2", LOAD_SKILL, json!({ "name": "release" })),
        reply("done"),
    ]);
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    app.set_error_handler(bevy_ecs::error::warn)
        .insert_resource(SkillRoots(vec![root.clone()]))
        .add_plugins((AgentPlugin, JournalPlugin, SkillsPlugin))
        .init_resource::<Ended>()
        .add_observer(|_: On<TurnEnded>, mut ended: ResMut<Ended>| ended.0 += 1);
    let agent = connect(&model).map(|connection| app.world_mut().spawn((Agent, connection)).id());
    app.finish();
    app.update();
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    let ask = Deliver::new(agent, "Ship version 2.", DeliveryMode::Steer);
    app.world_mut().trigger(ask);
    run_until(&mut app, &wakes, |world| world.resource::<Ended>().0 > 0);

    let sent: Vec<String> = model
        .requests()
        .iter()
        .map(|request| serde_json::to_string(request).unwrap_or_default())
        .collect();
    assert_eq!(sent.len(), 3, "{sent:?}");
    let [first, second, third] = sent.as_slice() else {
        return;
    };
    assert!(first.contains(HEADING), "{first}");
    assert!(first.contains("- release: How to cut a release"), "{first}");
    assert!(!first.contains("Run ./release.sh"), "{first}");
    assert!(
        !first.contains("broken"),
        "a broken skill is skipped: {first}"
    );
    assert!(second.contains("Run ./release.sh"), "{second}");
    let notes = root.join("release").join("notes.md");
    let linked = format!("[the notes]({})", notes.display());
    assert!(second.contains(&linked), "the link is resolved: {second}");
    assert!(third.contains("already loaded above"), "{third}");
    assert_eq!(third.matches("Run ./release.sh").count(), 1, "{third}");
    let loaded = app.world().get::<LoadedSkills>(agent);
    assert_eq!(
        loaded.map(|loaded| loaded.session.clone()),
        Some(vec!["release".to_owned()])
    );
    assert!(loaded.is_some_and(|loaded| loaded.turn.is_empty()));
    std::fs::remove_dir_all(root).ok();
}

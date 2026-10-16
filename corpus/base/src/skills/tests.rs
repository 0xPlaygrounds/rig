use std::path::Path;

use rig_core::test_utils::MockCompletionModel;
use rig_harness::watch::WatchPlugin;
use rig_harness_test_support::MemoryStore;
use serde_json::json;

use super::*;
use rig_harness_test_support::{
    Ended, app_on, calls, connect, log_ended, reply, run_until, scratch_home,
};

/// Writes a skill `name` under `skills` with `description` and `body`.
fn write_skill(skills: &Path, name: &str, description: &str, body: &str) -> Option<()> {
    let dir = skills.join(name);
    std::fs::create_dir_all(&dir).ok()?;
    let text = format!("---\nname: {name}\ndescription: {description}\n---\n{body}\n");
    std::fs::write(dir.join("SKILL.md"), text).ok()
}

/// A project in `root` whose `.rig/skills` has `release`, with a notes
/// file its body links to, and a skill whose frontmatter is broken.
fn project(root: &Path) -> Option<()> {
    let skills = root.join(".rig").join("skills");
    let body = "Run ./release.sh, after reading [the notes](notes.md).";
    write_skill(&skills, "release", "How to cut a release", body)?;
    std::fs::write(skills.join("release").join("notes.md"), "Tag it first.").ok()?;
    std::fs::create_dir_all(skills.join("broken")).ok()?;
    std::fs::write(skills.join("broken").join("SKILL.md"), "no frontmatter").ok()
}

/// The skill list in the system prompt.
fn listing(world: &World) -> String {
    let sections = world.try_query::<&PromptSection>();
    let mut texts = sections.iter().flat_map(|query| query.iter_manual(world));
    texts
        .find(|section| section.tag == "skills")
        .map(|section| section.text.clone())
        .unwrap_or_default()
}

#[test]
fn a_skill_is_listed_loaded_once_per_turn_and_watched() {
    let home = scratch_home("corpus-skills");
    let root = home.root().to_path_buf();
    assert!(project(&root).is_some(), "the project is written");
    let model = MockCompletionModel::from_stream_turns([
        calls("1", LOAD_SKILL, json!({ "name": "release" })),
        calls("2", LOAD_SKILL, json!({ "name": "release" })),
        reply("done"),
        calls("3", LOAD_SKILL, json!({ "name": "release" })),
        reply("shipped"),
    ]);
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    app.set_error_handler(rig_harness::error::warn)
        .insert_resource(ProjectRoot(root.clone()))
        .insert_resource(RigHome(home))
        .add_plugins((AgentPlugin, SessionLogPlugin, WatchPlugin, SkillsPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    let agent = connect(&model).map(|connection| app.world_mut().spawn((Agent, connection)).id());
    app.finish();
    app.update();
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    let ask = Deliver::new(agent, "Ship version 2.", DeliveryMode::Steer);
    app.world_mut().trigger(ask);
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });

    // `/skill` sends the body with the user's text, in the turn it starts.
    let typed = Deliver::new(agent, "/skill release Ship it.", DeliveryMode::Steer);
    app.world_mut().trigger(typed);
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() > 1
    });

    let sent: Vec<String> = model
        .requests()
        .iter()
        .map(|request| serde_json::to_string(request).unwrap_or_default())
        .collect();
    assert_eq!(sent.len(), 5, "{sent:?}");
    let [first, second, third, fourth, fifth] = sent.as_slice() else {
        return;
    };
    assert!(first.contains(HEADING), "{first}");
    assert!(first.contains("- release: How to cut a release"), "{first}");
    assert!(!first.contains("Run ./release.sh"), "{first}");
    assert!(!first.contains("broken"), "a broken skill is skipped");
    assert!(second.contains("Run ./release.sh"), "{second}");
    let notes = root.join(".rig/skills/release/notes.md");
    let linked = format!("[the notes]({})", notes.display());
    assert!(second.contains(&linked), "the link is resolved: {second}");
    assert!(third.contains("already loaded above"), "{third}");
    assert_eq!(third.matches("Run ./release.sh").count(), 1, "{third}");
    assert!(fourth.contains("Ship it."), "{fourth}");
    assert_eq!(fourth.matches("Run ./release.sh").count(), 2, "{fourth}");
    let again = fifth.matches("already loaded above").count();
    assert_eq!(again, 2, "/skill loaded it in its turn: {fifth}");
    let loaded = app.world().get::<LoadedSkills>(agent);
    assert_eq!(
        loaded.map(|loaded| loaded.0.clone()),
        Some(vec!["release".to_owned()])
    );

    // A skill written now is listed once the watch sees it; one removed goes.
    let skills = root.join(".rig").join("skills");
    assert!(write_skill(&skills, "pdf", "PDF handling", "Use pdftotext.").is_some());
    std::fs::remove_dir_all(skills.join("release")).ok();
    run_until(&mut app, &wakes, |world| listing(world).contains("pdf"));
    let now = listing(app.world());
    assert!(now.contains("- pdf: PDF handling"), "{now}");
    assert!(!now.contains("release"), "{now}");
    std::fs::remove_dir_all(root).ok();
}

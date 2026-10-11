use std::path::{Path, PathBuf};

use rig_cassette::journal::MemoryStore;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::watch::WatchPlugin;
use serde_json::json;

use super::*;
use rig_harness_test_support::{Ended, app_on, calls, connect, log_ended, reply, run_until};

/// Writes a skill `name` under `skills` with `description` and `body`.
fn write_skill(skills: &Path, name: &str, description: &str, body: &str) -> Option<()> {
    let dir = skills.join(name);
    std::fs::create_dir_all(&dir).ok()?;
    let text = format!("---\nname: {name}\ndescription: {description}\n---\n{body}\n");
    std::fs::write(dir.join("SKILL.md"), text).ok()
}

/// A project beside the test binary (never the system's temporary
/// directory) whose `.rig/skills` has `release`, with a notes file its body
/// links to, and a skill whose frontmatter is broken.
fn project() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let root = exe
        .parent()?
        .join(format!("corpus-skills-{}", std::process::id()));
    let skills = root.join(".rig").join("skills");
    let body = "Run ./release.sh, after reading [the notes](notes.md).";
    write_skill(&skills, "release", "How to cut a release", body)?;
    std::fs::write(skills.join("release").join("notes.md"), "Tag it first.").ok()?;
    std::fs::create_dir_all(skills.join("broken")).ok()?;
    std::fs::write(skills.join("broken").join("SKILL.md"), "no frontmatter").ok()?;
    Some(root)
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
    let root = project();
    assert!(root.is_some(), "the project is written");
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
    app.set_error_handler(rig_harness::error::warn)
        .insert_resource(ProjectRoot(root.clone()))
        .insert_resource(RigHome(Home::at(root.join("home"))))
        .add_plugins((AgentPlugin, JournalPlugin, WatchPlugin, SkillsPlugin))
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
    assert!(!first.contains("broken"), "a broken skill is skipped");
    assert!(second.contains("Run ./release.sh"), "{second}");
    let notes = root.join(".rig/skills/release/notes.md");
    let linked = format!("[the notes]({})", notes.display());
    assert!(second.contains(&linked), "the link is resolved: {second}");
    assert!(third.contains("already loaded above"), "{third}");
    assert_eq!(third.matches("Run ./release.sh").count(), 1, "{third}");
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

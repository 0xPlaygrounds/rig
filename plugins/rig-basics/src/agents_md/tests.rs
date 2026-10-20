use std::path::{Path, PathBuf};

use rig_core::test_utils::MockCompletionModel;
use rig_ecs::prelude::*;
use rig_harness::harness_protocol::Home;
use rig_harness::services::{ProjectRoot, RigHome};
use rig_harness_test_support::{
    Ended, MemoryStore, app_on, connect, log_ended, reply, run_until, scratch_home,
};

use super::{AgentsMdPlugin, instruction_files};
use crate::prompt::DefaultPromptPlugin;

/// A user's home and a repository inside a directory that has an
/// `AGENTS.md` of its own, with `AGENTS.md` at the repository root and in
/// `a/b`, but none in `a`: the home, and the working directory `a/b`.
fn project(name: &str) -> Option<(PathBuf, PathBuf)> {
    let scratch = scratch_home(name).root().to_owned();
    let (home, outside, repository) = (
        scratch.join("home"),
        scratch.join("work"),
        scratch.join("work/repo"),
    );
    let dir = repository.join("a/b");
    std::fs::create_dir_all(&home).ok()?;
    std::fs::create_dir_all(repository.join(".git")).ok()?;
    std::fs::create_dir_all(&dir).ok()?;
    let files: [(&Path, &str); 4] = [
        (&home, "The user's own."),
        (&outside, "Above the repository."),
        (&repository, "The repository's."),
        (&dir, "The working directory's."),
    ];
    for (place, text) in files {
        std::fs::write(place.join("AGENTS.md"), text).ok()?;
    }
    Some((home, dir))
}

#[test]
fn the_files_run_from_the_users_to_the_nearest_and_stop_at_the_repository() {
    let made = project("agents-md-files");
    assert!(made.is_some(), "a scratch project");
    let Some((home, dir)) = made else {
        return;
    };
    let repository = dir.join("../..");
    let found = instruction_files(&dir, &home);
    let expected = [
        home.join("AGENTS.md"),
        repository.join("AGENTS.md"),
        dir.join("AGENTS.md"),
    ];
    let same = |a: &PathBuf, b: &PathBuf| a.canonicalize().ok() == b.canonicalize().ok();
    let matches =
        found.len() == expected.len() && found.iter().zip(&expected).all(|(a, b)| same(a, b));
    assert!(matches, "{found:?}");
}

/// The files are in the system prompt of every request, in order; the
/// default prompt names no instruction file.
#[test]
fn the_files_are_in_the_system_prompt() {
    let made = project("agents-md-prompt");
    assert!(made.is_some(), "a scratch project");
    let Some((home, dir)) = made else {
        return;
    };
    let model = MockCompletionModel::from_stream_turns([reply("1")]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(ProjectRoot(dir))
        .insert_resource(RigHome(Home::at(home)))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            DefaultPromptPlugin,
            AgentsMdPlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let Some(connection) = connect(&model) else {
        return;
    };
    let agent = app.world_mut().spawn((Agent, connection)).id();
    app.world_mut()
        .trigger(Deliver::new(agent, "hello", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    let prompt = model
        .requests()
        .first()
        .and_then(|request| request.system_instructions().map(str::to_owned))
        .unwrap_or_default();
    let at = |text: &str| prompt.find(text);
    let order = [
        at("The user's own."),
        at("The repository's."),
        at("The working directory's."),
    ];
    assert!(order.iter().all(Option::is_some), "{prompt}");
    assert!(order.is_sorted(), "{prompt}");
    assert!(!prompt.contains("Above the repository."), "{prompt}");
    assert!(prompt.contains("<project_instructions>"), "{prompt}");
}

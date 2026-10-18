use rig_core::test_utils::MockCompletionModel;
use rig_ecs::prelude::*;
use rig_harness::services::ProjectRoot;
use rig_harness_test_support::{Ended, MemoryStore, app_on, connect, log_ended, reply, run_until};

use rig_harness::worker::Unattended;

use super::{DEFAULT_SYSTEM_PROMPT, DefaultPromptPlugin, UNATTENDED_SYSTEM_PROMPT};

/// Every request of an agent without a prompt of its own carries rig's
/// with the project's `AGENTS.md` as it was at the first request, turn
/// after turn, through an edit; an agent with its own keeps it.
#[test]
fn the_prompt_holds_the_instructions_read_once_across_an_agents_md_edit() {
    let told = prompts(false);
    let rig = format!(
        "{DEFAULT_SYSTEM_PROMPT}\n\n# The project's instructions\n\n## {}\n\nUse tabs.",
        told.1.join("AGENTS.md").display()
    );
    let expected = [rig.as_str(), &rig, "You review code."];
    assert_eq!(told.0, expected.map(|prompt| Some(prompt.to_owned())));
    assert!(!rig.contains("read the project's instructions"));
}

/// A run nobody attends tells the model nobody replies, and not to ask.
#[test]
fn an_unattended_run_is_told_nobody_replies() {
    let (told, _) = prompts(true);
    let first = told.first().cloned().flatten().unwrap_or_default();
    assert!(first.starts_with(UNATTENDED_SYSTEM_PROMPT), "{first}");
    assert!(first.contains("Nobody will reply during this run"));
    assert!(!first.contains("Ask when the request is unclear"));
    assert!(first.contains("Use tabs."));
}

/// The system prompts of three requests: two of rig's agent around an
/// edit of `AGENTS.md`, one of an agent with its own; and the project.
fn prompts(unattended: bool) -> (Vec<Option<String>>, std::path::PathBuf) {
    let root = std::env::temp_dir().join(format!(
        "rig-basics-prompt-{}-{unattended}",
        std::process::id()
    ));
    let agents_md = root.join("AGENTS.md");
    let written = std::fs::create_dir_all(&root)
        .and_then(|()| std::fs::write(&agents_md, "Use tabs."))
        .is_ok();
    assert!(written, "a scratch project in {}", root.display());
    let model = MockCompletionModel::from_stream_turns([reply("1"), reply("2"), reply("3")]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(ProjectRoot(root.clone()))
        .insert_resource(Unattended(unattended))
        .add_plugins((AgentPlugin, SessionLogPlugin, DefaultPromptPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let Some(connection) = connect(&model) else {
        return (Vec::new(), root);
    };
    let ask = |app: &mut App, agent: Entity| {
        let world = app.world_mut();
        world.trigger(Deliver::new(agent, "hello", DeliveryMode::Steer));
        let ended = world.resource::<Ended>().0.len() + 1;
        run_until(app, &wakes, |world| {
            world.resource::<Ended>().0.len() == ended
        });
    };
    let rig = app.world_mut().spawn((Agent, connection.clone())).id();
    ask(&mut app, rig);
    let edited = std::fs::write(&agents_md, "Use spaces, never tabs.").is_ok();
    assert!(edited);
    ask(&mut app, rig);
    let reviewer = SystemPrompt("You review code.".to_owned());
    let reviewer = app.world_mut().spawn((Agent, reviewer, connection)).id();
    ask(&mut app, reviewer);
    std::fs::remove_dir_all(&root).ok();
    let prompts: Vec<Option<String>> = model
        .requests()
        .iter()
        .map(|request| request.system_instructions().map(str::to_owned))
        .collect();
    (prompts, root)
}

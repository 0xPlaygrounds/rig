use rig_core::test_utils::MockCompletionModel;
use rig_ecs::prelude::*;
use rig_harness::services::ProjectRoot;
use rig_harness_test_support::{Ended, MemoryStore, app_on, connect, log_ended, reply, run_until};

use super::{DEFAULT_SYSTEM_PROMPT, DefaultPromptPlugin};

/// Every request of an agent without a prompt of its own carries exactly
/// rig's, turn after turn, whatever the project's `AGENTS.md` says; an
/// agent with its own keeps it.
#[test]
fn the_prompt_is_the_constant_on_every_request_across_an_agents_md_edit() {
    let root = std::env::temp_dir().join(format!("rig-basics-prompt-{}", std::process::id()));
    let agents_md = root.join("AGENTS.md");
    let written = std::fs::create_dir_all(&root)
        .and_then(|()| std::fs::write(&agents_md, "Use tabs."))
        .is_ok();
    assert!(written, "a scratch project in {}", root.display());
    let model = MockCompletionModel::from_stream_turns([reply("1"), reply("2"), reply("3")]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(ProjectRoot(root.clone()))
        .add_plugins((AgentPlugin, SessionLogPlugin, DefaultPromptPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let Some(connection) = connect(&model) else {
        return;
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
    let told = [
        DEFAULT_SYSTEM_PROMPT,
        DEFAULT_SYSTEM_PROMPT,
        "You review code.",
    ];
    assert_eq!(prompts, told.map(|prompt| Some(prompt.to_owned())));
}

/// Instruction files are [`AgentsMdPlugin`](crate::agents_md::AgentsMdPlugin)'s:
/// without it the prompt names none.
#[test]
fn the_prompt_names_no_instruction_file() {
    for name in ["AGENTS.md", "CLAUDE.md"] {
        assert!(!DEFAULT_SYSTEM_PROMPT.contains(name), "{name}");
    }
}

use rig_cassette::journal::MemoryStore;
use rig_core::test_utils::MockCompletionModel;
use rig_ecs::prelude::*;
use rig_harness_test_support::{Ended, app_on, connect, log_ended, reply, run_until};

use rig_harness::harness_protocol::Home;

use super::{
    ContextEntry, DEFAULT_SYSTEM_PROMPT, ProjectContextPlugin, ProjectRoot, RigHome, default_prompt,
};

#[test]
fn an_agent_without_a_prompt_of_its_own_is_told_it_is_rig() {
    let model = MockCompletionModel::from_stream_turns([reply("hi"), reply("hello")]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.add_plugins((AgentPlugin, JournalPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .add_systems(AgentPass, default_prompt.in_set(Stage::Prepare));
    app.finish();
    app.update();
    let Some(connection) = connect(&model) else {
        return;
    };
    let reviewer = SystemPrompt("You review code.".to_owned());
    for prompt in [SystemPrompt::default(), reviewer] {
        let world = app.world_mut();
        let agent = world.spawn((Agent, prompt, connection.clone())).id();
        world.trigger(Deliver::new(agent, "hello", DeliveryMode::Steer));
        let ended = world.resource::<Ended>().0.len() + 1;
        run_until(&mut app, &wakes, |world| {
            world.resource::<Ended>().0.len() == ended
        });
    }
    let prompts: Vec<Option<String>> = model
        .requests()
        .iter()
        .map(|request| request.system_instructions().map(str::to_owned))
        .collect();
    let told = [DEFAULT_SYSTEM_PROMPT, "You review code."].map(|prompt| Some(prompt.to_owned()));
    assert_eq!(prompts, told);
}

#[test]
fn context_lists_what_other_plugins_put_in_the_context() {
    let root = std::env::temp_dir().join(format!("rig-basics-{}", std::process::id()));
    let mut app = App::new();
    app.insert_resource(RigHome(Home::at(root.join("home"))))
        .insert_resource(ProjectRoot(root.clone()))
        .add_plugins((AgentPlugin, ProjectContextPlugin));
    let world = app.world_mut();
    world.spawn(ContextEntry::new("nested rules", "src/AGENTS.md"));
    world.spawn(ContextEntry::new("unused", ""));
    let agent = world.spawn(Agent).id();
    world.trigger(RunCommand {
        entity: agent,
        line: "context".to_owned(),
    });
    world.flush();
    let mut notices = world.resource_mut::<Messages<Notice>>();
    let shown: Vec<String> = notices.drain().map(|notice| notice.text).collect();
    let shown = shown.join("\n");
    assert!(shown.ends_with("\nnested rules: src/AGENTS.md"), "{shown}");
    assert!(!shown.contains("unused"), "{shown}");
}

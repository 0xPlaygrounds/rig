use rig_core::test_utils::MockCompletionModel;
use rig_harness::settings::Settings;
use rig_harness_test_support::MemoryStore;

use super::*;
use rig_harness_test_support::{app_on, reply, run_until, scripted_models};

#[test]
fn debaters_answer_each_other_and_the_judges_verdict_reaches_the_agent() {
    let main = MockCompletionModel::from_stream_turns([reply("unused")]);
    let pro = MockCompletionModel::from_stream_turns([reply("P1"), reply("P2")]);
    let con = MockCompletionModel::from_stream_turns([reply("C1"), reply("C2")]);
    let judge = MockCompletionModel::from_stream_turns([reply("Verdict: P")]);
    let models = scripted_models([
        ("debate-main/m", &main),
        ("debate-pro/m", &pro),
        ("debate-con/m", &con),
        ("debate-judge/m", &judge),
    ]);

    let (mut app, wakes) = app_on(&MemoryStore::default());
    let settings = "[debate]\npro = \"debate-pro/m\"\ncon = \"debate-con/m\"\n\
                    judge = \"debate-judge/m\"\n";
    app.insert_resource(models)
        .insert_resource(Settings::from_toml(settings))
        .add_plugins((AgentPlugin, SessionLogPlugin, DebatePlugin));
    let host = app
        .world_mut()
        .spawn((Agent, ModelChoice("debate-main/m".to_owned())))
        .id();
    app.finish();
    app.update();

    let line = "debate --rounds 2 Should the parser be hand-written?".to_owned();
    app.world_mut().trigger(RunCommand::new(host, line));
    let said = |world: &World| {
        let conversation = world.get::<Conversation>(host);
        format!(
            "{:?}",
            conversation.map(|conversation| conversation.messages())
        )
    };
    run_until(&mut app, &wakes, |world| {
        !world.entity(host).contains::<Debate>() && said(world).contains("Verdict: P")
    });

    let asked = |model: &MockCompletionModel, at: usize| format!("{:?}", model.requests().get(at));
    assert!(asked(&pro, 1).contains("C1"), "{}", asked(&pro, 1));
    assert!(asked(&con, 1).contains("P1"), "{}", asked(&con, 1));
    let judged = asked(&judge, 0);
    for answer in ["P1", "P2", "C1", "C2"] {
        assert!(judged.contains(answer), "{judged}");
    }
    assert!(said(app.world()).contains("Verdict: P"));
    assert_eq!(main.request_count(), 0, "a note starts no turn");
    let world = app.world_mut();
    let mut debaters = world.query::<(&Debater, &SpawnedBy)>();
    let spawned: Vec<Entity> = debaters.iter(world).map(|(_, by)| by.0).collect();
    assert_eq!(spawned, [host; 3], "three debaters, spawned by the agent");
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(DebatePlugin);
}

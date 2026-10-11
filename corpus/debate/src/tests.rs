use rig_cassette::journal::MemoryStore;
use rig_core::test_utils::MockCompletionModel;

use super::*;
use rig_harness_test_support::{app_on, connect, reply, run_until};

#[test]
fn debaters_answer_each_other_and_the_judges_verdict_reaches_the_agent() {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.add_plugins((AgentPlugin, JournalPlugin, DebatePlugin))
        .insert_resource(DebateConfig {
            pro: Some("openai/gpt-5.1".to_owned()),
            con: Some("anthropic/claude-sonnet-4-5".to_owned()),
            judge: Some("deepseek/deepseek-flash".to_owned()),
            rounds: None,
        });
    let main = MockCompletionModel::from_stream_turns([reply("unused")]);
    let host = connect(&main).map(|connection| {
        let choice = ModelChoice(connection.spec.reference());
        app.world_mut().spawn((Agent, connection, choice)).id()
    });
    let host = host.unwrap_or(Entity::PLACEHOLDER);
    app.finish();
    app.update();

    let line = "debate --rounds 2 Should the parser be hand-written?".to_owned();
    app.world_mut().trigger(RunCommand { entity: host, line });
    app.world_mut().flush();
    let debate = app.world().get::<Debate>(host).cloned();
    assert!(debate.is_some(), "no debate started");
    let Some(debate) = debate else {
        return;
    };
    // Each part answers by its own scripted model.
    let pro = MockCompletionModel::from_stream_turns([reply("P1"), reply("P2")]);
    let con = MockCompletionModel::from_stream_turns([reply("C1"), reply("C2")]);
    let judge = MockCompletionModel::from_stream_turns([reply("Verdict: P")]);
    for (agent, model) in [
        (debate.pro, &pro),
        (debate.con, &con),
        (debate.judge, &judge),
    ] {
        if let Some(connection) = connect(model) {
            app.world_mut().entity_mut(agent).insert(connection);
        }
    }
    let said = |world: &World| {
        let conversation = world.get::<Conversation>(host);
        format!("{:?}", conversation.map(Conversation::messages))
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
    let spawned = app.world().get::<Spawned>(host);
    let spawned: Vec<Entity> = spawned.map(|of| of.iter().collect()).unwrap_or_default();
    assert_eq!(spawned, debate.debaters());
}

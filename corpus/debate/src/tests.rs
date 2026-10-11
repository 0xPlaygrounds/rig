use rig_cassette::journal::MemoryStore;
use rig_core::catalog::{Catalog, ModelSpec};
use rig_core::operation::Completion;
use rig_core::providers::registry::ProviderId;
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ModelAdapter;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::settings::Settings;

use super::*;
use rig_harness_test_support::{app_on, reply, run_until};

/// Adds `model` to `models` as the catalog model `<vendor>/m`, which
/// connects to it.
fn scripted(models: &mut Models, vendor: &str, model: &MockCompletionModel) -> Option<()> {
    let vendor = ProviderId::declare(vendor).ok()?;
    let model = model.clone();
    models
        .0
        .catalog_mut()
        .insert(ModelSpec::new(vendor, "m").with_tools(true));
    models
        .0
        .provide(vendor, move |spec: &ModelSpec, _: &Catalog| {
            let model = ModelAdapter::<Completion>::new(spec.reference(), model.clone());
            Ok(ErasedHandler::new(model))
        });
    Some(())
}

#[test]
fn debaters_answer_each_other_and_the_judges_verdict_reaches_the_agent() {
    let main = MockCompletionModel::from_stream_turns([reply("unused")]);
    let pro = MockCompletionModel::from_stream_turns([reply("P1"), reply("P2")]);
    let con = MockCompletionModel::from_stream_turns([reply("C1"), reply("C2")]);
    let judge = MockCompletionModel::from_stream_turns([reply("Verdict: P")]);
    let mut models = Models::default();
    let parts = [
        ("debate-main", &main),
        ("debate-pro", &pro),
        ("debate-con", &con),
        ("debate-judge", &judge),
    ];
    let listed = parts
        .iter()
        .all(|(vendor, model)| scripted(&mut models, vendor, model).is_some());
    assert!(listed, "the scripted models are in the catalog");

    let (mut app, wakes) = app_on(&MemoryStore::default());
    let settings = "[debate]\npro = \"debate-pro/m\"\ncon = \"debate-con/m\"\n\
                    judge = \"debate-judge/m\"\n";
    app.insert_resource(models)
        .insert_resource(Settings::from_toml(settings))
        .add_plugins((AgentPlugin, JournalPlugin, DebatePlugin));
    let host = app
        .world_mut()
        .spawn((Agent, ModelChoice("debate-main/m".to_owned())))
        .id();
    app.finish();
    app.update();

    let line = "debate --rounds 2 Should the parser be hand-written?".to_owned();
    app.world_mut().trigger(RunCommand { entity: host, line });
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
    let world = app.world_mut();
    let mut debaters = world.query::<(&Debater, &SpawnedBy)>();
    let spawned: Vec<Entity> = debaters.iter(world).map(|(_, by)| by.0).collect();
    assert_eq!(spawned, [host; 3], "three debaters, spawned by the agent");
}

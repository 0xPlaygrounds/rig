use http::{HeaderMap, HeaderValue, StatusCode};
use rig_core::ProviderResponseError;
use rig_core::operation::Completion;
use rig_core::serve::adapters::ModelAdapter;
use rig_core::test_utils::{MockCompletionModel, MockError, MockStreamEvent};
use rig_harness::harness_protocol::SessionId;
use rig_harness_test_support::{
    Collected, Ended, app_in, collect, first_agent, log_ended, reply, run_until, scratch_home,
    scripted_models,
};

use super::*;

/// The three keys' values, which must never show.
const SECRETS: [&str; 3] = ["sk-personal-1111", "sk-team-2222", "sk-spare-3333"];

/// A reply that fails as the provider answered `status`, with `headers`.
fn failing(status: StatusCode, headers: HeaderMap) -> Vec<MockStreamEvent> {
    let body = r#"{"error":{"message":"no"}}"#;
    let error = ProviderResponseError::new(status, body).with_headers(Some(headers));
    vec![MockStreamEvent::Error(MockError::ProviderResponse(error))]
}

/// Runs frames until `turns` turns ended, then one more for the journal.
fn until_ended(app: &mut App, wakes: &std::sync::mpsc::Receiver<()>, turns: usize) {
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() >= turns
    });
    app.update();
}

/// The texts of the notices so far.
fn notices(app: &App) -> Vec<(NoticeLevel, String)> {
    let notices = &app.world().resource::<Collected<Notice>>().0;
    notices
        .iter()
        .map(|notice| (notice.level, notice.text.clone()))
        .collect()
}

#[test]
fn a_rate_limited_and_a_refused_key_fail_over_to_the_spare() {
    // Key 1 is rate-limited for 30 s, key 2 refused, key 3 answers.
    let mut retry_after = HeaderMap::new();
    retry_after.insert("retry-after", HeaderValue::from_static("30"));
    let personal = MockCompletionModel::from_stream_turns([failing(
        StatusCode::TOO_MANY_REQUESTS,
        retry_after,
    )]);
    let team = MockCompletionModel::from_stream_turns([failing(
        StatusCode::UNAUTHORIZED,
        HeaderMap::new(),
    )]);
    let spare = MockCompletionModel::from_stream_turns([reply("From the spare."), reply("Again.")]);

    let home = scratch_home("ho-api-key-pool");
    let id = SessionId::generate();
    std::fs::create_dir_all(home.root()).ok();
    let pool = "[standin]\nkeys = [{ env = \"HO_KEY_POOL_PERSONAL\" }, \
                { env = \"HO_KEY_POOL_TEAM\" }, { file = \"spare.key\" }]\n";
    std::fs::write(home.root().join("key-pool.toml"), pool).ok();
    std::fs::write(home.root().join("spare.key"), SECRETS[2]).ok();
    // SAFETY: the crate's only test, and no other thread reads these names.
    unsafe {
        std::env::set_var("HO_KEY_POOL_PERSONAL", SECRETS[0]);
        std::env::set_var("HO_KEY_POOL_TEAM", SECRETS[1]);
    }

    let (mut app, wakes) = app_in(&home, &id);
    let models = [personal.clone(), team.clone(), spare.clone()];
    app.insert_resource(scripted_models([("standin/model", &spare)]))
        .insert_resource(KeyConnect(Arc::new(move |spec, _, key, _| {
            let index = SECRETS.iter().position(|secret| *secret == key.expose());
            let model = index.and_then(|index| models.get(index)).cloned();
            let model = model.ok_or_else(|| ConnectError::Provider("unknown key".into()))?;
            let model = ModelAdapter::<Completion>::new(spec.reference(), model);
            Ok(ErasedHandler::new(model))
        })))
        .add_plugins((AgentPlugin, JournalPlugin, KeyPoolPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<Notice>(&mut app);
    collect::<Recorded>(&mut app);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let model = "standin/model".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
    });
    app.update();

    app.world_mut()
        .trigger(Deliver::new(agent, "Hello.", DeliveryMode::Steer));
    until_ended(&mut app, &wakes, 1);
    let ended = &app.world().resource::<Ended>().0;
    let answer = match ended.first() {
        Some(TurnOutcome::Answered(message)) => final_answer(message),
        _ => None,
    };
    assert_eq!(answer.as_deref(), Some("From the spare."), "{ended:?}");
    let errors: Vec<String> = notices(&app)
        .into_iter()
        .filter(|(level, _)| *level == NoticeLevel::Error)
        .map(|(_, text)| text)
        .collect();
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(
        errors.iter().all(|text| text.contains("HO_KEY_POOL_TEAM")),
        "{errors:?}"
    );
    let retried = notices(&app)
        .into_iter()
        .any(|(_, text)| text.contains("Retrying"));
    assert!(!retried, "{:?}", notices(&app));

    // The next turn goes straight to the spare.
    app.world_mut()
        .trigger(Deliver::new(agent, "And again.", DeliveryMode::Steer));
    until_ended(&mut app, &wakes, 2);
    let counts = [&personal, &team, &spare].map(MockCompletionModel::request_count);
    assert_eq!(counts, [1, 1, 2]);

    app.world_mut().trigger(RunCommand::new(agent, "keys"));
    app.update();
    let shown = notices(&app);
    let line = |reference: &str| {
        let keys = shown
            .iter()
            .rev()
            .find(|(_, text)| text.contains("requests"));
        let lines = keys.map(|(_, text)| text.lines()).into_iter().flatten();
        lines
            .filter(|line| line.contains(reference))
            .collect::<String>()
    };
    assert!(
        line("HO_KEY_POOL_PERSONAL").contains("cooling 3"),
        "{shown:?}"
    );
    assert!(line("HO_KEY_POOL_TEAM").contains("disabled"), "{shown:?}");
    assert!(line("spare.key").contains("ok, 2 requests"), "{shown:?}");

    // No key's value in a notice, an effect record or the session's files.
    let records = &app.world().resource::<Collected<Recorded>>().0;
    let records: Vec<String> = records
        .iter()
        .map(|record| format!("{:?}", record.0))
        .collect();
    let mut saved = String::new();
    for entry in walk(home.session(&id).path()) {
        saved.push_str(&std::fs::read_to_string(entry).unwrap_or_default());
    }
    assert!(
        saved.contains("HO_KEY_POOL_TEAM"),
        "the usage is saved by reference"
    );
    for secret in SECRETS {
        assert!(!shown.iter().any(|(_, text)| text.contains(secret)));
        assert!(!records.iter().any(|record| record.contains(secret)));
        assert!(!saved.contains(secret));
    }
}

/// Every file under `dir`.
fn walk(dir: &std::path::Path) -> Vec<PathBuf> {
    let entries = std::fs::read_dir(dir).into_iter().flatten().flatten();
    entries
        .flat_map(|entry| {
            let path = entry.path();
            if path.is_dir() {
                walk(&path)
            } else {
                vec![path]
            }
        })
        .collect()
}

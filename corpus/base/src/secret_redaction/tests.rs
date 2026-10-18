use std::io;
use std::path::{Path, PathBuf};

use rig_basics::coding_tools::{ReadTool, WriteTool};
use rig_basics::telemetry::{EffectLog, EffectLogPlugin};
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::{Home, SessionId};
use rig_harness_test_support::{
    Collected, Ended, app_in, calls, collect, connect, first_agent, log_ended, reply, run_until,
    scratch_home,
};

use super::*;

const TOKEN: &str = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
const NEW_TOKEN: &str = "ghp_ZYXWVUTSRQPONMLKJIHGFEDCBA9876543210";
/// In the system prompt after the restart.
const PASSWORD: &str = "DEPLOY_PASSWORD=hunter2-for-prod";

/// Runs an agent of the session `id` in `home`, restored if it has one,
/// answered by `model`, with the read and write tools, the effect log, the
/// plugin and `prompt` in its system prompt, until its turn ends; what it
/// streamed, and its effect log.
fn run(
    home: &Home,
    id: &SessionId,
    model: &MockCompletionModel,
    prompt: &str,
) -> io::Result<(String, String)> {
    let (mut app, wakes) = app_in(home, id);
    app.add_plugins((
        AgentPlugin,
        SessionLogPlugin,
        EffectLogPlugin,
        ReadTool,
        WriteTool,
        SecretRedactionPlugin,
    ))
    .init_resource::<Ended>()
    .add_observer(log_ended);
    collect::<StreamDelta>(&mut app);
    app.finish();
    app.update();
    let world = app.world_mut();
    let (Some(agent), Some(connection)) = (first_agent(world), connect(model)) else {
        return Err(io::Error::other("no agent or model"));
    };
    world.entity_mut(agent).insert(connection);
    world.spawn(PromptSection::new(400, "deploy", prompt));
    world.trigger(Deliver::new(agent, "go", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    app.update();
    let streamed = app.world().resource::<Collected<StreamDelta>>().0.iter();
    let streamed = streamed
        .filter_map(|delta| match &delta.event {
            StreamEvent::Text { text, .. } | StreamEvent::Reasoning { text, .. } => {
                Some(text.as_str())
            }
            StreamEvent::Arguments { json, .. } => Some(json.as_str()),
            _ => None,
        })
        .collect();
    let log = app.world().resource::<EffectLog>().read()?;
    Ok((streamed, serde_json::to_string(&log)?))
}

/// The requests `model` was sent, as JSON.
fn sent(model: &MockCompletionModel) -> Vec<String> {
    let requests = model.requests();
    let requests = requests.iter().map(serde_json::to_string);
    requests.map(Result::unwrap_or_default).collect()
}

/// Every file under `dir`, as text.
fn read_all(dir: &Path) -> io::Result<String> {
    let mut text = String::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        match path.is_dir() {
            true => text.push_str(&read_all(&path)?),
            false => text.push_str(&String::from_utf8_lossy(&std::fs::read(&path)?)),
        }
    }
    Ok(text)
}

/// What the two runs left.
struct Seen {
    /// The requests of the first run, then of the run after the restart.
    sent: Vec<String>,
    sent_again: Vec<String>,
    /// The first run's streamed text and effect log.
    streamed: String,
    effects: String,
    /// What `out.txt` holds, and whether `again.txt` was written.
    written: String,
    wrote_again: bool,
    /// Every file of the session's directory.
    session: String,
}

/// The model reads `cfg.txt`, which holds a token, writes its placeholder
/// to `out.txt` and says the token. After a restart, with a password in
/// the system prompt, it reads a new token and writes the old placeholder
/// to `again.txt`.
fn run_the_script() -> io::Result<Seen> {
    let home = scratch_home("corpus-secret-redaction");
    let files: PathBuf = std::env::current_exe()?.with_file_name("corpus-secret-redaction-files");
    std::fs::remove_dir_all(&files).ok();
    std::fs::create_dir_all(&files)?;
    let path = |name: &str| files.join(name);
    std::fs::write(path("cfg.txt"), format!("token={TOKEN}\n"))?;
    std::fs::write(path("new.txt"), format!("token={NEW_TOKEN}\n"))?;
    let id = SessionId::generate();

    let model = MockCompletionModel::from_stream_turns([
        calls("1", "read", serde_json::json!({ "path": path("cfg.txt") })),
        calls(
            "2",
            "write",
            serde_json::json!({ "path": path("out.txt"), "content": "‹SECRET_1›" }),
        ),
        reply(&format!("Copied {TOKEN} for you.")),
    ]);
    let (streamed, effects) = run(&home, &id, &model, "Be brief.")?;

    let restarted = MockCompletionModel::from_stream_turns([
        calls("3", "read", serde_json::json!({ "path": path("new.txt") })),
        calls(
            "4",
            "write",
            serde_json::json!({ "path": path("again.txt"), "content": "‹SECRET_1›" }),
        ),
        reply("It is gone."),
    ]);
    run(&home, &id, &restarted, &format!("Deploy with {PASSWORD}."))?;
    let seen = Seen {
        sent: sent(&model),
        sent_again: sent(&restarted),
        streamed,
        effects,
        written: std::fs::read_to_string(path("out.txt"))?,
        wrote_again: path("again.txt").exists(),
        session: read_all(home.session(&id).path())?,
    };
    std::fs::remove_dir_all(&files).ok();
    Ok(seen)
}

/// The token never leaves the machine nor reaches a log or the screen, the
/// file gets it, and after a restart its placeholder is unknown and its
/// number is not given to a new secret.
#[test]
fn a_secret_read_goes_out_as_a_placeholder_and_is_written_back_for_real() {
    let seen = run_the_script();
    assert!(seen.is_ok(), "the script ran");
    let Ok(seen) = seen else {
        return;
    };
    let sent = |requests: &[String], at: usize| requests.get(at).cloned().unwrap_or_default();
    assert_eq!(seen.sent.len(), 3, "read, write, answer: {:?}", seen.sent);
    let after_read = sent(&seen.sent, 1);
    assert!(after_read.contains("‹SECRET_1›"), "{after_read}");
    assert_eq!(seen.written, TOKEN);
    assert!(
        seen.streamed.contains("Copied ‹SECRET_1› for"),
        "{}",
        seen.streamed
    );
    assert!(seen.effects.contains("‹SECRET_1›"), "{}", seen.effects);

    // The password in the prompt is the session's second secret.
    let new_read = sent(&seen.sent_again, 1);
    assert!(new_read.contains("‹SECRET_3›"), "{new_read}");
    let refused = sent(&seen.sent_again, 2);
    assert!(refused.contains("not a known secret"), "{refused}");
    assert!(!seen.wrote_again);

    for request in seen.sent.iter().chain(&seen.sent_again) {
        assert!(!request.contains("ghp_"), "{request}");
        assert!(!request.contains("hunter2"), "{request}");
    }
    assert!(seen.session.contains("‹SECRET_1›"));
    assert!(!seen.session.contains("ghp_"), "{}", seen.session);
    assert!(!seen.session.contains("hunter2"), "{}", seen.session);
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(SecretRedactionPlugin);
}

/// The plugins that edit what a request sends, loaded together.
#[test]
fn stages_are_ordered_with_the_other_request_editors() {
    rig_defaults_test_support::assert_stages_ordered((
        SecretRedactionPlugin,
        corpus_shared::few_shot_examples::FewShotExamplesPlugin,
        corpus_ho::stale_read_elision::StaleReadElisionPlugin,
    ));
}

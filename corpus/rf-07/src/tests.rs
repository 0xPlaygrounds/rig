use super::*;
use rig_core::test_utils::{
    MockCompletionModel, MockError, MockStreamEvent, mock_final_with_total_tokens,
};
use serde_json::json;

/// `assert!` for a test that returns a `Result`.
macro_rules! check {
    ($cond:expr $(,)?) => {
        if !$cond {
            return Err(format!("check failed: {}", stringify!($cond)).into());
        }
    };
    ($cond:expr, $($msg:tt)+) => {
        if !$cond {
            return Err(format!($($msg)+).into());
        }
    };
}

/// `assert_eq!` for a test that returns a `Result`.
macro_rules! check_eq {
    ($a:expr, $b:expr $(,)?) => {{
        let (a, b) = (&$a, &$b);
        if a != b {
            return Err(format!("{a:?} != {b:?}").into());
        }
    }};
    ($a:expr, $b:expr, $($msg:tt)+) => {{
        let (a, b) = (&$a, &$b);
        if a != b {
            return Err(format!("{a:?} != {b:?}: {}", format!($($msg)+)).into());
        }
    }};
}

const KEY: &str = "sk-proj-abcdefghijklmnop1234";

fn call(id: &str, row: &str, tokens: u64) -> Vec<MockStreamEvent> {
    vec![
        MockStreamEvent::text("writing "),
        MockStreamEvent::tool_call(id, "write_row", json!({ "row": row })),
        MockStreamEvent::FinalResponse(mock_final_with_total_tokens(tokens)),
    ]
}

fn text(text: &str, tokens: u64) -> Vec<MockStreamEvent> {
    vec![
        MockStreamEvent::text(text),
        MockStreamEvent::text(" for now"),
        MockStreamEvent::FinalResponse(mock_final_with_total_tokens(tokens)),
    ]
}

fn program() -> Program {
    Program {
        preamble: "You clean up the customer table.".into(),
        prompt: "Clean up the customer table.".into(),
        max_turns: 6,
        streaming: true,
        observer: None,
    }
}

fn scratch(name: &str) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    // Beside the test binary in the target directory, never in /tmp.
    let exe = std::env::current_exe()?;
    let dir = exe
        .parent()
        .ok_or("the test binary has a directory")?
        .join(format!("rf-07-{}", std::process::id()))
        .join(name);
    let _ = fs::remove_dir_all(&dir);
    Ok(dir)
}

fn files(dir: &Path) -> Result<BTreeMap<String, Vec<u8>>, Box<dyn std::error::Error>> {
    let mut out = BTreeMap::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        out.insert(path.display().to_string(), fs::read(&path)?);
    }
    Ok(out)
}

#[tokio::test]
async fn a_recorded_run_replays_forks_and_refuses_from_its_files()
-> Result<(), Box<dyn std::error::Error>> {
    // Record: four streamed turns, three rows written, a secret in a row.
    let original = scratch("original")?;
    let model = MockCompletionModel::from_stream_turns([
        call("c1", &format!("alice OPENAI_API_KEY={KEY}"), 10),
        call("c2", "bob", 20),
        call("c3", "carol", 30),
        text("done", 40),
    ]);
    let rows = Rows::default();
    let ending = record(&program(), model.clone(), rows.clone(), &original, true).await?;
    check!(matches!(&ending, Ending::Answer { output, .. } if output.contains("done")));
    check_eq!((rows.count(), model.request_count()), (3, 4));
    let log_text = fs::read_to_string(original.join("log.json"))?;
    check!(
        !log_text.contains(KEY),
        "the key is redacted before writing"
    );
    check!(original.join("checkpoint-2.json").exists());

    // Replay from turn 2, from the files alone: identical, no tool body
    // ran, the model was never asked.
    let verdict = replay(&program(), &original, 2).await?;
    check_eq!(verdict, Verdict::Identical, "{verdict}");
    check_eq!((rows.count(), model.request_count()), (3, 4));

    // A changed preamble is refused up front; an added observer is not.
    let changed = Program {
        preamble: "You delete the customer table.".into(),
        ..program()
    };
    let refused = replay(&changed, &original, 2).await;
    check!(
        matches!(&refused, Err(RerunError::Incompatible(why)) if why.contains("preamble")),
        "{refused:?}"
    );
    let observer = Observer::default();
    let observed = Program {
        observer: Some(observer.clone()),
        ..program()
    };
    check_eq!(replay(&observed, &original, 2).await?, Verdict::Identical);
    check!(observer.0.load(Ordering::SeqCst) > 0, "the observer ran");

    // Fork from turn 2 against another model: new turns, a new directory,
    // the original untouched.
    let before = files(&original)?;
    let forked = scratch("forked")?;
    let other =
        MockCompletionModel::from_stream_turns([call("d3", "dave", 300), text("forked", 400)]);
    let ending = fork(
        &program(),
        &original,
        2,
        other.clone(),
        rows.clone(),
        &forked,
    )
    .await?;
    check!(matches!(&ending, Ending::Answer { output, .. } if output.contains("forked")));
    check_eq!(
        other.request_count(),
        2,
        "turns 3 and 4 are the new model's"
    );
    check_eq!(
        rows.count(),
        5,
        "turn 2's pending call and turn 3's ran live"
    );
    check_eq!(files(&original)?, before, "the original is byte-identical");
    check_eq!(replay(&program(), &forked, 3).await?, Verdict::Identical);

    // A stream that failed at turn 3 replays as the same items, then the
    // same error.
    let failing = scratch("failing")?;
    let mut broken = call("e3", "erin", 30);
    broken.truncate(1);
    broken.push(MockStreamEvent::Error(MockError::provider(
        "upstream reset",
    )));
    let model =
        MockCompletionModel::from_stream_turns([call("e1", "a", 10), call("e2", "b", 20), broken]);
    let ending = record(&program(), model, Rows::default(), &failing, false).await?;
    check!(matches!(&ending, Ending::Error(error) if error.contains("upstream reset")));
    check_eq!(replay(&program(), &failing, 2).await?, Verdict::Identical);

    // A truncated log, or a checkpoint of another version, is refused.
    let mut corrupt = fs::read_to_string(original.join("checkpoint-2.json"))?;
    corrupt = corrupt.replacen(
        &format!(
            "\"format\": {}",
            rig_cassette::effect_log::CHECKPOINT_FORMAT
        ),
        "\"format\": 999",
        1,
    );
    fs::write(forked.join("checkpoint-2.json"), corrupt)?;
    let refused = load(&forked, 2).err().map(|error| error.to_string());
    check!(
        refused
            .as_deref()
            .is_some_and(|why| why.contains("format 999")),
        "{refused:?}"
    );
    let truncated = log_text
        .get(..log_text.len().saturating_sub(100))
        .ok_or("the log is longer than 100 bytes")?;
    fs::write(original.join("log.json"), truncated)?;
    check!(matches!(
        load(&original, 2),
        Err(RerunError::Corrupt { .. })
    ));

    let _ = fs::remove_dir_all(original.parent().ok_or("a scratch root")?);
    Ok(())
}

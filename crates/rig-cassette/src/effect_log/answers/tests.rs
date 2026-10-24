use rig_core::effect::{Answered, EffectId, EffectKind, EffectRecord, HandlerKey};
use rig_core::error::ErrorKind;

use super::{Divergence, RecordedAnswers};
use crate::effect_log::{EffectLog, RequestCheck};

fn call(args: &str) -> EffectKind {
    EffectKind::ToolCall {
        name: "read".into(),
        args: args.into(),
    }
}

fn custom(payload: &str) -> EffectKind {
    EffectKind::Custom {
        kind: "embed-note".into(),
        payload: payload.into(),
    }
}

fn record(id: u64, scope: Option<&str>, kind: EffectKind, answer: &str) -> EffectRecord {
    let mut record = EffectRecord::new(
        EffectId::from_raw(id),
        HandlerKey::from("handler"),
        kind,
        Ok(Answered::Custom {
            payload: answer.into(),
        }),
    );
    record.scope = scope.map(Into::into);
    record
}

fn answered(record: Result<EffectRecord, Divergence>) -> Option<String> {
    match record.ok()?.outcome {
        Ok(Answered::Custom { payload }) => payload.as_str().map(str::to_owned),
        _ => None,
    }
}

/// Each scope's effects of each kind are answered in their recorded order,
/// scopes and kinds apart, and a record answers once.
#[test]
fn answers_follow_each_scope_and_kind_in_order() {
    let log = EffectLog::from_records(vec![
        record(0, Some("lead"), call("{\"path\":\"a\"}"), "lead a"),
        record(1, Some("helper"), call("{\"path\":\"a\"}"), "helper a"),
        record(2, Some("lead"), custom("note"), "noted"),
        record(3, Some("lead"), call("{\"path\":\"a\"}"), "lead a again"),
        record(4, None, custom("note"), "app noted"),
    ]);
    let mut answers = RecordedAnswers::new(&log);
    assert_eq!(answers.remaining(), 5);
    let a = call("{\"path\":\"a\"}");
    assert_eq!(
        answered(answers.answer(Some("lead"), &a)).as_deref(),
        Some("lead a")
    );
    assert_eq!(
        answered(answers.answer(Some("helper"), &a)).as_deref(),
        Some("helper a")
    );
    assert_eq!(
        answered(answers.answer(Some("lead"), &a)).as_deref(),
        Some("lead a again")
    );
    assert_eq!(
        answered(answers.answer(None, &custom("note"))).as_deref(),
        Some("app noted")
    );
    let exhausted = answers.answer(Some("helper"), &a);
    assert!(
        exhausted.is_err_and(|Divergence { expected, report }| expected.is_none()
            && report.kind == ErrorKind::Divergence
            && report.message.contains("no more"))
    );
    assert_eq!(answers.remaining(), 1);
}

/// Calls made side by side match their records whatever order they come
/// in; a call no record matches diverges, carrying the next record and
/// naming the difference, and takes nothing; without checks, records
/// answer in order.
#[test]
fn a_call_takes_the_first_record_it_matches() {
    let log = EffectLog::from_records(vec![
        record(0, Some("lead"), call("{\"path\":\"a\"}"), "A"),
        record(1, Some("lead"), call("{\"path\":\"b\"}"), "B"),
    ]);
    let mut answers = RecordedAnswers::new(&log);
    assert_eq!(
        answered(answers.answer(Some("lead"), &call("{\"path\":\"b\"}"))).as_deref(),
        Some("B")
    );
    let diverged = answers.answer(Some("lead"), &call("{\"path\":\"c\"}"));
    assert!(diverged.is_err_and(|Divergence { expected, report }| {
        expected.is_some_and(|record| record.id == EffectId::from_raw(0))
            && report.kind == ErrorKind::Divergence
            && report.message.contains("arguments differ")
    }));
    assert_eq!(answers.remaining(), 1);

    let mut unchecked = RecordedAnswers::new(&log).checking(RequestCheck::Skip);
    assert_eq!(
        answered(unchecked.answer(Some("lead"), &call("{\"path\":\"z\"}"))).as_deref(),
        Some("A")
    );
}

/// A completion made again in a new session matches its record though its
/// prompt-cache key (per agent, so new) differs; any other difference
/// still diverges.
#[test]
fn a_new_cache_key_alone_does_not_diverge() {
    let completion = |key: &str, prompt: &str| {
        let mut request = rig_core::completion::CompletionRequest::new(prompt);
        request.options.cache_key = Some(key.to_owned());
        EffectKind::Completion {
            request: request.into(),
            stream: true,
        }
    };
    let log = EffectLog::from_records(vec![record(
        0,
        Some("lead"),
        completion("agent-1", "hi"),
        "hello",
    )]);
    let mut answers = RecordedAnswers::new(&log);
    assert!(
        answers
            .answer(Some("lead"), &completion("agent-2", "bye"))
            .is_err_and(|divergence| divergence.report.kind == ErrorKind::Divergence)
    );
    assert_eq!(
        answered(answers.answer(Some("lead"), &completion("agent-2", "hi"))).as_deref(),
        Some("hello")
    );
}

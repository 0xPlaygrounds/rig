//! Deterministic, side-effect-free checkpoint programs shared by both runtimes.
//! Tokens force four dependent tool turns; the large result is generated from
//! its arguments alone, so restoring a scene needs no hidden fixture state.
#![allow(dead_code, reason = "checkpoint cells run on five provider columns")]

use rig_core::effect::EffectKind;

use rig_core::effect::Outcome;

use rig_cassette::effect_log::EffectLog;

use rig_core::tool::Tool;

use rig_core::tool::ToolContext;

use serde::{Deserialize, Serialize};

use super::cells::{CELL, Cell, ToolKind};
use super::corpus::Program;

pub(crate) const PREAMBLE: &str = "Follow the requested tool protocol exactly. Never invent a tool result. Do not call tools again after completing the protocol. Your final answer must be exactly checkpoint-complete.";
const LOOP_PROMPT: &str = "Call checkpoint_step with token=start. Read its next_token and call checkpoint_step with that token, one call per model turn. Continue until its result says done=true. Then answer checkpoint-complete. Do not guess tokens or batch dependent calls.";
const BASE: Program = Program {
    preamble: Some(PREAMBLE),
    prompt: LOOP_PROMPT,
    temperature: Some(0.0),
    max_tokens: Some(2048),
    max_turns: Some(8),
    tool_concurrency: Some(3),
    ..Program::DEFAULT
};

pub(crate) const MULTI_TURN_UNARY: Cell = Cell {
    name: "checkpoint_multi_turn_unary",
    program: BASE,
    tools: &[ToolKind::CheckpointStep],
    events: true,
    ..CELL
};
pub(crate) const MULTI_TURN_STREAMED: Cell = Cell {
    name: "checkpoint_multi_turn_streamed",
    program: Program {
        streamed: true,
        ..BASE
    },
    ..MULTI_TURN_UNARY
};
pub(crate) const PARALLEL_BATCH: Cell = Cell {
    name: "checkpoint_parallel_batch",
    program: Program {
        prompt: "In a single parallel batch call checkpoint_batch exactly three times, in order with slot=0, slot=1, slot=2. One call intentionally fails; do not retry it. After all three results arrive, answer checkpoint-complete.",
        ..BASE
    },
    tools: &[ToolKind::CheckpointBatch],
    events: true,
    ..CELL
};
pub(crate) const LARGE_RESULT: Cell = Cell {
    name: "checkpoint_large_result",
    program: Program {
        prompt: "Call checkpoint_large exactly once with no arguments. Once its result arrives, answer checkpoint-complete. Do not repeat or summarize the payload.",
        ..BASE
    },
    tools: &[ToolKind::CheckpointLarge],
    events: true,
    ..CELL
};

pub(crate) fn tool_turns(cell: &Cell) -> usize {
    match cell.name {
        "checkpoint_multi_turn_unary" | "checkpoint_multi_turn_streamed" => 4,
        "checkpoint_parallel_batch" | "checkpoint_large_result" => 1,
        other => panic!("not a checkpoint cell: {other}"),
    }
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub(crate) struct CheckpointError(pub &'static str);

#[derive(Deserialize, Serialize)]
pub(crate) struct StepArgs {
    pub(crate) token: String,
}
#[derive(Deserialize, Serialize)]
pub(crate) struct CheckpointStep;
const TOKENS: [&str; 4] = ["start", "cobalt-17", "orchard-29", "harbor-43"];
fn step_output(token: &str) -> Result<serde_json::Value, CheckpointError> {
    let step = TOKENS
        .iter()
        .position(|value| *value == token)
        .ok_or(CheckpointError("unknown checkpoint token"))?;
    Ok(serde_json::json!({"step": step + 1, "next_token": TOKENS.get(step + 1), "done": step == 3}))
}
impl Tool for CheckpointStep {
    const NAME: &'static str = "checkpoint_step";
    type Error = CheckpointError;
    type Args = StepArgs;
    type Output = serde_json::Value;
    fn description(&self) -> String {
        "Advance one dependent checkpoint step. Start with token=start; subsequent tokens come only from the previous result.".into()
    }
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type":"object","properties":{"token":{"type":"string"}},"required":["token"],"additionalProperties":false})
    }
    async fn call(&self, _: &mut ToolContext, args: StepArgs) -> Result<Self::Output, Self::Error> {
        step_output(&args.token)
    }
}

#[derive(Deserialize, Serialize)]
pub(crate) struct BatchArgs {
    pub(crate) slot: usize,
}
#[derive(Deserialize, Serialize)]
pub(crate) struct CheckpointBatch;
impl Tool for CheckpointBatch {
    const NAME: &'static str = "checkpoint_batch";
    type Error = CheckpointError;
    type Args = BatchArgs;
    type Output = String;
    fn description(&self) -> String {
        "Read one deterministic batch slot (0, 1, or 2). Slot 1 intentionally returns an error; do not retry it.".into()
    }
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type":"object","properties":{"slot":{"type":"integer","enum":[0,1,2]}},"required":["slot"],"additionalProperties":false})
    }
    async fn call(&self, _: &mut ToolContext, args: BatchArgs) -> Result<String, Self::Error> {
        match args.slot {
            0 => Ok("checkpoint-slot-zero".into()),
            1 => Err(CheckpointError("checkpoint-slot-one-intentional-error")),
            2 => Ok("checkpoint-slot-two".into()),
            _ => Err(CheckpointError("invalid checkpoint slot")),
        }
    }
}

#[derive(Deserialize, Serialize)]
pub(crate) struct EmptyArgs {}
#[derive(Deserialize, Serialize)]
pub(crate) struct CheckpointLarge;
pub(crate) fn large_result() -> String {
    "0123456789abcdef".repeat(3072)
}
impl Tool for CheckpointLarge {
    const NAME: &'static str = "checkpoint_large";
    type Error = CheckpointError;
    type Args = EmptyArgs;
    type Output = String;
    fn description(&self) -> String {
        "Return an immutable 49152-byte fixture. Call exactly once; no external side effects."
            .into()
    }
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type":"object","properties":{},"additionalProperties":false})
    }
    async fn call(&self, _: &mut ToolContext, _: EmptyArgs) -> Result<String, Self::Error> {
        Ok(large_result())
    }
}

/// Assert the actual tool arguments, outputs, turn separation and stream events.
/// Full producer/native effect-log parity remains an independent caller gate.
pub(crate) fn assert_log(cell: &Cell, log: &EffectLog) {
    let requests: Vec<_> = log
        .records
        .iter()
        .filter(|record| matches!(record.kind, EffectKind::Completion { .. }))
        .collect();
    assert_eq!(
        requests.len(),
        tool_turns(cell) + 1,
        "{}: exact completion count",
        cell.name
    );
    if cell.program.streamed {
        use rig_core::message::AssistantContent;

        use rig_core::streaming::BlockKind;

        use rig_core::streaming::Delta;

        use rig_core::streaming::StreamEvent;

        for (turn, record) in requests.iter().enumerate() {
            assert!(matches!(
                record.kind,
                EffectKind::Completion { stream: true, .. }
            ));
            let events = record
                .events
                .as_ref()
                .expect("actual provider stream delivery is retained");
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(event, StreamEvent::Final(_)))
                    .count(),
                1,
                "one actual terminal per completion"
            );
            if turn < tool_turns(cell) {
                // Gemini REST delivers function calls atomically; it need not
                // fabricate partial argument deltas to prove stream delivery.
                let delivered: Vec<_> = events
                    .iter()
                    .filter_map(|event| match event {
                        StreamEvent::BlockEnd {
                            id,
                            block: Some(AssistantContent::ToolCall(_)),
                            ..
                        } => Some(id),
                        _ => None,
                    })
                    .collect();
                assert_eq!(
                    delivered.len(),
                    1,
                    "one completed tool call delivered in this dependent turn"
                );
                assert!(events.iter().any(|event| matches!(event,
                    StreamEvent::BlockStart { id, kind: BlockKind::ToolCall }
                    | StreamEvent::BlockDelta { id, delta: Delta::ToolName { .. } | Delta::ToolArguments { .. } }
                    if id == delivered[0]
                )), "the completed call has actual matching tool-block or delta delivery");
            } else {
                assert!(events.iter().any(|event| matches!(event, StreamEvent::BlockDelta { delta: Delta::Text { text }, .. } if !text.is_empty())), "actual final answer text was streamed");
            }
        }
    }
    let tools: Vec<_> = log
        .records
        .iter()
        .filter_map(|record| match (&record.kind, &record.outcome) {
            (EffectKind::ToolCall { name, args }, Ok(Outcome::ToolResult { result })) => Some((
                name.as_str(),
                serde_json::from_str::<serde_json::Value>(args).expect("tool JSON"),
                result,
            )),
            (EffectKind::ToolCall { .. }, other) => {
                panic!("the tool outcome is published: {other:?}")
            }
            _ => None,
        })
        .collect();
    match cell.name {
        "checkpoint_multi_turn_unary" | "checkpoint_multi_turn_streamed" => {
            assert_eq!(tools.len(), 4, "each dependent tool runs once");
            for (i, (name, args, result)) in tools.iter().enumerate() {
                assert_eq!(*name, "checkpoint_step");
                assert_eq!(*args, serde_json::json!({"token":TOKENS[i]}));
                assert!(!result.is_error());
                assert_eq!(
                    serde_json::from_str::<serde_json::Value>(&result.output().render())
                        .expect("step JSON"),
                    step_output(TOKENS[i]).expect("known token")
                );
            }
            let families: Vec<_> = log
                .records
                .iter()
                .map(|record| record.kind.family())
                .collect();

            use rig_core::effect::EffectFamily::Completion as C;

            use rig_core::effect::EffectFamily::Tool as T;

            assert_eq!(
                families,
                [C, T, C, T, C, T, C, T, C],
                "dependent calls occupy four separate tool turns"
            );
        }
        "checkpoint_parallel_batch" => {
            assert_eq!(tools.len(), 3);
            for (slot, (name, args, result)) in tools.iter().enumerate() {
                assert_eq!(*name, "checkpoint_batch");
                assert_eq!(
                    *args,
                    serde_json::json!({"slot":slot}),
                    "provider call order"
                );
                assert_eq!(
                    result.is_error(),
                    slot == 1,
                    "the middle call fails without retry"
                );
            }
            assert_eq!(tools[0].2.output().render(), "checkpoint-slot-zero");
            assert_eq!(tools[2].2.output().render(), "checkpoint-slot-two");
        }
        "checkpoint_large_result" => {
            assert_eq!(tools.len(), 1);
            assert_eq!(tools[0].0, "checkpoint_large");
            assert_eq!(tools[0].1, serde_json::json!({}));
            assert!(!tools[0].2.is_error());
            assert_eq!(
                tools[0].2.output().render(),
                large_result(),
                "complete 48KiB tool bytes"
            );
        }
        other => panic!("not a checkpoint cell: {other}"),
    }
    assert!(
        super::corpus::golden_answer(log)
            .trim()
            .ends_with("checkpoint-complete"),
        "the provider finishes with the checkpoint marker"
    );
}

/// Retain scrubbed records before semantic assertions, including rejected live
/// shapes. The caller supplies a unique external directory for every attempt.
pub(crate) fn write_attempt(cell: &Cell, log: &EffectLog) {
    let Some(directory) = std::env::var_os("RIG_CHECKPOINT_ATTEMPT_DIR") else {
        return;
    };
    let path = std::path::PathBuf::from(directory).join(format!("{}.effects.json", cell.name));
    std::fs::create_dir_all(path.parent().expect("attempt directory"))
        .expect("create evidence directory");
    let value = crate::cassettes::scrub_artifact(
        &serde_json::to_value(log).expect("effect log serializes"),
    );
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&value).expect("scrubbed log serializes"),
    )
    .expect("save attempt log");
    eprintln!("CHECKPOINT_ATTEMPT effects={}", path.display());
}

/// The first strict replay validates the program and saves complete evidence
/// before goldens exist. It explicitly does not claim golden parity. Subsequent
/// generation and verification use the ordinary producer golden callback.
pub(crate) async fn run_agent<M: rig_agent::completion::CompletionModel + Clone + 'static>(
    wire: &super::Wire<M>,
    cell: &Cell,
    golden: impl FnOnce(&EffectLog),
) -> EffectLog {
    let audit = std::env::var_os("CHECKPOINT_AUDIT_REPLAY").is_some();
    if audit {
        assert_eq!(
            std::env::var("RIG_PROVIDER_TEST_MODE").as_deref(),
            Ok("replay")
        );
        assert!(
            std::env::var_os("RIG_REGENERATE_GOLDEN").is_none(),
            "initial audit never generates goldens"
        );
        assert!(
            std::env::var_os("RIG_CHECKPOINT_ATTEMPT_DIR").is_some(),
            "initial replay must save evidence"
        );
    }
    super::agent::run_agent(wire, cell, |log| {
        if audit {
            assert_log(cell, log);
            write_attempt(cell, log);
            eprintln!("CHECKPOINT_AUDIT_REPLAY {}: strict transport/program validation only; no golden parity claimed", cell.name);
        } else {
            golden(log);
        }
    }).await
}

/// Matcher sensitivity, deliberately separate from native continuation: send
/// actual recorded HTTP request bodies to the real replay server, alter only
/// the last byte of the large tool result, and prove the strict matcher refuses
/// it. Positive native cuts independently construct and send their own requests.
pub(crate) async fn assert_large_request_rejected(provider: &'static str, scenario: &'static str) {
    assert_request_body_rejected(provider, scenario, false).await;
}

/// Mutate the first actual tool-result continuation in the streamed loop.
/// This exercises the matcher, independently of native positive request parity.
pub(crate) async fn assert_stream_request_rejected(provider: &'static str, scenario: &'static str) {
    assert_request_body_rejected(provider, scenario, true).await;
}

async fn assert_request_body_rejected(
    provider: &'static str,
    scenario: &'static str,
    streamed: bool,
) {
    use futures::FutureExt;
    use std::panic::AssertUnwindSafe;

    assert_eq!(
        crate::cassettes::CassetteMode::current(),
        crate::cassettes::CassetteMode::Replay,
        "negative matcher tests never record"
    );
    let path = crate::cassettes::cassette_path(provider, scenario);
    let yaml = std::fs::read_to_string(&path).expect("the live recording exists");
    let interactions: Vec<serde_json::Value> = serde_yaml::Deserializer::from_str(&yaml)
        .map(|document| serde_json::Value::deserialize(document).expect("recorded YAML document"))
        .collect();
    assert_eq!(
        interactions.len(),
        if streamed { 5 } else { 2 },
        "the complete recorded tool program"
    );
    let bodies = crate::cassettes::recorded_interaction_bodies(provider, scenario);
    let payload = if streamed {
        "cobalt-17".to_owned()
    } else {
        large_result()
    };
    assert_eq!(payload.len(), if streamed { 9 } else { 49152 });
    assert_eq!(
        bodies[1].0.matches(&payload).count(),
        1,
        "the actual second request carries the whole tool body exactly once"
    );
    let start = bodies[1].0.find(&payload).expect("the full payload");
    let at = start + payload.len() - 1;
    let mut changed = bodies[1].0.as_bytes().to_vec();
    assert_eq!(changed[at], if streamed { b'7' } else { b'f' });
    changed[at] = b'X';
    let changed = String::from_utf8(changed).expect("one ASCII substitution");
    assert_eq!(changed.len(), bodies[1].0.len());
    assert_eq!(
        changed
            .bytes()
            .zip(bodies[1].0.bytes())
            .filter(|(left, right)| left != right)
            .count(),
        1,
        "only the last result byte changed"
    );
    serde_json::from_str::<serde_json::Value>(&changed).expect("mutation preserves JSON syntax");

    let cassette = crate::cassettes::ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        provider,
        scenario,
        "https://checkpoint.invalid",
    )
    .await;
    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("local HTTP client");
    async fn post(
        client: &reqwest::Client,
        provider: &str,
        base: &str,
        interaction: &serde_json::Value,
        body: &str,
    ) -> (u16, String) {
        let request = &interaction["when"];
        assert_eq!(request["method"], "POST");
        let mut url = reqwest::Url::parse(base).expect("local replay URL");
        assert_eq!(
            url.host_str(),
            Some("127.0.0.1"),
            "only the real local replay server receives the probe"
        );
        url.set_path(request["path"].as_str().expect("recorded request path"));
        for query in request["query_param"]
            .as_array()
            .expect("recorded query parameters")
        {
            url.query_pairs_mut().append_pair(
                query["name"].as_str().expect("query name"),
                query["value"].as_str().expect("query value"),
            );
        }
        // Recordings intentionally omit credentials; satisfy the real replay
        // policy with fixed dummy headers, never credentials from the host.
        let mut builder = match provider {
            "anthropic" => client
                .post(url)
                .header("x-api-key", "checkpoint-replay-only"),
            "openai" | "deepseek" => client
                .post(url)
                .header("authorization", "Bearer checkpoint-replay-only"),
            "gemini" => client.post(url),
            other => panic!("not a checkpoint provider: {other}"),
        };
        for header in request["header"].as_array().expect("recorded headers") {
            builder = builder.header(
                header["name"].as_str().expect("header name"),
                header["value"].as_str().expect("header value"),
            );
        }
        let response = builder
            .body(body.to_owned())
            .send()
            .await
            .expect("local replay response");
        let status = response.status().as_u16();
        let body = response.text().await.expect("consume local response body");
        (status, body)
    }
    let base = cassette.base_url();
    let first = post(&client, provider, &base, &interactions[0], &bodies[0].0).await;
    assert_eq!(
        u64::from(first.0),
        interactions[0]["then"]["status"]
            .as_u64()
            .expect("recorded first status"),
        "unmodified first request matches: {}",
        first.1
    );
    let rejected = post(&client, provider, &base, &interactions[1], &changed).await;
    assert_eq!(
        rejected.0, 404,
        "last-byte result mutation is rejected by real strict matching: {}",
        rejected.1
    );
    let diagnostic: serde_json::Value =
        serde_json::from_str(&rejected.1).expect("real matcher diagnostic");
    let candidate = &diagnostic["candidates"][1];
    assert_eq!(
        candidate["body_matches"], false,
        "the body specifically fails matching"
    );
    for field in [
        "method_matches",
        "path_matches",
        "query_matches",
        "headers_match",
        "required_headers_match",
    ] {
        assert_eq!(
            candidate[field], true,
            "all other request attributes match: {field}"
        );
    }
    for (interaction, body) in interactions.iter().zip(&bodies).skip(1) {
        let original = post(&client, provider, &base, interaction, &body.0).await;
        assert_eq!(
            u64::from(original.0),
            interaction["then"]["status"]
                .as_u64()
                .expect("recorded status"),
            "original request still matches; the miss consumed nothing: {}",
            original.1
        );
    }
    let failed = AssertUnwindSafe(cassette.finish())
        .catch_unwind()
        .await
        .expect_err("finish remembers the rejected mutation");
    let message = failed
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| {
            failed
                .downcast_ref::<&str>()
                .map(|value| (*value).to_owned())
        })
        .expect("matcher failure message");
    assert!(
        message.contains("received unexpected replay request(s)"),
        "specific replay mismatch: {message}"
    );
    assert!(
        !message.contains("left unused interactions"),
        "all original requests were consumed: {message}"
    );
    eprintln!(
        "CHECKPOINT_NEGATIVE provider={provider} scenario={scenario} result_offset={} body_offset={at} changed_bytes=1 status=404 original_requests_consumed={}",
        payload.len() - 1,
        interactions.len()
    );
}

/// Retain the JSON scene and effect head used at a fresh-world cut. Scrubbing
/// applies only to these external copies; restoration uses the original encoded
/// strings. Both original and artifact hashes identify that distinction.
pub(crate) fn write_cut_evidence(cell: &Cell, cut: usize, encoded_scene: &str, encoded_head: &str) {
    use sha2::{Digest, Sha256};
    let Some(directory) = std::env::var_os("RIG_CHECKPOINT_ATTEMPT_DIR") else {
        return;
    };
    let directory = std::path::PathBuf::from(directory);
    std::fs::create_dir_all(&directory).expect("create cut evidence directory");
    let hash = |bytes: &[u8]| format!("{:x}", Sha256::digest(bytes));
    let mut artifacts = Vec::new();
    for (kind, source) in [("scene", encoded_scene), ("head", encoded_head)] {
        let value: serde_json::Value = serde_json::from_str(source).expect("actual cut JSON");
        let scrubbed = crate::cassettes::scrub_artifact(&value);
        let bytes = serde_json::to_vec_pretty(&scrubbed).expect("scrubbed cut JSON");
        let filename = format!("{}-cut-{cut}.{kind}.json", cell.name);
        let path = directory.join(&filename);
        let text = std::str::from_utf8(&bytes).expect("JSON is UTF-8");
        assert!(
            crate::cassettes::artifact_safety_failures(&path, text).is_empty(),
            "external cut evidence contains no sensitive data"
        );
        std::fs::write(&path, &bytes).expect("write cut artifact");
        artifacts.push(serde_json::json!({
            "kind": kind, "file": filename,
            "source_bytes": source.len(), "source_sha256": hash(source.as_bytes()),
            "artifact_bytes": bytes.len(), "artifact_sha256": hash(&bytes),
            "scrubbed_value_changed": scrubbed != value,
        }));
    }
    let metadata = serde_json::json!({
        "cell": cell.name, "cut_tool_turns": cut, "streamed": cell.program.streamed,
        "test_thread": std::thread::current().name(),
        "restoration_input": "original encoded scene and head; external copies are scrubbed",
        "artifacts": artifacts,
    });
    let path = directory.join(format!("{}-cut-{cut}.metadata.json", cell.name));
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&metadata).expect("cut metadata JSON"),
    )
    .expect("write cut metadata");
    eprintln!("CHECKPOINT_CUT_EVIDENCE {}", path.display());
}

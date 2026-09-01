//! Response-shape dimensions the generation matrices do not own: reasoning,
//! multiple candidates, log probabilities, and the finish-reason vocabulary.
//!
//! **Server**: the default configuration — `unsloth/Qwen3-1.7B-GGUF` Q4_K_M,
//! `--jinja --seed 42 --temp 0 -c 4096`, `llama-server` b10499-6d05498.
//!
//! | Cell | Dimension | Pinned |
//! | --- | --- | --- |
//! | [`reasoning_content_reaches_the_caller_on_both_transports`] | `reasoning_content` | a non-OpenAI message field, reaching the caller as reasoning on both transports — a block when blocking, correlated deltas when streaming |
//! | [`n_greater_than_one_answers_from_candidate_zero_on_both_transports`] | `n > 1` | the two transports pick the *same* candidate |
//! | [`logprobs_survive_into_the_raw_response`] | `logprobs` | preserved on `raw`, absent from the normalized view |
//! | [`the_finish_reason_vocabulary_is_covered_end_to_end`] | `finish_reason` | all three values llama.cpp can emit are recorded somewhere in this suite |
//!
//! # The reasoning cell is a parity check, not a shape check
//!
//! llama.cpp puts hidden reasoning in `reasoning_content`, a field OpenAI does
//! not define. The two transports deliver it at different *granularities* —
//! one [`Reasoning`](rig::message::Reasoning) block when blocking, a run of
//! correlated `ReasoningDelta`s when streaming — which is rig's normal
//! streaming shape and not a disagreement. What the cell requires is that
//! neither transport lets it fall into the plain-text surface, which is the
//! failure the blocking mapping's own comment says it exists to prevent.
//!
//! # Why `n > 1` earns a cell here
//!
//! Most providers in rig's matrix either reject `n > 1` or are never asked.
//! llama.cpp accepts it and returns real, separately-indexed candidates on
//! both transports, which makes it one of the few places the shared layer's
//! candidate-selection rule can actually be observed. The streaming path
//! interleaves candidates in one SSE stream distinguished only by
//! `choices[].index`, and the shared decoder selects index 0 — "taking each
//! chunk's first choice would concatenate every candidate into one garbled
//! answer, while the blocking path answers the same request from candidate 0
//! alone". This cell is that claim, measured.

use rig::client::CompletionClient;
use rig::completion::CompletionModel;
use rig::message::AssistantContent;
use serde_json::{Value, json};

use crate::cassettes::{
    recorded_json_request, recorded_sse_json_frames, recorded_statuses_and_bodies,
};
use crate::support::assistant_text_response;

use super::super::cassette_support::*;

/// A prompt Qwen3 answers with a visible `<think>` pass, so
/// `reasoning_content` is populated.
const REASONING_PROMPT: &str = "What is 2+2? Answer briefly.";

/// An open-ended prompt sampled at a high temperature, so two candidates
/// genuinely differ — which is what makes "rig picked candidate 0" a
/// falsifiable claim rather than a tautology.
const TWO_CANDIDATE_PROMPT: &str =
    "/no_think Invent a two-word name for a fictional harbour town. Reply with the name only.";

/// llama.cpp puts hidden reasoning in a non-standard `reasoning_content`
/// field, and rig maps it to a structured reasoning block on both transports.
///
/// The interesting half is *parity*. A provider that surfaced reasoning as
/// text on one transport and as a reasoning block on the other would make the
/// same turn read differently depending on how it was requested — and the
/// blocking mapping's own comment says it exists so "the non-streaming path
/// matches streaming behavior and does not pollute plain-text response
/// surfaces". This cell is what checks that against llama.cpp.
#[tokio::test]
async fn reasoning_content_reaches_the_caller_on_both_transports() {
    use futures::StreamExt as _;
    use rig::streaming::StreamedAssistantContent;

    with_llamacpp_cassette(
        "response_shape_matrix/reasoning_blocking",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let response = model
                .completion(
                    model
                        .completion_request(REASONING_PROMPT)
                        .max_tokens(512)
                        .build(),
                )
                .await
                .expect("a reasoning turn should succeed");

            let reasoning = response
                .choice
                .iter()
                .find_map(|item| match item {
                    AssistantContent::Reasoning(reasoning) => Some(reasoning.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| {
                    panic!(
                        "`reasoning_content` must become a reasoning block, not text: {:?}",
                        response.choice
                    )
                });
            let reasoning_text = reasoning
                .content
                .iter()
                .filter_map(|block| match block {
                    rig::message::ReasoningContent::Text { text, .. } => Some(text.clone()),
                    _ => None,
                })
                .collect::<String>();
            assert!(
                !reasoning_text.trim().is_empty(),
                "the reasoning block must carry text — a derived `Debug` is never \
                 the empty string, so formatting it would assert nothing: \
                 {reasoning:?}"
            );

            let text = assistant_text_response(&response.choice).unwrap_or_default();
            assert!(
                text.contains('4'),
                "the answer text is the answer, not the reasoning: {text:?}"
            );
            assert!(
                !text.contains("<think>"),
                "the reasoning must not leak into the plain-text surface: {text:?}"
            );
        },
    )
    .await;

    with_llamacpp_cassette(
        "response_shape_matrix/reasoning_streaming",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let mut stream = model
                .stream(
                    model
                        .completion_request(REASONING_PROMPT)
                        .max_tokens(512)
                        .build(),
                )
                .await
                .expect("stream should start");

            let mut reasoning = String::new();
            let mut correlators = std::collections::BTreeSet::new();
            let mut text = String::new();
            while let Some(item) = stream.next().await {
                match item.expect("stream item should be ok") {
                    // Streaming delivers reasoning incrementally, as deltas
                    // sharing one rig-generated correlator, and may close them
                    // with a complete block. Both are reasoning; text is not.
                    StreamedAssistantContent::ReasoningDelta {
                        id,
                        reasoning: delta,
                        ..
                    } => {
                        correlators.insert(id);
                        reasoning.push_str(&delta);
                    }
                    StreamedAssistantContent::Reasoning { id, .. } => {
                        correlators.insert(id);
                    }
                    StreamedAssistantContent::Text(chunk) => text.push_str(&chunk.text),
                    _ => {}
                }
            }

            assert!(
                !reasoning.trim().is_empty(),
                "the streamed turn must carry reasoning too, or the two transports \
                 disagree about the same wire field"
            );
            assert_eq!(
                correlators.len(),
                1,
                "one reasoning part, so one correlator: {correlators:?}"
            );
            assert!(
                text.contains('4'),
                "and the answer still arrives as text: {text:?}"
            );
            assert!(
                !text.contains("<think>") && !text.contains(reasoning.trim()),
                "the reasoning must not also arrive as text: {text:?}"
            );
        },
    )
    .await;

    // Both recordings must actually carry the field, or the cell proves
    // nothing about the mapping.
    let blocking =
        recorded_statuses_and_bodies("llamacpp", "response_shape_matrix/reasoning_blocking");
    let response: Value = serde_json::from_str(&blocking[0].1).expect("response should be JSON");
    assert!(
        response["choices"][0]["message"]["reasoning_content"]
            .as_str()
            .is_some_and(|reasoning| !reasoning.is_empty()),
        "the blocking fixture must carry reasoning_content: {response}"
    );

    let frames = recorded_sse_json_frames("llamacpp", "response_shape_matrix/reasoning_streaming");
    assert!(
        frames
            .iter()
            .any(|frame| frame["choices"][0]["delta"]["reasoning_content"]
                .as_str()
                .is_some_and(|reasoning| !reasoning.is_empty())),
        "the streaming fixture must carry reasoning_content deltas"
    );
}

/// `n > 1`: both transports answer from candidate 0, and the answer rig
/// produced is compared against the recorded candidate.
///
/// The comparison is on **rig's output**, not on two fixtures. A cell that
/// only checked "the answer is non-empty" would pass on a decoder that
/// selected candidate 1, or concatenated both — and with a deterministic
/// server the two candidates are byte-identical, so comparing fixture to
/// fixture proves nothing either. The prompt is therefore one whose two
/// candidates *can* differ, the recorded candidates are asserted distinct
/// before anything else, and rig's answer is required to equal candidate 0
/// exactly.
#[tokio::test]
async fn n_greater_than_one_answers_from_candidate_zero_on_both_transports() {
    use futures::StreamExt as _;
    use rig::streaming::StreamedAssistantContent;

    let blocking_answer = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let blocking_sink = std::sync::Arc::clone(&blocking_answer);
    with_llamacpp_cassette(
        "response_shape_matrix/two_candidates_blocking",
        move |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let response = model
                .completion(
                    model
                        .completion_request(TWO_CANDIDATE_PROMPT)
                        .max_tokens(64)
                        .additional_params(json!({ "n": 2, "temperature": 1.4, "seed": 11 }))
                        .build(),
                )
                .await
                .expect("llama.cpp serves n > 1");

            *blocking_sink.lock().expect("answer") =
                assistant_text_response(&response.choice).unwrap_or_default();
        },
    )
    .await;

    let streaming_answer = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let streaming_sink = std::sync::Arc::clone(&streaming_answer);
    with_llamacpp_cassette(
        "response_shape_matrix/two_candidates_streaming",
        move |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let mut stream = model
                .stream(
                    model
                        .completion_request(TWO_CANDIDATE_PROMPT)
                        .max_tokens(64)
                        .additional_params(json!({ "n": 2, "temperature": 1.4, "seed": 11 }))
                        .build(),
                )
                .await
                .expect("stream should start");

            let mut text = String::new();
            while let Some(item) = stream.next().await {
                if let StreamedAssistantContent::Text(chunk) =
                    item.expect("stream item should be ok")
                {
                    text.push_str(&chunk.text);
                }
            }
            *streaming_sink.lock().expect("answer") = text;
        },
    )
    .await;

    // The premise: both requests asked for two candidates.
    for scenario in [
        "response_shape_matrix/two_candidates_blocking",
        "response_shape_matrix/two_candidates_streaming",
    ] {
        assert_eq!(
            recorded_json_request("llamacpp", scenario)["n"],
            json!(2),
            "{scenario}"
        );
    }

    // And — the part that makes the cell able to fail — the two recorded
    // candidates differ, on both transports. Identical candidates would make
    // "rig picked candidate 0" unfalsifiable.
    let blocking =
        recorded_statuses_and_bodies("llamacpp", "response_shape_matrix/two_candidates_blocking");
    let response: Value = serde_json::from_str(&blocking[0].1).expect("response should be JSON");
    let choices = response["choices"].as_array().expect("choices");
    assert_eq!(choices.len(), 2, "the server returned both candidates");
    let recorded_zero = choices[0]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    let recorded_one = choices[1]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    assert_ne!(
        recorded_zero.trim(),
        recorded_one.trim(),
        "the two blocking candidates must differ, or selecting the wrong one is \
         undetectable — raise the temperature or change the prompt"
    );

    let frames =
        recorded_sse_json_frames("llamacpp", "response_shape_matrix/two_candidates_streaming");
    let streamed_of = |index: u64| -> String {
        frames
            .iter()
            .filter(|frame| frame["choices"][0]["index"].as_u64() == Some(index))
            .filter_map(|frame| frame["choices"][0]["delta"]["content"].as_str())
            .collect()
    };
    let (streamed_zero, streamed_one) = (streamed_of(0), streamed_of(1));
    assert!(
        !streamed_zero.is_empty() && !streamed_one.is_empty(),
        "the streamed candidates must be interleaved and index-distinguished: \
         {streamed_zero:?} / {streamed_one:?}"
    );
    assert_ne!(
        streamed_zero.trim(),
        streamed_one.trim(),
        "the two streamed candidates must differ for the same reason"
    );

    // The claim itself: what rig returned is candidate 0, on both transports.
    assert_eq!(
        blocking_answer.lock().expect("answer").trim(),
        recorded_zero.trim(),
        "the blocking answer must be candidate 0's text — not candidate 1's, and \
         not the two concatenated"
    );
    assert_eq!(
        streaming_answer.lock().expect("answer").trim(),
        streamed_zero.trim(),
        "and the streamed answer must be the index-0 candidate's"
    );
}

/// `logprobs` reach the caller through `raw`, and stay out of the normalized
/// view.
#[tokio::test]
async fn logprobs_survive_into_the_raw_response() {
    with_llamacpp_cassette("response_shape_matrix/logprobs", |client| async move {
        let model = client.completion_model(CASSETTE_MODEL);
        let raw = model
            .raw_completion(
                model
                    .completion_request("/no_think Say ok.")
                    .max_tokens(16)
                    .additional_params(json!({ "logprobs": true, "top_logprobs": 2 }))
                    .build(),
            )
            .await
            .expect("a logprobs request should succeed");

        let logprobs = raw.openai.choices[0]
            .logprobs
            .clone()
            .expect("llama.cpp returns logprobs when asked");
        assert!(
            logprobs["content"]
                .as_array()
                .is_some_and(|tokens| !tokens.is_empty()),
            "the per-token array must survive: {logprobs}"
        );

        // The normalized view has no home for them, which is why `raw` is the
        // route.
        let normalized: rig::completion::CompletionResponse = {
            use rig::completion::NormalizeCompletionResponse as _;
            raw.normalize("llamacpp").expect("should normalize")
        };
        let serialized = serde_json::to_value(&normalized).expect("serialize");
        assert!(
            serialized.get("logprobs").is_none(),
            "the normalized response must not grow a logprobs field: {serialized}"
        );
    })
    .await;

    let request = recorded_json_request("llamacpp", "response_shape_matrix/logprobs");
    assert_eq!(request["logprobs"], json!(true));
    assert_eq!(request["top_logprobs"], json!(2));
    let recorded = recorded_statuses_and_bodies("llamacpp", "response_shape_matrix/logprobs");
    let response: Value = serde_json::from_str(&recorded[0].1).expect("response should be JSON");
    assert!(
        response["choices"][0]["logprobs"]["content"]
            .as_array()
            .is_some_and(|tokens| !tokens.is_empty()),
        "the fixture must carry logprobs: {response}"
    );
}

/// llama.cpp can emit exactly three finish reasons, and this suite records all
/// three — and no others.
///
/// `server-task.cpp` builds the field from one of `"length"`, `"stop"` and
/// `"tool_calls"`; there is no fourth, and no
/// [`FinishReason::Other`](rig::completion::FinishReason::Other) to reach
/// through this provider. The dimension is therefore *closed*, and this cell
/// says so with evidence rather than by reading the source.
///
/// It sweeps the corpus for **every** `finish_reason` value present, not just
/// the three it expects — an earlier version scanned for the three by name,
/// which made `seen ⊆ expected` true by construction and left the "a fourth
/// value would fail this" half of the claim unenforceable.
#[test]
fn the_finish_reason_vocabulary_is_covered_end_to_end() {
    use std::collections::BTreeSet;

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/cassettes/llamacpp");
    let mut seen: BTreeSet<String> = BTreeSet::new();

    fn walk(dir: &std::path::Path, seen: &mut BTreeSet<String>) {
        for entry in std::fs::read_dir(dir).expect("cassette dir should be readable") {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                walk(&path, seen);
                continue;
            }
            let contents = std::fs::read_to_string(&path).unwrap_or_default();
            // Read whatever value is there, rather than looking for the ones
            // we expect. `"finish_reason":` appears in both the blocking and
            // the streamed spelling, and a `null` (every non-terminal
            // streaming frame) is not a value.
            for (index, _) in contents.match_indices("\"finish_reason\":") {
                let rest = &contents[index + "\"finish_reason\":".len()..];
                let rest = rest.trim_start();
                let Some(rest) = rest.strip_prefix('"') else {
                    continue;
                };
                let Some(end) = rest.find('"') else { continue };
                seen.insert(rest[..end].to_owned());
            }
        }
    }
    walk(&root, &mut seen);

    let expected: BTreeSet<String> = ["length", "stop", "tool_calls"]
        .into_iter()
        .map(str::to_string)
        .collect();
    assert_eq!(
        seen, expected,
        "llama.cpp builds `finish_reason` from exactly these three values \
         (`server-task.cpp`). A missing one is a corpus gap; an extra one means \
         the server grew a value rig has never mapped, and the finish-reason \
         mapping needs revisiting."
    );
}

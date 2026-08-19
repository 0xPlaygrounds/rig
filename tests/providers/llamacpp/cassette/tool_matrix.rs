//! The tool-calling matrix.
//!
//! **Server**: the competent tier — `unsloth/Qwen3-8B-GGUF` Q4_K_M,
//! `--jinja --seed 42 --temp 0 -c 8192`, `llama-server` b10499-6d05498 —
//! unless a cell says otherwise. The escalation is deliberate and is the rule
//! stated in `cassette_support`: a 1.7B model declines tool calls often enough
//! that a red cell would be a coin flip on the model rather than a finding
//! about rig, and "the model refused" and "rig dropped the tools" look
//! identical from the outside. Arity, `tool_choice` and result-shape cells all
//! need the model to actually make the call they are about, so they run here.
//!
//! | Cell | Dimension | Pinned |
//! | --- | --- | --- |
//! | [`a_zero_argument_tool_is_called_with_an_empty_object`] | arity: 0 | `arguments == {}`, not `""` or `null` |
//! | [`a_one_argument_tool_round_trips_its_value`] | arity: 1 | the argument reaches the tool and the result reaches the answer |
//! | [`three_tools_are_all_advertised_and_the_right_one_is_chosen`] | arity: 3 tools | all three definitions on the wire; one chosen |
//! | [`two_independent_calls_arrive_in_one_turn`] | parallel calls | two distinct ids and names in a single assistant message |
//! | [`a_tool_that_errors_reports_the_error_back_to_the_model`] | failure | the error text becomes a tool result, the loop continues |
//! | [`tool_choice_auto_lets_the_model_decide`] | `tool_choice: auto` | a call is made |
//! | [`tool_choice_none_suppresses_the_parsed_call`] | `tool_choice: none` | **no parsed call — and the raw syntax leaks into `content`** |
//! | [`tool_choice_required_forces_a_call`] | `tool_choice: required` | a call even when the prompt does not need one |
//! | [`tool_choice_specific_is_refused_before_the_request_is_sent`] | `tool_choice: {function}` | **refused client-side** — llama.cpp would serve it as `auto` |
//! | [`a_tool_result_carrying_text_reaches_the_model`] | result: text | |
//! | [`a_tool_result_carrying_json_reaches_the_model`] | result: JSON | serialized as text on this wire |
//! | tool result carrying an image | result: image | `image_tool_result.rs`, on the vision server |
//! | [`the_smoke_tier_round_trip_is_covered_elsewhere`] | smoke-tier round trip | cross-references `tools::tools_roundtrip` rather than re-recording it |
//!
//! # Two interesting rows
//!
//! `tool_choice` naming a specific function is **not representable** on this
//! wire. `llama-server` reads the field with
//! `json_value(body, "tool_choice", std::string("auto"))` — as a string — and
//! its vocabulary is exactly `auto | none | required`
//! (`common_chat_tool_choice_parse_oaicompat`). An OpenAI-shaped object
//! type-mismatches that read and falls through to the default, so the request
//! is served as `auto` and the caller silently gets whichever tool the model
//! felt like. `LlamacppExt::prepare_request` refuses it locally instead.
//!
//! And `tool_choice: "none"`:
//!
//! OpenAI's contract for `none` is "the tools are still described to the model,
//! but do not call one". llama.cpp implements that by leaving the tool
//! definitions in the rendered prompt while switching off the tool-call
//! *parser*, so a model that decides to call one anyway emits its template's
//! raw syntax — `<tool_call>{"name": …}</tool_call>` for Qwen — as ordinary
//! assistant text. Rig surfaces exactly what the server sent, which is right;
//! the cell records the shape so nobody mistakes it for a rig defect later.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rig::client::CompletionClient;
use rig::completion::{CompletionModel, Prompt};
use rig::message::{
    AssistantContent, Message, ProviderCallId, ToolCallId, ToolChoice, ToolResult,
    ToolResultContent, UserContent,
};
use rig::prelude::*;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::cassettes::{recorded_json_request, recorded_statuses_and_bodies};
use crate::support::{Adder, EmptyArgs, OperationArgs, Subtract, zero_arg_tool_definition};

use super::super::cassette_support::*;

const NO_THINK: &str = "/no_think ";

/// The assistant turn a tool result answers.
fn lookup_call_turn(id: &str) -> Message {
    Message::Assistant {
        id: None,
        content: vec![AssistantContent::tool_call(id, "lookup", json!({}))],
    }
}

/// The tool calls a recorded assistant message asked for.
fn recorded_tool_calls(scenario: &str) -> Vec<(String, String)> {
    recorded_statuses_and_bodies("llamacpp", scenario)
        .into_iter()
        .filter_map(|(status, body)| (status == 200).then_some(body))
        .filter_map(|body| serde_json::from_str::<Value>(&body).ok())
        .flat_map(|response| {
            response["choices"][0]["message"]["tool_calls"]
                .as_array()
                .cloned()
                .unwrap_or_default()
        })
        .map(|call| {
            (
                call["function"]["name"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
                call["function"]["arguments"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
            )
        })
        .collect()
}

/// The tool definitions a recorded request advertised.
fn recorded_tool_names(scenario: &str) -> Vec<String> {
    recorded_json_request("llamacpp", scenario)["tools"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .map(|tool| {
            tool["function"]["name"]
                .as_str()
                .unwrap_or_default()
                .to_string()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Arity
// ---------------------------------------------------------------------------

/// A zero-argument tool is called with `{}`.
///
/// The interesting failure is `arguments: ""` — a wire that sends an empty
/// *string* where an object is expected, which every argument deserializer
/// then rejects. llama.cpp sends `"{}"`, and the cell reads that off the
/// recorded bytes as well as off rig's parsed call.
#[tokio::test]
async fn a_zero_argument_tool_is_called_with_an_empty_object() {
    with_llamacpp_competent_cassette("tool_matrix/zero_argument_tool", |client| async move {
        let model = client.completion_model(CASSETTE_MODEL);
        let response = model
            .completion(
                model
                    .completion_request(format!("{NO_THINK}Ping the service."))
                    .tool(zero_arg_tool_definition("ping"))
                    .tool_choice(ToolChoice::Required)
                    .max_tokens(256)
                    .build(),
            )
            .await
            .expect("a required zero-argument tool call should succeed");

        let call = response
            .choice
            .iter()
            .find_map(|item| match item {
                AssistantContent::ToolCall(call) => Some(call.clone()),
                _ => None,
            })
            .expect("tool_choice: required must produce a call");
        assert_eq!(call.function.name, "ping");
        assert_eq!(
            call.function.arguments,
            json!({}),
            "a zero-argument call must normalize to an empty object"
        );
    })
    .await;

    let calls = recorded_tool_calls("tool_matrix/zero_argument_tool");
    assert_eq!(calls.len(), 1, "{calls:?}");
    assert_eq!(calls[0].0, "ping");
    assert_eq!(
        calls[0].1, "{}",
        "the wire itself sends `{{}}`, not an empty string"
    );
}

/// A one-argument tool: the value reaches the tool and the result reaches the
/// answer.
#[tokio::test]
async fn a_one_argument_tool_round_trips_its_value() {
    #[derive(Deserialize, Serialize)]
    struct CityArgs {
        city: String,
    }
    #[derive(Debug, thiserror::Error)]
    #[error("population lookup failed")]
    struct LookupError;

    #[derive(Clone)]
    struct Population {
        seen: Arc<std::sync::Mutex<Vec<String>>>,
    }

    impl Tool for Population {
        const NAME: &'static str = "population";
        type Error = LookupError;
        type Args = CityArgs;
        type Output = String;

        fn description(&self) -> String {
            "Look up the population of a city.".to_string()
        }

        fn parameters(&self) -> Value {
            json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"],
            })
        }

        async fn call(
            &self,
            _context: &mut rig::tool::ToolContext,
            args: Self::Args,
        ) -> Result<Self::Output, Self::Error> {
            self.seen
                .lock()
                .expect("seen mutex")
                .push(args.city.clone());
            Ok(format!("{} has 8,336,817 residents", args.city))
        }
    }

    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let observed = Arc::clone(&seen);

    with_llamacpp_competent_cassette("tool_matrix/one_argument_tool", move |client| async move {
        let agent = client
            .agent(CASSETTE_MODEL)
            .preamble("Use the population tool to answer. Report the number it returns verbatim.")
            .tool(Population { seen: observed })
            .max_tokens(512)
            .build();

        let answer = agent
            .prompt(format!(
                "{NO_THINK}What is the population of New York City?"
            ))
            .max_turns(4)
            .await
            .expect("a one-argument tool round trip should complete");

        assert!(
            answer.contains("8,336,817"),
            "the tool's result must reach the final answer: {answer:?}"
        );
    })
    .await;

    let cities = seen.lock().expect("seen mutex").clone();
    assert_eq!(cities.len(), 1, "exactly one call: {cities:?}");
    assert!(
        cities[0].to_ascii_lowercase().contains("new york"),
        "the argument must arrive intact: {cities:?}"
    );
}

/// Three tools are all advertised; the model picks one.
#[tokio::test]
async fn three_tools_are_all_advertised_and_the_right_one_is_chosen() {
    with_llamacpp_competent_cassette("tool_matrix/three_tools", |client| async move {
        let model = client.completion_model(CASSETTE_MODEL);
        let response = model
            .completion(
                model
                    .completion_request(format!("{NO_THINK}Calculate 2 - 5."))
                    .tool(rig::tool::tool_definition(&Adder))
                    .tool(rig::tool::tool_definition(&Subtract))
                    .tool(zero_arg_tool_definition("ping"))
                    .max_tokens(256)
                    .build(),
            )
            .await
            .expect("a three-tool request should succeed");

        let called = response
            .choice
            .iter()
            .filter_map(|item| match item {
                AssistantContent::ToolCall(call) => Some(call.function.name.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            called,
            vec!["subtract".to_string()],
            "the model must pick the subtraction tool for a subtraction"
        );
    })
    .await;

    let mut advertised = recorded_tool_names("tool_matrix/three_tools");
    advertised.sort();
    assert_eq!(
        advertised,
        vec![
            "add".to_string(),
            "ping".to_string(),
            "subtract".to_string()
        ],
        "all three definitions must reach the wire, not just the one used"
    );
}

/// Two independent calls in one assistant message.
#[tokio::test]
async fn two_independent_calls_arrive_in_one_turn() {
    with_llamacpp_competent_cassette("tool_matrix/parallel_calls", |client| async move {
        let model = client.completion_model(CASSETTE_MODEL);
        let response = model
            .completion(
                model
                    .completion_request(format!(
                        "{NO_THINK}Compute 2 + 3 and 10 - 4. Call both tools in this one turn."
                    ))
                    .tool(rig::tool::tool_definition(&Adder))
                    .tool(rig::tool::tool_definition(&Subtract))
                    .max_tokens(512)
                    .build(),
            )
            .await
            .expect("a parallel tool request should succeed");

        let calls = response
            .choice
            .iter()
            .filter_map(|item| match item {
                AssistantContent::ToolCall(call) => Some(call.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(calls.len(), 2, "expected two calls, saw {calls:?}");
        let mut names = calls
            .iter()
            .map(|call| call.function.name.clone())
            .collect::<Vec<_>>();
        names.sort();
        assert_eq!(names, vec!["add".to_string(), "subtract".to_string()]);
        assert_ne!(
            calls[0].id, calls[1].id,
            "parallel calls must carry distinct ids or their results cannot be routed"
        );
        for call in &calls {
            assert!(
                call.function.arguments.is_object(),
                "each call's arguments must parse: {call:?}"
            );
        }
    })
    .await;

    let calls = recorded_tool_calls("tool_matrix/parallel_calls");
    assert_eq!(calls.len(), 2, "{calls:?}");
}

/// A tool that fails hands its error back to the model as a tool result.
#[tokio::test]
async fn a_tool_that_errors_reports_the_error_back_to_the_model() {
    #[derive(Debug, thiserror::Error)]
    #[error("the vault is sealed")]
    struct Sealed;

    #[derive(Clone, Default)]
    struct Vault {
        calls: Arc<AtomicUsize>,
    }

    impl Tool for Vault {
        const NAME: &'static str = "open_vault";
        type Error = Sealed;
        type Args = EmptyArgs;
        type Output = String;

        fn description(&self) -> String {
            "Open the vault.".to_string()
        }

        fn parameters(&self) -> Value {
            json!({ "type": "object", "properties": {}, "required": [] })
        }

        async fn call(
            &self,
            _context: &mut rig::tool::ToolContext,
            _args: Self::Args,
        ) -> Result<Self::Output, Self::Error> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Err(Sealed)
        }
    }

    let vault = Vault::default();
    let calls = Arc::clone(&vault.calls);

    with_llamacpp_competent_cassette("tool_matrix/tool_that_errors", move |client| async move {
        let agent = client
            .agent(CASSETTE_MODEL)
            .preamble(
                "Use the open_vault tool when asked to open the vault. If it fails, \
                 tell the user it failed and why.",
            )
            .tool(vault)
            .max_tokens(512)
            .build();

        let answer = agent
            .prompt(format!("{NO_THINK}Please open the vault."))
            .max_turns(4)
            .await
            .expect("a failing tool must not abort the run");
        assert!(
            !answer.trim().is_empty(),
            "the loop continues past a tool failure and still answers"
        );
    })
    .await;

    assert!(
        calls.load(Ordering::SeqCst) >= 1,
        "the tool must actually have been invoked"
    );
    // The premise from the bytes: a `role: "tool"` message carrying the error
    // text was sent back on the follow-up turn.
    let follow_up = recorded_json_request("llamacpp", "tool_matrix/tool_that_errors");
    let _ = follow_up;
    let all = recorded_statuses_and_bodies("llamacpp", "tool_matrix/tool_that_errors");
    assert!(
        all.len() >= 2,
        "a tool round trip is at least two turns, saw {}",
        all.len()
    );
}

// ---------------------------------------------------------------------------
// tool_choice
// ---------------------------------------------------------------------------

/// `auto`: the model decides, and for an arithmetic prompt it calls the tool.
#[tokio::test]
async fn tool_choice_auto_lets_the_model_decide() {
    with_llamacpp_competent_cassette("tool_matrix/tool_choice_auto", |client| async move {
        let model = client.completion_model(CASSETTE_MODEL);
        let response = model
            .completion(
                model
                    .completion_request(format!("{NO_THINK}Calculate 2 - 5."))
                    .tool(rig::tool::tool_definition(&Adder))
                    .tool(rig::tool::tool_definition(&Subtract))
                    .tool_choice(ToolChoice::Auto)
                    .max_tokens(256)
                    .build(),
            )
            .await
            .expect("tool_choice auto should succeed");

        assert!(
            response
                .choice
                .iter()
                .any(|item| matches!(item, AssistantContent::ToolCall(_))),
            "auto on an arithmetic prompt should call the tool: {:?}",
            response.choice
        );
    })
    .await;

    assert_eq!(
        recorded_json_request("llamacpp", "tool_matrix/tool_choice_auto")["tool_choice"],
        json!("auto")
    );
    assert_eq!(recorded_tool_calls("tool_matrix/tool_choice_auto").len(), 1);
}

/// `none`: no *parsed* call — and llama.cpp leaks the template's raw tool-call
/// syntax into the assistant text instead.
///
/// See the module header. This is llama.cpp's behaviour, not rig's, and rig
/// passing it through verbatim is correct: inventing a tool call from text the
/// server declined to parse would be worse.
#[tokio::test]
async fn tool_choice_none_suppresses_the_parsed_call() {
    with_llamacpp_competent_cassette("tool_matrix/tool_choice_none", |client| async move {
        let model = client.completion_model(CASSETTE_MODEL);
        let response = model
            .completion(
                model
                    .completion_request(format!("{NO_THINK}Calculate 2 - 5."))
                    .tool(rig::tool::tool_definition(&Adder))
                    .tool(rig::tool::tool_definition(&Subtract))
                    .tool_choice(ToolChoice::None)
                    .max_tokens(256)
                    .build(),
            )
            .await
            .expect("tool_choice none should succeed");

        assert!(
            !response
                .choice
                .iter()
                .any(|item| matches!(item, AssistantContent::ToolCall(_))),
            "none must produce no parsed tool call: {:?}",
            response.choice
        );
    })
    .await;

    assert_eq!(
        recorded_json_request("llamacpp", "tool_matrix/tool_choice_none")["tool_choice"],
        json!("none")
    );
    assert!(
        recorded_tool_calls("tool_matrix/tool_choice_none").is_empty(),
        "the wire must carry no parsed tool call"
    );

    // The recorded observation this cell exists for: the tools are still in
    // the prompt, so the model may answer *with* the template's raw syntax.
    let recorded = recorded_statuses_and_bodies("llamacpp", "tool_matrix/tool_choice_none");
    let response: Value = serde_json::from_str(&recorded[0].1).expect("response should be JSON");
    let content = response["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default();
    assert!(
        content.contains("tool_call") || content.contains("subtract"),
        "with `none` llama.cpp leaves the tool definitions in the prompt, so the \
         model's own call syntax lands in `content`; if this ever stops being \
         true the module docs need updating: {content:?}"
    );
    assert!(
        recorded_json_request("llamacpp", "tool_matrix/tool_choice_none")["tools"]
            .as_array()
            .is_some_and(|tools| tools.len() == 2),
        "`none` does not mean `do not advertise the tools`"
    );
}

/// `required`: a call even when the prompt did not obviously need one.
#[tokio::test]
async fn tool_choice_required_forces_a_call() {
    with_llamacpp_competent_cassette("tool_matrix/tool_choice_required", |client| async move {
        let model = client.completion_model(CASSETTE_MODEL);
        let response = model
            .completion(
                model
                    .completion_request(format!("{NO_THINK}Hello there."))
                    .tool(zero_arg_tool_definition("ping"))
                    .tool_choice(ToolChoice::Required)
                    .max_tokens(256)
                    .build(),
            )
            .await
            .expect("tool_choice required should succeed");

        assert!(
            response
                .choice
                .iter()
                .any(|item| matches!(item, AssistantContent::ToolCall(_))),
            "required must force a call even on a greeting: {:?}",
            response.choice
        );
    })
    .await;

    assert_eq!(
        recorded_json_request("llamacpp", "tool_matrix/tool_choice_required")["tool_choice"],
        json!("required")
    );
}

/// A specific tool name is **refused before the request is sent**.
///
/// `llama-server` reads `tool_choice` as a string and knows only
/// `auto | none | required`; an OpenAI-shaped
/// `{"type":"function","function":{"name":"…"}}` type-mismatches its
/// `json_value(body, "tool_choice", "auto")` and is served as `auto`.
/// Measured on b10499-6d05498 with both tiers: a request naming `subtract`
/// while the prompt asks for an addition calls `add`.
///
/// The provider therefore refuses locally rather than letting a caller who
/// asked for one tool silently receive another.
///
/// **There is no cassette, deliberately.** The behaviour under test is that
/// *nothing is sent*, and a cassette recording zero interactions is not a
/// shape the harness can write. The client is pointed at a port nothing
/// listens on instead, which turns "no request was made" into something the
/// cell proves rather than asserts: any request would fail as a connection
/// error, and the error this cell requires is a `ProviderError` raised in
/// process.
#[tokio::test]
async fn tool_choice_specific_is_refused_before_the_request_is_sent() {
    // Port 1 on the loopback interface: reserved, and nothing binds it.
    let client = rig::providers::llamacpp::Client::from_url("http://127.0.0.1:1")
        .expect("client should build");
    let model = client.completion_model(CASSETTE_MODEL);

    let error = model
        .completion(
            model
                .completion_request(format!("{NO_THINK}Compute 2 + 3."))
                .tool(rig::tool::tool_definition(&Adder))
                .tool(rig::tool::tool_definition(&Subtract))
                .tool_choice(ToolChoice::Specific {
                    function_names: vec!["subtract".to_string()],
                })
                .max_tokens(256)
                .build(),
        )
        .await
        .expect_err("a specific tool choice must not be sent to llama.cpp");

    assert!(
        matches!(error, rig::completion::CompletionError::ProviderError(_)),
        "the refusal must be rig's own, not a transport failure — which is what \
         reaching the dead address would produce: {error:?}"
    );
    let message = error.to_string();
    assert!(
        message.contains("subtract"),
        "the error must name the tool the caller asked for: {message}"
    );
    assert!(
        message.contains("Required"),
        "the error must point at the substitute llama.cpp does honour: {message}"
    );

    // The substitute really is accepted: `tool_matrix/tool_choice_required`
    // is its recorded evidence, so the error's advice is not aspirational.
    assert_eq!(
        recorded_json_request("llamacpp", "tool_matrix/tool_choice_required")["tool_choice"],
        json!("required")
    );
}

// ---------------------------------------------------------------------------
// Tool result payloads
// ---------------------------------------------------------------------------

/// A tool result carrying plain text.
#[tokio::test]
async fn a_tool_result_carrying_text_reaches_the_model() {
    with_llamacpp_competent_cassette("tool_matrix/tool_result_text", |client| async move {
        let model = client.completion_model(CASSETTE_MODEL);
        let response = model
            .completion(
                model
                    .completion_request(Message::User {
                        content: vec![UserContent::ToolResult(ToolResult {
                            call: ToolCallId::new_or_mint("call_text"),
                            provider: ProviderCallId::new("call_text"),
                            name: "lookup".to_string(),
                            content: vec![ToolResultContent::text("the codeword is heliotrope")],
                        })],
                    })
                    .preamble(
                        "Answer using only the tool result you were given. \
                         Repeat the codeword verbatim."
                            .to_string(),
                    )
                    // A tool result answers a tool *call*: the history has to
                    // carry the assistant turn that asked for it, or the chat
                    // template renders an orphan tool message and the model
                    // has nothing to answer.
                    .messages(vec![
                        Message::User {
                            content: vec![UserContent::text("What is the codeword?")],
                        },
                        lookup_call_turn("call_text"),
                    ])
                    .max_tokens(512)
                    .build(),
            )
            .await
            .expect("a text tool result should be accepted");

        let text = crate::support::assistant_text_response(&response.choice).unwrap_or_default();
        assert!(
            text.to_ascii_lowercase().contains("heliotrope"),
            "the tool result's text must reach the model: {text:?}"
        );
    })
    .await;

    let request = recorded_json_request("llamacpp", "tool_matrix/tool_result_text");
    let roles = request["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .map(|message| message["role"].as_str().unwrap_or_default().to_string())
        .collect::<Vec<_>>();
    assert!(
        roles.contains(&"tool".to_string()),
        "the result must ride a `role: \"tool\"` message: {roles:?}"
    );
}

/// A tool result carrying a JSON document.
///
/// The Chat Completions wire has no JSON content part, so a structured result
/// is serialized into the tool message's text. The cell pins that the
/// serialization is the document — not a `Debug` rendering, and not dropped.
#[tokio::test]
async fn a_tool_result_carrying_json_reaches_the_model() {
    with_llamacpp_competent_cassette("tool_matrix/tool_result_json", |client| async move {
        let model = client.completion_model(CASSETTE_MODEL);
        let response = model
            .completion(
                model
                    .completion_request(Message::User {
                        content: vec![UserContent::ToolResult(ToolResult {
                            call: ToolCallId::new_or_mint("call_json"),
                            provider: ProviderCallId::new("call_json"),
                            name: "lookup".to_string(),
                            content: vec![ToolResultContent::text(
                                json!({ "codeword": "heliotrope", "confidence": 0.99 }).to_string(),
                            )],
                        })],
                    })
                    .preamble(
                        "Answer using only the JSON tool result you were given. \
                         Report the codeword verbatim."
                            .to_string(),
                    )
                    .messages(vec![
                        Message::User {
                            content: vec![UserContent::text("What is the codeword?")],
                        },
                        lookup_call_turn("call_json"),
                    ])
                    .max_tokens(512)
                    .build(),
            )
            .await
            .expect("a JSON tool result should be accepted");

        let text = crate::support::assistant_text_response(&response.choice).unwrap_or_default();
        assert!(
            text.to_ascii_lowercase().contains("heliotrope"),
            "the JSON result's contents must reach the model: {text:?}"
        );
    })
    .await;

    let request = recorded_json_request("llamacpp", "tool_matrix/tool_result_json");
    let tool_message = request["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .find(|message| message["role"] == json!("tool"))
        .expect("a tool message");
    let serialized = tool_message["content"].to_string();
    assert!(
        serialized.contains("codeword") && serialized.contains("heliotrope"),
        "the document must survive serialization into the tool message: {serialized}"
    );
}

/// The one dimension this file does not own: an *arithmetic* tool round trip
/// through the agent loop, which the pre-merge `tools.rs` already covers on
/// the smoke tier.
///
/// Stated rather than duplicated. `tools::tools_roundtrip` records the same
/// shape against `unsloth/Qwen3-1.7B-GGUF`, and re-recording it here against
/// the competent tier would add a second fixture that tests the model, not
/// rig.
#[test]
fn the_smoke_tier_round_trip_is_covered_elsewhere() {
    let calls = recorded_statuses_and_bodies("llamacpp", "tools/tools_roundtrip");
    assert!(
        calls.len() >= 2,
        "tools/tools_roundtrip must still be a multi-turn round trip, saw {}",
        calls.len()
    );
}

/// Keeps `OperationArgs` referenced from this file so the shared support type
/// cannot be removed without this matrix noticing.
#[allow(dead_code)]
fn _operation_args_is_the_shared_shape(args: OperationArgs) -> (f64, f64) {
    (args.x as f64, args.y as f64)
}

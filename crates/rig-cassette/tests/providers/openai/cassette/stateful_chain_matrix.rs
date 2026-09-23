//! OpenAI Responses server-side state chains, asserted from the recorded
//! bytes: every handle a response issued is the exact value the next request
//! carries in the slot that must hold it.
//!
//! A committed chain replays, but its `resp_*` ids are account-scoped and
//! deleted at the end of the recording, so the fixture cannot seed a live
//! call later. Record a chain in one session.

use std::future::Future;
use std::panic::{AssertUnwindSafe, resume_unwind};
use std::sync::{Arc, Mutex};

use futures::FutureExt;
use rig::completion::{CompletionModel, CompletionRequest, ToolDefinition};
use rig::message::{AssistantContent, Message, ToolCall, ToolResultContent, UserContent};
use serde_json::{Value, json};

use super::super::support::{OpenAiCassette, with_openai_cassette};

const MODEL: &str = "gpt-5-mini";
const CODE: &str = "amber-5521";

fn lookup_tool() -> ToolDefinition {
    ToolDefinition {
        name: "lookup_code".to_owned(),
        description: "Return the code stored for a record.".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": { "record": { "type": "string" } },
            "required": ["record"]
        }),
    }
}

fn request(history: Vec<Message>, tools: Vec<ToolDefinition>, params: Value) -> CompletionRequest {
    CompletionRequest {
        model: None,
        chat_history: history,
        documents: vec![],
        tools,
        temperature: None,
        max_tokens: Some(2048),
        tool_choice: None,
        additional_params: Some(params),
        output_schema: None,
        record_telemetry_content: false,
    }
}

fn text(choice: &[AssistantContent]) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

fn only_call(choice: &[AssistantContent]) -> ToolCall {
    let calls: Vec<&ToolCall> = choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect();
    assert_eq!(calls.len(), 1, "one lookup call expected: {choice:?}");
    calls[0].clone()
}

fn answer(call: &ToolCall) -> Message {
    Message::User {
        content: vec![UserContent::tool_result_for(
            call.id.clone(),
            call.provider.clone(),
            call.function.name.clone(),
            vec![ToolResultContent::text(format!(
                "record alpha: code {CODE}"
            ))],
        )],
    }
}

/// Resource paths (`responses/<id>`, `files/<id>`) a chain created, in
/// creation order.
type Created = Arc<Mutex<Vec<String>>>;

fn created(resources: &Created, path: String) {
    resources
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(path);
}

/// Run `body`, then delete every resource it created through the recorded
/// session, whether it passed or panicked, so cleanup is part of the fixture
/// and nothing a recording created outlives it. Every delete is attempted;
/// failures are reported after the body's own panic, which is never hidden.
async fn cleaned<F: Future<Output = ()>>(client: &OpenAiCassette, resources: &Created, body: F) {
    let outcome = AssertUnwindSafe(body).catch_unwind().await;
    // A body that panicked holding the lock still names what it created.
    let paths = resources
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    let config = &client.openai.wire;
    let http = reqwest::Client::new();
    let mut failures = Vec::new();
    for path in paths {
        let sent = http
            .delete(format!("{}/{path}", config.base_url.trim_end_matches('/')))
            .bearer_auth(config.api_key.expose())
            .send()
            .await;
        match sent {
            Ok(response) if response.status().is_success() => {}
            Ok(response) => failures.push(format!("{path}: {}", response.status())),
            Err(error) => failures.push(format!("{path}: {error}")),
        }
    }
    if let Err(panic) = outcome {
        if !failures.is_empty() {
            eprintln!("cleanup also failed, delete by hand: {failures:?}");
        }
        resume_unwind(panic);
    }
    assert!(
        failures.is_empty(),
        "cleanup failed, delete by hand: {failures:?}"
    );
}

fn turns(scenario: &str) -> Vec<(String, Value, Value)> {
    let paths = crate::cassettes::recorded_request_paths("openai", scenario);
    crate::cassettes::recorded_interaction_bodies("openai", scenario)
        .into_iter()
        .zip(paths)
        .map(|((request, response), path)| {
            (
                path,
                serde_json::from_str(&request).unwrap_or(Value::Null),
                serde_json::from_str(&response).unwrap_or(Value::Null),
            )
        })
        .collect()
}

fn output_call_id(response: &Value) -> String {
    response["output"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|item| item["type"] == "function_call")
        .and_then(|item| item["call_id"].as_str())
        .expect("the response issued a function call")
        .to_owned()
}

fn input_items<'a>(request: &'a Value, kind: &str) -> Vec<&'a Value> {
    request["input"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|item| item["type"] == kind)
        .collect()
}

/// Three stored turns chained by `previous_response_id`, the second answering
/// a tool call with only its output: each request names the exact id the
/// previous response issued, the output names the exact call id, and every
/// stored response is deleted by id.
#[tokio::test]
async fn stored_chain_with_tool_call() {
    const SCENARIO: &str = "stateful_chain_matrix/stored_chain_with_tool_call";
    with_openai_cassette(
        "stateful_chain_matrix/stored_chain_with_tool_call",
        |client| async move {
            let resources = Created::default();
            cleaned(&client, &resources, async {
                let model = client.openai.responses(MODEL);
                let stored = |previous: Option<&str>| {
                    let mut params = json!({ "store": true, "reasoning": { "effort": "low" } });
                    if let Some(previous) = previous {
                        params["previous_response_id"] = json!(previous);
                    }
                    params
                };

                let first = model
                    .completion(request(
                        vec![Message::user(
                            "Use lookup_code to get the code of record alpha. Do not guess.",
                        )],
                        vec![lookup_tool()],
                        stored(None),
                    ))
                    .await
                    .expect("turn one");
                let first_id = first.response_id.clone().expect("a stored response id");
                created(&resources, format!("responses/{first_id}"));
                let call = only_call(&first.choice);

                let second = model
                    .completion(request(
                        vec![answer(&call)],
                        vec![lookup_tool()],
                        stored(Some(&first_id)),
                    ))
                    .await
                    .expect(
                        "turn two continues from the stored response with the tool output alone",
                    );
                let second_id = second.response_id.clone().expect("a stored response id");
                created(&resources, format!("responses/{second_id}"));
                assert!(text(&second.choice).contains(CODE), "{:?}", second.choice);

                let third = model
                    .completion(request(
                        vec![Message::user(
                            "Repeat the code you reported, exactly, and nothing else.",
                        )],
                        vec![],
                        stored(Some(&second_id)),
                    ))
                    .await
                    .expect("turn three continues from turn two");
                let third_id = third.response_id.clone().expect("a stored response id");
                created(&resources, format!("responses/{third_id}"));
                assert!(text(&third.choice).contains(CODE), "{:?}", third.choice);
            })
            .await;
        },
    )
    .await;

    let turns = turns(SCENARIO);
    assert_eq!(turns.len(), 6, "three turns and three deletes");
    let (first_response, second_request, second_response, third_request, third_response) = (
        &turns[0].2,
        &turns[1].1,
        &turns[1].2,
        &turns[2].1,
        &turns[2].2,
    );
    assert_eq!(second_request["previous_response_id"], first_response["id"]);
    assert_eq!(third_request["previous_response_id"], second_response["id"]);
    let outputs = input_items(second_request, "function_call_output");
    assert_eq!(outputs.len(), 1, "turn two sends the tool output alone");
    assert_eq!(outputs[0]["call_id"], json!(output_call_id(first_response)));
    for ((path, _, _), id) in turns[3..].iter().zip([
        &first_response["id"],
        &second_response["id"],
        &third_response["id"],
    ]) {
        assert!(
            path.ends_with(&format!("/responses/{}", id.as_str().expect("id"))),
            "delete {path} names the stored response {id}"
        );
    }
}

/// A conversation that starts stored and chained, then continues statelessly
/// by resending the whole history: the stateless request must carry the
/// reasoning items (by id and encrypted content) and the call exactly as the
/// stored turns issued them.
#[tokio::test]
async fn stored_then_stateless_mid_conversation() {
    const SCENARIO: &str = "stateful_chain_matrix/stored_then_stateless_mid_conversation";
    with_openai_cassette(
        "stateful_chain_matrix/stored_then_stateless_mid_conversation",
        |client| async move {
            let resources = Created::default();
            cleaned(&client, &resources, async {
                let model = client.openai.responses(MODEL);
                let params = |previous: Option<&str>, store: bool| {
                    let mut params = json!({
                        "store": store,
                        "reasoning": { "effort": "low" },
                        "include": ["reasoning.encrypted_content"]
                    });
                    if let Some(previous) = previous {
                        params["previous_response_id"] = json!(previous);
                    }
                    params
                };
                let prompt =
                    Message::user("Use lookup_code to get the code of record alpha. Do not guess.");

                let first = model
                    .completion(request(
                        vec![prompt.clone()],
                        vec![lookup_tool()],
                        params(None, true),
                    ))
                    .await
                    .expect("turn one");
                let first_id = first.response_id.clone().expect("a stored response id");
                created(&resources, format!("responses/{first_id}"));
                let call = only_call(&first.choice);
                let tool_answer = answer(&call);

                let second = model
                    .completion(request(
                        vec![tool_answer.clone()],
                        vec![lookup_tool()],
                        params(Some(&first_id), true),
                    ))
                    .await
                    .expect("turn two chains");
                let second_id = second.response_id.clone().expect("a stored response id");
                created(&resources, format!("responses/{second_id}"));

                let history = vec![
                    prompt,
                    Message::Assistant {
                        id: first.message_id.clone(),
                        content: first.choice.clone(),
                    },
                    tool_answer,
                    Message::Assistant {
                        id: second.message_id.clone(),
                        content: second.choice.clone(),
                    },
                    Message::user("Repeat the code you reported, exactly, and nothing else."),
                ];
                let third = model
                    .completion(request(history, vec![lookup_tool()], params(None, false)))
                    .await
                    .expect("turn three continues statelessly from the full history");
                assert!(text(&third.choice).contains(CODE), "{:?}", third.choice);
            })
            .await;
        },
    )
    .await;

    let turns = turns(SCENARIO);
    let (first_response, third_request) = (&turns[0].2, &turns[2].1);
    assert!(third_request.get("previous_response_id").is_none());
    let first_reasoning = first_response["output"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|item| item["type"] == "reasoning")
        .expect("turn one reasoned");
    let replayed = input_items(third_request, "reasoning");
    assert!(
        replayed
            .iter()
            .any(|item| item["id"] == first_reasoning["id"]
                && item["encrypted_content"] == first_reasoning["encrypted_content"]),
        "turn one's reasoning item is replayed with its exact id and ciphertext"
    );
    let calls = input_items(third_request, "function_call");
    assert!(
        calls
            .iter()
            .any(|item| item["call_id"] == json!(output_call_id(first_response))),
        "turn one's call is replayed with its exact call id"
    );
}

/// A PDF uploaded to the Files API, referenced by its id in Responses content
/// on two turns, then deleted. The upload expires after an hour as a backstop.
#[tokio::test]
async fn file_id_chain() {
    const SCENARIO: &str = "stateful_chain_matrix/file_id_chain";
    with_openai_cassette("stateful_chain_matrix/file_id_chain", |client| async move {
        let resources = Created::default();
        cleaned(&client, &resources, async {
            let config = &client.openai.wire;
            let bytes = std::fs::read(crate::support::PDF_FIXTURE_PATH).expect("fixture PDF");
            let form = reqwest::multipart::Form::new()
                .text("purpose", "user_data")
                .text("expires_after[anchor]", "created_at")
                .text("expires_after[seconds]", "3600")
                .part(
                    "file",
                    reqwest::multipart::Part::bytes(bytes)
                        .file_name("rig-pages.pdf")
                        .mime_str("application/pdf")
                        .expect("PDF MIME"),
                );
            let uploaded: Value = reqwest::Client::new()
                .post(format!("{}/files", config.base_url.trim_end_matches('/')))
                .bearer_auth(config.api_key.expose())
                .multipart(form)
                .send()
                .await
                .expect("upload is sent")
                .json()
                .await
                .expect("upload reply is JSON");
            let file_id = uploaded["id"]
                .as_str()
                .expect("an uploaded file id")
                .to_owned();
            created(&resources, format!("files/{file_id}"));

            let document = Message::User {
                content: vec![
                    UserContent::Document(rig::message::Document {
                        data: rig::message::DocumentSourceKind::file_id(&file_id),
                        media_type: Some(rig::message::DocumentMediaType::PDF),
                        additional_params: None,
                    }),
                    UserContent::text("What is the title on the first page? Answer briefly."),
                ],
            };
            let model = client.openai.responses("gpt-4.1-mini");
            let params = json!({ "store": false });
            let first = model
                .completion(request(vec![document.clone()], vec![], params.clone()))
                .await
                .expect("turn one reads the file by id");
            let history = vec![
                document,
                Message::Assistant {
                    id: first.message_id.clone(),
                    content: first.choice.clone(),
                },
                Message::user("How many pages does the attached PDF have? Answer with a number."),
            ];
            let second = model
                .completion(request(history, vec![], params))
                .await
                .expect("turn two still reads the file by id");
            assert!(
                !text(&second.choice).trim().is_empty(),
                "{:?}",
                second.choice
            );
        })
        .await;
    })
    .await;

    let turns = turns(SCENARIO);
    assert_eq!(turns.len(), 4, "upload, two turns, delete");
    let file_id = turns[0].2["id"].as_str().expect("the upload's id");
    assert!(file_id.starts_with("file-") && !file_id.contains("REDACTED"));
    for (path, request, _) in &turns[1..3] {
        assert!(path.ends_with("/responses"), "{path}");
        let referenced: Vec<&str> = request["input"]
            .as_array()
            .into_iter()
            .flatten()
            .flat_map(|item| item["content"].as_array().into_iter().flatten())
            .filter(|part| part["type"] == "input_file")
            .filter_map(|part| part["file_id"].as_str())
            .collect();
        assert_eq!(
            referenced,
            [file_id],
            "the input_file part carries the uploaded id"
        );
    }
    assert!(
        turns[3].0.ends_with(&format!("/files/{file_id}")),
        "the upload is deleted"
    );
}

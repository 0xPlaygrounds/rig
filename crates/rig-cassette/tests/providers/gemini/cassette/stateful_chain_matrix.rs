//! Gemini server-side state chains, asserted from the recorded bytes: every
//! handle a response issued is the exact value later requests carry.
//!
//! Handles are account-scoped and deleted at the end of each recording, so a
//! committed chain replays but cannot seed a live call later. Record a chain
//! in one session.

use std::future::Future;
use std::panic::{AssertUnwindSafe, resume_unwind};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::FutureExt;

use rig::completion::{CompletionModel, CompletionRequest, ToolDefinition};
use rig::message::{
    AssistantContent, Document, DocumentMediaType, DocumentSourceKind, Message, ToolCall,
    ToolResultContent, UserContent,
};
use rig::providers::gemini;
use rig::providers::gemini::cached_content::{CacheExpiry, NewCachedContent};
use rig::providers::gemini::interactions_api::AdditionalParameters;
use serde_json::{Value, json};

use super::super::support::{
    BoundGemini, always_deleting_cached_contents, with_gemini_interactions_cassette,
    with_gemini_prompt_caching_cassette,
};

const CACHE_MODEL: &str = gemini::completion::GEMINI_2_5_FLASH;
const INTERACTIONS_MODEL: &str = "gemini-3-flash-preview";
const CODE: &str = "amber-5521";

fn turns(scenario: &str) -> Vec<(String, Value, Value)> {
    let paths = crate::cassettes::recorded_request_paths("gemini", scenario);
    crate::cassettes::recorded_interaction_bodies("gemini", scenario)
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

fn text(choice: &[AssistantContent]) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

const FILE_TEXT: &str =
    "Warehouse note. The ordering token is violet-needle. The shelf code is K-4471.";

fn ask_with(history: Vec<Message>) -> CompletionRequest {
    CompletionRequest {
        model: None,
        chat_history: history,
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: Some(1024),
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    }
}

fn ask(prompt: &str) -> CompletionRequest {
    CompletionRequest {
        model: None,
        chat_history: vec![Message::user(prompt)],
        documents: vec![],
        tools: vec![],
        temperature: Some(0.0),
        max_tokens: Some(64),
        tool_choice: None,
        additional_params: Some(json!({
            "generationConfig": { "thinkingConfig": { "thinkingBudget": 0 } }
        })),
        output_schema: None,
        record_telemetry_content: false,
    }
}

/// One cache through its whole life: create, generate against it, extend its
/// TTL, find it by paging the list one entry at a time, delete it, and prove
/// the deleted handle is refused. Every later request names the exact
/// handle `create` issued, and every page request the exact cursor the
/// previous page returned.
#[tokio::test]
async fn cached_content_lifecycle_chain() {
    const SCENARIO: &str = "stateful_chain_matrix/cached_content_lifecycle_chain";
    with_gemini_prompt_caching_cassette(
        "stateful_chain_matrix/cached_content_lifecycle_chain",
        |client: BoundGemini| async move {
            let caches = client.cached_contents();
            let created = caches
                .create(
                    NewCachedContent::new(CACHE_MODEL)
                        .system_instruction(format!(
                            "The code of record alpha is {CODE}.\n{}",
                            crate::cache_conformance::cache_padding(45)
                        ))
                        .display_name("rig-stateful-chain")
                        .expiry(CacheExpiry::ttl(Duration::from_secs(120))),
                )
                .await
                .expect("create");
            // A second cache makes a one-entry page list span two pages, so the
            // cursor is followed for real.
            let sibling = caches
                .create(
                    NewCachedContent::new(CACHE_MODEL)
                        .system_instruction(crate::cache_conformance::cache_padding(45))
                        .display_name("rig-stateful-chain-sibling")
                        .expiry(CacheExpiry::ttl(Duration::from_secs(120))),
                )
                .await;
            let mut handles = vec![created.name.clone()];
            if let Ok(sibling) = &sibling {
                handles.push(sibling.name.clone());
            }
            let name = created.name.clone();
            let generator = client.clone();

            always_deleting_cached_contents(&client, &handles, async {
                if let Err(error) = &sibling {
                    panic!("creating the sibling cache: {error}");
                }
                let model = generator
                    .clone()
                    .completion(CACHE_MODEL)
                    .map_wire(|wire| wire.with_cached_content(name.clone()));
                let reply = model
                    .completion(ask(
                        "What is the code of record alpha? Reply with the code only.",
                    ))
                    .await
                    .expect("generation against the cache");
                assert!(text(&reply.choice).contains(CODE), "{:?}", reply.choice);

                caches
                    .update_expiry(&name, CacheExpiry::ttl(Duration::from_secs(240)))
                    .await
                    .expect("extend the TTL");
                let listed = caches.list_with_page_size(1).await.expect("paged list");
                assert!(
                    listed.iter().any(|entry| entry.name == name),
                    "the cache is listed"
                );
            })
            .await;

            let refused = client
                .completion(CACHE_MODEL)
                .map_wire(|wire| wire.with_cached_content(created.name.clone()))
                .completion(ask("What is the code of record alpha?"))
                .await;
            let refused = refused.expect_err("a deleted cache handle must be refused");
            let report = rig::error::ErrorReport::from(&refused);
            assert!(
                matches!(refused, rig::error::ProviderError::ProviderResponse(_))
                    && report.http_status == Some(403),
                "a deleted handle is refused by the provider as not found or denied: {report:?}"
            );
        },
    )
    .await;

    let turns = turns(SCENARIO);
    let name = turns[0].2["name"]
        .as_str()
        .expect("create issued a name")
        .to_owned();
    let mut saw = (false, false, false, false);
    for (path, request, _) in &turns[1..] {
        if path.contains(":generateContent") {
            assert_eq!(
                request["cachedContent"],
                json!(name),
                "generate names the handle"
            );
            saw.0 = true;
        } else if path.ends_with(&format!("/{name}")) && request.get("ttl").is_some() {
            saw.1 = true;
        } else if path.ends_with("/cachedContents") && request.is_null() {
            saw.2 = true;
        } else if path.ends_with(&format!("/{name}")) {
            saw.3 = true;
        }
    }
    assert!(
        saw.0 && saw.1 && saw.2 && saw.3,
        "generate, update, list and delete all name the handle: {saw:?}"
    );
    let pages: Vec<(Vec<(String, String)>, Option<String>)> = {
        let queries = crate::cassettes::recorded_request_query_pairs("gemini", SCENARIO);
        turns
            .iter()
            .zip(queries)
            .filter(|((path, _, _), query)| {
                path.ends_with("/cachedContents") && query.iter().any(|(key, _)| key == "pageSize")
            })
            .map(|((_, _, response), query)| {
                (query, response["nextPageToken"].as_str().map(str::to_owned))
            })
            .collect()
    };
    for window in pages.windows(2) {
        let issued = window[0]
            .1
            .as_deref()
            .expect("a page before another names a cursor");
        assert!(
            window[1]
                .0
                .iter()
                .any(|(key, value)| key == "pageToken" && value == issued),
            "the next page request carries the exact cursor the previous page issued"
        );
    }
    assert!(
        pages.len() >= 2,
        "the one-entry list spans pages, so a cursor was followed: {} page(s)",
        pages.len()
    );
    assert!(
        pages.last().is_some_and(|(_, cursor)| cursor.is_none()),
        "paging ran to the end"
    );
    // The refused generation is the last request, after the handle's delete.
    let last = &turns[turns.len() - 1];
    assert!(
        last.0.contains(":generateContent") && last.1["cachedContent"] == json!(name),
        "the last request uses the deleted handle"
    );
    assert!(
        turns[..turns.len() - 1]
            .iter()
            .any(|(path, request, _)| path.ends_with(&format!("/{name}")) && request.is_null()),
        "the handle was deleted before it was used again"
    );
}

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

fn only_call(choice: &[AssistantContent]) -> ToolCall {
    choice
        .iter()
        .find_map(|content| match content {
            AssistantContent::ToolCall(call) => Some(call.clone()),
            _ => None,
        })
        .expect("a lookup call")
}

/// Run `body`, then delete every interaction it stored through the recorded
/// session, whether it passed or panicked. Every delete is attempted;
/// failures are reported after the body's own panic, which is never hidden.
async fn deleting_interactions<F: Future<Output = ()>>(
    client: &BoundGemini,
    stored: &Arc<Mutex<Vec<String>>>,
    body: F,
) {
    let outcome = AssertUnwindSafe(body).catch_unwind().await;
    // A body that panicked holding the lock still names what it created.
    let ids = stored
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    let http = reqwest::Client::new();
    let mut failures = Vec::new();
    for id in ids {
        let sent = http
            .delete(format!(
                "{}/v1beta/interactions/{id}",
                client.wire.base_url.trim_end_matches('/')
            ))
            .header("x-goog-api-key", client.wire.api_key.expose())
            .send()
            .await;
        match sent {
            Ok(response) if response.status().is_success() => {}
            Ok(response) => failures.push(format!("{id}: {}", response.status())),
            Err(error) => failures.push(format!("{id}: {error}")),
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

/// Three stored interactions chained by `previous_interaction_id`, the second
/// answering a tool call: each request names the exact id the previous
/// interaction issued, the result names the exact call id, and every stored
/// interaction is deleted by id.
#[tokio::test]
async fn interactions_chain_with_tool_call() {
    const SCENARIO: &str = "stateful_chain_matrix/interactions_chain_with_tool_call";
    with_gemini_interactions_cassette(
        "stateful_chain_matrix/interactions_chain_with_tool_call",
        |client| async move {
            let stored = Arc::new(Mutex::new(Vec::new()));
            let keep = |id: &str| {
                stored
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(id.to_owned());
            };
            deleting_interactions(&client, &stored, async {
                let model = client
                    .clone()
                    .map_wire(|config| config.interactions(INTERACTIONS_MODEL));
                let params = |previous: Option<String>| {
                    serde_json::to_value(AdditionalParameters {
                        store: Some(true),
                        previous_interaction_id: previous,
                        ..Default::default()
                    })
                    .expect("params serialize")
                };

                let first = model
                    .completion(
                        model
                            .completion_request(
                                "Use lookup_code to get the code of record alpha. Do not guess.",
                            )
                            .tool(lookup_tool())
                            .additional_params(params(None))
                            .build(),
                    )
                    .await
                    .expect("turn one");
                let first_id = first.response_id.clone().expect("an interaction id");
                keep(&first_id);
                let call = only_call(&first.choice);

                let second = model
                    .completion(
                        model
                            .completion_request(Message::from(UserContent::tool_result_for(
                                call.id.clone(),
                                call.provider.clone(),
                                call.function.name.clone(),
                                vec![ToolResultContent::text(format!(
                                    "record alpha: code {CODE}"
                                ))],
                            )))
                            .additional_params(params(Some(first_id)))
                            .build(),
                    )
                    .await
                    .expect("turn two answers the call");
                let second_id = second.response_id.clone().expect("an interaction id");
                keep(&second_id);

                let third = model
                    .completion(
                        model
                            .completion_request(
                                "Repeat the code you reported, exactly, and nothing else.",
                            )
                            .additional_params(params(Some(second_id)))
                            .build(),
                    )
                    .await
                    .expect("turn three continues");
                keep(third.response_id.as_deref().expect("an interaction id"));
                assert!(text(&third.choice).contains(CODE), "{:?}", third.choice);
            })
            .await;
        },
    )
    .await;

    let turns = turns(SCENARIO);
    assert_eq!(turns.len(), 6, "three interactions and three deletes");
    for (index, (path, _, _)) in turns[3..].iter().enumerate() {
        let id = turns[index].2["id"].as_str().expect("an interaction id");
        assert!(
            path.ends_with(&format!("/interactions/{id}")),
            "delete {path} names stored interaction {id}"
        );
    }
    assert_eq!(turns[1].1["previous_interaction_id"], turns[0].2["id"]);
    assert_eq!(turns[2].1["previous_interaction_id"], turns[1].2["id"]);
    let issued_call = turns[0].2["steps"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|item| item["type"] == "function_call")
        .and_then(|item| item["id"].as_str())
        .expect("turn one issued a call id")
        .to_owned();
    let result = turns[1].1["input"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|item| item["type"] == "function_result")
        .expect("turn two sends a result");
    assert_eq!(result["call_id"], json!(issued_call));
}

/// A text document uploaded to the Files API, referenced by its URI in
/// `fileData` on two turns, then deleted. The file expires on its own after
/// 48 hours as a backstop; the delete runs whether the body passes or
/// panics. The document is text because replay matches a request body only
/// when it is UTF-8 (or multipart, which it does not record), and a binary
/// media upload is neither.
#[tokio::test]
async fn file_uri_chain() {
    const SCENARIO: &str = "stateful_chain_matrix/file_uri_chain";
    with_gemini_interactions_cassette(
        "stateful_chain_matrix/file_uri_chain",
        |client| async move {
            let base = client.wire.base_url.trim_end_matches('/').to_owned();
            let key = client.wire.api_key.expose().to_owned();
            let http = reqwest::Client::new();
            let uploaded: Value = http
                .post(format!("{base}/upload/v1beta/files?uploadType=media"))
                .header("x-goog-api-key", &key)
                .header("content-type", "text/plain")
                .body(FILE_TEXT)
                .send()
                .await
                .expect("upload is sent")
                .json()
                .await
                .expect("upload reply is JSON");
            let name = uploaded["file"]["name"]
                .as_str()
                .expect("a file name")
                .to_owned();
            let uri = uploaded["file"]["uri"]
                .as_str()
                .expect("a file uri")
                .to_owned();

            let body = async {
                let document = Message::User {
                    content: vec![
                        UserContent::Document(Document {
                            data: DocumentSourceKind::Url(uri.clone()),
                            media_type: Some(DocumentMediaType::TXT),
                            additional_params: None,
                        }),
                        UserContent::text(
                            "What is the ordering token in the attached file? Reply with the token only.",
                        ),
                    ],
                };
                let model = client.completion(gemini::completion::GEMINI_2_5_FLASH);
                let first = model
                    .completion(ask_with(vec![document.clone()]))
                    .await
                    .expect("turn one reads the file by uri");
                assert!(
                    text(&first.choice).contains("violet-needle"),
                    "{:?}",
                    first.choice
                );
                let history = vec![
                    document,
                    Message::Assistant {
                        id: first.message_id.clone(),
                        content: first.choice.clone(),
                    },
                    Message::user(
                        "What is the shelf code in the same attached file? Reply with the code only.",
                    ),
                ];
                let second = model
                    .completion(ask_with(history))
                    .await
                    .expect("turn two still reads the file by uri");
                assert!(
                    text(&second.choice).contains("K-4471"),
                    "{:?}",
                    second.choice
                );
            };
            let outcome = AssertUnwindSafe(body).catch_unwind().await;
            let deleted = http
                .delete(format!("{base}/v1beta/{name}"))
                .header("x-goog-api-key", &key)
                .send()
                .await
                .map(|response| response.status());
            if let Err(panic) = outcome {
                eprintln!("cleanup of {name}: {deleted:?}");
                resume_unwind(panic);
            }
            assert!(
                deleted.as_ref().is_ok_and(|status| status.is_success()),
                "cleanup failed, delete {name} by hand: {deleted:?}"
            );
        },
    )
    .await;

    let turns = turns(SCENARIO);
    assert_eq!(turns.len(), 4, "upload, two turns, delete");
    let uri = turns[0].2["file"]["uri"]
        .as_str()
        .expect("the upload's uri");
    assert!(
        !uri.contains("REDACTED"),
        "the file uri is recorded verbatim"
    );
    let name = turns[0].2["file"]["name"]
        .as_str()
        .expect("the upload's name");
    for (path, request, _) in &turns[1..3] {
        assert!(path.contains(":generateContent"), "{path}");
        let referenced: Vec<&str> = request["contents"]
            .as_array()
            .into_iter()
            .flatten()
            .flat_map(|content| content["parts"].as_array().into_iter().flatten())
            .filter_map(|part| part["fileData"]["fileUri"].as_str())
            .collect();
        assert_eq!(
            referenced,
            [uri],
            "the fileData part carries the uploaded uri"
        );
    }
    assert!(
        turns[3].0.ends_with(&format!("/{name}")),
        "the upload is deleted"
    );
}

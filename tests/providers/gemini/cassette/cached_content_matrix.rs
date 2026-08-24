//! The `cachedContents` create matrix.
//!
//! Five dimensions change the create body: payload (system instruction /
//! contents / both), tool count (none / one / three), tool choice (absent /
//! auto), display name (absent / present), and expiry (relative TTL / absolute
//! timestamp / provider default). Their full product is 108 combinations.
//!
//! **All 108 are asserted, and only 8 are recorded.** That split is deliberate,
//! and it took recording all 108 to see why: Gemini's create *response* carries
//! only `name`, `model`, `displayName`, `createTime`, `updateTime`, `expireTime`
//! and `usageMetadata`. It never echoes `tools` or `toolConfig`. So a recorded
//! cell cannot tell whether the tools it asked for were sent at all — the cache
//! could discard them entirely and all 72 tool-bearing fixtures would still have
//! passed. What those cells actually pinned was the *request* body, through the
//! cassette matcher.
//!
//! So the request body is pinned directly instead, by
//! [`every_create_combination_serializes_as_expected`], which covers all 108
//! combinations for free and does assert the tool count — verified by dropping
//! the tools and watching it fail, which the fixtures did not. The 8 recorded
//! cells cover one arm of each dimension, keeping the provider's *acceptance* of
//! each shape under test, plus the edge cells below which pin behaviour no
//! serialization test could reach.
//!
//! That is 164 KB instead of 1.3 MB, for strictly more coverage.
//!
//! Every recorded cell deletes what it created, on the failure path too, because
//! storage bills until it does. Cells whose checks panic run them inside
//! `always_deleting_cached_contents` (`tests/providers/gemini/support.rs`),
//! which issues the delete even when an assertion panics; the rest get the same
//! guarantee by hand, returning a `Result` from their checks and deleting
//! before they panic on it.
//!
//! Behaviours pinned here were measured against the live API first:
//!
//! * the create minimum really is 1,024 tokens (720 is rejected with
//!   `min_total_token_count`, 1,080 is accepted);
//! * `ttl` and `expireTime` together are a 400 — the API models expiration as a
//!   `oneof`, which is exactly what `CacheExpiry` makes unrepresentable;
//! * omitting both defaults to one hour;
//! * a `toolConfig` with no `tools` is accepted;
//! * an unqualified model id is a 400, which is why `NewCachedContent::new`
//!   qualifies it.
//!
//! # Recording
//!
//! ```text
//! RIG_PROVIDER_TEST_MODE=record cargo test -p rig --test gemini --all-features \
//!     cached_content_matrix -- --test-threads=1
//! ```

use rig::client::CompletionClient as _;
use rig::client::DefaultTransportBuilder as _;
use rig::providers::gemini;
use rig::providers::gemini::cached_content::{CacheExpiry, CachedContent, NewCachedContent};
use std::time::Duration;

use super::super::support::{
    always_deleting_cached_contents, assert_recorded_requests_read_from_a_cache,
    assert_recorded_response_contains, with_gemini_prompt_caching_cassette,
};

/// Deterministic, committed timestamp for the absolute-expiry arm. A computed
/// "now + 10 minutes" would churn the fixture on every re-record.
const ABSOLUTE_EXPIRY: &str = "2030-01-01T00:00:00Z";

/// Gemini's documented create minimum, confirmed live.
const MIN_CACHEABLE_TOKENS: u64 = 1024;

const CACHE_MODEL: &str = gemini::completion::GEMINI_2_5_FLASH;

/// Padding sized just over the minimum, so the matrix is as cheap as it can be
/// while still being a legal cache.
fn pad() -> String {
    crate::cache_conformance::cache_padding(45)
}

fn probe_tool(name: &str) -> gemini::completion::gemini_api_types::Tool {
    use gemini::completion::gemini_api_types::{FunctionDeclaration, Tool};

    // Built directly rather than deserialized: `Tool` is `Serialize`-only.
    Tool {
        function_declarations: vec![FunctionDeclaration {
            name: name.to_owned(),
            description: "matrix probe tool".to_owned(),
            parameters: gemini::completion::gemini_api_types::tool_parameters_to_schema(
                serde_json::json!({
                    "type": "object",
                    "properties": { "topic": { "type": "string" } },
                    "required": ["topic"]
                }),
            )
            .expect("probe tool schema should convert"),
        }],
        code_execution: None,
    }
}

/// One cell of the matrix.
#[derive(Clone, Copy)]
struct Cell {
    payload: &'static str,
    tools: usize,
    tool_config: bool,
    display_name: bool,
    expiry: &'static str,
}

fn build(cell: Cell) -> NewCachedContent {
    let mut request = NewCachedContent::new(CACHE_MODEL);

    request = match cell.payload {
        "sys" => request.system_instruction(pad()),
        "contents" => request.content(pad()),
        _ => request.system_instruction(pad()).content(pad()),
    };

    if cell.tools > 0 {
        let names = ["lookup_alpha", "lookup_beta", "lookup_gamma"];
        request = request.tools(names[..cell.tools].iter().map(|n| probe_tool(n)).collect());
    }
    if cell.tool_config {
        request = request.tool_config(gemini::completion::gemini_api_types::ToolConfig {
            function_calling_config: Some(
                gemini::completion::gemini_api_types::FunctionCallingMode::Auto,
            ),
        });
    }
    if cell.display_name {
        request = request.display_name("rig-matrix");
    }
    request = match cell.expiry {
        "ttl" => request.expiry(CacheExpiry::ttl(Duration::from_secs(120))),
        "abs" => request.expiry(CacheExpiry::expire_time(ABSOLUTE_EXPIRY)),
        _ => request,
    };
    request
}

/// Create the cache, assert the invariants every cell shares, then delete it.
///
/// The delete runs on the failure path too: a matrix this size would leak a lot
/// of billed caches if one assertion took the test down with it.
async fn run_cell(client: gemini::Client, cell: Cell) {
    let created: CachedContent = client
        .cached_contents()
        .create(build(cell))
        .await
        .expect("creating a cached content should succeed");

    let outcome = check(&created, cell);

    client
        .cached_contents()
        .delete(&created.name)
        .await
        .expect("delete should succeed — storage bills until it lands");

    if let Err(failure) = outcome {
        panic!("{failure}");
    }
}

fn check(created: &CachedContent, cell: Cell) -> Result<(), String> {
    if !created.name.starts_with("cachedContents/") {
        return Err(format!(
            "a handle should be `cachedContents/<id>`, got {:?}",
            created.name
        ));
    }
    if created.model != format!("models/{CACHE_MODEL}") {
        return Err(format!(
            "the cache must be bound to the model it was created for, got {:?}",
            created.model
        ));
    }

    let stored = created
        .usage_metadata
        .as_ref()
        .map(|usage| usage.total_token_count)
        .unwrap_or_default();
    if stored < MIN_CACHEABLE_TOKENS {
        return Err(format!(
            "a cache storing {stored} tokens is under the {MIN_CACHEABLE_TOKENS}-token minimum, \
             so this matrix's padding has drifted and every cell proves less than it claims"
        ));
    }

    let Some(expiry) = created.expire_time.as_deref() else {
        return Err(
            "every cached content should report an expiry, however it was requested".into(),
        );
    };
    if cell.expiry == "abs" && expiry != ABSOLUTE_EXPIRY {
        return Err(format!(
            "an absolute expiry should be echoed back verbatim, got {expiry:?}"
        ));
    }
    if cell.display_name && created.display_name.as_deref() != Some("rig-matrix") {
        return Err(format!(
            "the display name should round-trip, got {:?}",
            created.display_name
        ));
    }
    Ok(())
}

/// The create matrix, asserted on the serialized request body — all 108
/// combinations, no network.
///
/// This replaced 108 recorded cells, and asserts strictly more than they did.
/// Gemini's create *response* carries only `name`, `model`, `displayName`,
/// `createTime`, `updateTime`, `expireTime` and `usageMetadata` — it never
/// echoes `tools` or `toolConfig`. So a recorded cell could not tell whether the
/// tools it asked for were sent at all: the cache could discard them entirely
/// and all 72 tool-bearing fixtures would still have passed. What those cells
/// really pinned was the request body, via the cassette matcher — and that is
/// exactly what this asserts, for free, without 1.3 MB of near-identical YAML.
///
/// Eight representative cells remain recorded, one per arm of each dimension,
/// to keep the provider's *acceptance* of each shape under test.
#[test]
fn every_create_combination_serializes_as_expected() {
    let mut checked = 0usize;

    for payload in ["sys", "contents", "both"] {
        for tools in [0usize, 1, 3] {
            for tool_config in [false, true] {
                for display_name in [false, true] {
                    for expiry in ["ttl", "abs", "default"] {
                        let cell = Cell {
                            payload,
                            tools,
                            tool_config,
                            display_name,
                            expiry,
                        };
                        let body = serde_json::to_value(build(cell)).expect("serialize");
                        let object = body.as_object().expect("object");
                        let label = format!(
                            "{payload}/{tools}tools/cfg={tool_config}/name={display_name}/{expiry}"
                        );

                        assert_eq!(
                            object.get("model").and_then(|v| v.as_str()),
                            Some("models/gemini-2.5-flash"),
                            "{label}: the model must be qualified — Gemini 400s a bare id"
                        );

                        assert_eq!(
                            object.contains_key("systemInstruction"),
                            payload != "contents",
                            "{label}: systemInstruction presence"
                        );
                        assert_eq!(
                            object.contains_key("contents"),
                            payload != "sys",
                            "{label}: contents presence"
                        );

                        // The dimension the recorded cells were blind to.
                        match object.get("tools") {
                            None => assert_eq!(tools, 0, "{label}: tools missing"),
                            Some(value) => assert_eq!(
                                value.as_array().expect("tools array").len(),
                                tools,
                                "{label}: tool count on the wire"
                            ),
                        }
                        assert_eq!(
                            object.contains_key("toolConfig"),
                            tool_config,
                            "{label}: toolConfig presence"
                        );
                        assert_eq!(
                            object.contains_key("displayName"),
                            display_name,
                            "{label}: displayName presence"
                        );

                        // Expiry is a `oneof` on the wire: Gemini answers a body
                        // carrying both with a 400, so exactly one may appear.
                        assert_eq!(
                            object.contains_key("ttl"),
                            expiry == "ttl",
                            "{label}: ttl presence"
                        );
                        assert_eq!(
                            object.contains_key("expireTime"),
                            expiry == "abs",
                            "{label}: expireTime presence"
                        );

                        checked += 1;
                    }
                }
            }
        }
    }

    assert_eq!(
        checked, 108,
        "the serialization matrix should be exhaustive"
    );
}

/// An `Agent` that owns tools does not get to read from a cache by default —
/// and the error has to say why.
///
/// [`NewCachedContent::tools`] reads like it makes a cached tool set usable from
/// an agent. It does not, and the reason is structural rather than a missing
/// feature: rig's agent derives the declarations it sends and the handles it
/// dispatches through from one registry snapshot, so a registered tool is always
/// advertised and a call to an unadvertised tool is an invalid tool call, never
/// a dispatch. An ordinary agent turn with a tool therefore carries `tools`, and
/// `cachedContent` alongside `tools` is refused. The unit tests in `rig-core`
/// cannot see this: they build a request directly, and so can simply leave the
/// tools out.
///
/// "By default" is load-bearing. An empty `RequestPatch::active_tools`
/// allow-list empties the snapshot for a turn, and such a request *is* accepted
/// — see `an_agent_that_suppresses_its_tools_may_read_from_a_cache`. What it
/// does not do is make anything callable, which is why the remedy in the message
/// still cannot be "move them into the cache".
///
/// The second assertion is the point of the finding this pins: the bare "move
/// them into the cache" remedy is right for a system instruction and wrong here,
/// because the cached declarations would be unreachable from the agent that was
/// told to create them.
///
/// No cassette and no socket — the guard fires while the request is being built.
/// The client still points at an unroutable address so that a regression fails
/// as a local assertion instead of reaching the live API from a unit test.
#[tokio::test]
async fn an_agent_with_tools_cannot_read_from_a_cache() {
    use rig::agent::AgentBuilder;
    use rig::completion::Prompt as _;

    use super::super::tools_support::CountingPing;

    let client = gemini::Client::builder()
        .api_key("not-a-real-key")
        .base_url("http://127.0.0.1:1")
        .build()
        .expect("client should build");

    let agent = AgentBuilder::new(
        client
            .completion_model(CACHE_MODEL)
            .with_cached_content("cachedContents/agent-guard"),
    )
    .tool(CountingPing::default())
    .build();

    let message = agent
        .prompt("hi")
        .await
        .expect_err("an agent that advertises tools cannot also read from a cache")
        .to_string();

    assert!(
        message.contains("also set tools"),
        "the failure should name the tool set as the conflict: {message}"
    );
    assert!(
        message.contains("declarations only"),
        "`move them into the cache` is not the remedy for tools — the message must say the \
         cached declarations are unreachable from an agent: {message}"
    );
}

/// The carve-out the `declares_functions` gate exists for, measured rather than
/// assumed: a cache carrying a **provider-hosted** tool is usable from an agent
/// that declares nothing itself.
///
/// This is the claim that makes the tools caveat conditional — "provider-hosted
/// tools such as `codeExecution` run on Gemini's side and are fine to keep in
/// the cache". `codeExecution` needs no tool loop, so unlike a function
/// declaration it loses nothing by living somewhere the agent cannot dispatch
/// from. The unit test pins that the caveat is *withheld* for this shape; only a
/// recording can say whether the underlying claim is true, i.e. that Gemini
/// accepts the tool in a cache and still runs it for a request that declares
/// nothing.
#[tokio::test]
async fn a_cache_carrying_a_provider_hosted_tool_is_usable_from_an_agent() {
    use rig::agent::AgentBuilder;
    use rig::completion::Prompt as _;
    use rig::providers::gemini::completion::gemini_api_types::{CodeExecution, Tool};

    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/edge_cached_code_execution",
        |client| async move {
            let cache = client
                .cached_contents()
                .create(
                    NewCachedContent::new(CACHE_MODEL)
                        .system_instruction(pad())
                        .tools(vec![Tool {
                            function_declarations: Vec::new(),
                            code_execution: Some(CodeExecution {}),
                        }])
                        .display_name("rig-cached-code-execution")
                        .expiry(CacheExpiry::ttl(Duration::from_secs(180))),
                )
                .await
                .expect("a cache carrying only a provider-hosted tool should be created");

            let handles = [cache.name.clone()];
            always_deleting_cached_contents(&client, &handles, async {
                let agent = AgentBuilder::new(
                    client
                        .completion_model(CACHE_MODEL)
                        .with_cached_content(cache.name.clone()),
                )
                .build();

                let answer = agent
                    .prompt(
                        "Use the code execution tool to compute 7919 * 6841. Reply with only the \
                         number.",
                    )
                    .max_turns(3)
                    .await
                    .expect(
                        "an agent declaring nothing may read from a cache carrying a \
                         provider-hosted tool",
                    );

                assert!(
                    answer.contains("54173879"),
                    "the cached codeExecution tool should have run on Gemini's side: {answer:?}"
                );
            })
            .await;
        },
    )
    .await;

    // The answer alone proves nothing — a model can multiply two numbers in its
    // head and return the same string with no tool involved at all. The cell's
    // claim is that the tool *in the cache* ran, so it has to be read off the
    // recorded bytes: Python generated and executed on Gemini's side, for a
    // request that declared no tools of its own.
    assert_recorded_response_contains(
        "cached_content_matrix/edge_cached_code_execution",
        &["executableCode", "codeExecutionResult"],
    );
    assert_recorded_requests_read_from_a_cache("cached_content_matrix/edge_cached_code_execution");
}

/// The exception to the cell above, recorded because it is the one that would
/// otherwise be an unchecked claim in the docs.
///
/// An empty `RequestPatch::active_tools` allow-list empties the tool snapshot
/// for a turn, so a tool-holding agent builds a request with no `tools` and the
/// handle is accepted — by rig, and, as this recording shows, by Gemini. The
/// docs say so; without a cell they would be asserting a behaviour nothing
/// exercises, and the neighbouring cell's "an agent with tools is refused"
/// would read as absolute when it is not.
///
/// What it does *not* buy is dispatch: the suppressed tools are gone for the
/// turn and the cache's are unreachable regardless, which is why the remedy in
/// the conflict message still cannot be "move them into the cache".
#[tokio::test]
async fn an_agent_that_suppresses_its_tools_may_read_from_a_cache() {
    use rig::agent::{AgentBuilder, RequestPatch};
    use rig::completion::Prompt as _;

    use super::super::hook_stress_support::ApplyPatch;
    use super::super::tools_support::CountingPing;

    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/edge_agent_active_tools_suppressed",
        |client| async move {
            let cache = client
                .cached_contents()
                .create(
                    NewCachedContent::new(CACHE_MODEL)
                        .system_instruction(pad())
                        .display_name("rig-active-tools")
                        .expiry(CacheExpiry::ttl(Duration::from_secs(180))),
                )
                .await
                .expect("create should succeed");

            let handles = [cache.name.clone()];
            always_deleting_cached_contents(&client, &handles, async {
                let agent = AgentBuilder::new(
                    client
                        .completion_model(CACHE_MODEL)
                        .with_cached_content(cache.name.clone()),
                )
                .tool(CountingPing::default())
                .build();

                let answer = agent
                    .prompt("Reply with exactly: suppressed")
                    .add_hook(ApplyPatch(
                        RequestPatch::new()
                            .active_tools(Vec::<String>::new())
                            .temperature(0.0),
                    ))
                    .await
                    .expect(
                        "an empty active_tools allow-list drops the tool set, so the handle is \
                         accepted — this is the documented exception",
                    );

                assert!(
                    answer.to_lowercase().contains("suppressed"),
                    "the turn should have run against the cache: {answer:?}"
                );
            })
            .await;
        },
    )
    .await;

    // The whole claim is about what left the process: a `tools` field
    // suppressed and a handle sent. A passing prompt says neither.
    assert_recorded_requests_read_from_a_cache(
        "cached_content_matrix/edge_agent_active_tools_suppressed",
    );
}

#[tokio::test]
async fn sys_notools_nocfg_noname_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_notools_nocfg_noname_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 0,
                    tool_config: false,
                    display_name: false,
                    expiry: "ttl",
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn sys_notools_nocfg_noname_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_notools_nocfg_noname_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 0,
                    tool_config: false,
                    display_name: false,
                    expiry: "abs",
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn sys_notools_nocfg_noname_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_notools_nocfg_noname_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 0,
                    tool_config: false,
                    display_name: false,
                    expiry: "default",
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn sys_notools_nocfg_named_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_notools_nocfg_named_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 0,
                    tool_config: false,
                    display_name: true,
                    expiry: "ttl",
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn sys_onetool_nocfg_noname_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_onetool_nocfg_noname_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 1,
                    tool_config: false,
                    display_name: false,
                    expiry: "ttl",
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn sys_threetools_autocfg_noname_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_threetools_autocfg_noname_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 3,
                    tool_config: true,
                    display_name: false,
                    expiry: "ttl",
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn contents_notools_nocfg_noname_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_notools_nocfg_noname_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 0,
                    tool_config: false,
                    display_name: false,
                    expiry: "ttl",
                },
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn both_notools_nocfg_noname_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_notools_nocfg_noname_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 0,
                    tool_config: false,
                    display_name: false,
                    expiry: "ttl",
                },
            )
            .await;
        },
    )
    .await;
}

// ---------------------------------------------------------------------------
// Edges the create matrix cannot reach
// ---------------------------------------------------------------------------

/// A cache under the minimum is refused by the provider, citing the minimum.
///
/// The create matrix pads every cell just over 1,024 tokens; this is the cell
/// that proves that number is real rather than folklore. If Gemini lowers the
/// floor, this fails and the matrix's padding can come down with it.
#[tokio::test]
async fn a_cache_below_the_minimum_is_refused_by_the_provider() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/edge_below_minimum",
        |client| async move {
            let refusal = client
                .cached_contents()
                .create(
                    NewCachedContent::new(CACHE_MODEL)
                        .system_instruction(crate::cache_conformance::cache_padding(4))
                        .expiry(CacheExpiry::ttl(Duration::from_secs(120))),
                )
                .await;

            // The day Gemini lowers the floor this cell fails — but it fails
            // having deleted the cache it just paid to create, rather than
            // leaking one out of an `expect_err`.
            let error = match refusal {
                Err(error) => error,
                Ok(created) => {
                    let _ = client.cached_contents().delete(&created.name).await;
                    panic!("a cache under the minimum should be refused");
                }
            };

            let message = error.to_string();
            assert!(
                message.contains("too small") || message.contains("min_total_token_count"),
                "the provider should say the cache is too small: {message}"
            );
        },
    )
    .await;
}

/// An `expireTime` already in the past is **accepted**.
///
/// Surprising, and worth pinning precisely because it is: Gemini does not
/// validate that the timestamp is in the future, so a caller can create a cache
/// that is dead on arrival and pay to store it. rig cannot prevent this without
/// a clock, and guessing the caller's intent would be worse — so it is recorded
/// as provider behaviour rather than patched over.
#[tokio::test]
async fn an_expiry_in_the_past_is_accepted_by_the_provider() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/edge_expiry_in_the_past",
        |client| async move {
            let created = client
                .cached_contents()
                .create(
                    NewCachedContent::new(CACHE_MODEL)
                        .system_instruction(pad())
                        .expiry(CacheExpiry::expire_time("2020-01-01T00:00:00Z")),
                )
                .await
                .expect("gemini accepts an expiry in the past");

            let outcome = (created.expire_time.as_deref() == Some("2020-01-01T00:00:00Z"))
                .then_some(())
                .ok_or_else(|| {
                    format!(
                        "the past timestamp should be echoed back verbatim, got {:?}",
                        created.expire_time
                    )
                });

            // Delete before asserting: a failed assertion in record mode would
            // otherwise leave a billed cache behind.
            let _ = client.cached_contents().delete(&created.name).await;
            outcome.unwrap_or_else(|failure| panic!("{failure}"));
        },
    )
    .await;
}

/// Omitting both expiry forms defaults to one hour.
#[tokio::test]
async fn omitting_expiry_defaults_to_one_hour() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/edge_default_expiry_is_one_hour",
        |client| async move {
            let created = client
                .cached_contents()
                .create(NewCachedContent::new(CACHE_MODEL).system_instruction(pad()))
                .await
                .expect("a cache with no expiry should be created");

            let handles = [created.name.clone()];
            always_deleting_cached_contents(&client, &handles, async {
                let (create_time, expire_time) = (
                    created.create_time.as_deref().expect("create time"),
                    created.expire_time.as_deref().expect("expire time"),
                );
                // Compare the hour fields rather than parsing RFC 3339: the default
                // is one hour, and both stamps are same-day UTC.
                let hour = |stamp: &str| stamp[11..13].parse::<i32>().expect("hour");
                let gap = (hour(expire_time) - hour(create_time)).rem_euclid(24);
                if gap != 1 {
                    panic!(
                        "the documented default is one hour: created {create_time}, expires \
                         {expire_time}"
                    );
                }
            })
            .await;
        },
    )
    .await;
}

/// `list` follows the cursor when the page is smaller than the collection.
///
/// The pagination loop and its cursor-does-not-advance guard were unreachable
/// while only one cache existed. Creating three and asking the provider for
/// them a page at a time exercises the loop for real rather than trusting it by
/// inspection.
#[tokio::test]
async fn list_follows_the_cursor_across_pages() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/edge_list_pagination",
        |client| async move {
            let caches = client.cached_contents();
            // The creates are collected rather than unwrapped: a second cache
            // that fails to create still has to delete the first, or the very
            // failure leaks three times the storage a passing run does.
            let mut created = Vec::new();
            let mut create_failure = None;
            for index in 0..3 {
                match caches
                    .create(
                        NewCachedContent::new(CACHE_MODEL)
                            .system_instruction(pad())
                            .display_name(format!("rig-page-{index}"))
                            .expiry(CacheExpiry::ttl(Duration::from_secs(180))),
                    )
                    .await
                {
                    Ok(cache) => created.push(cache.name),
                    Err(error) => {
                        create_failure = Some(error);
                        break;
                    }
                }
            }

            always_deleting_cached_contents(&client, &created, async {
                if let Some(error) = create_failure {
                    panic!("creating a page fixture should succeed: {error}");
                }

                // Page size 1 with three caches means three pages and a cursor
                // followed twice — the loop runs for real instead of returning
                // everything in one response as pageSize=1000 does.
                let listed = caches
                    .list_with_page_size(1)
                    .await
                    .expect("paginated list should succeed");
                for name in &created {
                    assert!(
                        listed.iter().any(|entry| entry.name == *name),
                        "every created cache should appear in the listing: {name} missing from \
                         {} entries",
                        listed.len()
                    );
                }
            })
            .await;
        },
    )
    .await;
}

/// Deleting a handle twice reports the second as gone rather than succeeding.
#[tokio::test]
async fn deleting_twice_reports_the_second_as_expired() {
    use rig::providers::gemini::cached_content::CachedContentError;

    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/edge_double_delete",
        |client| async move {
            let caches = client.cached_contents();
            let created = caches
                .create(
                    NewCachedContent::new(CACHE_MODEL)
                        .system_instruction(pad())
                        .expiry(CacheExpiry::ttl(Duration::from_secs(120))),
                )
                .await
                .expect("create should succeed");

            caches.delete(&created.name).await.expect("first delete");
            let error = caches
                .delete(&created.name)
                .await
                .expect_err("a second delete should not silently succeed");
            let CachedContentError::Expired { name, message } = &error else {
                panic!("a handle that is already gone should report Expired: {error:?}");
            };
            assert_eq!(*name, created.name);
            assert!(
                message.contains("not found") || message.contains("permission"),
                "Expired should carry the provider's own message — a 403 also covers a disabled \
                 key or a project without the API enabled, and the message is the only text that \
                 says which: {message}"
            );
        },
    )
    .await;
}

/// Extending expiry with an absolute timestamp uses the `expireTime` update
/// mask, not the `ttl` one.
#[tokio::test]
async fn update_expiry_accepts_an_absolute_timestamp() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/edge_update_expiry_absolute",
        |client| async move {
            let caches = client.cached_contents();
            let created = caches
                .create(
                    NewCachedContent::new(CACHE_MODEL)
                        .system_instruction(pad())
                        .expiry(CacheExpiry::ttl(Duration::from_secs(120))),
                )
                .await
                .expect("create should succeed");

            let handles = [created.name.clone()];
            always_deleting_cached_contents(&client, &handles, async {
                let updated = caches
                    .update_expiry(&created.name, CacheExpiry::expire_time(ABSOLUTE_EXPIRY))
                    .await
                    .expect("updating to an absolute expiry should succeed");
                assert_eq!(updated.expire_time.as_deref(), Some(ABSOLUTE_EXPIRY));
            })
            .await;
        },
    )
    .await;
}

/// Streaming against a cache handle.
///
/// The recorded explicit-cache cells all drive the blocking surface. Cache
/// counters arrive on a different frame when streaming, and the accumulator has
/// to carry them to the final response — a bug class that has bitten this
/// provider family before — so the streamed path needs its own recording rather
/// than an assumption that it matches.
#[tokio::test]
async fn streaming_against_a_cache_reports_the_cache_read() {
    use futures::StreamExt;
    use rig::streaming::StreamedAssistantContent;

    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/use_streaming",
        |client| async move {
            let cache = client
                .cached_contents()
                .create(
                    NewCachedContent::new(CACHE_MODEL)
                        .system_instruction(pad())
                        .expiry(CacheExpiry::ttl(Duration::from_secs(180))),
                )
                .await
                .expect("create should succeed");

            let handles = [cache.name.clone()];
            always_deleting_cached_contents(&client, &handles, async {
                let model = client
                    .completion_model(CACHE_MODEL)
                    .with_cached_content(cache.name.clone());

                let request = rig::completion::CompletionRequest {
                    preamble: None,
                    chat_history: vec![rig::message::Message::User {
                        content: vec![rig::message::UserContent::text(
                            "Reply with exactly: streamed",
                        )],
                    }],
                    documents: vec![],
                    tools: vec![],
                    temperature: Some(0.0),
                    max_tokens: Some(16),
                    tool_choice: None,
                    additional_params: Some(serde_json::json!({
                        "generationConfig": { "thinkingConfig": { "thinkingBudget": 0 } }
                    })),
                    model: None,
                    output_schema: None,
                    record_telemetry_content: false,
                };

                let mut stream = rig::completion::CompletionModel::stream(&model, request)
                    .await
                    .expect("streamed cached-content request should start");
                let mut usage = None;
                while let Some(item) = stream.next().await {
                    if let StreamedAssistantContent::Final(response) =
                        item.expect("stream item should succeed")
                    {
                        usage = Some(response.usage);
                    }
                }

                let usage = usage.expect(
                    "the stream should carry final usage; losing it on the streaming path is the \
                     exact bug this cell exists to catch",
                );
                let ratio = usage.cached_input_tokens as f64 / usage.input_tokens as f64;
                assert!(
                    ratio >= 0.95,
                    "a streamed request against a cache handle should read essentially the whole \
                     prefix from cache, got {} of {} ({:.1}%)",
                    usage.cached_input_tokens,
                    usage.input_tokens,
                    ratio * 100.0
                );
            })
            .await;
        },
    )
    .await;
}

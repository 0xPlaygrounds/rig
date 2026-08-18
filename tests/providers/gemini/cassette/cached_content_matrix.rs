//! The `cachedContents` create matrix — every shape rig can build.
//!
//! One cell per combination of the five dimensions that change the create body:
//! payload (system instruction / contents / both), tool count (none / one /
//! three), tool choice (absent / auto), display name (absent / present), and
//! expiry (relative TTL / absolute timestamp / provider default).
//!
//! The full Cartesian product is 108 cells. That is affordable because each one
//! creates a cache just over the 1,024-token minimum and deletes it immediately,
//! so the whole matrix is a few hundred thousand input tokens on Flash and
//! essentially no storage.
//!
//! Each cell asserts the same invariants — the handle's shape, the model
//! binding, the stored token count against the documented minimum, and that an
//! expiry came back — so a provider change to any of them fails loudly across
//! the whole matrix rather than in one hand-picked scenario.
//!
//! Every cell deletes what it created, including on the failure path, because
//! storage bills until it does.
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
use rig::providers::gemini;
use rig::providers::gemini::cached_content::{CacheExpiry, CachedContent, NewCachedContent};
use std::time::Duration;

use super::super::support::with_gemini_prompt_caching_cassette;

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
            .await
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
            .await
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
            .await
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
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn sys_notools_nocfg_named_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_notools_nocfg_named_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 0,
                    tool_config: false,
                    display_name: true,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn sys_notools_nocfg_named_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_notools_nocfg_named_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 0,
                    tool_config: false,
                    display_name: true,
                    expiry: "default",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn sys_notools_autocfg_noname_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_notools_autocfg_noname_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 0,
                    tool_config: true,
                    display_name: false,
                    expiry: "ttl",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn sys_notools_autocfg_noname_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_notools_autocfg_noname_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 0,
                    tool_config: true,
                    display_name: false,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn sys_notools_autocfg_noname_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_notools_autocfg_noname_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 0,
                    tool_config: true,
                    display_name: false,
                    expiry: "default",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn sys_notools_autocfg_named_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_notools_autocfg_named_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 0,
                    tool_config: true,
                    display_name: true,
                    expiry: "ttl",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn sys_notools_autocfg_named_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_notools_autocfg_named_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 0,
                    tool_config: true,
                    display_name: true,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn sys_notools_autocfg_named_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_notools_autocfg_named_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 0,
                    tool_config: true,
                    display_name: true,
                    expiry: "default",
                },
            )
            .await
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
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn sys_onetool_nocfg_noname_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_onetool_nocfg_noname_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 1,
                    tool_config: false,
                    display_name: false,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn sys_onetool_nocfg_noname_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_onetool_nocfg_noname_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 1,
                    tool_config: false,
                    display_name: false,
                    expiry: "default",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn sys_onetool_nocfg_named_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_onetool_nocfg_named_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 1,
                    tool_config: false,
                    display_name: true,
                    expiry: "ttl",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn sys_onetool_nocfg_named_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_onetool_nocfg_named_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 1,
                    tool_config: false,
                    display_name: true,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn sys_onetool_nocfg_named_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_onetool_nocfg_named_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 1,
                    tool_config: false,
                    display_name: true,
                    expiry: "default",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn sys_onetool_autocfg_noname_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_onetool_autocfg_noname_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 1,
                    tool_config: true,
                    display_name: false,
                    expiry: "ttl",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn sys_onetool_autocfg_noname_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_onetool_autocfg_noname_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 1,
                    tool_config: true,
                    display_name: false,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn sys_onetool_autocfg_noname_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_onetool_autocfg_noname_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 1,
                    tool_config: true,
                    display_name: false,
                    expiry: "default",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn sys_onetool_autocfg_named_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_onetool_autocfg_named_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 1,
                    tool_config: true,
                    display_name: true,
                    expiry: "ttl",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn sys_onetool_autocfg_named_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_onetool_autocfg_named_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 1,
                    tool_config: true,
                    display_name: true,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn sys_onetool_autocfg_named_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_onetool_autocfg_named_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 1,
                    tool_config: true,
                    display_name: true,
                    expiry: "default",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn sys_threetools_nocfg_noname_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_threetools_nocfg_noname_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 3,
                    tool_config: false,
                    display_name: false,
                    expiry: "ttl",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn sys_threetools_nocfg_noname_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_threetools_nocfg_noname_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 3,
                    tool_config: false,
                    display_name: false,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn sys_threetools_nocfg_noname_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_threetools_nocfg_noname_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 3,
                    tool_config: false,
                    display_name: false,
                    expiry: "default",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn sys_threetools_nocfg_named_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_threetools_nocfg_named_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 3,
                    tool_config: false,
                    display_name: true,
                    expiry: "ttl",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn sys_threetools_nocfg_named_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_threetools_nocfg_named_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 3,
                    tool_config: false,
                    display_name: true,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn sys_threetools_nocfg_named_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_threetools_nocfg_named_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 3,
                    tool_config: false,
                    display_name: true,
                    expiry: "default",
                },
            )
            .await
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
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn sys_threetools_autocfg_noname_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_threetools_autocfg_noname_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 3,
                    tool_config: true,
                    display_name: false,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn sys_threetools_autocfg_noname_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_threetools_autocfg_noname_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 3,
                    tool_config: true,
                    display_name: false,
                    expiry: "default",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn sys_threetools_autocfg_named_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_threetools_autocfg_named_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 3,
                    tool_config: true,
                    display_name: true,
                    expiry: "ttl",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn sys_threetools_autocfg_named_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_threetools_autocfg_named_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 3,
                    tool_config: true,
                    display_name: true,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn sys_threetools_autocfg_named_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/sys_threetools_autocfg_named_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "sys",
                    tools: 3,
                    tool_config: true,
                    display_name: true,
                    expiry: "default",
                },
            )
            .await
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
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_notools_nocfg_noname_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_notools_nocfg_noname_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 0,
                    tool_config: false,
                    display_name: false,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_notools_nocfg_noname_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_notools_nocfg_noname_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 0,
                    tool_config: false,
                    display_name: false,
                    expiry: "default",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_notools_nocfg_named_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_notools_nocfg_named_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 0,
                    tool_config: false,
                    display_name: true,
                    expiry: "ttl",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_notools_nocfg_named_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_notools_nocfg_named_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 0,
                    tool_config: false,
                    display_name: true,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_notools_nocfg_named_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_notools_nocfg_named_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 0,
                    tool_config: false,
                    display_name: true,
                    expiry: "default",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_notools_autocfg_noname_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_notools_autocfg_noname_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 0,
                    tool_config: true,
                    display_name: false,
                    expiry: "ttl",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_notools_autocfg_noname_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_notools_autocfg_noname_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 0,
                    tool_config: true,
                    display_name: false,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_notools_autocfg_noname_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_notools_autocfg_noname_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 0,
                    tool_config: true,
                    display_name: false,
                    expiry: "default",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_notools_autocfg_named_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_notools_autocfg_named_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 0,
                    tool_config: true,
                    display_name: true,
                    expiry: "ttl",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_notools_autocfg_named_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_notools_autocfg_named_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 0,
                    tool_config: true,
                    display_name: true,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_notools_autocfg_named_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_notools_autocfg_named_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 0,
                    tool_config: true,
                    display_name: true,
                    expiry: "default",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_onetool_nocfg_noname_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_onetool_nocfg_noname_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 1,
                    tool_config: false,
                    display_name: false,
                    expiry: "ttl",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_onetool_nocfg_noname_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_onetool_nocfg_noname_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 1,
                    tool_config: false,
                    display_name: false,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_onetool_nocfg_noname_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_onetool_nocfg_noname_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 1,
                    tool_config: false,
                    display_name: false,
                    expiry: "default",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_onetool_nocfg_named_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_onetool_nocfg_named_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 1,
                    tool_config: false,
                    display_name: true,
                    expiry: "ttl",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_onetool_nocfg_named_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_onetool_nocfg_named_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 1,
                    tool_config: false,
                    display_name: true,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_onetool_nocfg_named_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_onetool_nocfg_named_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 1,
                    tool_config: false,
                    display_name: true,
                    expiry: "default",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_onetool_autocfg_noname_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_onetool_autocfg_noname_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 1,
                    tool_config: true,
                    display_name: false,
                    expiry: "ttl",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_onetool_autocfg_noname_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_onetool_autocfg_noname_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 1,
                    tool_config: true,
                    display_name: false,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_onetool_autocfg_noname_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_onetool_autocfg_noname_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 1,
                    tool_config: true,
                    display_name: false,
                    expiry: "default",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_onetool_autocfg_named_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_onetool_autocfg_named_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 1,
                    tool_config: true,
                    display_name: true,
                    expiry: "ttl",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_onetool_autocfg_named_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_onetool_autocfg_named_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 1,
                    tool_config: true,
                    display_name: true,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_onetool_autocfg_named_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_onetool_autocfg_named_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 1,
                    tool_config: true,
                    display_name: true,
                    expiry: "default",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_threetools_nocfg_noname_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_threetools_nocfg_noname_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 3,
                    tool_config: false,
                    display_name: false,
                    expiry: "ttl",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_threetools_nocfg_noname_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_threetools_nocfg_noname_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 3,
                    tool_config: false,
                    display_name: false,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_threetools_nocfg_noname_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_threetools_nocfg_noname_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 3,
                    tool_config: false,
                    display_name: false,
                    expiry: "default",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_threetools_nocfg_named_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_threetools_nocfg_named_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 3,
                    tool_config: false,
                    display_name: true,
                    expiry: "ttl",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_threetools_nocfg_named_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_threetools_nocfg_named_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 3,
                    tool_config: false,
                    display_name: true,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_threetools_nocfg_named_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_threetools_nocfg_named_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 3,
                    tool_config: false,
                    display_name: true,
                    expiry: "default",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_threetools_autocfg_noname_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_threetools_autocfg_noname_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 3,
                    tool_config: true,
                    display_name: false,
                    expiry: "ttl",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_threetools_autocfg_noname_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_threetools_autocfg_noname_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 3,
                    tool_config: true,
                    display_name: false,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_threetools_autocfg_noname_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_threetools_autocfg_noname_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 3,
                    tool_config: true,
                    display_name: false,
                    expiry: "default",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_threetools_autocfg_named_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_threetools_autocfg_named_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 3,
                    tool_config: true,
                    display_name: true,
                    expiry: "ttl",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_threetools_autocfg_named_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_threetools_autocfg_named_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 3,
                    tool_config: true,
                    display_name: true,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn contents_threetools_autocfg_named_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/contents_threetools_autocfg_named_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "contents",
                    tools: 3,
                    tool_config: true,
                    display_name: true,
                    expiry: "default",
                },
            )
            .await
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
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_notools_nocfg_noname_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_notools_nocfg_noname_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 0,
                    tool_config: false,
                    display_name: false,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_notools_nocfg_noname_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_notools_nocfg_noname_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 0,
                    tool_config: false,
                    display_name: false,
                    expiry: "default",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_notools_nocfg_named_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_notools_nocfg_named_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 0,
                    tool_config: false,
                    display_name: true,
                    expiry: "ttl",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_notools_nocfg_named_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_notools_nocfg_named_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 0,
                    tool_config: false,
                    display_name: true,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_notools_nocfg_named_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_notools_nocfg_named_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 0,
                    tool_config: false,
                    display_name: true,
                    expiry: "default",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_notools_autocfg_noname_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_notools_autocfg_noname_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 0,
                    tool_config: true,
                    display_name: false,
                    expiry: "ttl",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_notools_autocfg_noname_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_notools_autocfg_noname_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 0,
                    tool_config: true,
                    display_name: false,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_notools_autocfg_noname_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_notools_autocfg_noname_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 0,
                    tool_config: true,
                    display_name: false,
                    expiry: "default",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_notools_autocfg_named_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_notools_autocfg_named_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 0,
                    tool_config: true,
                    display_name: true,
                    expiry: "ttl",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_notools_autocfg_named_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_notools_autocfg_named_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 0,
                    tool_config: true,
                    display_name: true,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_notools_autocfg_named_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_notools_autocfg_named_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 0,
                    tool_config: true,
                    display_name: true,
                    expiry: "default",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_onetool_nocfg_noname_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_onetool_nocfg_noname_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 1,
                    tool_config: false,
                    display_name: false,
                    expiry: "ttl",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_onetool_nocfg_noname_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_onetool_nocfg_noname_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 1,
                    tool_config: false,
                    display_name: false,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_onetool_nocfg_noname_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_onetool_nocfg_noname_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 1,
                    tool_config: false,
                    display_name: false,
                    expiry: "default",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_onetool_nocfg_named_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_onetool_nocfg_named_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 1,
                    tool_config: false,
                    display_name: true,
                    expiry: "ttl",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_onetool_nocfg_named_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_onetool_nocfg_named_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 1,
                    tool_config: false,
                    display_name: true,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_onetool_nocfg_named_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_onetool_nocfg_named_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 1,
                    tool_config: false,
                    display_name: true,
                    expiry: "default",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_onetool_autocfg_noname_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_onetool_autocfg_noname_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 1,
                    tool_config: true,
                    display_name: false,
                    expiry: "ttl",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_onetool_autocfg_noname_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_onetool_autocfg_noname_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 1,
                    tool_config: true,
                    display_name: false,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_onetool_autocfg_noname_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_onetool_autocfg_noname_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 1,
                    tool_config: true,
                    display_name: false,
                    expiry: "default",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_onetool_autocfg_named_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_onetool_autocfg_named_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 1,
                    tool_config: true,
                    display_name: true,
                    expiry: "ttl",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_onetool_autocfg_named_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_onetool_autocfg_named_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 1,
                    tool_config: true,
                    display_name: true,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_onetool_autocfg_named_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_onetool_autocfg_named_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 1,
                    tool_config: true,
                    display_name: true,
                    expiry: "default",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_threetools_nocfg_noname_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_threetools_nocfg_noname_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 3,
                    tool_config: false,
                    display_name: false,
                    expiry: "ttl",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_threetools_nocfg_noname_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_threetools_nocfg_noname_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 3,
                    tool_config: false,
                    display_name: false,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_threetools_nocfg_noname_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_threetools_nocfg_noname_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 3,
                    tool_config: false,
                    display_name: false,
                    expiry: "default",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_threetools_nocfg_named_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_threetools_nocfg_named_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 3,
                    tool_config: false,
                    display_name: true,
                    expiry: "ttl",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_threetools_nocfg_named_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_threetools_nocfg_named_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 3,
                    tool_config: false,
                    display_name: true,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_threetools_nocfg_named_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_threetools_nocfg_named_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 3,
                    tool_config: false,
                    display_name: true,
                    expiry: "default",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_threetools_autocfg_noname_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_threetools_autocfg_noname_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 3,
                    tool_config: true,
                    display_name: false,
                    expiry: "ttl",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_threetools_autocfg_noname_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_threetools_autocfg_noname_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 3,
                    tool_config: true,
                    display_name: false,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_threetools_autocfg_noname_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_threetools_autocfg_noname_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 3,
                    tool_config: true,
                    display_name: false,
                    expiry: "default",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_threetools_autocfg_named_ttl() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_threetools_autocfg_named_ttl",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 3,
                    tool_config: true,
                    display_name: true,
                    expiry: "ttl",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_threetools_autocfg_named_abs() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_threetools_autocfg_named_abs",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 3,
                    tool_config: true,
                    display_name: true,
                    expiry: "abs",
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn both_threetools_autocfg_named_default() {
    with_gemini_prompt_caching_cassette(
        "cached_content_matrix/both_threetools_autocfg_named_default",
        |client| async move {
            run_cell(
                client,
                Cell {
                    payload: "both",
                    tools: 3,
                    tool_config: true,
                    display_name: true,
                    expiry: "default",
                },
            )
            .await
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
            let error = client
                .cached_contents()
                .create(
                    NewCachedContent::new(CACHE_MODEL)
                        .system_instruction(crate::cache_conformance::cache_padding(4))
                        .expiry(CacheExpiry::ttl(Duration::from_secs(120))),
                )
                .await
                .expect_err("a cache under the minimum should be refused");

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

            assert_eq!(
                created.expire_time.as_deref(),
                Some("2020-01-01T00:00:00Z"),
                "the past timestamp should be echoed back verbatim"
            );

            let _ = client.cached_contents().delete(&created.name).await;
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

            let (create_time, expire_time) = (
                created.create_time.as_deref().expect("create time"),
                created.expire_time.as_deref().expect("expire time"),
            );
            // Compare the hour fields rather than parsing RFC 3339: the default
            // is one hour, and both stamps are same-day UTC.
            let hour = |stamp: &str| stamp[11..13].parse::<i32>().expect("hour");
            assert_eq!(
                (hour(expire_time) - hour(create_time)).rem_euclid(24),
                1,
                "the documented default is one hour: created {create_time}, expires {expire_time}"
            );

            let _ = client.cached_contents().delete(&created.name).await;
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
            let mut created = Vec::new();
            for index in 0..3 {
                created.push(
                    caches
                        .create(
                            NewCachedContent::new(CACHE_MODEL)
                                .system_instruction(pad())
                                .display_name(format!("rig-page-{index}"))
                                .expiry(CacheExpiry::ttl(Duration::from_secs(180))),
                        )
                        .await
                        .expect("creating a page fixture should succeed")
                        .name,
                );
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
                    "every created cache should appear in the listing: {name} missing from {} \
                     entries",
                    listed.len()
                );
            }

            for name in &created {
                let _ = caches.delete(name).await;
            }
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
            assert!(
                matches!(&error, CachedContentError::Expired { name, .. } if *name == created.name),
                "a handle that is already gone should report Expired: {error:?}"
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

            let updated = caches
                .update_expiry(&created.name, CacheExpiry::expire_time(ABSOLUTE_EXPIRY))
                .await
                .expect("updating to an absolute expiry should succeed");
            assert_eq!(updated.expire_time.as_deref(), Some(ABSOLUTE_EXPIRY));

            let _ = caches.delete(&created.name).await;
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

            let _ = client.cached_contents().delete(&cache.name).await;
        },
    )
    .await;
}

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

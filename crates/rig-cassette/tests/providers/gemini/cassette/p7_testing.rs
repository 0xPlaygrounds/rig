//! P7 of design record 0002, unchanged: a maintainer's long-horizon cassette
//! and an app developer's offline test on scripted Gemini replies.

use std::path::Path;
use std::time::Duration;

use rig::AgentBuilder;
use rig::cassette::gemini::{Exchanges, Scripted, reply};
use rig::cassette::http::{CassetteSpec, ProviderCassette, cassette_path};
use rig::providers::gemini::{self, CacheExpiry, GeminiConfig, NewCachedContent};
use rig::tool::{PortableTool, tool_definition};
use serde::Deserialize;
use serde_json::{Value, json};

// Above Gemini's 1,024-token cache minimum.
const SUPPORT_PREAMBLE: &str = include_str!("support_preamble.md");

#[derive(Deserialize)]
struct OrderArgs {
    order_id: String,
}

struct LookupOrder;

impl PortableTool for LookupOrder {
    const NAME: &'static str = "lookup_order";
    type Args = OrderArgs;
    type Output = Value;
    type Error = std::convert::Infallible;

    fn description(&self) -> String {
        "Status of an order by id.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({ "type": "object", "properties": { "order_id": { "type": "string" } }, "required": ["order_id"] })
    }

    async fn call(&self, args: OrderArgs) -> Result<Value, Self::Error> {
        Ok(json!({ "order_id": args.order_id, "status": "refunded", "refunded_on": "2026-05-02" }))
    }
}

// Maintainer: recorded once against live Gemini 3.8 Flash, replayed strictly in CI.
#[tokio::test]
async fn support_agent_long_horizon() {
    const SCENARIO: &str = "long_horizon/support_agent";
    let root = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/cassettes"));
    let fixture = cassette_path(root, "gemini", SCENARIO);

    let cassette = ProviderCassette::start(
        root,
        "gemini",
        CassetteSpec::new(SCENARIO),
        gemini::BASE_URL,
    )
    .await;
    let client = GeminiConfig::new(cassette.api_key(gemini::API_KEY_ENV))
        .with_base_url(cassette.base_url())
        .client();

    let caches = client.cached_contents();
    let prefix = caches
        .create(
            NewCachedContent::new(gemini::GEMINI_3_8_FLASH)
                .system_instruction(SUPPORT_PREAMBLE)
                .tools([tool_definition(&LookupOrder)])
                .expiry(CacheExpiry::ttl(Duration::from_secs(60 * 60))),
        )
        .await
        .expect("cache created");
    let agent = AgentBuilder::new(
        client
            .completion(gemini::GEMINI_3_8_FLASH)
            .cached_content(prefix.clone()),
    )
    .preamble(SUPPORT_PREAMBLE)
    .tool(LookupOrder)
    .default_max_turns(4)
    .build();

    let mut history = Vec::new();
    for order in 1..=30 {
        agent
            .chat(format!("What happened to order A-{order}?"), &mut history)
            .await
            .expect("turn succeeds");
    }
    caches.delete(prefix.name()).await.expect("cache deleted");
    cassette.finish().await;

    let exchanges =
        Exchanges::from_cassette(&fixture).expect("every body parses as Google's schema");
    assert!(
        exchanges.len() >= 30,
        "a long-horizon run, got {}",
        exchanges.len()
    );
    exchanges
        .assert_replayed_verbatim()
        .expect("every signature and native part is re-sent unchanged");
    assert!(
        exchanges.unmodeled().is_empty(),
        "Gemini sent fields the mirror does not type"
    );
    for (turn, exchange) in exchanges.turns().iter().enumerate() {
        let request = &exchange.request;
        assert_eq!(
            request.cached_content.as_deref(),
            Some(prefix.name()),
            "turn {turn} skipped the cache"
        );
        assert!(
            request.tools.is_empty()
                && request.system_instruction.is_none()
                && request.tool_config.is_none(),
            "turn {turn} re-sent a prefix the cache owns"
        );
        let usage = exchange.usage().expect("usage reported");
        let input = usage.input_tokens.unwrap_or(0);
        let output = usage.output_tokens.unwrap_or(0);
        assert!(
            usage.cached_input_tokens.unwrap_or(0) > 0,
            "turn {turn} missed the cache"
        );
        assert!(usage.cached_input_tokens.unwrap_or(0) <= input);
        assert!(usage.reasoning_tokens.unwrap_or(0) <= output);
        assert_eq!(usage.total_tokens, Some(input + output));
    }
}

// App developer: scripted Gemini replies, no key, no network. The replies are
// Google's own JSON, so `Scripted::from_cassette(path)` replays a recording too.
#[tokio::test]
async fn refund_question_uses_lookup() {
    let scripted = Scripted::new([
        reply::function_call(
            "lookup_order",
            json!({ "order_id": "A-17" }),
            Some("c2lnLTE="),
        ),
        reply::text("Order A-17 was refunded on 2026-05-02."),
    ]);
    let client = GeminiConfig::new("offline").connect(scripted.clone());
    let agent = AgentBuilder::new(client.completion(gemini::GEMINI_3_8_FLASH))
        .tool(LookupOrder)
        .default_max_turns(3)
        .build();

    let response = agent
        .prompt("Where is my refund for A-17?")
        .await
        .expect("scripted run");
    assert!(response.output.contains("refunded"));

    let requests = scripted.requests();
    let replayed_call = requests
        .get(1)
        .and_then(|request| request.contents.get(1))
        .and_then(|content| content.parts.first())
        .expect("the second request replays the model's call");
    assert_eq!(replayed_call.thought_signature.as_deref(), Some("c2lnLTE="));
}

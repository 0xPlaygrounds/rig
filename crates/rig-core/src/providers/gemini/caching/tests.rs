use super::*;

/// A spec's cached-read ratio comes from its pricing; a spec with no price
/// keeps the default, as does every other parameter.
#[test]
fn a_policy_for_a_spec_reads_its_cached_ratio_from_its_pricing() {
    let gemini = crate::providers::registry::ProviderId::catalog(super::super::PROVIDER_NAME)
        .expect("a catalog vendor");
    let spec = crate::catalog::Catalog::builtin()
        .get_exact(gemini, "gemini-2.5-flash-image")
        .expect("listed");
    let image = AutoCache::for_spec(spec);
    assert!((image.cached_ratio - 0.25).abs() < 1e-9, "{image:?}");
    assert_eq!(
        AutoCache {
            cached_ratio: AutoCache::default().cached_ratio,
            ..image
        },
        AutoCache::default()
    );
    let unpriced = crate::catalog::ModelSpec::new(gemini, "x-rig-unlisted");
    assert_eq!(AutoCache::for_spec(&unpriced), AutoCache::default());
}

/// Prices the ratio cannot come from keep the default ratio: no cached-read
/// price, a negative one, and an input price of zero to divide by. A
/// positive pair gives their quotient.
#[test]
fn a_policy_keeps_the_default_ratio_for_prices_it_cannot_divide() {
    let pricing = |input, cache_read| crate::catalog::Pricing {
        input,
        output: 1.0,
        cache_read,
        cache_write: None,
    };
    for unusable in [
        pricing(1.0, None),
        pricing(1.0, Some(-0.1)),
        pricing(0.0, Some(0.0)),
    ] {
        assert_eq!(
            AutoCache::priced(Some(&unusable)),
            AutoCache::default(),
            "{unusable:?}"
        );
    }
    assert_eq!(AutoCache::priced(None), AutoCache::default());
    let priced = AutoCache::priced(Some(&pricing(2.0, Some(0.5))));
    assert!((priced.cached_ratio - 0.25).abs() < 1e-9, "{priced:?}");
}

/// A body's digests and token estimates, pinned: a saved [`Lease`] names its
/// prefix by digest, so a change to how bodies are read must not move them,
/// and the estimates steer when a cache is made.
#[test]
fn digests_and_estimates_are_pinned() {
    let body = br#"{"contents":[{"parts":[{"text":"Hi","thought":false}],"role":"user"},{"parts":[{"thoughtSignature":"c2lnbmF0dXJl","functionCall":{"args":{"b":1.5,"a":[1,2.0]},"name":"read"}}],"role":"model"},{"parts":[{"functionResponse":{"name":"read","response":{"result":"fn main() {}\n"}},"thought":false}],"role":"user"}],"generationConfig":{"temperature":0.0},"safetySettings":null,"toolConfig":null,"systemInstruction":{"parts":[{"text":"Be brief.","thought":false}],"role":"model"},"tools":[{"functionDeclarations":[{"name":"read","description":"Read a file.","parametersJsonSchema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}],"codeExecution":null}]}"#;
    let parsed = parse(body).expect("the body parses");
    let Digested {
        d,
        prefix,
        contents: estimates,
    } = digests("gemini-3.8-flash", &parsed);
    assert_eq!(
        d,
        [
            "ea4a272c8066da518a40f5600614c38b178580e5cb4e9056d6554ad49d7c5172",
            "497aa92d0d210ddfe4842f186746d1f2e845ff8197a2ede95749ef8795282d6e",
            "452fe3da26f9e23d58e6ef98f94ff2de85b4c1d90de7909f145275b42b4feaa7",
            "ce7230db1669c2c1c1e10ad42765296121cee0375336be07f438d04d526b50f7",
        ]
    );
    assert_eq!((prefix, estimates), (63, vec![13, 22, 29]));
}

/// A body's repeated key keeps its first place and its last value, as a
/// map does, and a `cachedContent` that is not null marks the body cached.
#[test]
fn a_parsed_body_reads_a_repeated_key_as_a_map_does() {
    let parsed =
        parse(br#"{"model":"a","contents":[],"cachedContent":"cachedContents/c","model":"b"}"#)
            .expect("the body parses");
    assert!(parsed.has_cached_content);
    let rest: Vec<(&str, &str)> = parsed
        .rest
        .iter()
        .map(|(key, value)| (key.as_str(), value.get()))
        .collect();
    assert_eq!(
        rest,
        [
            ("model", r#""b""#),
            ("cachedContent", r#""cachedContents/c""#)
        ]
    );
    let uncached = parse(br#"{"contents":[],"cachedContent":null}"#).expect("the body parses");
    assert!(!uncached.has_cached_content);
}

/// A book shows its policy and cache-name prefix, not the caches it holds.
#[test]
fn a_book_shows_its_policy_and_prefix() {
    let book = CacheBook::new(AutoCache::default()).with_display_prefix("run-");
    assert_eq!(book.policy(), AutoCache::default());
    let shown = format!("{book:?}");
    assert!(
        shown.starts_with("CacheBook { policy: AutoCache {")
            && shown.ends_with(r#"display_prefix: "run-", .. }"#),
        "{shown}"
    );
}

/// Answers `cachedContents` creates with `cachedContents/cache-<n>` and every
/// other request with a short reply, recording the cache each one read.
#[derive(Clone, Default)]
struct CacheServer(Arc<Mutex<(usize, Vec<Option<String>>)>>);

impl crate::driver::Transport<super::super::completion::GenerateContent> for CacheServer {
    fn send(
        &self,
        payload: crate::wire::Encoded,
        _: crate::driver::Exchange,
    ) -> crate::driver::Opening<crate::wire::WireFrame> {
        let body = crate::test_utils::json_body(&payload.request);
        let mut seen = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        let reply = if payload.request.uri().path().ends_with("/cachedContents") {
            seen.0 += 1;
            serde_json::json!({
                "name": format!("cachedContents/cache-{}", seen.0),
                "usageMetadata": {"totalTokenCount": 2048}
            })
        } else {
            let read = body.get("cachedContent").and_then(|name| name.as_str());
            seen.1.push(read.map(str::to_owned));
            serde_json::json!({
                "candidates": [{
                    "content": {"role": "model", "parts": [{"text": "ok"}]},
                    "finishReason": "STOP"
                }],
                "usageMetadata": {"promptTokenCount": 2050, "candidatesTokenCount": 1}
            })
        };
        crate::driver::Opening::ready(crate::driver::Opened::new(futures::stream::iter([Ok(
            crate::wire::WireFrame::Text(reply.to_string()),
        )])))
    }
}

/// A shared-prefix cache that expired is replaced by the next call that
/// would read it, and that call reads the replacement.
#[tokio::test]
async fn an_expired_shared_prefix_cache_is_replaced() {
    use crate::completion::CompletionRequest;
    use crate::providers::gemini::{GeminiConfig, completion::GenerateContent};
    use std::sync::atomic::{AtomicU64, Ordering};

    let clock = Arc::new(AtomicU64::new(1000));
    let reading = clock.clone();
    let book =
        CacheBook::new(AutoCache::default()).with_clock(move || reading.load(Ordering::SeqCst));
    let server = CacheServer::default();
    let model = crate::driver::Model::new(
        GenerateContent::new(GeminiConfig::new("test-key"), "gemini-2.5-flash"),
        server.clone(),
    )
    .caching(&book);
    let ask = |n: u64| CompletionRequest::new(format!("question {n}")).preamble("x".repeat(8192));

    // The second conversation on the prefix shares it, so it is cached.
    for n in 0..2 {
        model.call(ask(n)).await.expect("the call answers");
    }
    clock.store(1000 + 3600, Ordering::SeqCst);
    model.call(ask(2)).await.expect("the call answers");

    let seen = server.0.lock().unwrap_or_else(PoisonError::into_inner);
    assert_eq!(
        seen.1,
        [
            None,
            Some("cachedContents/cache-1".to_owned()),
            Some("cachedContents/cache-2".to_owned())
        ]
    );
}

/// A late 403, retire or extend for an old cache leaves the replacement
/// that took its digest as it is.
#[test]
fn a_late_callback_for_an_old_cache_spares_its_replacement() {
    let old = Lease {
        name: "cachedContents/old".to_owned(),
        digest: "prefix".to_owned(),
        covers: 0,
        tokens: 2048,
        expires_at: 999,
        model: "gemini-2.5-flash".to_owned(),
    };
    let replacement = Lease {
        name: "cachedContents/new".to_owned(),
        expires_at: 4600,
        ..old.clone()
    };
    let callbacks: [fn(&CacheBook, &Lease); 3] =
        [CacheBook::lost, CacheBook::retired, |book, lease| {
            book.extended(lease, 100)
        }];
    for callback in callbacks {
        let book = CacheBook::new(AutoCache::default()).with_clock(|| 1000);
        book.restore(vec![old.clone(), replacement.clone()]);
        callback(&book, &old);
        assert_eq!(book.leases(), std::slice::from_ref(&replacement));
    }
}

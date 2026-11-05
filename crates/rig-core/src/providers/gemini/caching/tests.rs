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

#[derive(Default)]
struct CacheRequests {
    creates: usize,
    reads: Vec<Option<String>>,
}

#[derive(Clone, Default)]
struct ExpiryTransport(Arc<Mutex<CacheRequests>>);

impl crate::driver::Transport<super::super::completion::GenerateContent> for ExpiryTransport {
    fn send(
        &self,
        payload: crate::wire::Encoded,
        _: crate::driver::Exchange,
    ) -> crate::driver::Opening<crate::wire::WireFrame> {
        use crate::driver::Opened;
        use crate::error::ProviderError;
        use crate::wire::{Body, WireFrame};
        use serde_json::{Value, json};

        let requests = self.0.clone();
        crate::driver::Opening::new(async move {
            let Body::Bytes(bytes) = payload.request.body() else {
                return Err(ProviderError::request("expected a JSON request"));
            };
            let body: Value = serde_json::from_slice(bytes)
                .map_err(|error| ProviderError::request(error.to_string()))?;
            let mut requests = requests.lock().unwrap_or_else(PoisonError::into_inner);
            let reply = if payload.request.uri().path().ends_with("/cachedContents")
                && payload.request.method() == http::Method::POST
            {
                requests.creates += 1;
                json!({
                    "name": format!("cachedContents/cache-{}", requests.creates),
                    "usageMetadata": {"totalTokenCount": 2048}
                })
            } else {
                let cached = body.get("cachedContent").and_then(Value::as_str);
                requests.reads.push(cached.map(str::to_owned));
                json!({
                    "candidates": [{
                        "content": {"role": "model", "parts": [{"text": "ok"}]},
                        "finishReason": "STOP", "index": 0
                    }],
                    "usageMetadata": {
                        "promptTokenCount": 2050, "candidatesTokenCount": 1,
                        "totalTokenCount": 2051,
                        "cachedContentTokenCount": if cached.is_some() { 2048 } else { 0 }
                    }
                })
            };
            let bytes = serde_json::to_vec(&reply)
                .map_err(|error| ProviderError::request(error.to_string()))?;
            Ok(Opened::new(futures::stream::iter([Ok(WireFrame::Bytes(
                bytes,
            ))])))
        })
    }
}

/// The book's clock and creation deduplication are local policy, so a
/// controlled transport checks expiry boundaries without recording provider traffic.
#[test]
fn shared_prefix_replacement_follows_the_expiry_margin() -> anyhow::Result<()> {
    use std::sync::atomic::{AtomicU64, Ordering};

    use crate::completion::CompletionRequest;
    use crate::driver::Model;
    use crate::providers::gemini::{GeminiConfig, completion::GenerateContent};

    futures::executor::block_on(async {
        for replacement_time in [4595, 4600, 4620] {
            let clock = Arc::new(AtomicU64::new(1000));
            let reading = clock.clone();
            let book = CacheBook::new(AutoCache::default())
                .with_clock(move || reading.load(Ordering::SeqCst));
            let transport = ExpiryTransport::default();
            let model = Model::new(
                GenerateContent::new(GeminiConfig::new("test-key"), "gemini-2.5-flash"),
                transport.clone(),
            )
            .caching(&book);
            let request = |question| {
                CompletionRequest::new(format!("question {question}")).preamble("x".repeat(8192))
            };

            for question in 0..3 {
                model.call(request(question)).await?;
            }
            let old = book
                .leases()
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("the shared prefix should create a cache"))?;
            anyhow::ensure!(old.covers == 0 && old.expires_at == 4600);

            clock.store(4594, Ordering::SeqCst);
            model.call(request(3)).await?;
            anyhow::ensure!(book.leases().as_slice() == std::slice::from_ref(&old));

            clock.store(replacement_time, Ordering::SeqCst);
            model.call(request(4)).await?;
            model.call(request(5)).await?;
            let leases = book.leases();
            anyhow::ensure!(leases.len() == 1);
            let replacement = leases
                .first()
                .ok_or_else(|| anyhow::anyhow!("the shared prefix should retain a replacement"))?;
            anyhow::ensure!(
                replacement.name == "cachedContents/cache-2"
                    && replacement.digest == old.digest
                    && replacement.covers == 0
                    && replacement.expires_at == replacement_time + 3600,
                "wrong replacement at {replacement_time}: {replacement:?}"
            );
            let requests = transport.0.lock().unwrap_or_else(PoisonError::into_inner);
            anyhow::ensure!(
                requests.creates == 2,
                "creates at {replacement_time}: {}",
                requests.creates
            );
            let reads: Vec<_> = requests.reads.iter().map(Option::as_deref).collect();
            anyhow::ensure!(
                reads
                    == [
                        None,
                        Some("cachedContents/cache-1"),
                        Some("cachedContents/cache-1"),
                        Some("cachedContents/cache-1"),
                        Some("cachedContents/cache-2"),
                        Some("cachedContents/cache-2"),
                    ],
                "reads at {replacement_time}: {reads:?}"
            );
        }
        Ok(())
    })
}

#[test]
fn a_waiting_cache_creation_rechecks_a_replacements_expiry() -> anyhow::Result<()> {
    use std::sync::atomic::{AtomicU64, Ordering};

    use crate::completion::CompletionRequest;
    use crate::driver::Model;
    use crate::providers::gemini::{GeminiConfig, completion::GenerateContent};

    futures::executor::block_on(async {
        let clock = Arc::new(AtomicU64::new(1000));
        let reading = clock.clone();
        let book =
            CacheBook::new(AutoCache::default()).with_clock(move || reading.load(Ordering::SeqCst));
        let transport = ExpiryTransport::default();
        let model = Model::new(
            GenerateContent::new(GeminiConfig::new("test-key"), "gemini-2.5-flash"),
            transport.clone(),
        )
        .caching(&book);
        let request = |question| {
            CompletionRequest::new(format!("question {question}")).preamble("x".repeat(8192))
        };
        for question in 0..3 {
            model.call(request(question)).await?;
        }
        let old = book
            .leases()
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("the shared prefix should create a cache"))?;

        clock.store(4595, Ordering::SeqCst);
        let guard = book.create.lock().await;
        let mut waiting = Box::pin(model.call(request(3)));
        anyhow::ensure!(futures::poll!(waiting.as_mut()).is_pending());
        book.restore(vec![Lease {
            name: "cachedContents/racing".to_owned(),
            expires_at: 4601,
            ..old
        }]);
        clock.store(4602, Ordering::SeqCst);
        drop(guard);
        waiting.await?;

        let requests = transport.0.lock().unwrap_or_else(PoisonError::into_inner);
        anyhow::ensure!(requests.creates == 2);
        anyhow::ensure!(
            requests.reads.last().and_then(Option::as_deref) == Some("cachedContents/cache-2")
        );
        Ok(())
    })
}

#[test]
fn delayed_cache_events_leave_the_replacement_intact() {
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
    for action in ["lost", "retired", "extended"] {
        let book = CacheBook::new(AutoCache::default()).with_clock(|| 1000);
        book.restore(vec![old.clone()]);
        book.restore(vec![replacement.clone()]);
        match action {
            "lost" => book.lost(&old),
            "retired" => book.retired(&old),
            _ => book.extended(&old, 100),
        }
        assert_eq!(
            book.leases(),
            std::slice::from_ref(&replacement),
            "{action}"
        );

        match action {
            "lost" => book.lost(&replacement),
            "retired" => book.retired(&replacement),
            _ => book.extended(&replacement, 100),
        }
        if action == "extended" {
            assert_eq!(
                book.leases(),
                [Lease {
                    expires_at: 1100,
                    ..replacement.clone()
                }]
            );
        } else {
            assert!(book.leases().is_empty(), "{action}");
        }
    }
}

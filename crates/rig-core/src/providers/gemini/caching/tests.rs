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

/// Answers `cachedContents` creates with `cachedContents/cache-<n>`, extends
/// with a 403 and every other request with a short reply, recording the cache
/// each one read. A create waits for `.1`. A cache named in `.2` answers a
/// read with 403.
#[derive(Clone, Default)]
struct CacheServer(
    Arc<Mutex<(usize, Vec<Option<String>>)>>,
    Arc<futures::lock::Mutex<()>>,
    Arc<Mutex<HashSet<String>>>,
);

impl crate::driver::Transport<super::super::completion::GenerateContent> for CacheServer {
    fn send(
        &self,
        payload: crate::wire::Encoded,
        _: crate::driver::Exchange,
    ) -> crate::driver::Opening<crate::wire::WireFrame> {
        use crate::driver::{Opened, Opening};
        use crate::error::ProviderError;
        use crate::wire::WireFrame;

        let refused =
            |status| Opening::failed(ProviderError::from_http_response(status, "refused"));
        let reply = |text: String| Opened::new(futures::stream::iter([Ok(WireFrame::Text(text))]));
        let path = payload.request.uri().path();
        if payload.request.method() == http::Method::PATCH {
            return refused(http::StatusCode::FORBIDDEN);
        }
        if path.ends_with("/cachedContents") {
            let (seen, gate) = (self.0.clone(), self.1.clone());
            return Opening::new(async move {
                drop(gate.lock().await);
                let mut seen = seen.lock().unwrap_or_else(PoisonError::into_inner);
                seen.0 += 1;
                let cache = serde_json::json!({
                    "name": format!("cachedContents/cache-{}", seen.0),
                    "usageMetadata": {"totalTokenCount": 2048}
                });
                Ok(reply(cache.to_string()))
            });
        }
        let body = crate::test_utils::json_body(&payload.request);
        let read = body.get("cachedContent").and_then(|name| name.as_str());
        let mut seen = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        seen.1.push(read.map(str::to_owned));
        if read.is_some_and(|name| self.2.lock().expect("refused").contains(name)) {
            let error = ProviderError::from_http_response(http::StatusCode::FORBIDDEN, "refused");
            return Opening::ready(Opened::new(futures::stream::iter([Err(error)])));
        }
        Opening::ready(reply(
            serde_json::json!({
                "candidates": [{
                    "content": {"role": "model", "parts": [{"text": "ok"}]},
                    "finishReason": "STOP"
                }],
                "usageMetadata": {"promptTokenCount": 2050, "candidatesTokenCount": 1}
            })
            .to_string(),
        ))
    }
}

/// Answers every `cachedContents` delete once `.1` is free, recording the
/// names. While `.2` is set it answers 500.
#[derive(Clone, Default)]
struct DeleteServer(
    Arc<Mutex<Vec<String>>>,
    Arc<futures::lock::Mutex<()>>,
    Arc<std::sync::atomic::AtomicBool>,
);

impl crate::driver::Transport<super::super::cached_content::CachedContents> for DeleteServer {
    fn send(
        &self,
        payload: crate::wire::Encoded,
        _: crate::driver::Exchange,
    ) -> crate::driver::Opening<crate::wire::WireFrame> {
        let path = payload.request.uri().path().trim_start_matches("/v1beta/");
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(path.to_owned());
        let (gate, refused) = (self.1.clone(), self.2.clone());
        crate::driver::Opening::new(async move {
            drop(gate.lock().await);
            if refused.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(crate::error::ProviderError::from_http_response(
                    http::StatusCode::INTERNAL_SERVER_ERROR,
                    "refused",
                ));
            }
            Ok(crate::driver::Opened::new(futures::stream::iter([Ok(
                crate::wire::WireFrame::Text("{}".to_owned()),
            )])))
        })
    }
}

/// A shared-prefix cache that a later call read is replaced by the next call
/// after it expires, and that call reads the replacement. One nobody read
/// again is replaced only after a run of calls shows a read would pay, so no
/// spacing of calls keeps making caches nobody reads. A cache that would not
/// live to its line's next call is never made, and a request whose extend
/// found its cache gone goes inline.
#[tokio::test]
async fn an_expired_shared_prefix_cache_is_replaced() {
    use crate::completion::CompletionRequest;
    use crate::providers::gemini::{GeminiConfig, completion::GenerateContent};
    use std::sync::atomic::{AtomicU64, Ordering};

    let clock = Arc::new(AtomicU64::new(1000));
    let clocked = |policy| {
        let reading = clock.clone();
        CacheBook::new(policy).with_clock(move || reading.load(Ordering::SeqCst))
    };
    let book = clocked(AutoCache::default());
    let server = CacheServer::default();
    let caching = |book: &CacheBook, server: &CacheServer| {
        crate::driver::Model::new(
            GenerateContent::new(GeminiConfig::new("test-key"), "gemini-2.5-flash"),
            server.clone(),
        )
        .caching(book)
    };
    let model = caching(&book, &server);
    let ask = |n: u64| CompletionRequest::new(format!("question {n}")).preamble("x".repeat(8192));

    // The second conversation on the prefix shares it, so it is cached.
    for n in 0..3 {
        model.call(ask(n)).await.expect("the call answers");
    }
    clock.store(1000 + 3600, Ordering::SeqCst);
    model.call(ask(3)).await.expect("the call answers");
    clock.fetch_add(2 * MAX_TTL_SECS, Ordering::SeqCst);
    for _ in 0..3 {
        model.call(ask(3)).await.expect("the call answers");
    }

    assert_eq!(
        server.0.lock().unwrap_or_else(PoisonError::into_inner).1,
        [
            None,
            Some("cachedContents/cache-1".to_owned()),
            Some("cachedContents/cache-1".to_owned()),
            Some("cachedContents/cache-2".to_owned()),
            None,
            None,
            Some("cachedContents/cache-3".to_owned())
        ]
    );

    // Two questions taking turns raise a 5 second TTL to twice their gap,
    // which still ends before each one's next call.
    let short = clocked(AutoCache {
        ttl: Duration::from_secs(EXPIRY_MARGIN_SECS),
        ..AutoCache::default()
    });
    let server = CacheServer::default();
    let model = caching(&short, &server);
    for n in 0..8 {
        model.call(ask(n % 2)).await.expect("the call answers");
        clock.fetch_add(2, Ordering::SeqCst);
    }
    assert_eq!(
        *server.0.lock().unwrap_or_else(PoisonError::into_inner),
        (0, vec![None; 8])
    );

    // A new conversation each TTL: every call is on a line of its own, and
    // none would read what the one before it made.
    let hourly = clocked(AutoCache::default());
    let server = CacheServer::default();
    let model = caching(&hourly, &server);
    for n in 0..8 {
        model.call(ask(n)).await.expect("the call answers");
        clock.fetch_add(3600, Ordering::SeqCst);
    }
    assert_eq!(
        *server.0.lock().unwrap_or_else(PoisonError::into_inner),
        (0, vec![None; 8])
    );

    // Two questions, then a pause past the TTL: the first pair's cache is
    // read by no other request, so no later pair replaces it.
    let bursts = clocked(AutoCache::default());
    let server = CacheServer::default();
    let model = caching(&bursts, &server);
    for n in 0..16 {
        model.call(ask(n)).await.expect("the call answers");
        clock.fetch_add(if n % 2 == 0 { 1 } else { 3700 }, Ordering::SeqCst);
    }
    assert_eq!(server.0.lock().expect("seen").0, 1);

    // Reading a cache close to expiry extends it first; the 403 means gone.
    let extending = clocked(AutoCache::default());
    let server = CacheServer::default();
    let model = caching(&extending, &server);
    for n in [0, 1] {
        model.call(ask(n)).await.expect("the call answers");
    }
    clock.fetch_add(3000, Ordering::SeqCst);
    model.call(ask(1)).await.expect("the call answers");
    assert_eq!(
        server.0.lock().expect("seen").1,
        [None, Some("cachedContents/cache-1".to_owned()), None]
    );
    assert!(extending.leases().is_empty());

    // A read the server refuses is resent inline, once.
    let lost = clocked(AutoCache::default());
    let server = CacheServer::default();
    let model = caching(&lost, &server);
    for n in [0, 1] {
        model.call(ask(n)).await.expect("the call answers");
    }
    let name = "cachedContents/cache-1".to_owned();
    server.2.lock().expect("refused").insert(name.clone());
    model.call(ask(1)).await.expect("the call answers");
    assert_eq!(
        server.0.lock().expect("seen").1,
        [None, Some(name.clone()), Some(name), None]
    );

    // A close waits for the create in flight and deletes its cache; the one
    // queued behind it never makes one.
    let closing = clocked(AutoCache::default());
    let server = CacheServer::default();
    let model = caching(&closing, &server);
    model.call(ask(0)).await.expect("the call answers");
    let held = server.1.lock().await;
    let (mut first, mut second) = (Box::pin(model.call(ask(1))), Box::pin(model.call(ask(2))));
    assert!(futures::poll!(first.as_mut()).is_pending());
    assert!(futures::poll!(second.as_mut()).is_pending());
    let deletes = crate::driver::Model::new(
        super::super::cached_content::CachedContents::new(GeminiConfig::new("test-key")),
        DeleteServer::default(),
    );
    let mut closed = Box::pin(closing.close(&deletes));
    assert!(futures::poll!(closed.as_mut()).is_pending());
    drop(held);
    first.await.expect("the call answers");
    second.await.expect("the call answers");
    closed.await;
    assert_eq!(server.0.lock().expect("seen").0, 1);
    assert_eq!(
        *deletes.transport.0.lock().expect("deleted"),
        ["cachedContents/cache-1"]
    );
    assert!(closing.leases().is_empty());

    // A close gives up on a stuck create in time, and on a stuck or refused
    // delete. Each cache stays the book's until a close confirms its delete.
    let stuck = clocked(AutoCache::default());
    let server = CacheServer::default();
    let model = caching(&stuck, &server);
    model.call(ask(0)).await.expect("the call answers");
    let held = server.1.lock().await;
    let mut landing = Box::pin(model.call(ask(1)));
    assert!(futures::poll!(landing.as_mut()).is_pending());
    stuck.close(&deletes).await;
    drop(held);
    landing.await.expect("the call answers");
    assert_eq!(stuck.leases().len(), 1);
    let held = deletes.transport.1.lock().await;
    stuck.close(&deletes).await;
    assert_eq!(stuck.leases().len(), 1);
    drop(held);
    deletes.transport.2.store(true, Ordering::SeqCst);
    stuck.close(&deletes).await;
    assert_eq!(stuck.leases().len(), 1);
    deletes.transport.2.store(false, Ordering::SeqCst);
    stuck.close(&deletes).await;
    assert!(stuck.leases().is_empty());
}

/// A late 403, retire or extend for an old cache leaves the replacement
/// that took its digest as it is. The old cache stays the book's to delete:
/// it may still be alive on the server, so closing the book deletes it too.
/// In any restore order the later expiry holds the digest, and a live cache
/// takes over when the one held is lost.
#[tokio::test]
async fn a_late_callback_for_an_old_cache_spares_its_replacement() {
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
            book.extended(lease, 100);
        }];
    for callback in callbacks {
        let book = CacheBook::new(AutoCache::default()).with_clock(|| 1000);
        book.restore(vec![old.clone(), replacement.clone()]);
        callback(&book, &old);
        assert_eq!(book.report().live, std::slice::from_ref(&replacement));
    }

    let earlier = Lease {
        name: "cachedContents/earlier".to_owned(),
        expires_at: 4000,
        ..old.clone()
    };
    let book = CacheBook::new(AutoCache::default()).with_clock(|| 1000);
    book.restore(vec![replacement.clone(), old.clone(), earlier.clone()]);
    assert_eq!(book.report().live, std::slice::from_ref(&replacement));
    book.extended(&earlier, 200);
    book.lost(&replacement);
    let extended = Lease {
        expires_at: 1200,
        ..earlier
    };
    assert_eq!(book.report().live, std::slice::from_ref(&extended));

    let book = CacheBook::new(AutoCache::default()).with_clock(|| 1000);
    book.restore(vec![replacement.clone(), old.clone()]);
    let server = DeleteServer::default();
    let config = crate::providers::gemini::GeminiConfig::new("test-key");
    book.close(&crate::driver::Model::new(
        super::super::cached_content::CachedContents::new(config),
        server.clone(),
    ))
    .await;
    let mut deleted = server.0.lock().expect("deleted").clone();
    deleted.sort_unstable();
    assert_eq!(deleted, [replacement.name, old.name]);
}

/// A transport that keeps each request's URL and key header, then lets
/// [`CacheServer`] answer it.
#[derive(Clone, Default)]
struct KeySpy(Arc<Mutex<Vec<(http::Uri, Option<String>)>>>, CacheServer);

impl crate::driver::Transport<super::super::completion::GenerateContent> for KeySpy {
    fn send(
        &self,
        payload: crate::wire::Encoded,
        exchange: crate::driver::Exchange,
    ) -> crate::driver::Opening<crate::wire::WireFrame> {
        let header = payload
            .request
            .headers()
            .get("x-goog-api-key")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((payload.request.uri().clone(), header));
        self.1.send(payload, exchange)
    }
}

/// A URL ends up in proxy and server logs, traces and error messages, so the
/// key travels only in the header: on every wire, and on the requests the
/// caching transport builds itself.
#[tokio::test]
async fn no_gemini_wire_puts_the_key_in_its_url() {
    use crate::completion::CompletionRequest;
    use crate::providers::gemini::cached_content::{
        CacheExpiry, CachedContentRequest, NewCachedContent,
    };
    use crate::providers::gemini::model_listing::{InteractionsModels, Models};
    use crate::providers::gemini::{GeminiConfig, completion::GenerateContent};
    use crate::transcription::TranscriptionRequestBuilder;
    use crate::wire::{Mode, Wire};

    const KEY: &str = "not-a-real-gemini-key";
    const MODEL: &str = "gemini-2.5-flash";
    let check = |wire: &str, uri: &http::Uri, header: Option<&str>| {
        assert!(
            !uri.to_string().contains(KEY),
            "{wire} put the key in its URL"
        );
        assert_eq!(header, Some(KEY), "{wire} sent no key header");
    };
    let gemini = GeminiConfig::new(KEY);
    let completion = || CompletionRequest::new("probe");
    let cached = gemini.cached_contents();
    let cache = |request| cached.encode(request, Mode::Unary);
    let name = || "cachedContents/abc".to_owned();
    let mut encoded = vec![
        (
            "completion",
            gemini.completion(MODEL).encode(completion(), Mode::Unary),
        ),
        (
            "streaming completion",
            gemini
                .completion(MODEL)
                .encode(completion(), Mode::Streaming),
        ),
        (
            "interactions",
            gemini.interactions(MODEL).encode(completion(), Mode::Unary),
        ),
        (
            "embedding",
            gemini
                .embedding("gemini-embedding-001", None)
                .encode(vec!["probe".to_owned()], Mode::Unary),
        ),
        (
            "transcription",
            gemini.transcription(MODEL).encode(
                TranscriptionRequestBuilder::new(b"audio".to_vec()).build(),
                Mode::Unary,
            ),
        ),
        (
            "cache create",
            cache(CachedContentRequest::Create(
                NewCachedContent::new(MODEL).content("corpus"),
            )),
        ),
        ("cache get", cache(CachedContentRequest::Get(name()))),
        (
            "cache list",
            cache(CachedContentRequest::List { page_token: None }),
        ),
        (
            "cache update",
            cache(CachedContentRequest::UpdateExpiry {
                name: name(),
                expiry: CacheExpiry::ttl(Duration::from_secs(60)),
            }),
        ),
        ("cache delete", cache(CachedContentRequest::Delete(name()))),
        (
            "model listing",
            Models::new(gemini.clone()).encode(None, Mode::Unary),
        ),
        (
            "interactions model listing",
            InteractionsModels::new(gemini.clone()).encode(None, Mode::Unary),
        ),
    ];
    #[cfg(feature = "image")]
    encoded.push((
        "image generation",
        gemini.image_generation(MODEL).encode(
            crate::image_generation::ImageGenerationRequestBuilder::new("probe").build(),
            Mode::Unary,
        ),
    ));
    encoded.push(("credential check", gemini.verify().encode((), Mode::Unary)));
    for (wire, encoded) in encoded {
        let request = encoded.expect("the request encodes").request;
        let header = request
            .headers()
            .get("x-goog-api-key")
            .and_then(|value| value.to_str().ok());
        check(wire, request.uri(), header);
    }

    // The second conversation on a prefix makes a cache, on a request the
    // caching transport builds without a wire.
    let spy = KeySpy::default();
    let model = crate::driver::Model::new(GenerateContent::new(gemini, MODEL), spy.clone())
        .caching(&CacheBook::new(AutoCache::default()));
    for n in 0..2 {
        let request = CompletionRequest::new(format!("question {n}")).preamble("x".repeat(8192));
        model.call(request).await.expect("the call answers");
    }
    let sent = std::mem::take(&mut *spy.0.lock().unwrap_or_else(PoisonError::into_inner));
    assert!(
        sent.iter()
            .any(|(uri, _)| uri.path().ends_with("/cachedContents")),
        "no cache was made: {sent:?}"
    );
    for (uri, header) in &sent {
        check("caching transport", uri, header.as_deref());
    }
}

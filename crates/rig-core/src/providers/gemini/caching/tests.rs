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

//! The client-side validation surface, exhaustively.
//!
//! Every cell here is free: it never opens a socket, so the full Cartesian
//! product is affordable where a recorded matrix would have to sample.

use super::*;

/// `models/x` from every spelling a caller might reach for.
#[test]
fn model_qualification_is_total_and_idempotent() {
    for (input, expected) in [
        ("gemini-2.5-flash", "models/gemini-2.5-flash"),
        ("models/gemini-2.5-flash", "models/gemini-2.5-flash"),
        ("", "models/"),
        ("models/", "models/"),
        ("tunedModels/x", "models/tunedModels/x"),
    ] {
        assert_eq!(qualify_model(input), expected, "input {input:?}");
    }
}

/// Every shape a handle can arrive in, against the path that `get`,
/// `update_expiry` and `delete` all build from it.
///
/// The refusals carry the weight. `abc?stale` is not a malformed URL the
/// provider would reject — `build_uri` appends the API key with `&` once a
/// `?` is present, so it is a valid `DELETE` of cache `abc`. `#` and `/`
/// mis-target the same way, and an empty id aims the request at the
/// collection endpoint.
#[test]
fn resource_path_accepts_server_assigned_ids_and_refuses_everything_else() {
    for (input, expected) in [
        // The shapes the API actually hands back, in both spellings, plus
        // the scrubbed spelling a replayed cassette feeds back to `delete`.
        ("n3v1qk0nqz9k", Some("/v1beta/cachedContents/n3v1qk0nqz9k")),
        (
            "cachedContents/n3v1qk0nqz9k",
            Some("/v1beta/cachedContents/n3v1qk0nqz9k"),
        ),
        (
            "cached-REDACTED_1",
            Some("/v1beta/cachedContents/cached-REDACTED_1"),
        ),
        ("abc?stale", None),
        ("abc#frag", None),
        ("abc/def", None),
        ("cachedContents/cachedContents/abc", None),
        ("abc%2Fdef", None),
        ("abc def", None),
        ("abc\n", None),
        ("..", None),
        ("", None),
        ("cachedContents/", None),
    ] {
        match (resource_path(input), expected) {
            (Ok(path), Some(expected)) => assert_eq!(path, expected, "input {input:?}"),
            (Err(CachedContentError::Invalid(message)), None) => assert!(
                message.contains(input),
                "input {input:?}: the refusal should quote the handle, got {message}"
            ),
            (outcome, _) => panic!("input {input:?}: unexpected {outcome:?}"),
        }
    }
}

/// Gemini's duration encoding across the range a caller might pass.
#[test]
fn ttl_encoding_covers_the_useful_range() {
    for (secs, nanos, expected) in [
        (0u64, 0u32, "0.000000000s"),
        (1, 0, "1.000000000s"),
        (60, 0, "60.000000000s"),
        (3_600, 0, "3600.000000000s"),
        (86_400, 0, "86400.000000000s"),
        (0, 500_000_000, "0.500000000s"),
    ] {
        assert_eq!(
            CacheExpiry::ttl_string(Duration::new(secs, nanos)),
            expected,
            "{secs}s {nanos}ns"
        );
    }
}

/// Expiry is one field on the wire, whichever order it is set in and
/// however many times.
#[test]
fn expiry_is_exclusive_under_every_ordering() {
    let orderings: Vec<Vec<CacheExpiry>> = vec![
        vec![CacheExpiry::ttl(Duration::from_secs(60))],
        vec![CacheExpiry::expire_time("2030-01-01T00:00:00Z")],
        vec![
            CacheExpiry::ttl(Duration::from_secs(60)),
            CacheExpiry::expire_time("2030-01-01T00:00:00Z"),
        ],
        vec![
            CacheExpiry::expire_time("2030-01-01T00:00:00Z"),
            CacheExpiry::ttl(Duration::from_secs(60)),
        ],
        vec![
            CacheExpiry::ttl(Duration::from_secs(1)),
            CacheExpiry::ttl(Duration::from_secs(2)),
        ],
    ];

    for ordering in orderings {
        let mut request = NewCachedContent::new("gemini-2.5-flash").content("corpus");
        for expiry in &ordering {
            request = request.expiry(expiry.clone());
        }
        let body = serde_json::to_value(&request).expect("serialize");
        let object = body.as_object().expect("object");
        let set = usize::from(object.contains_key("ttl"))
            + usize::from(object.contains_key("expireTime"));
        assert_eq!(
            set, 1,
            "exactly one expiry field should reach the wire: {body}"
        );

        // The last one set is the one that survives.
        match ordering.last().expect("non-empty") {
            CacheExpiry::Ttl(_) => assert!(object.contains_key("ttl"), "{body}"),
            CacheExpiry::ExpireTime(_) => {
                assert!(object.contains_key("expireTime"), "{body}");
            }
        }
    }
}

/// A cache with nothing in it would bill for storage and cache nothing.
#[test]
fn emptiness_is_rejected_but_either_payload_alone_suffices() {
    assert!(
        NewCachedContent::new("gemini-2.5-flash")
            .validate()
            .is_err(),
        "an empty cached content should be refused"
    );
    assert!(
        NewCachedContent::new("gemini-2.5-flash")
            .content("corpus")
            .validate()
            .is_ok()
    );
    assert!(
        NewCachedContent::new("gemini-2.5-flash")
            .system_instruction("be brief")
            .validate()
            .is_ok()
    );
    // Display name and expiry are not payload.
    assert!(
        NewCachedContent::new("gemini-2.5-flash")
            .display_name("x")
            .expiry(CacheExpiry::ttl(Duration::from_secs(60)))
            .validate()
            .is_err()
    );
}

/// Only the fields a caller actually set reach the wire.
#[test]
fn unset_fields_are_omitted_across_every_builder_combination() {
    let always_present = ["model"];
    for (label, request) in [
        (
            "content only",
            NewCachedContent::new("gemini-2.5-flash").content("corpus"),
        ),
        (
            "system only",
            NewCachedContent::new("gemini-2.5-flash").system_instruction("be brief"),
        ),
        (
            "both",
            NewCachedContent::new("gemini-2.5-flash")
                .content("corpus")
                .system_instruction("be brief"),
        ),
        (
            "named",
            NewCachedContent::new("gemini-2.5-flash")
                .content("corpus")
                .display_name("corpus-v1"),
        ),
    ] {
        let body = serde_json::to_value(&request).expect("serialize");
        let object = body.as_object().expect("object");
        for key in always_present {
            assert!(object.contains_key(key), "{label}: missing {key}");
        }
        for key in ["tools", "toolConfig", "ttl", "expireTime"] {
            assert!(
                !object.contains_key(key),
                "{label}: {key} was never set and must not be sent: {body}"
            );
        }
    }
}

/// Multiple content blocks accumulate in order — a corpus is usually more
/// than one document.
#[test]
fn content_blocks_accumulate_in_order() {
    let request = NewCachedContent::new("gemini-2.5-flash")
        .content("first")
        .content("second")
        .content("third");
    let body = serde_json::to_value(&request).expect("serialize");
    let contents = body
        .get("contents")
        .and_then(|value| value.as_array())
        .expect("contents array");
    assert_eq!(contents.len(), 3);
    let texts: Vec<&str> = contents
        .iter()
        .filter_map(|entry| {
            entry
                .get("parts")?
                .as_array()?
                .first()?
                .get("text")?
                .as_str()
        })
        .collect();
    assert_eq!(texts, vec!["first", "second", "third"]);
}

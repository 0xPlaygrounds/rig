use serde_json::json;

use super::{models, token_usage, uuidv7};

#[test]
fn listed_models_skip_diagnostics_and_deduplicate() {
    let listing = "\
muse: local session messaging disabled: session logging is required
muse-wonder-1.0
muse-wonder-full-1.0
muse-ember-1.0
muse-wonder-1.0

";
    assert_eq!(
        models(listing),
        vec![
            "muse-wonder-1.0".to_owned(),
            "muse-wonder-full-1.0".to_owned(),
            "muse-ember-1.0".to_owned(),
        ]
    );
}

#[test]
fn uuidv7_has_version_7_and_rfc4122_variant() {
    let id = uuidv7();
    let parsed = uuid::Uuid::parse_str(&id).ok();
    assert_eq!(parsed.map(|id| id.get_version_num()), Some(7));
    assert_eq!(parsed.map(|id| id.as_bytes()[8] & 0xc0), Some(0x80));
}

#[test]
fn token_usage_maps_prompt_and_cache_counters() {
    let params = json!({
        "promptTokens": 21626,
        "totalTokens": 21792,
        "usage": {
            "inputTokens": 262,
            "outputTokens": 166,
            "cachedTokens": 21364,
            "cacheReadTokens": 21364,
            "cacheWriteTokens": 0,
            "reasoningTokens": 40,
        }
    });
    let usage = token_usage(&params).unwrap_or_default();
    assert_eq!(usage.input_tokens, Some(21626));
    assert_eq!(usage.output_tokens, Some(166));
    assert_eq!(usage.cached_input_tokens, Some(21364));
    assert_eq!(usage.cache_creation_input_tokens, Some(0));
    assert_eq!(usage.reasoning_tokens, Some(40));
    assert_eq!(usage.total_tokens, Some(21792));
}

//! The merge that assembles a streamed Chat tool call from its fragments and
//! folds stored reasoning details back into a message.

/// The keys whose string fragments concatenate when a provider streams them:
/// text and reasoning, a call's arguments or custom input, audio's transcript
/// and data, a reasoning detail's text and summary. Every other string is an identifier,
/// a tag or a signature a later fragment restates.
const FRAGMENT_KEYS: [&str; 13] = [
    "content",
    "input",
    "refusal",
    "reasoning",
    "reasoning_content",
    "reasoning_text",
    "thinking",
    "tool_plan",
    "transcript",
    "data",
    "arguments",
    "text",
    "summary",
];

/// Merge one streamed fragment of a provider object into what arrived so
/// far: fragment strings ([`FRAGMENT_KEYS`]) append, arrays extend, objects
/// merge key by key. A value never replaces one of another JSON type, a
/// `null` or empty string never erases a value, and a literal `null`
/// argument placeholder gives way to the first real fragment.
pub(crate) fn merge_fields(
    target: &mut serde_json::Map<String, serde_json::Value>,
    delta: &serde_json::Map<String, serde_json::Value>,
) {
    use serde_json::Value;
    for (key, value) in delta {
        let fragment = FRAGMENT_KEYS.contains(&key.as_str());
        match (target.get_mut(key), value) {
            (Some(_), Value::Null) => {}
            (Some(Value::String(existing)), Value::String(more)) if fragment => {
                if existing.trim() == "null" && !more.trim().is_empty() {
                    existing.clear();
                }
                existing.push_str(more);
            }
            (Some(Value::String(existing)), Value::String(more))
                if more.is_empty() && !existing.is_empty() => {}
            (Some(Value::Array(existing)), Value::Array(more)) => {
                existing.extend(more.iter().cloned());
            }
            (Some(Value::Object(existing)), Value::Object(more)) => merge_fields(existing, more),
            (Some(existing), more)
                if !existing.is_null()
                    && std::mem::discriminant(existing) != std::mem::discriminant(more) => {}
            _ => {
                target.insert(key.clone(), value.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests;

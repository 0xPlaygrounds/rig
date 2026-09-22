use std::collections::BTreeSet;

use rig_cassette::effect_log::EffectLog;
use rig_core::{
    completion::Usage,
    effect::{EffectKind, Outcome},
};
use serde_json::{Value, json};

use super::super::{cells::ThinkingWire, long_loop};

pub(crate) fn assert_usage(thinking: ThinkingWire, log: &EffectLog) -> Value {
    let mut identities = BTreeSet::new();
    let mut rows = Vec::new();
    let mut usages = Vec::new();
    for record in &log.records {
        if !matches!(record.kind, EffectKind::Completion { .. }) {
            continue;
        }
        let identity = serde_json::to_string(&(&record.scope, record.id)).expect("effect identity");
        assert_unique(&mut identities, identity);
        let Ok(Outcome::Completion(response)) = &record.outcome else {
            panic!("successful task completion");
        };
        let usage = response.usage;
        let raw = long_loop::raw_usage(thinking, record);
        assert_prompt_usage(&usage, raw);
        let raw_usage = match thinking {
            ThinkingWire::Gemini => response
                .raw
                .get("usageMetadata")
                .or_else(|| response.raw.get("usage_metadata"))
                .expect("Gemini usage"),
            _ => response.raw.get("usage").expect("wire usage"),
        };
        let output_field = match thinking {
            ThinkingWire::OpenAiChat | ThinkingWire::DeepSeek => "completion_tokens",
            ThinkingWire::Gemini => "candidatesTokenCount",
            _ => "output_tokens",
        };
        assert_eq!(
            usage.output_tokens,
            raw_usage.get(output_field).and_then(Value::as_u64),
            "output usage decoded once"
        );
        let total = match thinking {
            ThinkingWire::Anthropic => raw
                .0
                .zip(usage.output_tokens)
                .map(|(input, output)| input + raw.1.unwrap_or(0) + raw.2.unwrap_or(0) + output),
            ThinkingWire::Gemini => raw_usage.get("totalTokenCount").and_then(Value::as_u64),
            _ => raw_usage.get("total_tokens").and_then(Value::as_u64),
        };
        assert_eq!(
            usage.total_tokens, total,
            "provider-specific total accounting"
        );
        if !matches!(thinking, ThinkingWire::Anthropic)
            && let (Some(input), Some(cached)) = (usage.input_tokens, usage.cached_input_tokens)
        {
            assert!(cached <= input, "inclusive cache accounting");
        }
        rows.push(json!({"scope": record.scope, "id": record.id, "usage": usage}));
        usages.push(usage);
    }
    assert!(!usages.is_empty());
    let segments: Vec<_> = usages.chunks(usages.len().div_ceil(3)).map(|segment| {
        json!({"turns": segment.len(), "input": total_counter(segment, |u| u.input_tokens), "cache_read": total_counter(segment, |u| u.cached_input_tokens), "cache_creation": total_counter(segment, |u| u.cache_creation_input_tokens)})
    }).collect();
    json!({"turns": rows, "segments": segments,
        "totals": {"input": total_counter(&usages, |u| u.input_tokens), "output": total_counter(&usages, |u| u.output_tokens), "total": total_counter(&usages, |u| u.total_tokens), "cache_read": total_counter(&usages, |u| u.cached_input_tokens), "cache_creation": total_counter(&usages, |u| u.cache_creation_input_tokens)}})
}

pub(crate) fn assert_totals(usages: &[Usage], actual: Usage) {
    assert_eq!(actual, reported_totals(usages), "cumulative reported usage");
}

pub(crate) fn reported_totals(usages: &[Usage]) -> Usage {
    fn sum(usages: &[Usage], field: fn(&Usage) -> Option<u64>) -> Option<u64> {
        let values: Vec<_> = usages.iter().filter_map(field).collect();
        if values.is_empty() {
            return None;
        }
        Some(
            values
                .into_iter()
                .try_fold(0_u64, |total, value| total.checked_add(value))
                .expect("bounded reported total"),
        )
    }
    Usage {
        input_tokens: sum(usages, |u| u.input_tokens),
        output_tokens: sum(usages, |u| u.output_tokens),
        total_tokens: sum(usages, |u| u.total_tokens),
        cached_input_tokens: sum(usages, |u| u.cached_input_tokens),
        cache_creation_input_tokens: sum(usages, |u| u.cache_creation_input_tokens),
        reasoning_tokens: sum(usages, |u| u.reasoning_tokens),
        tool_use_prompt_tokens: sum(usages, |u| u.tool_use_prompt_tokens),
    }
}

fn assert_unique(seen: &mut BTreeSet<String>, identity: String) {
    assert!(seen.insert(identity), "completion usage counted twice");
}

fn assert_prompt_usage(usage: &Usage, raw: (Option<u64>, Option<u64>, Option<u64>)) {
    assert_eq!(
        (
            usage.input_tokens,
            usage.cached_input_tokens,
            usage.cache_creation_input_tokens
        ),
        raw,
        "raw cache counters must match normalized usage"
    );
}

fn total_counter(usages: &[Usage], field: fn(&Usage) -> Option<u64>) -> Value {
    let values: Vec<_> = usages.iter().map(field).collect();
    let reported = values
        .iter()
        .flatten()
        .try_fold(0_u64, |sum, value| sum.checked_add(*value))
        .expect("bounded usage sum");
    let missing = values.iter().filter(|value| value.is_none()).count();
    json!({"reported_sum": if missing == values.len() { None } else { Some(reported) }, "missing_turns": missing, "complete_total": if missing == 0 { Some(reported) } else { None }})
}

#[cfg(test)]
#[path = "cache/tests.rs"]
mod tests;

//! The Responses section serializes every field under `"openai.responses"`,
//! and none of them writes a body leaf the request or its generation
//! options own.

use super::*;
use crate::completion::ProviderOptions;
use crate::providers::openai::extension::{OpenAiExt, OpenAiOptions};
use serde_json::json;

fn fully_set() -> OpenAiResponsesOptions {
    OpenAiResponsesOptions::default()
        .reasoning_summary(ReasoningSummary::Detailed)
        .reasoning_mode(ReasoningMode::Pro)
        .reasoning_context(ReasoningContext::AllTurns)
        .include([
            Include::FileSearchCallResults,
            Include::MessageOutputTextLogprobs,
        ])
        .conversation("conv_1")
        .truncation(Truncation::Disabled)
        .context_management([ContextManagement::compaction(None)])
        .prompt_cache_comparison("resp_0")
        .background(true)
        .max_tool_calls(4)
        .top_logprobs(5)
        .access_programs(AccessPrograms::cyber(CyberAccess::DaybreakBlue))
}

fn sections() -> Value {
    let entry = ProviderOptions::new()
        .with::<OpenAiExt>(&OpenAiOptions::default().responses(fully_set()))
        .expect("the options are sections");
    serde_json::to_value(entry.get::<OpenAiExt>()).expect("the sections serialize")
}

#[test]
fn a_fully_set_section_serializes_under_its_route() {
    assert_eq!(
        sections(),
        json!({"openai.responses": {
            "reasoning": {"summary": "detailed", "mode": "pro", "context": "all_turns"},
            "include": ["file_search_call.results", "message.output_text.logprobs"],
            "conversation": "conv_1",
            "truncation": "disabled",
            "context_management": [{"type": "compaction"}],
            "prompt_cache_options": {"comparison_response_id": "resp_0"},
            "background": true,
            "max_tool_calls": 4,
            "top_logprobs": 5,
            "access_programs": {"cyber": "daybreak_blue"}
        }})
    );
}

/// The JSON pointer of every leaf under `value`.
fn leaves(value: &Value, at: String, out: &mut Vec<String>) {
    match value {
        Value::Object(fields) if !fields.is_empty() => {
            for (key, value) in fields {
                leaves(value, format!("{at}/{key}"), out);
            }
        }
        _ => out.push(at),
    }
}

/// The body leaves the Responses wire writes itself or maps a
/// `GenerationOptions` field to (section 6.3 of `TYPED_OPTIONS.md`).
const RESERVED: &[&str] = &[
    "/model",
    "/input",
    "/instructions",
    "/max_output_tokens",
    "/temperature",
    "/tool_choice",
    "/tools",
    "/stream",
    "/text/format",
    "/text/verbosity",
    "/reasoning/effort",
    "/prompt_cache_options/ttl",
    "/prompt_cache_options/mode",
    "/prompt_cache_retention",
    "/service_tier",
    "/parallel_tool_calls",
    "/top_p",
];

/// No leaf of a fully set section is, contains or sits inside a reserved
/// leaf.
#[test]
fn no_field_writes_a_reserved_leaf() {
    let mut written = Vec::new();
    leaves(&sections()["openai.responses"], String::new(), &mut written);
    for leaf in &written {
        for reserved in RESERVED {
            assert!(
                leaf != reserved
                    && !leaf.starts_with(&format!("{reserved}/"))
                    && !reserved.starts_with(&format!("{leaf}/")),
                "{leaf} writes the reserved {reserved}"
            );
        }
    }
}

//! `CompletionRequest::new` and its setters build what #2600's request
//! builder built, and a response reads
//! back as text, reasoning, tool calls and the assistant turn. Every expected
//! value is written out here.

use super::*;
use crate::message::{Reasoning, ToolChoice};

/// A request with every field empty but the conversation: what the builder
/// built before any setter.
fn bare(chat_history: Vec<Message>) -> CompletionRequest {
    CompletionRequest {
        model: None,
        chat_history,
        documents: Vec::new(),
        tools: Vec::new(),
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
        accept_unknown_finish_reasons: false,
    }
}

/// Field-by-field equality: `CompletionRequest` is not `PartialEq`, and
/// `record_telemetry_content` is not serialized.
fn assert_same(actual: &CompletionRequest, expected: &CompletionRequest) {
    assert_eq!(
        serde_json::to_value(actual).ok(),
        serde_json::to_value(expected).ok()
    );
    assert_eq!(
        actual.record_telemetry_content,
        expected.record_telemetry_content
    );
}

fn tool(name: &str) -> ToolDefinition {
    ToolDefinition {
        name: crate::message::ToolName::new(name).expect("tool name"),
        description: format!("the {name} tool"),
        parameters: serde_json::json!({"type": "object"}),
    }
}

fn document(id: &str) -> Document {
    Document {
        id: id.to_owned(),
        text: format!("text of {id}"),
        additional_props: HashMap::new(),
    }
}

#[test]
fn new_is_the_one_user_message() {
    assert_same(
        &CompletionRequest::new("hi"),
        &bare(vec![Message::user("hi")]),
    );
}

#[test]
fn preamble_is_the_first_message_and_the_prompt_the_last() {
    let request = CompletionRequest::new("prompt")
        .message(Message::user("earlier"))
        .preamble("system");
    assert_same(
        &request,
        &bare(vec![
            Message::system("system"),
            Message::user("earlier"),
            Message::user("prompt"),
        ]),
    );
}

/// The setters apply in call order: a second preamble adds a system message
/// ahead of the first, where the builder kept only the last.
#[test]
fn a_second_preamble_goes_ahead_of_the_first() {
    let request = CompletionRequest::new("prompt")
        .preamble("first")
        .preamble("second");
    assert_same(
        &request,
        &bare(vec![
            Message::system("second"),
            Message::system("first"),
            Message::user("prompt"),
        ]),
    );
}

#[test]
fn message_and_messages_go_before_the_prompt_in_order() {
    let request = CompletionRequest::new("prompt")
        .message(Message::user("one"))
        .messages([Message::assistant("two"), Message::user("three")]);
    assert_same(
        &request,
        &bare(vec![
            Message::user("one"),
            Message::assistant("two"),
            Message::user("three"),
            Message::user("prompt"),
        ]),
    );
}

#[test]
fn document_and_documents_append_in_order() {
    let request = CompletionRequest::new("p")
        .document(document("a"))
        .documents([document("b"), document("c")]);
    let mut expected = bare(vec![Message::user("p")]);
    expected.documents = vec![document("a"), document("b"), document("c")];
    assert_same(&request, &expected);
}

#[test]
fn tool_and_tools_append_in_order() {
    let request = CompletionRequest::new("p")
        .tool(tool("a"))
        .tools(vec![tool("b"), tool("c")]);
    let mut expected = bare(vec![Message::user("p")]);
    expected.tools = vec![tool("a"), tool("b"), tool("c")];
    assert_same(&request, &expected);
}

#[test]
fn provider_tools_land_in_additional_params_tools() {
    let request = CompletionRequest::new("p")
        .additional_params(serde_json::json!({"top_p": 0.5}))
        .provider_tool(ProviderToolDefinition::new("web_search"))
        .provider_tools(vec![
            ProviderToolDefinition::new("code").with_config("x", serde_json::json!(1)),
        ]);
    let mut expected = bare(vec![Message::user("p")]);
    expected.additional_params = Some(serde_json::json!({
        "top_p": 0.5,
        "tools": [{"type": "web_search"}, {"type": "code", "x": 1}],
    }));
    assert_same(&request, &expected);
}

/// Provider tools live in `additional_params`, so parameters set after them
/// with a `tools` key, or cleared, replace them.
#[test]
fn additional_params_after_provider_tools_replace_them() {
    let replaced = CompletionRequest::new("p")
        .provider_tool(ProviderToolDefinition::new("web_search"))
        .additional_params(serde_json::json!({"tools": [{"type": "x"}]}));
    let mut expected = bare(vec![Message::user("p")]);
    expected.additional_params = Some(serde_json::json!({"tools": [{"type": "x"}]}));
    assert_same(&replaced, &expected);

    let cleared = CompletionRequest::new("p")
        .provider_tool(ProviderToolDefinition::new("web_search"))
        .additional_params(None);
    assert_same(&cleared, &bare(vec![Message::user("p")]));

    // Set first, the parameters keep their tools and the provider tools
    // follow them, as the builder sent them.
    let kept = CompletionRequest::new("p")
        .additional_params(serde_json::json!({"tools": [{"type": "x"}]}))
        .provider_tool(ProviderToolDefinition::new("web_search"));
    expected.additional_params = Some(serde_json::json!({
        "tools": [{"type": "x"}, {"type": "web_search"}],
    }));
    assert_same(&kept, &expected);
}

#[test]
fn additional_params_merge_and_none_clears() {
    let merged = CompletionRequest::new("p")
        .additional_params(serde_json::json!({"a": 1}))
        .additional_params(serde_json::json!({"b": 2}));
    let mut expected = bare(vec![Message::user("p")]);
    expected.additional_params = Some(serde_json::json!({"a": 1, "b": 2}));
    assert_same(&merged, &expected);

    let cleared = merged.additional_params(None);
    assert_same(&cleared, &bare(vec![Message::user("p")]));
}

#[test]
fn sampling_setters_set_and_none_clears() {
    let request = CompletionRequest::new("p")
        .temperature(0.2)
        .max_tokens(64)
        .tool_choice(ToolChoice::Required)
        .model("gpt-5.2")
        .record_content_telemetry(true);
    let mut expected = bare(vec![Message::user("p")]);
    expected.temperature = Some(0.2);
    expected.max_tokens = Some(64);
    expected.tool_choice = Some(ToolChoice::Required);
    expected.model = Some("gpt-5.2".to_owned());
    expected.record_telemetry_content = true;
    assert_same(&request, &expected);

    let cleared = request
        .temperature(None)
        .max_tokens(None)
        .model::<String>(None)
        .record_content_telemetry(false);
    let mut expected = bare(vec![Message::user("p")]);
    expected.tool_choice = Some(ToolChoice::Required);
    assert_same(&cleared, &expected);
}

#[test]
fn output_schema_sets_and_none_clears() {
    let schema = schemars::json_schema!({"type": "object", "title": "Answer"});
    let request = CompletionRequest::new("p").output_schema(schema.clone());
    let mut expected = bare(vec![Message::user("p")]);
    expected.output_schema = Some(schema);
    assert_same(&request, &expected);
    assert_same(
        &request.output_schema(None),
        &bare(vec![Message::user("p")]),
    );
}

#[test]
fn every_prompt_form_converts_to_the_same_request() {
    let expected = bare(vec![Message::user("hi")]);
    assert_same(&CompletionRequest::from("hi"), &expected);
    assert_same(&CompletionRequest::from("hi".to_owned()), &expected);
    assert_same(&CompletionRequest::from(Message::user("hi")), &expected);
    assert_same(
        &CompletionRequest::from(vec![Message::user("hi")]),
        &expected,
    );
}

fn response(choice: Vec<AssistantContent>) -> CompletionResponse {
    CompletionResponse::new(
        choice,
        Usage::default(),
        crate::message::Origin::new("test.api", "test", ""),
        serde_json::Value::Null,
    )
}

#[test]
fn reasoning_concatenates_reasoning_text_and_summaries_in_order() {
    let response = response(vec![
        AssistantContent::Reasoning(Reasoning::new("first, ")),
        AssistantContent::text("answer"),
        AssistantContent::Reasoning(Reasoning::new("then done")),
        AssistantContent::Reasoning(Reasoning {
            redacted: true,
            ..Reasoning::default()
        }),
    ]);
    assert_eq!(response.reasoning(), "first, then done");
    assert_eq!(response.text(), "answer");
}

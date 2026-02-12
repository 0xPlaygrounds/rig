//! Integration tests: 2-turn conversation with reasoning traces (streaming + non-streaming).
//!
//! Verifies that reasoning content from turn 1 survives roundtrip into turn 2's
//! chat history without the provider rejecting it.
//!
//! Each test requires the provider's API key as an env var.
//! Run with: `cargo test -p rig-core --test reasoning_roundtrip -- --ignored`

use futures::StreamExt;
use rig::OneOrMany;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{self, CompletionModel};
use rig::message::{AssistantContent, Message, Reasoning, UserContent};
use rig::streaming::StreamedAssistantContent;

struct TestAgent<M: CompletionModel> {
    model: M,
    preamble: String,
    additional_params: Option<serde_json::Value>,
}

const PREAMBLE: &str = "You are a helpful math tutor. Be concise.";

const TURN1_TEXT: &str = "\
A train leaves Station A at 60 km/h. Another train leaves Station B \
(300 km away) 30 minutes later at 90 km/h heading toward Station A. \
At what time do they meet, and how far from Station A? Show your work.";

const TURN2_TEXT: &str = "\
Now suppose both trains slow down by 10 km/h after traveling half \
the original distance. When do they meet now?";

/// Shared harness: runs a 2-turn streaming conversation with reasoning enabled.
///
/// Turn 1 sends a math word problem and collects reasoning + text.
/// Turn 2 sends a follow-up with the full assistant response (including reasoning)
/// in chat history, verifying the provider accepts it.
async fn run_reasoning_roundtrip_streaming<M>(agent: TestAgent<M>)
where
    M: CompletionModel,
    M::StreamingResponse: Send,
{
    // === Turn 1 ===
    let turn1_prompt = Message::User {
        content: OneOrMany::one(UserContent::text(TURN1_TEXT)),
    };

    let request = completion::CompletionRequest {
        preamble: Some(agent.preamble.clone()),
        chat_history: OneOrMany::one(turn1_prompt.clone()),
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: agent.additional_params.clone(),
        model: None,
        output_schema: None,
    };

    let mut stream = agent.model.stream(request).await.expect("Turn 1 stream");

    let mut assistant_content: Vec<AssistantContent> = vec![];
    let mut reasoning_count = 0usize;
    let mut reasoning_delta_count = 0usize;
    let mut reasoning_delta_text = String::new();
    let mut streamed_text = String::new();

    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(StreamedAssistantContent::Text(text)) => {
                streamed_text.push_str(&text.text);
            }
            Ok(StreamedAssistantContent::Reasoning(reasoning)) => {
                reasoning_count += 1;
                assistant_content.push(AssistantContent::Reasoning(reasoning));
            }
            Ok(StreamedAssistantContent::ReasoningDelta { reasoning, .. }) => {
                reasoning_delta_count += 1;
                reasoning_delta_text.push_str(&reasoning);
            }
            Ok(_) => {}
            Err(e) => panic!("Turn 1 stream error: {e}"),
        }
    }

    // Providers like Gemini 2.5 emit thinking as deltas without signatures;
    // assemble into a single Reasoning block for the chat history.
    if reasoning_count == 0 && !reasoning_delta_text.is_empty() {
        assistant_content.push(AssistantContent::Reasoning(Reasoning::new(
            &reasoning_delta_text,
        )));
    }

    let total_reasoning = reasoning_count + reasoning_delta_count;
    assert!(
        total_reasoning > 0,
        "No reasoning content received (0 blocks, 0 deltas). \
         Provider was configured for reasoning but returned none."
    );

    assert!(!streamed_text.is_empty(), "Turn 1 produced no text output.");

    // Build assistant message with reasoning + text for turn 2 history
    assistant_content.push(AssistantContent::text(&streamed_text));

    assert!(
        assistant_content
            .iter()
            .any(|c| matches!(c, AssistantContent::Reasoning(_))),
        "Assistant content for Turn 2 history has no Reasoning items"
    );

    let message_id = stream.message_id.clone();
    println!("  Message ID from stream: {message_id:?}");

    let turn1_assistant = Message::Assistant {
        id: message_id,
        content: OneOrMany::many(assistant_content).expect("non-empty"),
    };

    // === Turn 2 ===
    let turn2_prompt = Message::User {
        content: OneOrMany::one(UserContent::text(TURN2_TEXT)),
    };

    let request2 = completion::CompletionRequest {
        preamble: Some(agent.preamble.clone()),
        chat_history: OneOrMany::many(vec![turn1_prompt, turn1_assistant, turn2_prompt])
            .expect("non-empty"),
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: agent.additional_params.clone(),
        model: None,
        output_schema: None,
    };

    let mut stream2 = agent.model.stream(request2).await.expect("Turn 2 stream");

    let mut turn2_text = String::new();

    while let Some(chunk) = stream2.next().await {
        match chunk {
            Ok(StreamedAssistantContent::Text(text)) => {
                turn2_text.push_str(&text.text);
            }
            Ok(_) => {}
            Err(e) => panic!("Turn 2 stream error: {e}"),
        }
    }

    assert!(
        !turn2_text.is_empty(),
        "Turn 2 produced no text output. \
         Provider may have rejected the request with reasoning in chat history."
    );

    let trimmed = turn2_text.trim();
    assert!(
        trimmed.len() >= 20,
        "Turn 2 text suspiciously short ({} chars: {:?}). \
         Provider may not have processed the multi-turn context.",
        trimmed.len(),
        &trimmed[..trimmed.len().min(100)]
    );
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn test_openai_reasoning_roundtrip() {
    use rig::providers::openai;

    let client = openai::Client::from_env();
    run_reasoning_roundtrip_streaming(TestAgent {
        model: client.completion_model("gpt-5.2"),
        preamble: PREAMBLE.into(),
        additional_params: Some(serde_json::json!({
            "reasoning": { "effort": "medium" }
        })),
    })
    .await;
}

#[tokio::test]
#[ignore = "requires XAI_API_KEY — validate with grok-4-0725 once key is available"]
async fn test_xai_reasoning_roundtrip() {
    use rig::providers::xai;

    let client = xai::Client::from_env();
    run_reasoning_roundtrip_streaming(TestAgent {
        model: client.completion_model(xai::GROK_3_MINI),
        preamble: PREAMBLE.into(),
        additional_params: None,
    })
    .await;
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn test_anthropic_reasoning_roundtrip() {
    use rig::providers::anthropic;

    let client = anthropic::Client::from_env();
    run_reasoning_roundtrip_streaming(TestAgent {
        model: client.completion_model("claude-sonnet-4-5-20250929"),
        preamble: PREAMBLE.into(),
        additional_params: Some(serde_json::json!({
            "thinking": { "type": "enabled", "budget_tokens": 2048 }
        })),
    })
    .await;
}

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn test_gemini_reasoning_roundtrip() {
    use rig::providers::gemini;

    let client = gemini::Client::from_env();
    run_reasoning_roundtrip_streaming(TestAgent {
        model: client.completion_model("gemini-2.5-flash"),
        preamble: PREAMBLE.into(),
        additional_params: Some(serde_json::json!({
            "generationConfig": {
                "thinkingConfig": { "thinkingBudget": 2048, "includeThoughts": true }
            }
        })),
    })
    .await;
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn test_openrouter_reasoning_roundtrip() {
    use rig::providers::openrouter;

    let client = openrouter::Client::from_env();
    run_reasoning_roundtrip_streaming(TestAgent {
        model: client.completion_model("openai/gpt-5.2"),
        preamble: PREAMBLE.into(),
        additional_params: Some(serde_json::json!({
            "reasoning": { "effort": "medium" },
            "include_reasoning": true
        })),
    })
    .await;
}

// ==================== Non-Streaming Harness ====================

/// Non-streaming harness: runs a 2-turn non-streaming conversation with reasoning.
///
/// Turn 1: `model.completion()` → inspect `CompletionResponse.choice` for Reasoning items.
/// Turn 2: send follow-up with full assistant response (including reasoning) in chat history.
/// Provider accepting the request without a 400 proves reasoning roundtrip works.
async fn run_reasoning_roundtrip_nonstreaming<M>(agent: TestAgent<M>)
where
    M: CompletionModel,
{
    // === Turn 1 ===
    let turn1_prompt = Message::User {
        content: OneOrMany::one(UserContent::text(TURN1_TEXT)),
    };

    let request = completion::CompletionRequest {
        preamble: Some(agent.preamble.clone()),
        chat_history: OneOrMany::one(turn1_prompt.clone()),
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: agent.additional_params.clone(),
        model: None,
        output_schema: None,
    };

    let response = agent
        .model
        .completion(request)
        .await
        .expect("Turn 1 completion");

    // Collect reasoning and text from the response
    let mut has_reasoning = false;
    let mut text_parts = String::new();

    for content in response.choice.iter() {
        match content {
            AssistantContent::Reasoning(reasoning) => {
                has_reasoning = true;
                eprintln!(
                    "  [nonstreaming] Reasoning block: id={:?}, content_count={}",
                    reasoning.id,
                    reasoning.content.len()
                );
            }
            AssistantContent::Text(text) => {
                text_parts.push_str(&text.text);
            }
            _ => {}
        }
    }

    assert!(
        has_reasoning,
        "Turn 1 non-streaming response has no Reasoning blocks in choice. \
         Provider was configured for reasoning but returned none. \
         Choice items: {:?}",
        response
            .choice
            .iter()
            .map(|c| match c {
                AssistantContent::Text(_) => "Text",
                AssistantContent::ToolCall(_) => "ToolCall",
                AssistantContent::Reasoning(_) => "Reasoning",
                _ => "Other",
            })
            .collect::<Vec<_>>()
    );

    assert!(
        !text_parts.is_empty(),
        "Turn 1 non-streaming response has no text output."
    );

    eprintln!(
        "  [nonstreaming] Turn 1: {} chars text, message_id={:?}",
        text_parts.len(),
        response.message_id
    );

    // Build assistant message with full choice (reasoning + text) for Turn 2 history
    let turn1_assistant = Message::Assistant {
        id: response.message_id,
        content: response.choice,
    };

    // === Turn 2 ===
    let turn2_prompt = Message::User {
        content: OneOrMany::one(UserContent::text(TURN2_TEXT)),
    };

    let request2 = completion::CompletionRequest {
        preamble: Some(agent.preamble.clone()),
        chat_history: OneOrMany::many(vec![turn1_prompt, turn1_assistant, turn2_prompt])
            .expect("non-empty"),
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: agent.additional_params.clone(),
        model: None,
        output_schema: None,
    };

    let response2 = agent
        .model
        .completion(request2)
        .await
        .expect("Turn 2 completion — provider may have rejected reasoning in chat history");

    let turn2_text: String = response2
        .choice
        .iter()
        .filter_map(|c| match c {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect();

    assert!(
        !turn2_text.is_empty(),
        "Turn 2 non-streaming response has no text. \
         Provider may have rejected the request with reasoning in chat history."
    );

    let trimmed = turn2_text.trim();
    assert!(
        trimmed.len() >= 20,
        "Turn 2 text suspiciously short ({} chars: {:?}). \
         Provider may not have processed the multi-turn context.",
        trimmed.len(),
        &trimmed[..trimmed.len().min(100)]
    );

    eprintln!("  [nonstreaming] Turn 2: {} chars text — PASSED", trimmed.len());
}

// ==================== Non-Streaming Provider Tests ====================

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn test_openai_reasoning_roundtrip_nonstreaming() {
    use rig::providers::openai;

    let client = openai::Client::from_env();
    run_reasoning_roundtrip_nonstreaming(TestAgent {
        model: client.completion_model("gpt-5.2"),
        preamble: PREAMBLE.into(),
        additional_params: Some(serde_json::json!({
            "reasoning": { "effort": "medium" }
        })),
    })
    .await;
}

#[tokio::test]
#[ignore = "requires XAI_API_KEY — validate with grok-4-0725 once key is available"]
async fn test_xai_reasoning_roundtrip_nonstreaming() {
    use rig::providers::xai;

    let client = xai::Client::from_env();
    run_reasoning_roundtrip_nonstreaming(TestAgent {
        model: client.completion_model(xai::GROK_3_MINI),
        preamble: PREAMBLE.into(),
        additional_params: None,
    })
    .await;
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn test_anthropic_reasoning_roundtrip_nonstreaming() {
    use rig::providers::anthropic;

    let client = anthropic::Client::from_env();
    run_reasoning_roundtrip_nonstreaming(TestAgent {
        model: client.completion_model("claude-sonnet-4-5-20250929"),
        preamble: PREAMBLE.into(),
        additional_params: Some(serde_json::json!({
            "thinking": { "type": "enabled", "budget_tokens": 2048 }
        })),
    })
    .await;
}

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn test_gemini_reasoning_roundtrip_nonstreaming() {
    use rig::providers::gemini;

    let client = gemini::Client::from_env();
    run_reasoning_roundtrip_nonstreaming(TestAgent {
        model: client.completion_model("gemini-2.5-flash"),
        preamble: PREAMBLE.into(),
        additional_params: Some(serde_json::json!({
            "generationConfig": {
                "thinkingConfig": { "thinkingBudget": 2048, "includeThoughts": true }
            }
        })),
    })
    .await;
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn test_openrouter_reasoning_roundtrip_nonstreaming() {
    use rig::providers::openrouter;

    let client = openrouter::Client::from_env();
    run_reasoning_roundtrip_nonstreaming(TestAgent {
        model: client.completion_model("openai/gpt-5.2"),
        preamble: PREAMBLE.into(),
        additional_params: Some(serde_json::json!({
            "reasoning": { "effort": "medium" },
            "include_reasoning": true
        })),
    })
    .await;
}

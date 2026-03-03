/// Smoke test: 2-turn streaming conversation with reasoning traces.
///
/// Tests that reasoning content from turn 1 survives roundtrip into turn 2's
/// chat history without the provider rejecting it.
///
/// Usage:
///   cargo run --example reasoning_roundtrip_test -- <provider>
///
/// Providers: anthropic, gemini, openai, openrouter
///
/// Required env vars per provider:
///   anthropic:   ANTHROPIC_API_KEY
///   gemini:      GEMINI_API_KEY
///   openai:      OPENAI_API_KEY
///   openrouter:  OPENROUTER_API_KEY
use std::env;

use futures::StreamExt;
use rig::OneOrMany;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{self, CompletionModel};
use rig::message::{AssistantContent, Message, Reasoning, ReasoningContent, UserContent};
use rig::streaming::StreamedAssistantContent;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let provider = env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: reasoning_roundtrip_test <anthropic|gemini|openai|openrouter>");
        std::process::exit(1);
    });

    match provider.as_str() {
        "anthropic" => {
            let agent = build_anthropic()?;
            run_test(agent).await
        }
        "gemini" => {
            let agent = build_gemini()?;
            run_test(agent).await
        }
        "openai" => {
            let agent = build_openai()?;
            run_test(agent).await
        }
        "openrouter" => {
            let agent = build_openrouter()?;
            run_test(agent).await
        }
        other => {
            eprintln!("Unknown provider: {other}");
            std::process::exit(1);
        }
    }
}

struct TestAgent<M: CompletionModel> {
    model: M,
    preamble: String,
    additional_params: Option<serde_json::Value>,
}

fn build_anthropic() -> anyhow::Result<
    TestAgent<rig::providers::anthropic::completion::CompletionModel<reqwest::Client>>,
> {
    use rig::providers::anthropic;
    let client = anthropic::Client::from_env();
    Ok(TestAgent {
        model: client.completion_model("claude-sonnet-4-5-20250929"),
        preamble: "You are a helpful math tutor. Be concise.".into(),
        additional_params: Some(serde_json::json!({
            "thinking": { "type": "enabled", "budget_tokens": 2048 }
        })),
    })
}

fn build_gemini()
-> anyhow::Result<TestAgent<rig::providers::gemini::completion::CompletionModel<reqwest::Client>>> {
    use rig::providers::gemini;
    let client = gemini::Client::from_env();
    Ok(TestAgent {
        model: client.completion_model("gemini-2.5-flash"),
        preamble: "You are a helpful math tutor. Be concise.".into(),
        additional_params: Some(serde_json::json!({
            "generationConfig": {
                "thinkingConfig": { "thinkingBudget": 2048, "includeThoughts": true }
            }
        })),
    })
}

fn build_openai() -> anyhow::Result<
    TestAgent<rig::providers::openai::responses_api::ResponsesCompletionModel<reqwest::Client>>,
> {
    use rig::providers::openai;
    let client = openai::Client::from_env();
    Ok(TestAgent {
        model: client.completion_model("gpt-5.2"),
        preamble: "You are a helpful math tutor. Be concise.".into(),
        additional_params: Some(serde_json::json!({
            "reasoning": { "effort": "medium" }
        })),
    })
}

fn build_openrouter()
-> anyhow::Result<TestAgent<rig::providers::openrouter::CompletionModel<reqwest::Client>>> {
    use rig::providers::openrouter;
    let client = openrouter::Client::from_env();
    Ok(TestAgent {
        model: client.completion_model("openai/gpt-5.2"),
        preamble: "You are a helpful math tutor. Be concise.".into(),
        additional_params: Some(serde_json::json!({
            "reasoning": { "effort": "medium" },
            "include_reasoning": true
        })),
    })
}

async fn run_test<M>(agent: TestAgent<M>) -> anyhow::Result<()>
where
    M: CompletionModel,
    M::StreamingResponse: Send,
{
    // === Turn 1 ===
    println!("=== TURN 1: Sending prompt with reasoning enabled ===\n");

    let turn1_prompt = Message::User {
        content: OneOrMany::one(UserContent::text(
            "A train leaves Station A at 60 km/h. Another train leaves Station B \
             (300 km away) 30 minutes later at 90 km/h heading toward Station A. \
             At what time do they meet, and how far from Station A? Show your work.",
        )),
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

    let mut stream = agent.model.stream(request).await?;

    let mut assistant_content: Vec<AssistantContent> = vec![];
    let mut reasoning_count = 0;
    let mut reasoning_delta_count = 0;
    let mut reasoning_delta_text = String::new();
    let mut text_chunks = 0;
    let mut streamed_text = String::new();

    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(StreamedAssistantContent::Text(text)) => {
                print!("{}", text.text);
                std::io::Write::flush(&mut std::io::stdout())?;
                streamed_text.push_str(&text.text);
                text_chunks += 1;
            }
            Ok(StreamedAssistantContent::Reasoning(reasoning)) => {
                reasoning_count += 1;
                let display = reasoning.display_text();
                if !display.is_empty() {
                    print!("\x1b[2m{display}\x1b[0m"); // dim for reasoning
                    std::io::Write::flush(&mut std::io::stdout())?;
                }
                assistant_content.push(AssistantContent::Reasoning(reasoning));
            }
            Ok(StreamedAssistantContent::ReasoningDelta { reasoning, .. }) => {
                reasoning_delta_count += 1;
                if !reasoning.is_empty() {
                    print!("\x1b[2m{reasoning}\x1b[0m");
                    std::io::Write::flush(&mut std::io::stdout())?;
                    reasoning_delta_text.push_str(&reasoning);
                }
            }
            Ok(StreamedAssistantContent::ToolCall { .. }) => {
                println!("[tool call]");
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("\nStream error: {e}");
                return Err(e.into());
            }
        }
    }
    println!();

    // If we only got deltas (no full Reasoning blocks), build one from accumulated text.
    // This happens with providers like Gemini 2.5 that emit thinking without signatures.
    if reasoning_count == 0 && !reasoning_delta_text.is_empty() {
        assistant_content.push(AssistantContent::Reasoning(Reasoning::new(
            &reasoning_delta_text,
        )));
        reasoning_count = 1; // treat assembled block as one
    }

    let total_reasoning_events = reasoning_count + reasoning_delta_count;
    println!("\n=== TURN 1 STATS ===");
    println!("  Reasoning blocks received: {reasoning_count}");
    println!("  Reasoning deltas received: {reasoning_delta_count}");
    println!("  Text chunks received: {text_chunks}");
    println!("  Streamed text length: {} chars", streamed_text.len());

    // Dump reasoning content details
    let mut total_reasoning_chars = 0usize;
    for (i, content) in assistant_content.iter().enumerate() {
        if let AssistantContent::Reasoning(r) = content {
            println!("  Reasoning block {i}:");
            println!("    id: {:?}", r.id);
            for (j, rc) in r.content.iter().enumerate() {
                let variant = match rc {
                    ReasoningContent::Text { text, signature } => {
                        total_reasoning_chars += text.len();
                        format!(
                            "Text({}chars, signature={})",
                            text.len(),
                            signature
                                .as_ref()
                                .map(|s| format!("{}chars", s.len()))
                                .unwrap_or("None".into())
                        )
                    }
                    ReasoningContent::Summary(s) => {
                        total_reasoning_chars += s.len();
                        format!("Summary({}chars)", s.len())
                    }
                    ReasoningContent::Encrypted(e) => {
                        total_reasoning_chars += e.len();
                        format!("Encrypted({}chars)", e.len())
                    }
                    ReasoningContent::Redacted { data } => {
                        total_reasoning_chars += data.len();
                        format!("Redacted({}chars)", data.len())
                    }
                    other => format!("{other:?}"),
                };
                println!("    content[{j}]: {variant}");
            }
        }
    }

    // --- STRICT ASSERTIONS ---

    // 1. Must have received reasoning (we explicitly configured it for every provider)
    if total_reasoning_events == 0 {
        anyhow::bail!(
            "FAIL: No reasoning content received (0 blocks, 0 deltas). \
             Provider was configured for reasoning but returned none."
        );
    }

    // 2. Reasoning must have non-trivial content
    if total_reasoning_chars < 10 {
        anyhow::bail!(
            "FAIL: Reasoning content too short ({total_reasoning_chars} chars). \
             Expected substantial reasoning for a multi-step math problem."
        );
    }

    // 3. Must have received text output
    if text_chunks == 0 || streamed_text.is_empty() {
        anyhow::bail!("FAIL: Turn 1 produced no text output.");
    }

    // Build chat history for turn 2 using actual streamed text
    assistant_content.push(AssistantContent::text(&streamed_text));

    // 4. Verify the assistant content we're about to send contains reasoning
    let reasoning_in_history = assistant_content
        .iter()
        .any(|c| matches!(c, AssistantContent::Reasoning(_)));
    assert!(
        reasoning_in_history,
        "BUG: assistant_content for Turn 2 history has no Reasoning items"
    );

    let reasoning_items_count = assistant_content
        .iter()
        .filter(|c| matches!(c, AssistantContent::Reasoning(_)))
        .count();
    println!("  Reasoning items in Turn 2 history: {reasoning_items_count}");

    // Use the provider-assigned message ID (e.g. OpenAI `msg_` ID) so the
    // Responses API can pair reasoning input items with their output item.
    let message_id = stream.message_id.clone();
    println!("  Message ID from stream: {message_id:?}");

    let turn1_assistant = Message::Assistant {
        id: message_id,
        content: OneOrMany::many(assistant_content).expect("non-empty"),
    };

    let turn2_prompt = Message::User {
        content: OneOrMany::one(UserContent::text(
            "Now suppose both trains slow down by 10 km/h after traveling half \
             the original distance. When do they meet now?",
        )),
    };

    // === Turn 2 ===
    println!("\n=== TURN 2: Sending follow-up (reasoning traces in chat history) ===\n");

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

    let mut stream2 = agent.model.stream(request2).await?;

    let mut turn2_reasoning = 0;
    let mut turn2_reasoning_deltas = 0;
    let mut turn2_text = 0;
    let mut turn2_streamed_text = String::new();

    while let Some(chunk) = stream2.next().await {
        match chunk {
            Ok(StreamedAssistantContent::Text(text)) => {
                print!("{}", text.text);
                std::io::Write::flush(&mut std::io::stdout())?;
                turn2_streamed_text.push_str(&text.text);
                turn2_text += 1;
            }
            Ok(StreamedAssistantContent::Reasoning(reasoning)) => {
                turn2_reasoning += 1;
                let display = reasoning.display_text();
                if !display.is_empty() {
                    print!("\x1b[2m{display}\x1b[0m");
                    std::io::Write::flush(&mut std::io::stdout())?;
                }
            }
            Ok(StreamedAssistantContent::ReasoningDelta { reasoning, .. }) => {
                turn2_reasoning_deltas += 1;
                if !reasoning.is_empty() {
                    print!("\x1b[2m{reasoning}\x1b[0m");
                    std::io::Write::flush(&mut std::io::stdout())?;
                }
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("\nTurn 2 stream error: {e}");
                return Err(e.into());
            }
        }
    }
    println!();

    let turn2_total_reasoning = turn2_reasoning + turn2_reasoning_deltas;
    println!("\n=== TURN 2 STATS ===");
    println!("  Reasoning blocks: {turn2_reasoning}");
    println!("  Reasoning deltas: {turn2_reasoning_deltas}");
    println!("  Text chunks: {turn2_text}");
    println!(
        "  Streamed text length: {} chars",
        turn2_streamed_text.len()
    );

    // Turn 2 must produce text output (provider accepted the request)
    if turn2_text == 0 || turn2_streamed_text.is_empty() {
        anyhow::bail!(
            "FAIL: Turn 2 produced no text output. \
             Provider may have rejected the request with reasoning in chat history."
        );
    }

    // Turn 2 text must be non-trivial (not just whitespace or a one-word response)
    let trimmed = turn2_streamed_text.trim();
    if trimmed.len() < 20 {
        anyhow::bail!(
            "FAIL: Turn 2 text suspiciously short ({} chars: {:?}). \
             Provider may not have processed the multi-turn context.",
            trimmed.len(),
            &trimmed[..trimmed.len().min(100)]
        );
    }

    println!("\n=== ROUNDTRIP TEST PASSED ===");
    println!(
        "  Turn 1: {} reasoning events, {} text chars",
        total_reasoning_events,
        streamed_text.len()
    );
    println!(
        "  Turn 2: {} reasoning events, {} text chars (accepted reasoning in history)",
        turn2_total_reasoning,
        turn2_streamed_text.len()
    );

    Ok(())
}

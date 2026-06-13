//! Remaining tool-pipeline behaviors: agents mounted as tools (sub-agent
//! delegation) and the non-streaming multimodal tool-result path.

use rig::client::CompletionClient;
use rig::completion::{Chat, Message};
use rig::message::{ToolResultContent, UserContent};
use rig::providers::gemini;

use super::super::support::with_gemini_cassette;
use super::super::tools_support::BadgeImageTool;
use crate::support::assert_nonempty_response;

#[tokio::test]
async fn agent_as_tool_invokes_subagent() {
    with_gemini_cassette("agent_tools_misc/agent_as_tool_invokes_subagent", |client| async move {
        let translator = client
            .agent(gemini::completion::GEMINI_2_5_FLASH)
            .name("french_translator")
            .description("Translates short English phrases into French.")
            .preamble("You translate the user's text into French. Reply with the translation only.")
            .temperature(0.0)
            .build();

        let coordinator = client
            .agent(gemini::completion::GEMINI_2_5_FLASH)
            .preamble(
                "You delegate every translation request to the french_translator tool, then report its output to the user.",
            )
            .temperature(0.0)
            .tool(translator)
            .default_max_turns(3)
            .build();

        let mut history = Vec::<Message>::new();
        let response = coordinator
            .chat("Translate 'good morning' into French.", &mut history)
            .await
            .expect("delegated translation should succeed");

        let called_subagent = history.iter().any(|message| {
            super::super::agent_run_support::assistant_tool_call_names(message)
                .iter()
                .any(|name| name == "french_translator")
        });
        assert!(
            called_subagent,
            "the sub-agent should be invoked under its agent name: {history:?}"
        );
        assert!(
            response.to_ascii_lowercase().contains("bonjour"),
            "the sub-agent's translation should reach the final answer: {response:?}"
        );
    })
    .await;
}

#[tokio::test]
async fn nonstreaming_tool_result_image_part_round_trips() {
    with_gemini_cassette(
        "agent_tools_misc/nonstreaming_tool_result_image_part_round_trips",
        |client| async move {
            // Multimodal function responses need a Gemini 3 model; 2.5-flash
            // rejects them with a 400.
            let agent = client
                .agent(gemini::completion::GEMINI_3_FLASH_PREVIEW)
                .preamble(
                    "You must call the fetch_badge_image tool, inspect the returned image, and describe its dominant color in one short sentence.",
                )
                .temperature(0.0)
                .tool(BadgeImageTool)
                .default_max_turns(2)
                .build();

            let mut history = Vec::<Message>::new();
            let response = agent
                .chat(
                    "Fetch the attendee badge image and tell me its dominant color.",
                    &mut history,
                )
                .await
                .expect("image tool prompt should succeed");

            let image_results = history
                .iter()
                .filter_map(|message| match message {
                    Message::User { content } => Some(content.iter()),
                    _ => None,
                })
                .flatten()
                .filter_map(|content| match content {
                    UserContent::ToolResult(tool_result) => Some(tool_result.content.iter()),
                    _ => None,
                })
                .flatten()
                .filter(|content| matches!(content, ToolResultContent::Image(_)))
                .count();

            assert_eq!(
                image_results, 1,
                "the top-level image JSON output should become an image tool-result part"
            );
            assert_nonempty_response(&response);
        },
    )
    .await;
}

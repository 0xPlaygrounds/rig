//! DeepSeek strict function calling, live: a schemars-derived tool with an
//! optional struct and an optional list reaches `/beta` in the schema strict
//! mode takes, beside a map tool strict mode cannot express, which goes
//! without it.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use rig::completion::Message;
use rig::providers::deepseek::DEEPSEEK_V4_FLASH;
use rig::tool::Tool;
use schemars::JsonSchema;
use serde::Deserialize;

use super::support::with_deepseek_cassette;

#[derive(Debug, Deserialize, JsonSchema)]
struct Author {
    name: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct FindBooksArgs {
    /// What the books are about.
    topic: String,
    /// Who wrote them, when the user names an author.
    author: Option<Author>,
    /// The most books to return.
    limit: Option<u32>,
    /// Words the books should mention, when the user names any.
    keywords: Option<Vec<String>>,
}

/// Records the arguments strict mode guarantees match the schema.
#[derive(Default)]
struct FindBooks {
    calls: Arc<Mutex<Vec<FindBooksArgs>>>,
}

impl Tool for FindBooks {
    const NAME: &'static str = "find_books";
    type Error = std::convert::Infallible;
    type Args = FindBooksArgs;
    type Output = String;

    fn description(&self) -> String {
        "Find books by topic and optionally by author.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(FindBooksArgs)).expect("a schema")
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.calls.lock().expect("unpoisoned").push(args);
        Ok("A Wizard of Earthsea (1968)".to_owned())
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct TagBooksArgs {
    /// A label for each book title.
    tags: HashMap<String, String>,
}

/// A map argument, which strict mode cannot express.
struct TagBooks;

impl Tool for TagBooks {
    const NAME: &'static str = "tag_books";
    type Error = std::convert::Infallible;
    type Args = TagBooksArgs;
    type Output = String;

    fn description(&self) -> String {
        "Attach labels to book titles.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(TagBooksArgs)).expect("a schema")
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(format!("tagged {} books", args.tags.len()))
    }
}

#[tokio::test]
async fn an_agent_calls_a_strict_tool_and_answers() {
    with_deepseek_cassette(
        "strict_tools/an_agent_calls_a_strict_tool_and_answers",
        |client| async move {
            let mut model = client.chat(DEEPSEEK_V4_FLASH);
            model.wire = model.wire.with_strict_tools();
            let finder = FindBooks::default();
            let calls = finder.calls.clone();
            let agent = rig::AgentBuilder::new(model)
                .preamble("Answer book questions by calling find_books first.")
                .max_tokens(1024)
                .tool(finder)
                .tool(TagBooks)
                .default_max_turns(3)
                .build();

            let reply = agent
                .chat(
                    "Find one book by Ursula K. Le Guin about wizards, then tell me its title.",
                    &mut Vec::<Message>::new(),
                )
                .await
                .expect("the strict turn completes");

            let calls = calls.lock().expect("unpoisoned");
            let call = calls.first().expect("find_books was called");
            assert!(!call.topic.is_empty());
            assert!(call.limit.is_none_or(|limit| limit >= 1), "{call:?}");
            assert!(
                call.keywords
                    .as_ref()
                    .is_none_or(|words| words.iter().all(|w| !w.is_empty())),
                "{call:?}"
            );
            assert!(
                call.author
                    .as_ref()
                    .is_some_and(|author| author.name.contains("Le Guin")),
                "the inlined optional struct carries the author: {call:?}"
            );
            assert!(reply.output().contains("Earthsea"), "{}", reply.output());
        },
    )
    .await;
}

/// The control: a wire without strict tools stays off `/beta` and answers.
#[tokio::test]
async fn a_plain_turn_still_answers() {
    with_deepseek_cassette(
        "strict_tools/a_plain_turn_still_answers",
        |client| async move {
            let agent = rig::AgentBuilder::new(client.chat(DEEPSEEK_V4_FLASH))
                .max_tokens(256)
                .build();
            let reply = agent
                .chat(
                    "Reply with the single word: harbor",
                    &mut Vec::<Message>::new(),
                )
                .await
                .expect("the plain turn completes");
            assert!(
                reply.output().to_lowercase().contains("harbor"),
                "{}",
                reply.output()
            );
        },
    )
    .await;
}

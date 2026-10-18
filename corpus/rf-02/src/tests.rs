use std::convert::Infallible;
use std::sync::Arc;

use rig_agent::tool::{Tool, ToolContext};
use rig_agent::{Agent, AgentBuilder};
use rig_core::message::{
    AssistantContent, AssistantMessage, DocumentSourceKind, Image, Message, Reasoning, Text,
    ToolCall, ToolFunction, ToolName, ToolResultContent, UserContent,
};
use rig_core::test_utils::{MockCompletionModel, MockTurn};

use super::*;

/// The value of a `Result`, or the test fails with its error.
macro_rules! ok {
    ($result:expr) => {
        match $result {
            Ok(value) => value,
            Err(error) => {
                assert_eq!(Some(error.to_string()), None);
                return;
            }
        }
    };
}

#[derive(Clone)]
struct LookupOrder;

impl Tool for LookupOrder {
    const NAME: &'static str = "lookup_order";
    type Args = Value;
    type Output = String;
    type Error = Infallible;

    fn description(&self) -> String {
        "Look an order up".to_owned()
    }

    fn parameters(&self) -> Value {
        serde_json::json!({"type": "object", "properties": {"order": {"type": "integer"}}})
    }

    async fn call(&self, _: &mut ToolContext, args: Value) -> Result<String, Infallible> {
        Ok(format!(
            "order {} shipped",
            args.get("order").unwrap_or(&Value::Null)
        ))
    }
}

fn agent(model: &MockCompletionModel, preamble: &str) -> Agent {
    AgentBuilder::new(model.clone())
        .preamble(preamble)
        .tool(LookupOrder)
        .default_max_turns(4)
        .build()
}

/// The texts of a request's messages, in order.
fn texts(model: &MockCompletionModel) -> Vec<String> {
    let requests = model.requests();
    let last = requests.last().map(|request| request.chat_history.clone());
    let texts = last.into_iter().flatten().map(|message| match &*message {
        Message::Assistant(turn) => turn.text(),
        message => message.rag_text().unwrap_or_default(),
    });
    texts.collect()
}

fn setup() -> (Arc<InMemoryConversations>, Arc<dyn Conversations>) {
    let repo = Arc::new(InMemoryConversations::default());
    let shared: Arc<dyn Conversations> = repo.clone();
    (repo, shared)
}

#[tokio::test]
async fn the_repository_is_the_only_copy() {
    let (repo, shared) = setup();
    repo.create("c1", "alice", "Orders", &["support"]);
    let helper = MockCompletionModel::from_turns([
        MockTurn::tool_call("call-1", "lookup_order", serde_json::json!({"order": 42})),
        MockTurn::text("Order 42 shipped."),
        MockTurn::text("Anything else?"),
    ]);
    let service = ChatService::new(shared.clone(), [("helper", agent(&helper, "Help."))]);
    let reply = ok!(service
        .send("alice", "c1", "Where is order 42?", "helper")
        .await);
    assert_eq!(reply, "Order 42 shipped.");

    // User, tool call, tool result, reply: all in the repository, and a
    // service built anew over it reads them back.
    let restarted = ChatService::new(shared, [("helper", agent(&helper, "Help."))]);
    let history = ok!(restarted.history("c1").await);
    let kinds: Vec<_> = history.iter().map(|(_, message)| kind(message)).collect();
    assert_eq!(kinds, ["user", "tool call", "tool result", "assistant"]);
    assert_eq!(service.history("c1").await.ok().as_ref(), Some(&history));

    // A redaction and a system note made through the repository reach the
    // next request.
    let first = history.first().map_or(0, |(id, _)| *id);
    ok!(repo.update("c1", first, ok!(encode(&Message::user("[redacted]")))));
    ok!(repo.append("c1", 4, vec![ok!(encode(&Message::system("Be brief.")))]));
    ok!(restarted.send("alice", "c1", "Thanks", "helper").await);
    let sent = texts(&helper);
    assert!(sent.iter().any(|text| text == "[redacted]"));
    assert!(!sent.iter().any(|text| text == "Where is order 42?"));
    assert_eq!(restarted.history("c1").await.map(|h| h.len()).ok(), Some(7));

    // Soft-deleted or someone else's: the app's error, no model call.
    let calls = helper.request_count();
    let denied = restarted.send("bob", "c1", "Hi", "helper").await;
    assert!(matches!(denied, Err(ChatError::Repo(RepoError::Forbidden))));
    ok!(repo.soft_delete("c1"));
    let deleted = restarted.send("alice", "c1", "Hi", "helper").await;
    assert!(matches!(deleted, Err(ChatError::Repo(RepoError::Deleted))));
    assert_eq!(helper.request_count(), calls);

    let row = ok!(repo.row("c1"));
    assert_eq!(
        (row.owner.as_str(), row.title.as_str()),
        ("alice", "Orders")
    );
    assert_eq!(row.tags, ["support"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 3)]
async fn agents_share_one_repository() {
    let (repo, shared) = setup();
    let helper = MockCompletionModel::from_turns(
        std::iter::repeat_with(|| MockTurn::text("from helper")).take(12),
    );
    let summarizer = MockCompletionModel::from_turns(
        std::iter::repeat_with(|| MockTurn::text("summary")).take(2),
    );
    let service = ChatService::new(
        shared,
        [
            ("helper", agent(&helper, "Help.")),
            ("summarizer", agent(&summarizer, "Summarize.")),
        ],
    );
    let ids: Vec<String> = (0..10).map(|i| format!("c{i}")).collect();
    for id in ids.iter().map(String::as_str).chain(["shared"]) {
        repo.create(id, "alice", id, &[]);
    }

    let distinct = futures::future::join_all(
        ids.iter()
            .map(|id| service.send("alice", id, "hello", "helper")),
    );
    let alternating = async {
        for agent in ["helper", "summarizer", "helper", "summarizer"] {
            service.send("alice", "shared", "next", agent).await?;
        }
        Ok::<_, ChatError>(())
    };
    let (sent, alternated) = futures::join!(distinct, alternating);
    ok!(alternated);
    for (id, reply) in ids.iter().zip(sent) {
        assert_eq!(reply.ok().as_deref(), Some("from helper"));
        assert_eq!(service.history(id).await.map(|h| h.len()).ok(), Some(2));
    }
    let shared = service.history("shared").await.map(|h| h.len()).ok();
    assert_eq!(shared, Some(8));
    let seen = texts(&summarizer);
    assert_eq!(seen.iter().filter(|text| *text == "from helper").count(), 2);
    assert_eq!(seen.iter().filter(|text| *text == "summary").count(), 1);
}

#[tokio::test]
async fn every_content_kind_round_trips() {
    let (repo, shared) = setup();
    repo.create("c", "alice", "Kinds", &[]);
    let call = ToolCall::from_wire(
        "call-7",
        ToolFunction::new(
            ok!(ToolName::new("lookup_order")),
            serde_json::json!({"order": 7, "filters": {"open": true, "tags": ["a", null]}}),
        ),
    );
    let image = Image {
        data: DocumentSourceKind::url("https://example.com/receipt.png"),
        ..Image::default()
    };
    let written = vec![
        Message::system("A note."),
        Message::User {
            content: vec![
                UserContent::Text(Text::new("See this")),
                UserContent::Image(image),
            ],
        },
        Message::Assistant(AssistantMessage::new(vec![
            AssistantContent::Reasoning(Reasoning::new("Look it up.")),
            AssistantContent::ToolCall(call.clone()),
        ])),
        Message::User {
            content: vec![UserContent::ToolResult(
                call.result(vec![ToolResultContent::Text(Text::new("shipped"))]),
            )],
        },
        Message::assistant("It shipped."),
    ];
    let bodies = ok!(written.iter().map(encode).collect::<Result<_, _>>());
    ok!(repo.append("c", 0, bodies));
    let service = ChatService::new(shared, std::iter::empty::<(String, Agent)>());
    let read: Vec<_> = ok!(service.history("c").await)
        .into_iter()
        .map(|(_, m)| m)
        .collect();
    assert_eq!(read, written);
}

fn kind(message: &Message) -> &'static str {
    match message {
        Message::System { .. } => "system",
        Message::User { content }
            if content
                .iter()
                .any(|part| matches!(part, UserContent::ToolResult(_))) =>
        {
            "tool result"
        }
        Message::User { .. } => "user",
        Message::Assistant(turn) if turn.tool_calls().next().is_some() => "tool call",
        Message::Assistant(_) => "assistant",
    }
}

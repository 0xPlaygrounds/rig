//! The classic agent runtime (rig-agent) driving a model, with no network:
//!
//! - `agent_stream`: an agent's streamed prompt answered by the recorded
//!   text reply with 2,000 deltas in 16 KB chunks that `wire::stream_serve`
//!   serves, per family: the gap between the two is what the runtime adds;
//! - `agent_parallel`: eight such prompts streaming at once on one agent
//!   (the Responses family);
//! - `agent_turns`: one turn over a history of 200 and 3,000 messages (the
//!   wire bench's coding session, with images): the model calls a tool, the
//!   tool answers, the model answers. A scripted model answers, so no wire
//!   encodes the requests and what is measured is the runtime's own work
//!   per turn.
//!
//! `cargo bench -p rig-cassette --bench agent --features http`.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::StreamExt as _;
use rig_agent::{Agent, AgentBuilder};
use rig_core::completion::{CompletionRequest, Message, Usage};
use rig_core::driver::{Exchange, Opened, Opening, Transport};
use rig_core::test_utils::{MockFrame, MockScript, MockStreamEvent};
use rig_core::tool::{Tool, ToolContext, ToolExecutionError, args_schema};
use serde::Deserialize;

#[allow(dead_code)]
mod common;
use common::{FAMILIES, Native, chunked, ensure, history, long_reply, or_exit, replaying};

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

/// An agent over `family`'s model answering every prompt with the
/// recorded text reply of 2,000 deltas, in 16 KB chunks.
fn streaming_agent(family: &str) -> Agent {
    let reply = chunked(&long_reply(family, 2_000), 16_384);
    AgentBuilder::new(replaying(family, reply)).build()
}

/// `agent`'s streamed answer to a question, drained: the items it streamed.
async fn stream(agent: &Agent) -> usize {
    let mut stream = agent.prompt("Tell me about Rust.").stream();
    let mut items = 0;
    while let Some(item) = stream.next().await {
        items += usize::from(item.is_ok());
    }
    items
}

/// A streamed prompt per family, drained.
#[divan::bench(args = FAMILIES, sample_count = 30)]
fn agent_stream(bencher: divan::Bencher, family: &str) {
    let agent = streaming_agent(family);
    let items = futures::executor::block_on(stream(&agent));
    ensure(items > 1_900, || format!("{family}: {items} items"));
    bencher.bench_local(|| futures::executor::block_on(stream(&agent)));
}

/// `RUNS` streamed prompts at once on one agent, all drained.
#[divan::bench(consts = [8], sample_count = 20)]
fn agent_parallel<const RUNS: usize>(bencher: divan::Bencher) {
    let agent = streaming_agent("responses");
    let all = || {
        let runs = (0..RUNS).map(|_| stream(&agent));
        futures::executor::block_on(futures::future::join_all(runs))
    };
    let items = all();
    ensure(items.iter().all(|items| *items > 1_900), || {
        format!("{items:?} items")
    });
    bencher.bench_local(all);
}

/// A scripted model's transport: it answers the calls in turn with
/// `replies`, round and round, and keeps nothing of the requests.
#[derive(Clone)]
struct Scripted {
    replies: Arc<[Vec<MockStreamEvent>]>,
    sent: Arc<AtomicUsize>,
}

impl Transport<MockScript> for Scripted {
    fn send(&self, _request: CompletionRequest, _exchange: Exchange) -> Opening<MockFrame> {
        let next = self.sent.fetch_add(1, Ordering::Relaxed) % self.replies.len().max(1);
        let reply = self.replies.get(next).cloned().unwrap_or_default();
        let frames = reply.into_iter().map(|event| Ok(MockFrame::Event(event)));
        Opening::ready(Opened::new(futures::stream::iter(frames)))
    }
}

#[derive(Deserialize, schemars::JsonSchema)]
struct EchoArgs {
    /// The text.
    text: String,
}

/// A tool that answers with its argument.
struct Echo;

impl Tool for Echo {
    const NAME: &'static str = "echo";
    type Args = EchoArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Echo the text.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<EchoArgs>()
    }

    async fn call(&self, _: &mut ToolContext, args: EchoArgs) -> Result<String, Self::Error> {
        Ok(args.text)
    }
}

/// One turn over a history of `MESSAGES` messages: the model calls `echo`,
/// it answers, and the model answers with text.
#[divan::bench(consts = [200, 3_000], sample_count = 20)]
fn agent_turns<const MESSAGES: usize>(bencher: divan::Bencher) {
    let usage = Usage::new().input_tokens(1_000).output_tokens(40);
    let call = serde_json::json!({ "text": "echo this with a little text" });
    let replies = [
        vec![
            MockStreamEvent::tool_call("call-echo", "echo", call),
            MockStreamEvent::final_response(usage),
        ],
        vec![
            MockStreamEvent::text("The tool said: echo this with a little text."),
            MockStreamEvent::final_response(usage),
        ],
    ];
    let transport = Scripted {
        replies: Arc::new(replies),
        sent: Arc::default(),
    };
    let model = rig_core::Model::new(MockScript::default(), transport).erase();
    let agent = AgentBuilder::new(model).tool(Echo).build();
    let history = history(MESSAGES / 4, &Native::of_none(), true);
    let turn = |history: Vec<Message>| {
        let question = "What does handle_1_1 return?";
        let run = agent.prompt(question).history(history).max_turns(2);
        futures::executor::block_on(async move { run.await })
    };
    let response = or_exit(turn(history.clone()), "the turn");
    ensure(response.messages.len() == 4, || {
        format!("{} messages in the turn", response.messages.len())
    });
    bencher
        .with_inputs(|| history.clone())
        .bench_local_values(turn);
}

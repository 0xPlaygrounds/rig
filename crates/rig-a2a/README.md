# rig-a2a

Serve a [Rig](https://crates.io/crates/rig) agent over the [Agent2Agent (A2A) protocol](https://a2a-protocol.org/latest/specification/), version 1.0, so any A2A client or agent can discover it and talk to it. The protocol comes from the official A2A Rust SDK ([`a2a-lf`](https://crates.io/crates/a2a-lf) and [`a2a-server-lf`](https://crates.io/crates/a2a-server-lf), from [a2aproject/a2a-rs](https://github.com/a2aproject/a2a-rs)); this crate turns an agent into an A2A agent executor and an Axum router.

```rust,ignore
use rig_a2a::A2aServer;
use rig_agent::AgentBuilder;
use rig_core::providers::openai::{self, OpenAI};

let model = OpenAI::from_env()?.completion(openai::GPT_5_4_MINI);
let agent = AgentBuilder::new(model).name("helper").build();

let app = A2aServer::new(agent, "http://localhost:3000").into_router();
let listener = tokio::net::TcpListener::bind("localhost:3000").await?;
rig_a2a::axum::serve(listener, app).await?;
```

The router serves:

- the agent card at `/.well-known/agent-card.json`;
- the JSON-RPC binding at `/jsonrpc` (`SendMessage`, `SendStreamingMessage`, `GetTask`, `ListTasks`, `CancelTask`, `SubscribeToTask`);
- the HTTP+JSON binding at `/rest` (`POST /message:send`, `POST /message:stream`, `GET /tasks/{id}`, `POST /tasks/{id}:cancel`, and the rest).

Streaming responses are server-sent events. Clients send the `A2A-Version: 1.0` header: the JSON-RPC binding reads a request without it as A2A 0.3 and rejects it, as the specification requires.

## Tasks and conversations

Each message starts a task. The agent's reply streams into one text artifact named `response`: text deltas are appended as they arrive, and the final answer replaces them, since a multi-turn run can stream text before its tool calls. The task then ends `completed`, `failed`, or `canceled`. A message with a part the agent cannot read ends `rejected`.

Messages that share a `contextId` share the agent's conversation history, including its tool calls and results. That history is kept in memory for the life of the server. Tasks live in the SDK's `InMemoryTaskStore` unless `A2aServer::task_store` names another `TaskStore`.

Parts become Rig user content: text and data parts as text, and file parts (raw bytes or URLs) as image, audio, video, or document content by their media type. The default card accepts `text/plain` and `application/json`; add the media types your model reads to `default_input_modes`.

## The agent card

The card takes the agent's name and description and declares one `chat` skill. Edit anything else before serving:

```rust,ignore
use rig_a2a::a2a::{AgentProvider, AgentSkill};

let server = A2aServer::new(agent, "https://agents.example.com").edit_card(|card| {
    card.version = "2.1.0".into();
    card.default_input_modes.push("image/png".into());
    card.provider = Some(AgentProvider {
        organization: "Example".into(),
        url: "https://example.com".into(),
    });
});
```

For push notifications, authentication, or gRPC, build the SDK's `DefaultRequestHandler` around `A2aExecutor::new(agent)` yourself.

## Example

`examples/a2a_server.rs` serves an OpenAI-backed agent at `http://localhost:3000`:

```sh
OPENAI_API_KEY=... cargo run -p rig-a2a --example a2a_server

curl -s http://localhost:3000/.well-known/agent-card.json
curl -sN http://localhost:3000/jsonrpc -H 'content-type: application/json' -H 'A2A-Version: 1.0' \
  -d '{"jsonrpc":"2.0","id":1,"method":"SendStreamingMessage","params":{"message":{"messageId":"m1","role":"ROLE_USER","parts":[{"text":"Hi!"}]}}}'
```

The `rig` facade re-exports this crate as `rig::a2a` behind its `a2a` feature.

Native-only: the server runs on tokio and Axum. Building for `wasm32` fails with a single explanatory `compile_error!`.

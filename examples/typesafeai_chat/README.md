# Interactive Jev + Rig chat

Run from the repository root in a shell that already exports `JEV_TOKEN`:

```sh
cargo run -p typesafeai_chat
```

Try: `I upgraded yesterday, but exports still say plan limit exceeded.`
Then add evidence in the next message. Jev sees the current input and the last
8 completed turns. Each call batches a typed route, an urgency rubric, and the
probability that clarification is needed. A generic `Assessment` struct implements `Query`, retaining the same fields for
questions and answers. `AssessmentQuery::new()` builds it once for reuse.
The default mode prints decisions and
a deterministic suggested action; Jev does not generate conversational prose.

For replies from an ordinary Rig agent, also export `OPENAI_API_KEY`, then run:

```sh
cargo run -p typesafeai_chat -- --agent
```

This calls Jev first, then uses its assessment to choose instructions for
`gpt-5.6-sol`. Your OpenAI endpoint must support that model and have available
credits. Failed requests print an error and leave conversation history unchanged.
Requests time out after 90 seconds. No secrets files are loaded by the example.

Use `/reset` to clear history, `/quit` or EOF to exit, and `--help` for usage
(which requires no credentials). The thresholds are illustrative application
policy; confidence measures distribution concentration, not correctness.

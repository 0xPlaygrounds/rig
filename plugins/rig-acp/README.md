# rig-acp

An optional native ACP front for the Rig coding agent. Enable it with:

```sh
rig plugin add rig_acp::AcpPlugin --crate rig-acp
rig acp
```

Configure an ACP client with executable `rig` and argument `acp`. The SDK
negotiates v2 when offered and falls back to v1. Stdout contains only ACP
frames; diagnostics use stderr. Sign in through Rig before starting the client.
The model and reasoning selectors use Rig's catalog and existing authentication.
Model discovery waits for saved preferences, lists reachable models, and puts the
selected model first. A selected model remains listed if its credentials become
unavailable.

Each open ACP session runs the same agent binary in a separate process, with
its own working directory, persisted Rig session, settings and MCP connections.
Session setup waits for every declared MCP server to connect and list its tools.
Stdio commands and streamable HTTP endpoints are supported, including supplied
environment variables and HTTP headers. Server tools use rig-rmcp and are named
`<server>__<tool>`. MCP credentials are passed through pipes and are not saved
in ACP session metadata. Session records and replay events live in `RIG_HOME/acp`.

V2 supports new, list, resume, close, prompt, cancel, updates and configuration.
V1 supports new, load, prompt, cancel and configuration. Resume requires the
original working directory and protocol version. Clients resupply MCP declarations
on resume. Tool inventories are discovered during setup; reopen the session
to pick up server tool-list changes. V2 prompt responses acknowledge acceptance; settled turns produce idle
state updates. V1 prompt responses wait for settlement. Blank prompts are rejected.
Slash commands run through Rig's command dispatcher; local commands settle without
a model call and refusals report their diagnostic. Pending tool calls precede
permission requests. V2 reports when approvals block foreground work.

Text and resource links are accepted. Images, audio, additional directories,
client filesystem/terminal execution, and general elicitation are not implemented.
Resource links are supplied to the model as references. Tool approval requests
are bridged to ACP permissions; other interactive questions stop the turn with
an explicit diagnostic. Existing Rig approval policy decides which tools ask.

The `scripted` example is a deterministic protocol-test agent. It requires no
provider credentials and is neither installed by the plugin nor included in the
published crate. Live client compatibility must be validated separately.

## Validation

```sh
CARGO_BUILD_JOBS=4 CARGO_INCREMENTAL=1 cargo nextest run --locked --profile local -p rig-acp
CARGO_BUILD_JOBS=4 CARGO_INCREMENTAL=1 cargo clippy --locked -p rig-acp --all-targets -- -D warnings
CARGO_BUILD_JOBS=4 CARGO_INCREMENTAL=1 cargo test --locked -p rig-acp --doc
CARGO_BUILD_JOBS=4 CARGO_INCREMENTAL=1 cargo build --locked -p rig-acp --example scripted
python3 plugins/rig-acp/tests/stdio.py target/debug/examples/scripted
```

The Python client needs permission to bind localhost for its HTTP MCP fixture.
It exercises both protocol versions, isolated working directories, replay after
reconnection, permission denial, cancellation of pending and executing tools,
HTTP authorization headers, blank input, recovery after a history-write failure,
local and refused commands, worker failure and resumption, EOF cleanup, and SIGTERM
with stdin open. The SDK negotiation tests run under nextest. A dedicated ACP CI
job runs these checks on Linux and macOS for every PR.

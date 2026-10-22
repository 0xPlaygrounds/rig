# Claude Cloak actual-agent acceptance

The default commands are offline and use only Python's standard library (Python
3.11 or newer, on Linux or macOS). They do not read authentication files, install programs, start a
server or contact a provider:

```sh
python3 -m unittest discover -s plugins/claude-cloak/tests/acceptance -v
python3 plugins/claude-cloak/tests/acceptance/run.py
python3 plugins/claude-cloak/tests/acceptance/run.py --analyze /path/to/private/evidence
```

Linux-only private-capture and live-preflight tests are skipped when Linux or `/dev/shm` is unavailable.
Offline analysis treats imported records as untrusted. It exports only validated
structural fields, including nested usage/fact/marker values; arbitrary headers,
bodies, identity strings and unknown fields are dropped. Evidence paths with
symlink components or symlink input/output files are rejected. Reports are
written using private temporary files and atomic replacement, not by following
an existing destination link.

An optional local fake-process smoke exercises two real fux/bubblewrap
launch/send/quit cycles and cleanup, using a temporary `/bin/sh` script and an
empty fake auth directory. It never launches Rig or contacts an API:

```sh
CLOAK_ACCEPTANCE_PROCESS_SMOKE=1 python3 -m unittest discover \
  -s plugins/claude-cloak/tests/acceptance -v
```

The smoke is Linux-only and skipped by default and when fux, bubblewrap, `/dev/shm` or user
namespaces are unavailable. Normal CI remains offline without process servers.

These tests verify the observer and driver, not Rig's inference implementation.
Actual agent-loop/mock-HTTP regressions belong to the Rust test suite. Mock usage
cannot prove server-side caching. No live execution of this maintained driver is
claimed by its offline tests.

## Authorized live execution

Obtain separate authorization for stored-credential use and charges before
running. Explain the selected scenarios and all three caps first. The live mode
requires Linux, bubblewrap (`bwrap`), `mitmdump` with its Python package, writable
`/dev/shm`, and a verified built Rig containing Claude Cloak. The TUI matrix also
requires `fux`. Nothing is installed automatically. Inspect the recorded source
revision and plugin manifest, and independently establish that the supplied
binary was built from them. A binary hash alone does not establish that binding.

The small mode launches two actual Rig print sessions with `--tools=` and the
same synthetic prefix. It measures fresh-prefix and repeated-prefix behavior;
it is not a multi-turn conversation test. The optional `--matrix` runs the
actual TUI on a dedicated, randomly named fux server:

1. Seed synthetic facts, then recall without restating values.
2. Read `facts.txt` successfully, then read a nonexistent, randomly named
   `probe-<random>.txt` without retry. The neutral name does not reveal its outcome.
3. Switch effort low to medium and recall; return to low.
4. Give a neutral answer, manually compact, recall and rewarm. Recall asks for
   the second read's filename, outcome and attempt count without providing those
   answers. Structural booleans must confirm the exact filename and one failure.
5. Optionally `--fork-resume`: fork, recall, quit, resume the same isolated
   session and recall. `/fork` creates an agent, not a separate session directory.
6. Start a separate session with the reduced trigger, seed, append a large newest
   message, observe a real compaction effect and final answer, recall old/new
   facts and rewarm.
7. Restart with an empty tool roster and repeat the same prefix in two fresh
   sessions.

A future authorized invocation has this form. Replace every uppercase argument;
this is not an instruction to use existing credentials now:

```sh
python3 plugins/claude-cloak/tests/acceptance/run.py \
  --live --confirm-stored-credentials --matrix --fork-resume \
  --binary /absolute/path/to/verified/rig \
  --source /absolute/path/to/verified/source \
  --plugin-config /absolute/path/to/build/plugins.toml \
  --credentials /absolute/path/to/original/anthropic-oauth \
  --build-record /private/build-record.json \
  --ledger /private/live-ledger.json \
  --output /absolute/path/outside/repository/NEW-private-evidence \
  --requests APPROVED_REQUEST_CAP \
  --output-tokens APPROVED_PER_REQUEST_OUTPUT_CAP \
  --seconds APPROVED_ELAPSED_SECONDS
```

The request cap is 2..24 Messages attempts at the proxy boundary, including
retries; elapsed time is 1..1800 seconds. The output cap is 1..128000 **advertised
maximum tokens per request**, not a promise of small actual output. The prior
Opus profile advertised 128000 for ordinary calls. A lower approved cap rejects
such a request locally instead of silently rewriting it. Requests ask for brief
answers, but the authorization must account for the worst-case cap. The full
matrix normally needs about 20 calls; retries or extra tools can exhaust its cap.
The driver stops at the first failed/unknown phase. There is no dollar estimate
or invoice assertion. Authentication refresh calls are not inference requests
and are not counted against that cap.

The proxy does not rewrite model inputs, cache controls or output budgets. It
narrows `Accept-Encoding` to `gzip, deflate, identity`, allowing the observer to
decode negotiated compression. It blocks wrong models, unexpected tools, excess
output limits and excess requests. Credential-like `sk-ant-` tokens in decoded
model JSON, including tool results and JSON-escaped text, are rejected locally
before forwarding with `credential_like_input`. Normal authorization headers
remain allowed. Matched values are never logged. This narrow guard is not a
general secret detector or a substitute for filesystem isolation. A model request bypassing the configured
proxy would not be counted: this is a child-scoped audit, not a network firewall.
Use only the verified transport that honors proxy/trust settings. No observed
calls is **unknown**, never a successful acceptance run.

## Isolation and privacy

The runner supplies an allowlisted child environment, isolated HOME, XDG
locations, RIG_HOME, workspace, settings and sessions. It never changes the
active user's model, plugins, configuration or conversations. Its generated
workspace contains only synthetic facts. The original `anthropic-oauth`
directory is bind-mounted into the child so normal atomic credential refresh
uses one store; refresh grants are never copied. The proxy CA and proxy variables
apply only to children. TLS verification is not disabled. No system trust or
proxy settings are changed. No existing fux server or pane is inspected. Every
fux start/controller/cleanup call uses the same mode-0700 `scratch/runtime` as
`XDG_RUNTIME_DIR`, keeping sockets and terminal journals in private tmpfs. The
server is started explicitly, and every launch has a unique tab name even when
previous exited panes still exist.

Only structural allowlisted summaries are exported. Neither authorization nor
cookies, raw headers, user identifiers, tool inputs, arbitrary error strings,
response text, thinking or signatures are persisted in evidence. Exact synthetic
input text is recorded separately as provenance. Response content is reduced to
synthetic-fact booleans. Provider request/message IDs, session identity, device
identity and the pinned human-fingerprint suffix are HMAC pseudonyms. The
private `.pseudonym-key` is created once with mode 0600 and reused on addon
reload; keep it private and do not include it in public artifacts.

Ordinary Rig session/effect logging still exists inside the child's private
`/dev/shm` scratch, including internal/signed content. It is not exported. The
runner imports only structural effect outcomes, then removes its owned scratch.
A force-kill or machine failure can leave private scratch behind; inspect and
remove only that run's directory. This does not claim raw Rig session storage is
redacted. Bubblewrap makes the host root read-only except for owned scratch and
the authorized credential bind; it is not a secrecy sandbox against an arbitrary
malicious binary. Supply a verified binary, not untrusted code. Proxy stdout,
stderr, flows and terminal screenshots are not saved. The only durable outputs
are the private sanitized evidence directory.

## Evidence format and assertions

`audit.jsonl` contains `request`, `headers`, `response` and `effect` records:

- `call`: monotonically increasing capture number, restored after addon reload.
- `scenario`: allowlisted driver phase, not model-supplied text.
- `requested_model` / `served_model`: expected model or `unexpected`, with exact
  equality checks against `claude-opus-5-5`.
- `markers`: safe indexed protocol path, ephemeral-type validity and 5m/1h TTL.
  Only actual tools/system/message-content/root marker slots count, in protocol
  order. Null controls and schema/input/nested-result data keys do not count.
  Full block contents are deliberately absent.
- `request_id` and `message_id`: stable pseudonyms of actual transport IDs. An
  effect joins by its actual provider request ID, never by timing or list order.
  Joined effect ID, purpose and usage are retained, not just a correlation flag.
- `human_fingerprint`: pseudonym of the pinned billing suffix, when present.
  Comparing it does not validate a billing checksum or current Claude Code.
- `usage`: numeric input/output/cache-read/cache-creation counters, including
  separate five-minute and one-hour creation buckets when returned.
- `terminal`, `final_answer`, `close`: a terminal model result followed by
  cancellation is distinguished from a response lacking its terminal. A normal
  EOF with an incomplete compression trailer or unfinished SSE frame fails the
  observer-integrity check, even after a terminal. Explicit terminal-plus-CANCEL
  retains its separate completed classification. Tool-use
  terminals are not final answers. `max_tokens` stops do not pass.
- `parse_error`, `compression_complete`, `pending_frame`: decoding evidence.
  Unsupported observer compression produces capture **unknown**, not a provider
  failure. It is not silently treated as uncompressed bytes. Gzip and
  RFC zlib-wrapped deflate are supported; raw nonstandard deflate is not.
- `stop_reason`: the exact allowlisted protocol reason, including `refusal` and
  `max_tokens`; an unrecognized value becomes `other` without exporting it.
- `local_guard`: a local rejection, never represented as a provider rejection.
- `facts`: exact synthetic-value matches in the response, without exporting it.
- `second_read`: exact randomized filename and failure-with-one-attempt evidence
  from the formatted recall response. No arbitrary response text is exported.

`audit.py` accepts ordinary effect records and delta envelopes' `record` field.
It reads outcomes only; it does not reconstruct delta-compressed input history
or implement a second production Anthropic decoder. Summary effects must have
purpose `compaction`. The TUI driver waits for a final wire response and its
matching settled effect before advancing. It checks positive cache reads on
specified warm phases, not fixed production counts or a completely cold first
call. It checks tool-result success/error evidence, effort, fact retention,
smaller post-manual-compaction input, summary presence during auto-trigger, and
an exact full synthetic newest-message substring in the post-summary request.
Optional fork/resume checks preserve fingerprint provenance and expected session
identity continuity. These are empirical acceptance assertions, not proof of
retention for arbitrary messages or a provider context limit.

`provenance.json` records commit, dirty-tree status, binary and plugin-manifest
hashes, model, settings, caps and synthetic inputs. `matrix-inputs.json` records
matrix prompts. `matrix.json` records phase pass/fail/unknown. `summary.json` is
the machine report; `REPORT.md` is the concise human report. Missing phases, responses, usage, identity or compaction evidence stay unknown.
Two absent identity values never establish identity continuity. Failures are retained rather than retried until
a green result appears. If a transient failure warrants another authorized run,
use a new evidence directory and keep both reports.

## Boundaries and remaining coverage

The reduced policy uses `reserve = 970000`, `summary_tokens = 1024`,
`keep_recent = 1024`, `keep_asked = 0`. It tests a local roughly 30k trigger, not
million-token server overflow. The live driver deliberately does not trigger
summary failure, deliberate token-refresh cancellation, malformed provider
output, TTL expiration, or expensive true context exhaustion. Those require
mock-HTTP regressions or separately approved expensive/long-running experiments.

The bounded matrix's process commands and sequencing are covered offline. When
fux is installed, its native `check` validates command grammar without executing
actions or starting a server. Timeout options follow their pane/options stack
receiver; status 124 is accepted busy state, with completion still bounded by
the wire/effect observer deadline. Status 3 is accepted only for an expected
`/quit`. The fake-process smoke also checks that the bubblewrap `/dev` mount
precedes rebinding `/dev/shm` scratch, so the workspace remains reachable. Live
fux interaction and provider behavior must still be exercised with explicit
approval. If a build changes command names, event purpose, output budget or
profile fingerprint format, the driver should fail/mark unknown instead of
inventing equivalence. The printed phase plan also serves as a manual checklist
for a dedicated fux tab if diagnosing an interactive failure. Do not switch to
a user's existing tab, weaken an assertion or add unbounded sleeps to make a
phase appear successful.

## Matched Rig and Claude Code runs

The maintained driver supports Claude Code's TUI with `--clients claude`, or
Rig alone with `--clients rig`. Separate runs share `--ledger` and use the same
synthetic prompts and workspace contents. `--compare` runs both, Claude Code
first, within one run's cap. Its default `--scope probes` runs seed, warm recall,
successful read and failed-read continuation. `--scope full --fork-resume` adds
the lifecycle matrix. Each client stops at its first failed or unknown phase
by default. The explicit Rig read-refusal continuation described below can
observe later independent phases while retaining that failure.
A planned second client is still observed after an upstream refusal; permanent
local guards stop the experiment. Keep the failed evidence.

Claude Code uses `--safe-mode`, empty setting sources and MCP configuration,
`--strict-mcp-config`, read-only tools, `--permission-mode dontAsk`, explicit
Opus 5.5/low effort and disabled suggestions. `--name "synthetic parity"`
suppresses automatic title generation. The child also sets
`CLAUDE_CODE_DISABLE_TERMINAL_TITLE=1` and
`CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`. These controls deliberately exclude
auxiliary title traffic from the primary comparison. The observer still counts
and blocks any such requests. `claude-haiku-5-5` is the explicit known title
model; additional known auxiliary IDs require `--blocked-auxiliary-model ID`.
Auxiliary calls are never forwarded. Unexpected models or tools stop the run.
Requested fallback models are also checked: only an explicit model-array
shape naming the approved primary model can pass. Default routing, malformed
fallback shapes and any other model are rejected locally as `fallback_model`.
Safe profiles record only known public fallback IDs or `unexpected`. Earlier
observer versions checked the primary request and served model only; a native
request could therefore advertise a different fallback without triggering
that guard. Historical captures retain that limitation, even when their
actual served model was the approved primary.
The Claude Code roster is exactly `Read` and `DeferredToolPlaceholder`; Rig's
is exactly `read`/`_read`. Permanent guards return HTTP 400
`invalid_request_error` with `x-should-retry: false`, rather than retryable 429s.
Expected auxiliary blocks have classification `blocked_auxiliary` and cannot
establish a passing primary scenario.

The Claude Code child reads its existing access token into memory and writes
minimal identity/onboarding configuration only into private tmpfs. It never
copies a refresh grant or writes the original configuration. The token must
remain valid through the bounded deadline plus one minute, otherwise startup
fails before inference. Refresh the original client store normally first if
needed. Supply its original home with `--claude-auth-home` and the exact isolated
CLI binary with `--claude-binary`. Offline preflight records its version, binary
and help hashes and verifies the required isolation/session flags.

`--claude-auth stored` is the default. For an account-controlled comparison,
`--claude-auth rig-access-token` loads the exact existing Rig access token from
the supplied `--credentials` directory inside the child. It requires validity
through the deadline plus one minute. The private Claude Code configuration
uses that store's account and optional organization UUID, retaining only its
own installation identity from the original Claude Code configuration. Other
account details and email are omitted. No refresh grant is copied and neither
host credential store is modified. Provenance records the selected auth mode
and booleans comparing the stored and selected accounts with Rig; no account
identifiers or tokens are exported. Different account entitlements remain a
confound until a comparison uses the same access token.

Rig's reduced automatic-compaction threshold is specific to Rig. Claude Code's
matching large-append session uses explicit `/compact` after the append, then
recalls old/new facts and rewarms. Its comparison report says
`manual-after-append`; it does not claim to exercise Claude Code's automatic
threshold. Claude Code forks using its supported `--resume --fork-session` TUI
launch, discovers the new private persisted session ID and resumes that fork.
No raw session IDs are exported.

Each read phase reconstructs the streamed tool input in memory and requires
exactly one invocation of the client's read tool on the expected synthetic
file. Absolute Claude Code paths must resolve inside the owned workspace.
Only invocation counts and target-match booleans persist. Historical tool
results cannot satisfy a new phase. Rig retains its seed fingerprint through
manual compaction, fork and resume. Claude Code can change its human-message
fingerprint after compaction; the observer records that transition, then
requires continuity through recall, rewarm, fork and resume.

Record the successful build after source changes finish:

```sh
python3 plugins/claude-cloak/tests/acceptance/run.py \
  --record-build /private/build-record.json \
  --binary /absolute/path/to/rig-agent --source /absolute/path/to/rig \
  --plugin-config /private/build-home/plugins.toml \
  --generated-main /private/build-home/project/src/main.rs \
  --build-command 'THE EXACT SUCCESSFUL BUILD COMMAND'
```

Live preflight checks the head, tracked diff, untracked file names/content hashes,
manifest, generated main and binary
against that record. It checks enabled registrations for Claude Cloak, TUI,
print/tool limits, read, sessions/agents, compaction, basic commands and effect
logging. It also runs the actual binary with `RIG_DESCRIBE` in an isolated home
without credentials, rejecting stage conflicts before inference. The recorded
command is a build provenance assertion, not a compiler attestation.

Add `--clients claude --scope probes --claude-binary PATH --claude-auth-home HOME`
to the live example for a Claude Code run. Run Rig separately with
`--clients rig --match-inputs /private/claude-run` and the same ledger. This
reuses the exact randomized missing filename. An explicit `--missing-file
probe-<16 hex digits>.txt` can also pin that synthetic input. Full matrices
generally need separate runs.
The locked ledger counts every Messages attempt, including auxiliary and local
blocks, across evidence directories and addon reloads. The default limit is
48 aggregate attempts. Once the user explicitly authorizes an increase, pass
`--aggregate-requests 96` or `--aggregate-requests 200` on each subsequent run
to use that explicitly authorized ceiling on the same ledger. The ledger
records the approved ceiling and each run's chosen
ceiling; counts, elapsed time and earlier run policies are preserved. Legacy
runs retain their 48-attempt ceiling. Provenance and policy record the selected
ceiling. Per-run limits remain 24 attempts, 128000 advertised output tokens
per request and 1800 aggregate live seconds. The last permitted attempt
is rejected locally as a reserve to stop before another turn. An unfinished
run remains active in the ledger for inspection. Never reset consumed budgets.

Before a model run, use `--matrix --clients claude --startup-only` with the
same live options to validate actual private TUI launch and prompt readiness.
This sends only `/quit`. The proxy rejects every Messages request with a local
nonretryable error during this mode. A passing startup report requires ready
clients and zero Messages attempts; it proves no model behavior. Startup
diagnostics contain fixed event codes only. Raw screens and exception text
are never exported. Short private server names keep nested Unix socket paths
within the platform limit.

The separate `--root-ttl-probe /absolute/compiled/claude_cloak_test_binary`
mode runs the ignored test
`transport::live::tests::real_root_ttl_regression_and_control` through this
same proxy and ledger. Use a non-matrix run with `--requests 3 --seconds 90`:
two Messages can be forwarded before the reserved final slot. Offline
preflight verifies that exact test exists. The test checks Long and Short
retention with explicit raw root 5m caching, and requires complete READY
answers. Its stdout/stderr are discarded; the report requires exit zero and
two observed successful end-turn READY responses. The compiled test binary's
hash is recorded alongside the normal verified Rig build.

The separate `--tool-result-probe /absolute/compiled/claude_cloak_test_binary
--captured-request /dev/shm/PRIVATE/rig-6.json` mode runs
`transport::live::tests::tool_result::real_tool_result_representation_probe`.
The captured original must remain an owned private file in private tmpfs
directories, with no symlinks. It is mounted read-only into the child and
never copied into the evidence directory. Use `--requests 3 --seconds 90
--output-tokens 128000`. The Rust probe permits exactly two requests: the
captured baseline and a variant changing only the failed text tool result's
array/string representation. `--probe-variant tool_result_string` is that
default. `--probe-variant thinking_binding_absent` instead removes only
`thinking.block_binding` from the same captured baseline; the field must be
present. Both variants retain all signed history, `is_error`, cache markers,
output limits and beta headers. The Rust probe independently checks the final
request bytes. Its safe detail report records the selected closed-enum
variant and `variant_only_declared_field_and_cch` control. Older string-only
control records are interpreted only as that string variant.
`--probe-variant final_error_cache_absent` instead removes only the final
failed tool result's cache marker. Another cache marker must remain. History,
tool-result contents, `is_error`, all other markers and their TTLs, output
limit and beta headers stay unchanged; caching remains enabled.
`--probe-variant request_class_main` instead adds only the documented
`x-claude-code-request-class: main` header. Body bytes and checksum remain
identical. Its report requires `body_equal` and
`headers_only_declared_change` to be true, with `headers_equal` false;
body-changing variants still require equal headers.
`--probe-variant previous_request_id --probe-metadata /dev/shm/PRIVATE/metadata.json`
adds only `cc_prev_req` with the captured Rig history's actual preceding provider
request ID to the billing block, then recomputes its checksum. Metadata must be
an owned private tmpfs file containing exactly `{"previous_request_id":"req_..."}`.
The ID must come from that Rig history, not from another client. The metadata
file is mounted read-only into private child scratch; its contents and path
are not exported into provenance or reports. This variant requires equal
headers and preserves all other body fields.
`--probe-variant neutral_default_identity` changes exactly one system-block
prefix from `You are rig, a coding agent` to `You are a coding agent` and
recomputes the checksum. The protocol identity and every functional instruction,
message, tool, cache marker and header remain unchanged. Missing or duplicate
prefixes are rejected. This variant takes no metadata file.
`--probe-order baseline-first` is the default. Explicit `--probe-order variant-first`
reverses only the send order after both requests have passed their byte and
header guards. Reports record the order and label each observed outcome as
`baseline` or `variant`; a reversed pair does not retry either condition.
This permits a counterbalanced diagnostic without changing the request pair.
The proxy retains normal model, read-tool, output and budget guards.
`tool-result-probe.json` records only exit status and timeout. A complete
diagnostic exits zero but its overall classification remains unknown; refusal
calls remain failed observations. This mode cannot establish matrix acceptance
or turn an earlier failed run into a passing run.

`profiles.jsonl` records allowlisted application headers/betas, body keys,
system/message structure, tool schema hashes, thinking/display controls,
diagnostics presence and marker TTL/scope topology. Literal text, schemas,
tool-result contents and signatures are not exported. An optional
`--checksum-verifier PATH` module exposes `verify(original_decoded_bytes)` and
returns `True`, `False` or `None`; only match/mismatch/unknown is saved. It sees
the exact decoded bytes before JSON reserialization. No verifier means unknown.

Responses record whether structured stop details and an explanation were
present, plus a closed refusal type/category. The official native schema names
`cyber` and `bio` categories while leaving the category string open; other
categories become `other`, and absent categories become `absent`. Arbitrary
category strings and explanation text are never exported.

Native correlation headers use a separate closed roster: session, prompt,
agent/parent-agent, request IDs and class, agent type, previous tool durations,
compaction indicators, signature, additional protection and wiggly-dove
metadata. The observer exports presence and private pseudonyms, plus verified
closed request-class roles and `teammate`/`custom` agent roles. Other role
values remain opaque. It records whether a present session header matches the
body's metadata session. Unknown header names and values are never exported;
only their distinct-name count remains after excluding known HTTP/auth/cookie
headers. Profile comparison checks presence and roles because separate runs
have different pseudonym keys. Tool-result profiles distinguish an absent
`is_error` field from explicit false and true values.

For private checksum investigation, `--raw-capture --pause-after-seed` retains
original decoded request bodies only in owned `/dev/shm` scratch. After each
client's seed, `paused.json` names that private directory. Inspect the originals,
then create an empty `continue-claude` or `continue-rig` file in the evidence
directory. The same deadline applies to the pause. Cleanup deletes raw scratch.
Never export these files or verify checksums from redacted or rebuilt JSON.

Use `--raw-capture --pause-after-phase read_error` to inspect the matched read
continuation before cleanup, whether that phase passes or fails.
`--pause-on-failure` pauses at the first failed or unknown phase. Both require
`--matrix`; `--pause-after-seed` remains available for seed inspection.
`paused.json` records the phase, unchanged outcome, private raw directory and
waiting/released state. Create `continue-CLIENT` to release the pause. The
original failure then stops the client as usual. Inspection uses the same
live deadline and never authorizes another request.

`--scope full --continue-after-read-refusal` is a separate explicit option for
Rig. It permits later independent phases after a complete HTTP 200 refusal
following exactly one read of the expected missing file and exactly one new
error result. Both read-phase responses must have complete SSE framing,
expected model/tool roster and valid cache markers, and both must correlate
with actual Rig effects. The TUI must return to input readiness. Local guards,
HTTP failures, truncated streams, unexpected models/tools and other failed
phases still stop the client. The next prompt defaults to the different effort
recall prompt. Explicit `--read-refusal-next manual_compact` instead advances
directly to the existing compaction command, leaving effort and neutral checks
unknown. Recall still requires all facts and the genuine failed-read outcome,
and its input must shrink relative to the refused read request.
`continuations.jsonl` records the reason and next phase without
text; `read_error` and the overall report remain failed. There is no retry of
the refused prompt and later assertions are unchanged.

Rig-only `--scope auto-controls` runs the existing isolated automatic-compaction
session and two fresh zero-tool controls. It uses the same seed, large append,
reduced trigger settings and assertions as the full matrix. It does not add
context or retry the failed conversation. Manual/error lifecycle, effort,
fork and resume checks remain unknown in this separate report; earlier failed
full reports are unchanged. Successful isolated controls therefore do not make
the overall full matrix pass. This scope excludes `--fork-resume`,
`--startup-only` and refusal continuation.

Rig-only `--scope full --collect-failures` explicitly collects failures across
the fixed full matrix. After a fully observed HTTP 200 refusal or normal
end-turn semantic assertion failure, it may advance to the next distinct
phase once actual effects settle and the TUI is ready. It requires complete
SSE framing, expected primary and fallback models, valid cache markers and
the expected tool dispatch. No-tool phases require zero tool invocations;
read phases reject wrong targets and repeated calls. A read refused before
dispatch can be collected as a failure, without inventing its outcome for
later retention checks. Those checks also require evidence of the actual
missing-file read and its single error result. A cancellation after a fully
validated terminal SSE frame may retain the completed response; cancellation
with missing framing or pending data cannot. HTTP/transport errors, local
guards, unknown capture,
incomplete responses and tool-dispatch violations still stop the run.
`failure-continuations.jsonl` records only the known phases, reason and proof
booleans. Each failed check and the overall failure remain visible even if
later phases pass. This mode injects no history and retries no phase. It
cannot be combined with startup-only or the narrower refusal continuations.

Compare separate runs offline:

```sh
python3 plugins/claude-cloak/tests/acceptance/run.py \
  --compare-reports /private/rig-run /private/claude-run \
  --output /private/NEW-comparison
```

`comparison.json` pairs primary requests by phase and prompt/tool-continuation
role, then position within that role. It reports structural equality booleans
and each verifier's result. Optional observations missing from either capture
are `null` (unknown), including fields added by newer observer versions.
Historical captures are not backfilled. Application prompts, tools, summaries and responses
can explain unequal structure. Those differences alone establish neither a
failure nor byte equality. Claude Code completion is wire evidence; only Rig
responses can join to settled Rig effects.

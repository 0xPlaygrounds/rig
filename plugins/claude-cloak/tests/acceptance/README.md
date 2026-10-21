# Claude Cloak actual-agent acceptance

The default commands are offline and use only Python's standard library (Python
3.10 or newer). They do not read authentication files, install programs, start a
server or contact a provider:

```sh
python3 -m unittest discover -s plugins/claude-cloak/tests/acceptance -v
python3 plugins/claude-cloak/tests/acceptance/run.py
python3 plugins/claude-cloak/tests/acceptance/run.py --analyze /path/to/private/evidence
```

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

The smoke is skipped by default and when fux, bubblewrap, `/dev/shm` or user
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

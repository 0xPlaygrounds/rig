# Fix the review findings on the stream-part-lifecycle PR (#2262, same branch)

Repo: rig workspace, branch `refactor/stream-part-lifecycle` (open PR #2262, base `main`,
head `cc46752ae`). Independent review confirmed six defects plus four items needing an
explicit decision. Fix them all on this branch, one commit per fix (same discipline as
the existing C→B→A commits). Cross-check against the two review documents if present:
`../../pr-2262-review-findings.md` and `../../pr-2262-solutions.md` (relative to repo
root, in `many_rigs/`).

## Context you need

The PR gave stream parts an explicit lifecycle: `RawStreamingChoice` speaks
`{Text, Reasoning, ToolInput} × {Start, Delta, End}`; `PartsAccumulator`
(`crates/rig-core/src/streaming/parts.rs`) reduces open-maps into an arrival-ordered
part list with entity-owned idempotence (finished-sets). Accumulation keys are opaque
`StreamPartId` values (`Wire`/`Minted{kind,index}`/`Composite`); durable provider
handles travel separately as `WireId` (whose only constructor rejects empty strings).
`ToolResult` gained `name: Option<String>` (the *executed* tool's name), populated by
the drivers on every construction route; `resolve_tool_result_names`
(`crates/rig-core/src/providers/internal/mod.rs`) is the back-compat shim for
`name: None` legacy histories, called by name-requiring serializers before building
requests.

Verification gates that must stay green after every fix: `cargo nextest run
--all-features` on default members; `cargo clippy --workspace --all-targets
--all-features -D warnings`; `cargo fmt --check`; `RUSTDOCFLAGS="-D warnings" cargo
doc`. Cassette replays must stay byte-identical except where a fix deliberately changes
a request (name it in the commit message, re-record honestly). Follow the record-first
doctrine in `tests/README.md` for any new cassette. No Claude attribution lines
anywhere.

## Fix 1 (P1, blocking) — OpenRouter id-less encrypted reasoning detail clobbers streamed reasoning text

`crates/rig-core/src/providers/openrouter/completion.rs:1546`
(`streaming_detail_reasoning`) keys an id-less `ReasoningDetails::Encrypted` as
`StreamPartId::Minted { kind: MintKind::Reasoning, index: 0 }` — the same key the
compat adapter uses for all `reasoning`/`reasoning_content` text deltas
(`crates/rig-core/src/providers/internal/openai_chat_completions_compatible.rs:502`).
The compat adapter closes reasoning only when *text* interleaves, so in the common
reasoning→tool-calls shape the text part is still open when the whole encrypted block
arrives; `reasoning_end`'s open-path restatement (`parts.rs:294-298`) replaces the
part, discarding all accumulated reasoning text. Pre-PR the detail keyed
`PartId::wire("")` — distinct — so both parts coexisted.

- Add a `MintKind` variant for encrypted/opaque reasoning payloads (e.g.
  `EncryptedReasoning`) and key the id-less detail as
  `Minted { kind: EncryptedReasoning, index: 0 }`. Precedent: pydantic-ai fixed this
  exact OpenRouter bug by namespacing minted keys per detail type
  (`reasoning_detail_{type}_{i}`). Do NOT mint a random per-block key, and do NOT
  introduce a new public part type — the block stays a reasoning part with the blob in
  `ReasoningContent::Encrypted`.
- Regression test at the accumulator/adapter level: reasoning text deltas → tool-call
  deltas (no text interleave) → id-less encrypted detail ⇒ final message contains BOTH
  the assembled text reasoning part and the encrypted part, in arrival order. Also keep
  the wire-id'd detail path (`Wire(id)`) untouched.

## Fix 2 (P2) — rig-vertexai never reads `ToolResult::name`

`crates/rig-vertexai/src/types/message.rs:77` still does
`set_name(tool_result.id.clone())`, and `resolve_tool_result_names` is never called
from rig-vertexai. Vertex sets `ToolCall.id` to the *original* function name
(`crates/rig-vertexai/src/types/completion_response.rs:86-88`), so repair-renamed flows
send the never-executed name, cross-provider OpenAI-shaped histories send `call_abc` as
the function name (the 84a43e9e #5 bug, fixed on gemini/ollama, alive here), and the
streamed invalid-name synthetic result (`id: ""`) sends an empty name.

- Call `resolve_tool_result_names` on the history at the Vertex request-build site
  (mirror how `gemini/completion.rs:239` and `ollama.rs:481` do it; mind the crate
  boundary — if the shim isn't reachable from rig-vertexai, re-export it through the
  same internal path the other serializers use, without widening the public API).
- The serializer reads `name` first (with the same `filter(|n| !n.is_empty())` guard
  the other three serializers use), falling back to `id` only when `name` is `None`.
- Check `rig-gemini-grpc` for the same gap; if its request builder sends a function
  name derived from an id, apply the identical fix.
- Tests: unit-level serializer test pinning `functionResponse.name` = executed name for
  a `name: Some` result, and the legacy-history fallback. If a Vertex cassette harness
  exists, pin the repair-rename shape like the REST-Gemini cassette does.

## Fix 3 (P2) — shape-3 shim precedence loses the executed name on legacy repair-renamed histories

In `resolve_tool_result_names` (`crates/rig-core/src/providers/internal/mod.rs:99-116`),
`identity_match` fires on `call_id` first, so a legacy result shaped
`{ id: "<executed name>", call_id: Some(matching call id), name: None }` resolves to
the *model's call name*, and the shape-3 branch whose comment claims to cover repair
renames is unreachable whenever `call_id` matches.

- New precedence, per the industry contract (result-carried name is authoritative;
  call-id pairing is association, not name resolution): when the identity match fired
  **via `call_id`** and `result.id` is non-empty, differs from the matched call's
  identity, AND differs from the matched call's function name — treat `id` as the
  executed name (`result.name = Some(result.id)`), still consuming that pending call.
  Identifier-match-wins remains correct when `id` was the matching identifier itself or
  when `id` equals the call's name.
- Update the shape comments so each branch's claim matches reachable behavior.
- Unit tests: (a) the legacy repair-renamed shape above resolves to the executed name;
  (b) the cross-provider `call_abc`-in-id shape still resolves to the call's function
  name (the fix this PR made — must not regress); (c) `id` equal to the call's name
  stays that name.

## Fix 4 (P3) — reasoning correlator outlives the part

`crates/rig-core/src/streaming/mod.rs:1098`: `reasoning_correlators.entry(id)` is never
invalidated, but the accumulator's key-reuse rule opens a NEW part under the same key
after finish — so on constant-minted-key wires (ollama, gemini REST, cohere, compat)
two distinct reasoning blocks share one correlator, violating the documented
"stable across the part's deltas and unique per run" contract (`mod.rs:2222-2226`).

- Bind the correlator's lifetime to the part's: keep `entry().or_insert_with(generate)`
  on the delta arm, and `remove(&id)` in every arm that finishes the part — the
  `ReasoningEnd` arm AND the whole-`Reasoning` (restatement) arm, including after the
  driver's `reasoning_end` call on the whole-block path. Precedent: vercel-ai-sdk's
  accumulator does `reasoningPartIndexes.delete(chunk.id)` on reasoning-end for exactly
  this reason. Note this is NOT subsumed by Fix 1 — constant keys legitimately recur
  across sequential blocks.
- Test: constant-key sequence `ReasoningDelta("A")` → `ReasoningEnd` → `Message` →
  `ReasoningDelta("B")` ⇒ the public `ReasoningDelta` events for A and B carry
  *different* correlators (and each block's own deltas share one).

## Fix 5 (P3) — warn-payload guard is weaker than its claim

`crates/rig-core/tests/driver_adoption.rs:559-566`: the line-based scan misses named
captures (`warn!(payload = ?x)`), format-string Debug (`warn!("{:?}", x)`), and
multi-line macro calls — its `line.contains("warn!(\n")` branch is dead code because
`lines()` strips newlines. Make the guard structural instead of patching the regex:

- Redact by construction: add a small internal newtype in the providers-internal layer
  (e.g. `RedactedPayload<'a>` or an owned equivalent) whose `Debug` and `Display` print
  only kind + byte size, wrapping the wire payload at the point adapters hold
  unmodeled frames. `warn_unmodeled` (`adapter.rs`) keeps its current output format but
  routes through the type; any accidental `?payload` capture of the wrapped value
  becomes harmless. Expose raw bytes only via one explicitly named method so misuse is
  grep-able.
- Mechanical backstop: `clippy.toml` `disallowed-macros` is per-workspace, so scope it
  the narrow way — if banning `tracing::warn`/`tracing::error` wholesale in rig-core is
  too broad, keep the source scan but make it honest: fix the dead multiline branch by
  scanning statement-joined text (or a `syn`-based visitor over macro invocations,
  failing closed on unparseable bodies), add the named-capture and `{:?}`-format
  patterns, and align the test's doc comment with what it actually enforces. Whichever
  you choose, the guard's stated claim and its coverage must match at the end.
- Keep the existing `driver_adoption` assertions that adapters route through
  `warn_unmodeled`.

## Fix 6 (P3) — Ollama's id-less premise is stale

`tests/cassettes/ollama/streaming_grammar/same_tool_twice.yaml:485-487` records real
wire ids (`"id":"call_kqpofucm"`, `call_REDACTED_1/2`) while `ollama.rs:400-403` and
the streaming_grammar test claim the wire is id-less; rig's `ollama::ToolCall`
(`ollama.rs:1095`) has no `id` field, so a provider-issued durable id is silently
discarded and the test pins the discard. Also `call_kqpofucm` escaped the id scrubber.

- Add `id: Option<String>` (serde `default`) to the Ollama wire `ToolCall`. Route it
  through `WireId::new` when present — streaming: real wire key + `tool_id`; blocking:
  populate `ToolCall.call_id`/id per the crate's conventions — keeping the current
  minted-key path as the fallback for older daemons that omit it. Echo the id on the
  tool-result message if Ollama's request schema accepts it; keep sending the function
  *name* in the result payload regardless (that's the Fix-2 contract). Precedent:
  pydantic-ai reads `c.id` when present and synthesizes only when absent;
  langchain-ollama's ignore-and-always-synthesize is the weaker pattern.
- Update the streaming_grammar test to assert the wire id is *preserved*, not absent;
  fix the stale comments; scrub `call_kqpofucm` → `call_REDACTED_0` consistently in the
  cassette (scrubber config too, so re-records stay scrubbed). If the assertion change
  requires it, re-record honestly and name the cassette in the commit message.

## Decided items (the maintainer has chosen — implement as written)

- **D1 — wire-sent bare ends yield the completed block.** Change the driver's yield
  rule to key on *synthesized-vs-wire-sent* rather than payload presence: a bare
  `ReasoningEnd` that the wire actually sent (e.g. Anthropic `content_block_stop` on an
  unsigned thinking block) yields the accumulator's completed `Reasoning` block to
  streaming consumers; only rig-*synthesized* bare ends stay silent. This restores the
  pre-PR consumer contract (a completed event per block, signed or not — vercel emits
  boundaries for every block). Track end provenance on the raw event (a
  synthesized/wire-sent bit or equivalent, set at the adapter that fabricates ends);
  update the driver comment at the yield split, and pin both sides with tests
  (anthropic unsigned stop ⇒ completed event; compat/ollama synthesized end ⇒ silent).
  Note the behavior change in MIGRATING.
- **D2 — compat adapter closes reasoning on tool calls too.** Align the compat adapter
  with the ollama adapter: synthesize `ReasoningEnd` before the first tool-call delta
  as well as before text (vercel's openai-compatible provider force-closes before any
  other part class). This also shrinks the Fix-1 window — land it before Fix 1. Pin
  with a compat-adapter unit test (reasoning deltas → tool-call delta ⇒ end emitted,
  block closed). Mind the D1 interaction: this synthesized end stays silent publicly.
- **D3 — make `StreamPartId` payloads private.** Make the type match the claim: wrap
  the variant payloads in private-field newtypes (or convert `StreamPartId` to an
  opaque struct with a private repr), keeping exactly `Eq + Hash + Clone + Debug`, so
  `if let StreamPartId::Wire(s) = id` can no longer extract the string. Add a trybuild
  case pinning non-matchability alongside the existing render/Display/Serialize cases.
  Update constructors (`StreamPartId::wire`, the `Minted`/`Composite` construction
  sites in adapters) as needed; breaking is fine on this already-breaking PR. MIGRATING
  keeps its strong "no accessor into the durable id space" wording — now literally true.
- **D4 — amend the PR body.** Edit the PR description (and the re-record commit note
  only if that commit is being rewritten anyway) so the repair-cassette entry reads
  "name change plus re-record usage drift" (`thoughtsTokenCount` 37→48,
  `totalTokenCount` 221→232). No code or cassette change.

## Order and hygiene

Order: Fix 4 (smallest, unblocks correlator tests) → D2 → Fix 1 (its test benefits
from D2) → D1 → Fix 3 → Fix 2 → Fix 6 → Fix 5 → D3 → D4. One commit each,
conventional-commit subjects matching the branch's style (`fix(streaming): …`,
`fix(vertexai): …`, `test(guards): …`). Update MIGRATING/CHANGELOGs only where public
behavior changes (D1 if behavior changes, Fix 6's new field if `ollama::ToolCall` is
public). Run the full gate set at the end and report the numbers.

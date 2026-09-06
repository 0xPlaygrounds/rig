# Anthropic reasoning and stop-sequence agent cases

The four original agent exceptions in reasoning_usage_matrix, stop_sequence_terminal_matrix and empty_stop_sequence_matrix execute through native ECS systems and the real provider bus. Original source edits widen sibling visibility only. The catalog identifies their sources and fixtures.

Reasoning uses Sonnet4.6, the original arithmetic preamble/prompt, max2048 and thinking budget1024. The native test selects the first actual completion outcome in turn order for the run. Shared Observed requires a recorded usage, positive recorded thinking tokens, exact reasoning-token equality, reasoning within output, and total = input + cached input + cache creation + output. Fixture inspection stays after wrapper teardown.

The streamed single stop sequence uses Haiku4.5, the original list prompt, max64 and stop charlie. Native success awaits the run result and rejects stream error items; the actual typed provider final must exist and finish with Stop. The unchanged post-wrapper helper checks recorded terminal stop_sequence=charlie.

Both empty-stop cases use Haiku4.5, the original immediate prompt, max32 and stop alpha. Blocking output must trim to empty. Streaming must complete successfully with output exactly empty and retain Stop in the actual provider terminal, despite having no assistant content. The original post-wrapper empty blocking/streaming validators remain unchanged. Each test has one native run; the reused provider-final collector selects the latest real final by dispatch sequence, without constructing an agent stream item.

No legacy agent runner or completion record executes in the native tests. Original undeclared turn budgets stay undeclared. These recorded workloads do not establish all provider settings or exhaustive parity. The catalog records correspondences; current tests establish results.

# Agent cache-growth parity

Six original agent workloads execute independently through the native ECS provider bus: Anthropic, OpenAI Chat Completions and Responses, Gemini, OpenRouter and Venice. The batch records exact original/native IDs and unchanged fixtures. Original source changes only widen sibling visibility for neutral configuration helpers.

Each native run uses the original probe preamble, CacheProbeLookupTool, AGENT_CACHE_PROMPT, temperature zero and six-turn limit. Probe completion-only settings are not inherited. Anthropic enables model prompt caching; Responses supplies the original prompt_cache_key. Other model settings follow their original agent builders.

The native helper collects successful actual completion outcomes belonging to this run, orders them by turn Order, and passes their usage into the unchanged CacheObservation validator. It neither executes a legacy agent nor synthesizes legacy completion records. The shared validator requires at least two completions, a turn reaching the provider's cached-input ratio floor, and every later turn retaining the floor. Gemini's floor is 0.75; the others are 0.80. Anthropic accounts for uncached, cached and cache-creation input; other providers use input tokens.

All six preserve the original post-wrapper prefix-stability assertion. All except OpenRouter preserve the original breakpoint/support assertion. Strict cassette matching and consumption remain enforced by the original wrappers. Acceptance concerns recorded cache-growth behavior; live TTL, cache economics and adjacent provider-only scenarios remain separate evidence.

The paired runs and independent final review are recorded in evidence/agent-cache-growth-report.json and evidence/agent-cache-growth-review.json. Passing execution and semantic review are distinct checks.

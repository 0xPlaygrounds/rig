//! Provider-local test layout checks to keep the provider-first tree normalized,
//! plus the provider-enumeration guard that closes the conformance registry's
//! residual gap (#2258 F3).

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use rig_core::test_utils::streaming_conformance::WIRE_FAMILIES;

fn provider_dirs() -> Vec<PathBuf> {
    let tests_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/providers");
    let mut dirs = fs::read_dir(tests_dir)
        .expect("tests directory should exist")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .filter(|path| {
            !matches!(
                path.file_name().and_then(|name| name.to_str()),
                Some("common" | "core" | "data")
            )
        })
        .collect::<Vec<_>>();
    dirs.sort();
    dirs
}

#[test]
fn provider_local_tests_do_not_use_legacy_prefixes() {
    for dir in provider_dirs() {
        let provider = dir
            .file_name()
            .and_then(|name| name.to_str())
            .expect("provider directory name should be valid utf-8");

        for entry in fs::read_dir(&dir)
            .expect("provider directory should be readable")
            .filter_map(|entry| entry.ok())
        {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
                continue;
            }

            let stem = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .expect("test file should have a valid utf-8 stem");
            if stem == "mod" {
                continue;
            }

            assert!(
                !stem.starts_with(&format!("{provider}_")),
                "provider-local test file {:?} still uses a provider prefix",
                path
            );
            assert!(
                !stem.starts_with("agent_with_"),
                "provider-local test file {:?} still uses an agent_with_* name",
                path
            );
        }
    }
}
// ---------------------------------------------------------------------------
// Provider enumeration: every provider maps to a wire family, or says why not.
//
// `streaming_conformance_registry.rs` proves every entry in `WIRE_FAMILIES`
// has a compiled suite. That is one-directional: a brand-new provider that
// simply never joins `WIRE_FAMILIES` dodges the registry entirely, because
// nothing enumerates providers. This test is the other direction — it walks
// the tree, not a list — so adding a provider without classifying its wire
// fails CI (#2258 F3, "residual gap").
// ---------------------------------------------------------------------------

/// How a provider's streaming wire is covered by the conformance corpus.
enum WireCoverage {
    /// The provider owns these `WIRE_FAMILIES` entries.
    Families(&'static [&'static str]),
    /// The provider has no family of its own, for the stated reason. A reason
    /// is mandatory: "reuses another provider's wire" and "has no streaming
    /// wire at all" are very different claims and both must be written down.
    Exempt(&'static str),
    /// The provider shares another family's *wire* but layers its own
    /// streaming *semantics* on it through a non-trivial
    /// `CompatibleStreamProfile` / `OpenAICompatibleProvider` hook
    /// (`detail_reasoning`, `should_evict`, ...). "Shares the wire" and
    /// "shares the wire and the semantics" are different claims: the shared
    /// suite exercises the grammar, so the entry must name where the
    /// profile-level semantics are covered.
    SharedWireOwnSemantics {
        /// Why the base wire is someone else's.
        wire: &'static str,
        /// Where the provider-specific streaming semantics are pinned
        /// (cassette suite / test module) — mandatory, so profile hooks
        /// cannot hide behind a wire-verbatim exemption.
        semantics_coverage: &'static str,
    },
}

use WireCoverage::{Exempt, Families, SharedWireOwnSemantics};

/// Wire shared by every `OpenAICompatibleProvider` implementor: the gateway
/// swaps the base URL and request tweaks, not the SSE grammar.
const OPENAI_CHAT_GATEWAY: &str = "OpenAI Chat Completions wire verbatim (implements `OpenAICompatibleProvider`, decoded by \
     `providers/internal/openai_chat_completions_compatible.rs`) — covered by the `openai_chat` \
     suite";

/// Vector-store integrations carry no model wire.
const VECTOR_STORE: &str = "vector store integration — no completion model, no streaming wire";

/// Every provider module under `crates/rig-core/src/providers/`, keyed by its
/// on-disk name without the `.rs` extension, and every `crates/rig-*` package.
/// Discovery walks the tree; this list only classifies what it finds, so a new
/// provider (or a new crate) is a failure until someone states its wire.
const PROVIDER_WIRES: &[(&str, WireCoverage)] = &[
    // --- rig-core provider modules with their own wire family --------------
    ("anthropic", Families(&["anthropic"])),
    ("chatgpt", Families(&["chatgpt"])),
    ("cohere", Families(&["cohere"])),
    // Copilot ships two wires (Responses and Chat Completions); both are
    // instantiated under the one `copilot` family name.
    ("copilot", Families(&["copilot"])),
    ("gemini", Families(&["gemini_rest", "gemini_interactions"])),
    ("ollama", Families(&["ollama"])),
    (
        "openai",
        Families(&[
            "openai_chat",
            "openai_responses",
            "openai_responses_websocket",
        ]),
    ),
    ("xai", Families(&["xai"])),
    // --- rig-core provider modules reusing another provider's wire ---------
    (
        "azure",
        Exempt(
            "Azure-hosted OpenAI — the OpenAI Chat Completions and Responses wires verbatim, \
             covered by `openai_chat` and `openai_responses`",
        ),
    ),
    ("deepseek", Exempt(OPENAI_CHAT_GATEWAY)),
    ("doubleword", Exempt(OPENAI_CHAT_GATEWAY)),
    ("groq", Exempt(OPENAI_CHAT_GATEWAY)),
    ("huggingface", Exempt(OPENAI_CHAT_GATEWAY)),
    ("hyperbolic", Exempt(OPENAI_CHAT_GATEWAY)),
    ("llamafile", Exempt(OPENAI_CHAT_GATEWAY)),
    ("minimax", Exempt(OPENAI_CHAT_GATEWAY)),
    ("mira", Exempt(OPENAI_CHAT_GATEWAY)),
    ("mistral", Exempt(OPENAI_CHAT_GATEWAY)),
    ("moonshot", Exempt(OPENAI_CHAT_GATEWAY)),
    (
        "openrouter",
        SharedWireOwnSemantics {
            wire: OPENAI_CHAT_GATEWAY,
            semantics_coverage: "`streaming_detail_reasoning` maps `reasoning_details` \
                 (`reasoning.encrypted` → `ReasoningContent::Encrypted`, tool-call \
                 decorations) — pinned by the recorded openrouter cassettes \
                 (`tests/providers/openrouter/cassette/streaming_tools.rs`, \
                 `reasoning_roundtrip`/`reasoning_tool_roundtrip`) and the \
                 `streaming_detail_reasoning` unit tests in \
                 `providers/openrouter/completion.rs`",
        },
    ),
    ("perplexity", Exempt(OPENAI_CHAT_GATEWAY)),
    ("together", Exempt(OPENAI_CHAT_GATEWAY)),
    ("venice", Exempt(OPENAI_CHAT_GATEWAY)),
    ("xiaomimimo", Exempt(OPENAI_CHAT_GATEWAY)),
    ("zai", Exempt(OPENAI_CHAT_GATEWAY)),
    (
        "voyageai",
        Exempt("embeddings-only provider — no completion model, so no streaming wire"),
    ),
    // --- workspace packages -------------------------------------------------
    ("rig-bedrock", Families(&["bedrock"])),
    ("rig-candle", Families(&["candle"])),
    ("rig-gemini-grpc", Families(&["gemini_grpc"])),
    (
        "rig-vertexai",
        Exempt(
            "Vertex AI Gemini, unary only — `stream()` returns `streaming_unsupported()`, so \
             there is no streaming wire to conform. Adding streaming here means adding a family.",
        ),
    ),
    (
        "rig-core",
        Exempt("hosts the provider modules enumerated above and the shared wire machinery"),
    ),
    (
        "rig-agent",
        Exempt("consumer-side agent runtime — reads the canonical grammar, never decodes a wire"),
    ),
    ("rig-derive", Exempt("proc-macro crate — no runtime wire")),
    (
        "rig-fastembed",
        Exempt("local embedding models only — no completion model, no streaming wire"),
    ),
    (
        "rig-memory",
        Exempt("agent memory store — no model provider, no streaming wire"),
    ),
    ("rig-helixdb", Exempt(VECTOR_STORE)),
    ("rig-lancedb", Exempt(VECTOR_STORE)),
    ("rig-milvus", Exempt(VECTOR_STORE)),
    ("rig-mongodb", Exempt(VECTOR_STORE)),
    ("rig-neo4j", Exempt(VECTOR_STORE)),
    ("rig-postgres", Exempt(VECTOR_STORE)),
    ("rig-qdrant", Exempt(VECTOR_STORE)),
    ("rig-s3vectors", Exempt(VECTOR_STORE)),
    ("rig-scylladb", Exempt(VECTOR_STORE)),
    ("rig-sqlite", Exempt(VECTOR_STORE)),
    ("rig-surrealdb", Exempt(VECTOR_STORE)),
    ("rig-vectorize", Exempt(VECTOR_STORE)),
];

/// Module names under `providers/` that are not providers.
const NON_PROVIDER_MODULES: &[&str] = &["mod", "internal"];

/// Every provider module name under `crates/rig-core/src/providers/`, plus
/// every `crates/rig-*` package directory — discovered from the tree, never
/// from a list.
fn discovered_units() -> BTreeSet<String> {
    // The `rig` facade is the workspace-root package.
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut units = BTreeSet::new();

    let providers = root.join("crates/rig-core/src/providers");
    for entry in fs::read_dir(&providers).expect("providers directory should be readable") {
        let path = entry.expect("provider entry should be readable").path();
        let name = if path.is_dir() {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(str::to_owned)
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .map(str::to_owned)
        } else {
            None
        };
        if let Some(name) = name
            && !NON_PROVIDER_MODULES.contains(&name.as_str())
        {
            units.insert(name);
        }
    }

    for entry in fs::read_dir(root.join("crates")).expect("crates directory should be readable") {
        let path = entry.expect("crate entry should be readable").path();
        if !path.is_dir() {
            continue;
        }
        if let Some(name) = path.file_name().and_then(|name| name.to_str())
            && name.starts_with("rig-")
        {
            units.insert(name.to_owned());
        }
    }

    units
}

/// Every provider module and workspace package is classified, and every
/// classification still points at something that exists.
#[test]
fn every_provider_maps_to_a_wire_family_or_a_justified_exemption() {
    let discovered = discovered_units();
    let classified: BTreeSet<&str> = PROVIDER_WIRES.iter().map(|(name, _)| *name).collect();

    let unclassified: Vec<&str> = discovered
        .iter()
        .map(String::as_str)
        .filter(|name| !classified.contains(name))
        .collect();
    assert!(
        unclassified.is_empty(),
        "unclassified provider(s)/package(s): {unclassified:?}. Add each to PROVIDER_WIRES in \
         tests/core/provider_layout.rs — either naming the WIRE_FAMILIES entry whose \
         streaming_conformance_suite! covers its wire, or an Exempt(reason) saying why it has \
         none. A provider that never joins WIRE_FAMILIES is invisible to the conformance \
         registry, which is exactly the gap this test exists to close.",
    );

    let stale: Vec<&&str> = classified
        .iter()
        .filter(|name| !discovered.contains(**name))
        .collect();
    assert!(
        stale.is_empty(),
        "PROVIDER_WIRES classifies provider(s)/package(s) that no longer exist: {stale:?}",
    );

    for (name, coverage) in PROVIDER_WIRES {
        match coverage {
            Families(families) => {
                assert!(
                    !families.is_empty(),
                    "{name} claims families but names none — use Exempt(reason) instead",
                );
                for family in *families {
                    assert!(
                        WIRE_FAMILIES.contains(family),
                        "{name} maps to wire family {family:?}, which is absent from \
                         WIRE_FAMILIES",
                    );
                }
            }
            Exempt(reason) => assert!(
                !reason.trim().is_empty(),
                "{name}'s exemption carries no justification",
            ),
            SharedWireOwnSemantics {
                wire,
                semantics_coverage,
            } => {
                assert!(
                    !wire.trim().is_empty(),
                    "{name}'s shared-wire claim carries no justification",
                );
                assert!(
                    semantics_coverage.contains("tests/")
                        || semantics_coverage.contains("providers/"),
                    "{name}'s semantics coverage must point at real test locations, got: \
                     {semantics_coverage}",
                );
            }
        }
    }
}

/// Every wire family is owned by a real provider. A family nobody claims is a
/// suite pinning a wire that no longer ships (or a typo in `PROVIDER_WIRES`).
#[test]
fn every_wire_family_is_claimed_by_a_provider() {
    let claimed: BTreeSet<&str> = PROVIDER_WIRES
        .iter()
        .filter_map(|(_, coverage)| match coverage {
            Families(families) => Some(*families),
            Exempt(_) | SharedWireOwnSemantics { .. } => None,
        })
        .flatten()
        .copied()
        .collect();

    let orphaned: Vec<&&str> = WIRE_FAMILIES
        .iter()
        .filter(|family| !claimed.contains(**family))
        .collect();
    assert!(
        orphaned.is_empty(),
        "wire families claimed by no provider: {orphaned:?}",
    );
}

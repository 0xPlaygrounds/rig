//! Generating `rig_reqwest::providers` — the alias tree that gives every
//! transport-generic rig-core provider type a default of the bundled reqwest
//! transport, so `openai::CompletionModel` means `…<ReqwestClient>` again in
//! type position.
//!
//! The tree was hand-written and 333 lines long, and its own module doc claimed
//! it was generated. This is the generator that makes the claim true.
//!
//! # What gets an alias
//!
//! Every public type under `rig_core::providers::<provider>` with exactly one
//! type parameter and no default on it — see `rustdoc.rs` for why that rule is
//! stated against rustdoc's output rather than the sources, and why both halves
//! of it matter.
//!
//! # What does not
//!
//! [`NOT_A_TRANSPORT`] is the one hand-maintained list left, and it is three
//! entries long. A payload envelope like `ApiResponse<T>` satisfies the rule
//! above — one parameter, no default — while its parameter is the *response
//! body*, not the transport. Defaulting it to `ReqwestClient` would compile and
//! be nonsense.
//!
//! The list cannot rot silently: every entry must still match the rule, or
//! generation fails and tells you to delete it. A rule that could distinguish
//! these two shapes without a list would be better; see the PR for the
//! reachability analysis I tried and why it is not worth its fragility yet.
//!
//! # Feature gates are derived
//!
//! The `audio` and `image` aliases are not listed anywhere here. The surface is
//! built once with no features and once per feature, and a type that appears
//! only in the latter gets that feature's `#[cfg]`. Add a modality feature to
//! rig-core and the tree picks it up as soon as it is added to [`MODALITIES`].

use crate::rustdoc::{TransportGenericType, transport_generic_types};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

/// The rig-core features that gate provider types, and which `rig-reqwest`
/// forwards under the same names.
const MODALITIES: &[&str] = &["audio", "image"];

/// Types that match the transport-generic rule but whose parameter is not a
/// transport. Each entry is `(provider, type, why)`.
const NOT_A_TRANSPORT: &[(&str, &str, &str)] = &[
    (
        "cohere",
        "ApiResponse",
        "response envelope: the parameter is the deserialized body, not the transport",
    ),
    (
        "gemini",
        "ApiResponse",
        "response envelope: the parameter is the deserialized body, not the transport",
    ),
    (
        "huggingface",
        "ApiResponse",
        "response envelope: the parameter is the deserialized body, not the transport",
    ),
];

/// The generated file, as it should appear on disk.
pub fn render(workspace: &Path) -> Result<String, String> {
    let base = transport_generic_types(workspace, &[])?;
    let mut gated: BTreeMap<(String, String), &'static str> = BTreeMap::new();
    let mut all = base.clone();

    for modality in MODALITIES {
        let with_modality = transport_generic_types(workspace, &[modality])?;
        for candidate in with_modality {
            if base.iter().any(|item| item.path == candidate.path) {
                continue;
            }
            gated.insert(
                (candidate.provider.clone(), candidate.name.clone()),
                modality,
            );
            all.push(candidate);
        }
    }

    all.sort();
    all.dedup();
    let all = apply_exclusions(all)?;

    let mut providers: BTreeMap<String, Vec<TransportGenericType>> = BTreeMap::new();
    for item in all {
        providers
            .entry(item.provider.clone())
            .or_default()
            .push(item);
    }

    Ok(emit(&providers, &gated))
}

/// Drop the known payload generics, and fail on an entry that no longer
/// describes anything — a stale exclusion is how a list like this rots.
fn apply_exclusions(types: Vec<TransportGenericType>) -> Result<Vec<TransportGenericType>, String> {
    let mut unused: Vec<&str> = Vec::new();
    for (provider, name, _) in NOT_A_TRANSPORT {
        if !types
            .iter()
            .any(|item| item.provider == *provider && item.name == *name)
        {
            unused.push(name);
        }
    }
    if !unused.is_empty() {
        return Err(format!(
            "NOT_A_TRANSPORT lists {unused:?}, which no longer match the transport-generic rule. \
             Delete the stale entries."
        ));
    }

    Ok(types
        .into_iter()
        .filter(|item| {
            !NOT_A_TRANSPORT
                .iter()
                .any(|(provider, name, _)| item.provider == *provider && item.name == *name)
        })
        .collect())
}

fn emit(
    providers: &BTreeMap<String, Vec<TransportGenericType>>,
    gated: &BTreeMap<(String, String), &'static str>,
) -> String {
    let mut out = String::new();
    out.push_str(HEADER);

    for (provider, types) in providers {
        let _ = writeln!(out, "\npub mod {provider} {{");
        let _ = writeln!(out, "    pub use rig_core::providers::{provider}::*;");
        for item in types {
            if let Some(feature) = gated.get(&(item.provider.clone(), item.name.clone())) {
                let _ = writeln!(out, "    #[cfg(feature = \"{feature}\")]");
                let _ = writeln!(
                    out,
                    "    #[cfg_attr(docsrs, doc(cfg(feature = \"{feature}\")))]"
                );
            }
            // The alias binds its own parameter, so rig-core's name for it
            // (`H` in most providers, `T` in a few) is not part of this
            // surface: spell it `H` everywhere rather than propagate the
            // inconsistency into the public API.
            let _ = writeln!(
                out,
                "    pub type {name}<H = super::DefaultHttp> = {path}<H>;",
                name = item.name,
                path = item.path,
            );
        }
        out.push_str("}\n");
    }

    out
}

const HEADER: &str = r#"//! Type-position aliases for every rig-core provider type that is generic over the
//! transport, defaulted to the bundled [`crate::ReqwestClient`]: `rig::providers::openai::CompletionModel`
//! means `…::CompletionModel<ReqwestClient>` again, so `Agent<openai::CompletionModel>` and
//! `let c: openai::Client = …` read as before rig-core lost its default transport.
//!
//! Each module re-exports everything from the rig-core provider module and then shadows the
//! transport-generic names with defaulted aliases (nested rig-core paths stay generic).
//!
//! Construction goes through [`crate::client::DefaultTransportClient`] /
//! [`crate::client::DefaultTransportBuilder`]: type-alias defaults do not apply in expression
//! position, so `openai::Client::new(..)` needs those traits, not these aliases.
//!
//! # Generated file — do not edit
//!
//! Regenerate with `cargo xtask generate-provider-aliases`; CI runs the same
//! command with `--check`. The source of truth is rig-core's own rustdoc
//! output, so a type that is generic over the transport gets an alias here
//! whether it was written by hand or produced by a macro, and a type whose
//! parameter already has a default (`ClientBuilder<H = Missing>`) is left
//! alone. See `xtask/src/aliases.rs`.

/// The bundled transport every alias here defaults to.
pub type DefaultHttp = crate::ReqwestClient;
"#;

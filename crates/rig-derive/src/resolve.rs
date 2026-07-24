//! Single authority for how Rig's crates are reachable from the expanding crate.
//!
//! Every path the macros emit and every fully qualified type they recognize is
//! derived from one [`CrateRefs`] resolved once per expansion. Nothing else in
//! the crate may call [`proc_macro_crate::crate_name`] or hardcode a Rig crate
//! name.

use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

/// How one of Rig's published crates can be referenced from the code the
/// macros expand into.
enum Reachable {
    /// The expanding crate *is* this crate: reference it as `crate`.
    Itself,
    /// A (possibly renamed) dependency: reference it by its resolved name.
    Named(String),
    /// Not a dependency of the expanding crate.
    Absent,
}

fn lookup(package: &str) -> Reachable {
    match crate_name(package) {
        Ok(FoundCrate::Itself) => Reachable::Itself,
        Ok(FoundCrate::Name(name)) => Reachable::Named(name),
        Err(_) => Reachable::Absent,
    }
}

fn root_tokens(reachable: &Reachable) -> Option<TokenStream> {
    match reachable {
        Reachable::Itself => Some(quote!(crate)),
        Reachable::Named(name) => {
            let ident = format_ident!("{name}");
            Some(quote!(::#ident))
        }
        Reachable::Absent => None,
    }
}

fn root_name(reachable: &Reachable) -> Option<String> {
    match reachable {
        Reachable::Itself => Some("crate".to_string()),
        Reachable::Named(name) => Some(name.clone()),
        Reachable::Absent => None,
    }
}

/// The resolved Rig crate topology for one macro expansion.
pub(crate) struct CrateRefs {
    /// Path to the portable core namespace: `crate`, `::rig_core`, or the
    /// explicit `core` module of the facade or runtime crate. This namespace
    /// re-exports `serde`, `serde_json`, and `schemars`, so all generated code
    /// resolves those through it instead of assuming the caller's Cargo.toml.
    pub(crate) core: TokenStream,
    /// Path to the classic runtime root (the crate exposing
    /// `tool::{Tool, ToolContext}`): `rig-agent` or the `rig` facade. `None`
    /// when neither is a dependency, in which case contextual tools cannot be
    /// generated and get a targeted error instead of an unresolved-crate one.
    pub(crate) agent: Option<TokenStream>,
    /// First path segments under which `<root>::tool::ToolContext` names the
    /// runtime context in the expanding crate.
    context_roots: Vec<String>,
    /// First path segments under which `<root>::agent::tool::ToolContext`
    /// names the runtime context (the facade's explicit runtime module).
    facade_roots: Vec<String>,
}

impl CrateRefs {
    pub(crate) fn resolve() -> Self {
        let core_dep = lookup("rig-core");
        let facade_dep = lookup("rig");
        let agent_dep = lookup("rig-agent");

        let core = root_tokens(&core_dep)
            .or_else(|| root_tokens(&facade_dep).map(|root| quote!(#root::core)))
            .or_else(|| root_tokens(&agent_dep).map(|root| quote!(#root::core)))
            .unwrap_or_else(|| quote!(::rig_core));

        let agent = root_tokens(&agent_dep).or_else(|| root_tokens(&facade_dep));

        let mut context_roots = Vec::new();
        let mut facade_roots = Vec::new();
        if let Some(name) = root_name(&agent_dep) {
            context_roots.push(name);
        }
        if let Some(name) = root_name(&facade_dep) {
            context_roots.push(name.clone());
            facade_roots.push(name);
        }

        Self {
            core,
            agent,
            context_roots,
            facade_roots,
        }
    }

    /// Whether `segments` is an unambiguous fully qualified path to the
    /// runtime `ToolContext` under any name the crates resolve to in this
    /// build — including Cargo renames and `crate` self-references.
    pub(crate) fn is_context_path(&self, segments: &[String]) -> bool {
        match segments {
            [root, tool, context] => {
                self.context_roots.iter().any(|known| known == root)
                    && tool == "tool"
                    && context == "ToolContext"
            }
            [root, agent, tool, context] => {
                self.facade_roots.iter().any(|known| known == root)
                    && agent == "agent"
                    && tool == "tool"
                    && context == "ToolContext"
            }
            _ => false,
        }
    }
}

/// Render a re-exported dependency path (e.g. `::rig::core` + `serde`) as the
/// string form serde/schemars `crate = "..."` attributes expect.
pub(crate) fn crate_attr_string(root: &TokenStream, item: &str) -> String {
    format!("{}::{item}", root.to_string().replace(' ', ""))
}

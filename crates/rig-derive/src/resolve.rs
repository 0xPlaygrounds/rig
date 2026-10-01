//! Resolves Rig dependency paths and recognized context-type paths for macro expansion.

use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

/// How `package` is referenced from the expanding crate: `crate` when it is
/// that crate, its (possibly renamed) dependency name, or `None` when it is
/// not a dependency.
fn root_name(package: &str) -> Option<String> {
    match crate_name(package).ok()? {
        FoundCrate::Itself => Some("crate".to_string()),
        FoundCrate::Name(name) => Some(name),
    }
}

/// The path prefix for a [`root_name`]: `crate` or `::name`.
fn root_tokens(name: &str) -> TokenStream {
    if name == "crate" {
        quote!(crate)
    } else {
        let ident = format_ident!("{name}");
        quote!(::#ident)
    }
}

/// The resolved Rig crate topology for one macro expansion.
pub(crate) struct CrateRefs {
    /// Path to the portable core namespace: `crate`, `::rig_core`, or the
    /// explicit `core` module of the facade or runtime crate. This namespace
    /// re-exports `serde`, `serde_json`, and `schemars`, so all generated code
    /// resolves those through it instead of assuming the caller's Cargo.toml.
    pub(crate) core: TokenStream,
    /// First path segments under which `<root>::tool::ToolContext` names the
    /// dispatch context in the expanding crate: `rig-core` (which owns it),
    /// and `rig-agent` / the `rig` facade (which re-export it).
    context_roots: Vec<String>,
    /// First path segments under which `<root>::agent::tool::ToolContext`
    /// names the runtime context (the facade's explicit runtime module).
    facade_roots: Vec<String>,
}

impl CrateRefs {
    pub(crate) fn resolve() -> Self {
        let core_dep = root_name("rig-core");
        let facade_dep = root_name("rig");
        let agent_dep = root_name("rig-agent");

        let core = match (&core_dep, facade_dep.as_ref().or(agent_dep.as_ref())) {
            (Some(core), _) => root_tokens(core),
            (None, Some(runtime)) => {
                let root = root_tokens(runtime);
                quote!(#root::core)
            }
            (None, None) => quote!(::rig_core),
        };

        let facade_roots = facade_dep.iter().cloned().collect();
        let context_roots = [core_dep, agent_dep, facade_dep]
            .into_iter()
            .flatten()
            .collect();

        Self {
            core,
            context_roots,
            facade_roots,
        }
    }

    /// Recognizes qualified context paths using resolved dependency names,
    /// including Cargo renames and `crate` self-references.
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

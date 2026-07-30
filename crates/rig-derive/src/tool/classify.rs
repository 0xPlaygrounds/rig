//! Detection of the removed runtime execution context on tool parameters.
//!
//! `ToolContext` and the contextual `Tool` trait no longer exist: every
//! `#[rig_tool]` function is a portable record. These helpers exist only so a
//! leftover context parameter is rejected with a migration pointer instead of
//! being silently treated as a model-facing argument (which would put it in
//! the advertised JSON schema).

use syn::{Attribute, Ident, Meta, Type};

use crate::resolve::CrateRefs;

/// Returns whether `ty` uses an unambiguous fully qualified path to Rig's
/// tool execution context.
///
/// Procedural macros cannot resolve imported type names. Matching only the
/// last `ToolContext` path segment would therefore steal unrelated application
/// types with the same name, so only paths rooted at a crate name Rig resolves
/// to in this build (including Cargo renames) are recognized. Imported aliases
/// use the explicit `#[rig(context)]` parameter marker instead.
fn is_tool_context_type(ty: &Type, refs: &CrateRefs) -> bool {
    let ty = match ty {
        Type::Group(group) => &*group.elem,
        Type::Paren(paren) => &*paren.elem,
        ty => ty,
    };

    let Type::Path(type_path) = ty else {
        return false;
    };
    let segments = type_path
        .path
        .segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect::<Vec<_>>();

    refs.is_context_path(&segments)
}

/// Whether a function parameter explicitly marks itself as Rig's runtime
/// context. The marker is removed from the emitted function.
pub(crate) fn has_tool_context_marker(attrs: &[Attribute]) -> syn::Result<bool> {
    let mut marked = false;
    for attr in attrs.iter().filter(|attr| attr.path().is_ident("rig")) {
        if marked {
            return Err(syn::Error::new_spanned(
                attr,
                "duplicate `#[rig(context)]` parameter marker",
            ));
        }

        let Meta::List(list) = &attr.meta else {
            return Err(syn::Error::new_spanned(
                attr,
                "expected `#[rig(context)]` on the runtime context parameter",
            ));
        };
        let marker: Ident = list.parse_args().map_err(|_| {
            syn::Error::new_spanned(
                attr,
                "expected `#[rig(context)]` on the runtime context parameter",
            )
        })?;
        if marker != "context" {
            return Err(syn::Error::new_spanned(
                marker,
                "the only supported parameter marker is `#[rig(context)]`",
            ));
        }
        marked = true;
    }
    Ok(marked)
}

/// Whether a function parameter is a leftover runtime-context parameter, in
/// any form: marked with `#[rig(context)]`, or typed as a fully qualified
/// `ToolContext` behind any number of references.
///
/// Reference form and mutability are not distinguished — every shape resolves
/// to the same migration error, so there is nothing to tell apart.
pub(crate) fn is_tool_context_parameter(
    ty: &Type,
    explicitly_marked: bool,
    refs: &CrateRefs,
) -> bool {
    if explicitly_marked {
        return true;
    }

    let mut ty = ty;
    loop {
        ty = match ty {
            Type::Group(group) => &*group.elem,
            Type::Paren(paren) => &*paren.elem,
            Type::Reference(reference) => &*reference.elem,
            ty => return is_tool_context_type(ty, refs),
        };
    }
}

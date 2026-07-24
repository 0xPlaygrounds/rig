//! Classification of function parameters as the runtime execution context.

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

/// Classify a function parameter as the distinguished execution context.
///
/// An owned or shared `ToolContext` is almost certainly an authoring mistake:
/// tools need the exact mutable context supplied by the runtime so result
/// metadata and mutations remain visible to the caller.
pub(crate) fn is_tool_context_parameter(
    ty: &Type,
    explicitly_marked: bool,
    refs: &CrateRefs,
) -> syn::Result<bool> {
    let ty = match ty {
        Type::Group(group) => &*group.elem,
        Type::Paren(paren) => &*paren.elem,
        ty => ty,
    };

    if let Type::Reference(reference) = ty
        && (explicitly_marked || is_tool_context_type(&reference.elem, refs))
    {
        if reference.mutability.is_none() {
            return Err(syn::Error::new_spanned(
                ty,
                "a `ToolContext` parameter must have type `&mut ToolContext`",
            ));
        }

        return Ok(true);
    }

    if explicitly_marked || is_tool_context_type(ty, refs) {
        return Err(syn::Error::new_spanned(
            ty,
            "a `ToolContext` parameter must have type `&mut ToolContext`",
        ));
    }

    Ok(false)
}

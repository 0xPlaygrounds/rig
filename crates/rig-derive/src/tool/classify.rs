//! Classification of function parameters as the runtime execution context.

use syn::{Attribute, Ident, Meta, Type};

use crate::resolve::CrateRefs;

/// Diagnostic for context parameters lacking a mutable reference.
const MUT_CONTEXT_MSG: &str = "a `ToolContext` parameter must have type `&mut ToolContext`";

/// Peel grouping (`Group`/`Paren`) wrappers off a type.
fn peel(ty: &Type) -> &Type {
    match ty {
        Type::Group(group) => &group.elem,
        Type::Paren(paren) => &paren.elem,
        ty => ty,
    }
}

/// Recognizes qualified context paths under resolved Rig dependencies.
/// Imported aliases need an explicit marker because macros cannot resolve them.
fn is_tool_context_type(ty: &Type, refs: &CrateRefs) -> bool {
    let Type::Path(type_path) = peel(ty) else {
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

/// Recognizes marked or qualified context parameters and rejects non-mutable
/// references or owned values. Other parameters return `false`.
pub(crate) fn is_tool_context_parameter(
    ty: &Type,
    explicitly_marked: bool,
    refs: &CrateRefs,
) -> syn::Result<bool> {
    let ty = peel(ty);

    if let Type::Reference(reference) = ty
        && (explicitly_marked || is_tool_context_type(&reference.elem, refs))
    {
        if reference.mutability.is_none() {
            return Err(syn::Error::new_spanned(ty, MUT_CONTEXT_MSG));
        }

        return Ok(true);
    }

    if explicitly_marked || is_tool_context_type(ty, refs) {
        return Err(syn::Error::new_spanned(ty, MUT_CONTEXT_MSG));
    }

    Ok(false)
}

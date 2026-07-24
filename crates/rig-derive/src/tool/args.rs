//! Strict grammar for `#[rig_tool(...)]` arguments.
//!
//! Every argument either matches the grammar or produces a spanned error;
//! nothing is silently ignored, and duplicates are rejected. Name validation
//! against the actual parameter list happens in `expand`, where the function
//! signature is available.

use syn::{
    Expr, ExprLit, Ident, Lit, Meta, Token,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
};

pub(crate) struct MacroArgs {
    pub(crate) name: Option<String>,
    pub(crate) description: Option<String>,
    /// `params(name = "description")` entries in declaration order. Idents are
    /// kept (not strings) so validation errors can point at the exact key.
    pub(crate) param_descriptions: Vec<(Ident, String)>,
    /// `required(...)` names; `None` means "derive from the parameter types".
    pub(crate) required: Option<Vec<Ident>>,
}

impl MacroArgs {
    pub(crate) fn description_for(&self, param: &str) -> Option<&str> {
        self.param_descriptions
            .iter()
            .find(|(ident, _)| ident == param)
            .map(|(_, description)| description.as_str())
    }
}

fn parse_string_literal(expr: &Expr, field_name: &str) -> syn::Result<String> {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Str(lit_str),
            ..
        }) => Ok(lit_str.value()),
        _ => Err(syn::Error::new_spanned(
            expr,
            format!("`{field_name}` must be a string literal"),
        )),
    }
}

fn validate_explicit_tool_name(name: &str, expr: &Expr) -> syn::Result<()> {
    if name.is_empty() || name.len() > 64 {
        return Err(syn::Error::new_spanned(
            expr,
            "`name` must be between 1 and 64 characters long",
        ));
    }

    let mut chars = name.chars();
    let Some(first_char) = chars.next() else {
        return Err(syn::Error::new_spanned(
            expr,
            "`name` must be between 1 and 64 characters long",
        ));
    };

    if !first_char.is_ascii_alphabetic() && first_char != '_' {
        return Err(syn::Error::new_spanned(
            expr,
            "`name` must start with an ASCII letter or underscore",
        ));
    }

    if chars.any(|ch| !ch.is_ascii_alphanumeric() && ch != '_' && ch != '-') {
        return Err(syn::Error::new_spanned(
            expr,
            "`name` may only contain ASCII letters, digits, underscores, or hyphens",
        ));
    }

    Ok(())
}

fn reject_duplicate<T>(
    slot: &Option<T>,
    spanned: impl quote::ToTokens,
    what: &str,
) -> syn::Result<()> {
    if slot.is_some() {
        return Err(syn::Error::new_spanned(
            spanned,
            format!("duplicate `{what}` argument"),
        ));
    }
    Ok(())
}

impl Parse for MacroArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut name = None;
        let mut description = None;
        let mut param_descriptions: Option<Vec<(Ident, String)>> = None;
        let mut required = None;

        if input.is_empty() {
            return Ok(MacroArgs {
                name,
                description,
                param_descriptions: Vec::new(),
                required,
            });
        }

        let meta_list: Punctuated<Meta, Token![,]> = Punctuated::parse_terminated(input)?;

        for meta in meta_list {
            match meta {
                Meta::NameValue(nv) => {
                    let ident = nv.path.get_ident().ok_or_else(|| {
                        syn::Error::new_spanned(
                            &nv.path,
                            "unsupported top-level #[rig_tool] argument",
                        )
                    })?;

                    match ident.to_string().as_str() {
                        "name" => {
                            reject_duplicate(&name, &nv.path, "name")?;
                            let parsed_name = parse_string_literal(&nv.value, "name")?;
                            validate_explicit_tool_name(&parsed_name, &nv.value)?;
                            name = Some(parsed_name);
                        }
                        "description" => {
                            reject_duplicate(&description, &nv.path, "description")?;
                            description = Some(parse_string_literal(&nv.value, "description")?);
                        }
                        _ => {
                            return Err(syn::Error::new_spanned(
                                &nv.path,
                                format!("unsupported top-level #[rig_tool] argument `{ident}`"),
                            ));
                        }
                    }
                }
                Meta::List(list) => {
                    let ident = list.path.get_ident().ok_or_else(|| {
                        syn::Error::new_spanned(
                            &list.path,
                            "unsupported top-level #[rig_tool] argument",
                        )
                    })?;

                    match ident.to_string().as_str() {
                        "params" => {
                            reject_duplicate(&param_descriptions, &list.path, "params(...)")?;
                            let nested: Punctuated<Meta, Token![,]> =
                                list.parse_args_with(Punctuated::parse_terminated)?;

                            let mut descriptions = Vec::new();
                            for meta in nested {
                                let Meta::NameValue(nv) = meta else {
                                    return Err(syn::Error::new_spanned(
                                        meta,
                                        "`params(...)` entries must have the form `name = \"description\"`",
                                    ));
                                };
                                let Some(param_ident) = nv.path.get_ident().cloned() else {
                                    return Err(syn::Error::new_spanned(
                                        &nv.path,
                                        "parameter descriptions must use identifier keys",
                                    ));
                                };
                                let value =
                                    parse_string_literal(&nv.value, &param_ident.to_string())?;
                                if descriptions
                                    .iter()
                                    .any(|(existing, _): &(Ident, String)| *existing == param_ident)
                                {
                                    return Err(syn::Error::new_spanned(
                                        &param_ident,
                                        format!(
                                            "duplicate `params(...)` entry for `{param_ident}`"
                                        ),
                                    ));
                                }
                                descriptions.push((param_ident, value));
                            }
                            param_descriptions = Some(descriptions);
                        }
                        "required" => {
                            reject_duplicate(&required, &list.path, "required(...)")?;
                            let required_variables: Punctuated<Ident, Token![,]> =
                                list.parse_args_with(Punctuated::parse_terminated)?;

                            let mut names: Vec<Ident> = Vec::new();
                            for ident in required_variables {
                                if names.contains(&ident) {
                                    return Err(syn::Error::new_spanned(
                                        &ident,
                                        format!("duplicate `required(...)` entry for `{ident}`"),
                                    ));
                                }
                                names.push(ident);
                            }
                            required = Some(names);
                        }
                        _ => {
                            return Err(syn::Error::new_spanned(
                                &list.path,
                                format!("unsupported top-level #[rig_tool] argument `{ident}`"),
                            ));
                        }
                    }
                }
                Meta::Path(path) => {
                    let message = if let Some(ident) = path.get_ident() {
                        format!("unsupported top-level #[rig_tool] argument `{ident}`")
                    } else {
                        "unsupported top-level #[rig_tool] argument".to_string()
                    };

                    return Err(syn::Error::new_spanned(path, message));
                }
            }
        }

        Ok(MacroArgs {
            name,
            description,
            param_descriptions: param_descriptions.unwrap_or_default(),
            required,
        })
    }
}

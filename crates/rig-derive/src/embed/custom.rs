use quote::ToTokens;
use syn::{ExprPath, meta::ParseNestedMeta};

use super::{EMBED, field_member};

const EMBED_WITH: &str = "embed_with";

/// Returns custom-annotated fields and their parsed function paths.
/// Malformed attributes return spanned errors.
pub(crate) fn custom_embed_fields(
    data_struct: &syn::DataStruct,
) -> syn::Result<Vec<(syn::Member, syn::ExprPath)>> {
    data_struct
        .fields
        .iter()
        .enumerate()
        .filter_map(|(index, field)| {
            field
                .attrs
                .iter()
                .filter_map(|attribute| match attribute.is_custom() {
                    Ok(true) => match attribute.expand_tag() {
                        Ok(path) => Some(Ok((field_member(index, field), path))),
                        Err(e) => Some(Err(e)),
                    },
                    Ok(false) => None,
                    Err(e) => Some(Err(e)),
                })
                .next()
        })
        .collect::<Result<Vec<_>, _>>()
}

trait CustomAttributeParser {
    /// Recognizes custom embedding attributes, rejecting unknown keys.
    fn is_custom(&self) -> syn::Result<bool>;

    /// Parses the custom function path from a string literal.
    fn expand_tag(&self) -> syn::Result<syn::ExprPath>;
}

impl CustomAttributeParser for syn::Attribute {
    fn is_custom(&self) -> syn::Result<bool> {
        match &self.meta {
            syn::Meta::List(meta) => {
                if meta.tokens.is_empty() {
                    return Ok(false);
                }
            }
            _ => return Ok(false),
        };

        if !self.path().is_ident(EMBED) {
            return Ok(false);
        }

        self.parse_nested_meta(|meta| {
            meta.value()?.parse::<syn::Expr>()?;

            if meta.path.is_ident(EMBED_WITH) {
                Ok(())
            } else {
                let path = meta.path.to_token_stream().to_string().replace(' ', "");
                Err(syn::Error::new_spanned(
                    meta.path,
                    format_args!("unknown embedding field attribute `{path}`"),
                ))
            }
        })?;

        Ok(true)
    }

    fn expand_tag(&self) -> syn::Result<syn::ExprPath> {
        fn function_path(meta: &ParseNestedMeta<'_>) -> syn::Result<ExprPath> {
            let expr = meta.value()?.parse::<syn::Expr>()?;
            let mut value = &expr;
            while let syn::Expr::Group(e) = value {
                value = &e.expr;
            }
            let string = if let syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Str(lit_str),
                ..
            }) = value
            {
                let suffix = lit_str.suffix();
                if !suffix.is_empty() {
                    return Err(syn::Error::new_spanned(
                        lit_str,
                        format!("unexpected suffix `{suffix}` on string literal"),
                    ));
                }
                lit_str.clone()
            } else {
                return Err(syn::Error::new_spanned(
                    value,
                    format!(
                        "expected {EMBED_WITH} attribute to be a string: `{EMBED_WITH} = \"...\"`"
                    ),
                ));
            };

            string.parse()
        }

        let mut custom_func_path = None;

        self.parse_nested_meta(|meta| match function_path(&meta) {
            Ok(path) => {
                custom_func_path = Some(path);
                Ok(())
            }
            Err(e) => Err(e),
        })?;

        custom_func_path.ok_or_else(|| {
            syn::Error::new_spanned(
                self,
                format!("expected {EMBED_WITH} attribute: `{EMBED_WITH} = \"...\"`"),
            )
        })
    }
}

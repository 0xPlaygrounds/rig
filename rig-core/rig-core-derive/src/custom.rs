use quote::ToTokens;
use syn::{meta::ParseNestedMeta, spanned::Spanned, ExprPath};

use crate::EMBED;

const EMBED_WITH: &str = "embed_with";

pub(crate) trait CustomAttributeParser {
    // Determine if field is tagged with an #[embed(embed_with = "...")] attribute.
    fn is_custom(&self) -> syn::Result<bool>;

    // Get the "..." part of the #[embed(embed_with = "...")] attribute.
    // Ex: If attribute is tagged with #[embed(embed_with = "my_embed")], returns "my_embed".
    fn expand_tag(&self) -> syn::Result<syn::ExprPath>;
}

impl CustomAttributeParser for syn::Attribute {
    fn is_custom(&self) -> syn::Result<bool> {
        // Check that the attribute is a list.
        match &self.meta {
            syn::Meta::List(meta) => {
                if meta.tokens.is_empty() {
                    return Ok(false);
                }
            }
            _ => return Ok(false),
        };

        // Check the first attribute tag (the first "embed")
        if !self.path().is_ident(EMBED) {
            return Ok(false);
        }

        self.parse_nested_meta(|meta| {
            // Parse the meta attribute as an expression. Need this to compile.
            meta.value()?.parse::<syn::Expr>()?;

            if meta.path.is_ident(EMBED_WITH) {
                Ok(())
            } else {
                let path = meta.path.to_token_stream().to_string().replace(' ', "");
                Err(syn::Error::new_spanned(
                    meta.path,
                    format_args!("unknown embedding field attribute `{}`", path),
                ))
            }
        })?;

        Ok(true)
    }

    fn expand_tag(&self) -> syn::Result<syn::ExprPath> {
        let mut custom_func_path = None;

        self.parse_nested_meta(|meta| match function_path(&meta) {
            Ok(path) => {
                custom_func_path = Some(path);
                Ok(())
            }
            Err(e) => Err(e),
        })?;

        Ok(custom_func_path.unwrap())
    }
}

fn function_path(meta: &ParseNestedMeta<'_>) -> syn::Result<ExprPath> {
    // #[embed(embed_with = "...")]
    let expr = meta.value()?.parse::<syn::Expr>().unwrap();
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
            return Err(syn::Error::new(
                lit_str.span(),
                format!("unexpected suffix `{}` on string literal", suffix),
            ));
        }
        lit_str.clone()
    } else {
        return Err(syn::Error::new(
            value.span(),
            format!(
                "expected {} attribute to be a string: `{} = \"...\"`",
                EMBED_WITH, EMBED_WITH
            ),
        ));
    };

    string.parse()
}

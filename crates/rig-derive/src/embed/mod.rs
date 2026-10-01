//! `#[derive(Embed)]`: implement `Embed` for structs with `#[embed]` fields.

use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use syn::{DataStruct, Meta, parse_quote};

use crate::resolve::CrateRefs;

const EMBED: &str = "embed";
const EMBED_WITH: &str = "embed_with";

pub(crate) fn expand_derive_embedding(input: &mut syn::DeriveInput) -> syn::Result<TokenStream> {
    let refs = CrateRefs::resolve();
    let core = &refs.core;
    let embed_trait = quote!(#core::embeddings::embed::Embed);

    let syn::Data::Struct(data_struct) = &input.data else {
        return Err(syn::Error::new_spanned(
            input,
            "Embed derive macro should only be used on structs",
        ));
    };
    reject_conflicting_embed_attributes(data_struct)?;

    let mut basic_targets = Vec::new();
    let mut custom_targets = Vec::new();
    for (index, field) in data_struct.fields.iter().enumerate() {
        // The index belongs to the entire struct, before annotation filtering.
        let member = field.ident.clone().map_or_else(
            || syn::Member::Unnamed(syn::Index::from(index)),
            syn::Member::Named,
        );
        if field.attrs.iter().any(is_basic) {
            let field_type = &field.ty;
            input
                .generics
                .make_where_clause()
                .predicates
                .push(parse_quote! { #field_type: #embed_trait });
            basic_targets.push(quote! { self.#member });
        } else if let Some(path) = custom_function_path(field)? {
            custom_targets.push(quote! {
                #path(embedder, self.#member.clone())?;
            });
        }
    }

    let name = &input.ident;
    if basic_targets.is_empty() && custom_targets.is_empty() {
        return Err(syn::Error::new_spanned(
            name,
            "Add at least one field tagged with #[embed] or #[embed(embed_with = \"...\")].",
        ));
    }

    let target_stream = quote! {
        #(#embed_trait::embed(&#basic_targets, embedder)?;)*;
        #(#custom_targets)*;
    };
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let r#gen = quote! {
        impl #impl_generics #embed_trait for #name #ty_generics #where_clause {
            fn embed(&self, embedder: &mut #core::embeddings::embed::TextEmbedder) -> Result<(), #core::embeddings::embed::EmbedError> {
                #target_stream;

                Ok(())
            }
        }
    };

    Ok(r#gen)
}

/// Whether `attr` is a bare `#[embed]`.
fn is_basic(attr: &syn::Attribute) -> bool {
    matches!(&attr.meta, Meta::Path(path) if path.is_ident(EMBED))
}

/// Rejects mixed basic/custom annotations and repeated custom annotations.
fn reject_conflicting_embed_attributes(data_struct: &DataStruct) -> syn::Result<()> {
    for field in &data_struct.fields {
        let basic = field.attrs.iter().any(is_basic);
        let mut custom_attrs = field
            .attrs
            .iter()
            .filter(|attr| matches!(&attr.meta, Meta::List(_)) && attr.path().is_ident(EMBED));
        let custom = custom_attrs.next().is_some();
        if basic && custom {
            return Err(syn::Error::new_spanned(
                field,
                "a field cannot combine `#[embed]` and `#[embed(embed_with = \"...\")]`",
            ));
        }
        if let Some(duplicate) = custom_attrs.next() {
            return Err(syn::Error::new_spanned(
                duplicate,
                "a field cannot have more than one `#[embed(embed_with = \"...\")]` attribute",
            ));
        }
    }
    Ok(())
}

/// The function path of `field`'s `#[embed(embed_with = "...")]`, if it has
/// one. Unknown keys and values that are not a suffix-free string literal are
/// errors; of repeated `embed_with` keys, every value is checked and the last
/// one wins.
fn custom_function_path(field: &syn::Field) -> syn::Result<Option<syn::ExprPath>> {
    let Some(attr) = field.attrs.iter().find(|attr| {
        attr.path().is_ident(EMBED)
            && matches!(&attr.meta, Meta::List(list) if !list.tokens.is_empty())
    }) else {
        return Ok(None);
    };

    let mut values = Vec::new();
    attr.parse_nested_meta(|meta| {
        let value = meta.value()?.parse::<syn::Expr>()?;
        if !meta.path.is_ident(EMBED_WITH) {
            let path = meta.path.to_token_stream().to_string().replace(' ', "");
            return Err(syn::Error::new_spanned(
                meta.path,
                format_args!("unknown embedding field attribute `{path}`"),
            ));
        }
        values.push(value);
        Ok(())
    })?;

    // Every key is checked before any value, as the key errors take precedence.
    let mut paths = values
        .iter()
        .map(function_path)
        .collect::<syn::Result<Vec<_>>>()?;
    paths.pop().map(Some).ok_or_else(|| {
        syn::Error::new_spanned(
            attr,
            format!("expected {EMBED_WITH} attribute: `{EMBED_WITH} = \"...\"`"),
        )
    })
}

/// Parses an `embed_with` value, a string literal holding a function path.
fn function_path(expr: &syn::Expr) -> syn::Result<syn::ExprPath> {
    let mut value = expr;
    while let syn::Expr::Group(e) = value {
        value = &e.expr;
    }
    let syn::Expr::Lit(syn::ExprLit {
        lit: syn::Lit::Str(lit_str),
        ..
    }) = value
    else {
        return Err(syn::Error::new_spanned(
            value,
            format!("expected {EMBED_WITH} attribute to be a string: `{EMBED_WITH} = \"...\"`"),
        ));
    };
    let suffix = lit_str.suffix();
    if !suffix.is_empty() {
        return Err(syn::Error::new_spanned(
            lit_str,
            format!("unexpected suffix `{suffix}` on string literal"),
        ));
    }
    lit_str.parse()
}

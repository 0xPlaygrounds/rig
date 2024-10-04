extern crate proc_macro;
use indoc::indoc;
use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::{
    meta::ParseNestedMeta, parse_macro_input, spanned::Spanned, Attribute, DataStruct, DeriveInput, Meta, ExprPath
};

// https://doc.rust-lang.org/book/ch19-06-macros.html#how-to-write-a-custom-derive-macro
// https://doc.rust-lang.org/reference/procedural-macros.html

const EMBED: &str = "embed";
const EMBED_WITH: &str = "embed_with";

#[proc_macro_derive(Embedding, attributes(embed))]
pub fn derive_embed_trait(item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as DeriveInput);

    impl_embeddable_macro(&input)
}

fn impl_embeddable_macro(input: &syn::DeriveInput) -> TokenStream {
    let name = &input.ident;

    let embeddings = match &input.data {
        syn::Data::Struct(data_struct) => {
            // let invoke_trait = invoke_trait(data_struct)
            //     .map(|field_name| {
            //         quote! {
            //             self.#field_name.embeddable()
            //         }
            //     })
            //     .collect::<Vec<_>>();
            custom_trait_implementation(data_struct)
        }
        _ => Ok(false),
    }
    .unwrap();

    let gen = quote! {
        impl Embeddable for #name {
            type Kind = String;

            fn embeddable(&self) -> Vec<String> {
                // vec![
                //     #(#embeddings),*
                // ].into_iter().flatten().collect()
                println!("{}", #embeddings);
                vec![]
            }
        }
    };
    gen.into()
}

fn custom_trait_implementation(data_struct: &DataStruct) -> Result<bool, syn::Error> {
    let t = data_struct
        .fields
        .clone()
        .into_iter()
        .for_each(|field| {
            let _t = field.attrs.clone().into_iter().map(|attr| {
                let t = if attr.path().is_ident(EMBED) {
                    attr.parse_nested_meta(|meta| {
                        if meta.path.is_ident(EMBED_WITH) {
                            let path = parse_embed_with(&meta)?;

                            let tokens = meta.path.into_token_stream();
                        };
                        Ok(())
                    })
                } else {
                    todo!()
                };
            }).collect::<Vec<_>>();
        });
    Ok(false)
}

fn parse_embed_with(meta: &ParseNestedMeta) -> Result<ExprPath, syn::Error> {
    // #[embed(embed_with = "...")]
    let expr = meta.value().unwrap().parse::<syn::Expr>().unwrap();
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
                format!("unexpected suffix `{}` on string literal", suffix)
            ))
        }
        lit_str.clone()
    } else {
        return Err(syn::Error::new(
            value.span(),
            format!("expected {} attribute to be a string: `{} = \"...\"`", EMBED_WITH, EMBED_WITH)
        ))
    };

    string.parse()
}

fn invoke_trait(data_struct: &DataStruct) -> impl Iterator<Item = syn::Ident> {
    data_struct.fields.clone().into_iter().filter_map(|field| {
        let found_embed = field
            .attrs
            .clone()
            .into_iter()
            .any(|attribute| match attribute {
                Attribute {
                    meta: Meta::Path(path),
                    ..
                } => path.is_ident("embed"),
                _ => false,
            });
        match found_embed {
            true => Some(field.ident.expect("")),
            false => None,
        }
    })
}

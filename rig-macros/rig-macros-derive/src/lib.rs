extern crate proc_macro;
use indoc::indoc;
use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::{
    meta::ParseNestedMeta, parse_macro_input, parse_quote, spanned::Spanned, Attribute, DataStruct, DeriveInput, ExprPath, Meta, Path
};

// https://doc.rust-lang.org/book/ch19-06-macros.html#how-to-write-a-custom-derive-macro
// https://doc.rust-lang.org/reference/procedural-macros.html

const EMBED: &str = "embed";
const EMBED_WITH: &str = "embed_with";

#[proc_macro_derive(Embedding, attributes(embed))]
pub fn derive_embed_trait(item: TokenStream) -> TokenStream {
    let mut input = parse_macro_input!(item as DeriveInput);

    impl_embeddable_macro(&mut input)
}

fn impl_embeddable_macro(input: &mut syn::DeriveInput) -> TokenStream {
    let name = &input.ident;

    let embeddings = match &input.data {
        syn::Data::Struct(data_struct) => {
            basic_embed_fields(data_struct).for_each(|(_, field_type)| {
                add_struct_bounds(&mut input.generics, &field_type)
            });
            
            // let basic_embed_fields = basic_embed_fields(data_struct)
            //     .map(|field_name| {
            //         quote! {
            //             self.#field_name.embeddable()
            //         }
            //     })
            //     .collect::<Vec<_>>();

            let func_names = custom_trait_implementation(data_struct).unwrap();

            false
        }
        _ => false,
    };

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let gen = quote! {
        impl #impl_generics Embeddable for #name #ty_generics #where_clause {
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
    eprintln!("Generated code:\n{}", gen);

    gen.into()
}

fn custom_trait_implementation(data_struct: &DataStruct) -> Result<Vec<ExprPath>, syn::Error> {
    Ok(data_struct
        .fields
        .clone()
        .into_iter()
        .map(|field| {
            let mut path = None;
            field.attrs.clone().into_iter().map(|attr| {
                if attr.path().is_ident(EMBED) {
                    attr.parse_nested_meta(|meta| {
                        if meta.path.is_ident(EMBED_WITH) {
                            path = Some(parse_embed_with(&meta)?);

                            // let tokens = meta.path.into_token_stream();
                        };
                        Ok(())
                    })
                } else {
                    Ok(())
                }
            }).collect::<Result<Vec<_>,_>>()?;
            Ok::<_, syn::Error>(path)
        }).collect::<Result<Vec<_>,_>>()?
        .into_iter()
        .filter_map(|i| i)
        .collect())
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

fn add_struct_bounds(generics: &mut syn::Generics, field_type: &syn::Type) {
    let where_clause = generics.make_where_clause();

    where_clause.predicates.push(parse_quote! {
        #field_type: Embeddable
    });
}


fn basic_embed_fields(data_struct: &DataStruct) -> impl Iterator<Item = (syn::Ident, syn::Type)> {
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
            true => Some((
                field.ident.expect(""),
                field.ty
            )),
            false => None,
        }
    })
}

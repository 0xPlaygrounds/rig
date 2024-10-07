extern crate proc_macro;
use proc_macro::TokenStream;
use syn::{parse_macro_input, DeriveInput};

mod embedding;

// https://doc.rust-lang.org/book/ch19-06-macros.html#how-to-write-a-custom-derive-macro
// https://doc.rust-lang.org/reference/procedural-macros.html

#[proc_macro_derive(Embedding, attributes(embed))]
pub fn derive_embed_trait(item: TokenStream) -> TokenStream {
    let mut input = parse_macro_input!(item as DeriveInput);

    embedding::expand_derive_embedding(&mut input)
}

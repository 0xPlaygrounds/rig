use deluxe::{ParseAttributes, ParseMetaItem};
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use std::collections::HashMap;
use syn::{DeriveInput, parse_macro_input};

#[derive(ParseMetaItem, Default, ParseAttributes)]
#[deluxe(attributes(client))]
struct ClientAttr {
    pub features: Option<Vec<String>>,
}

pub fn provider_client(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let ident = &input.ident;
    let attrs = ClientAttr::parse_attributes(&input.attrs).unwrap();
    let features: Vec<String> = attrs.features.unwrap_or_default();

    struct FeatureInfo {
        as_trait_name: &'static str,
    }
    let known_features = HashMap::from([
        (
            "completion",
            FeatureInfo {
                as_trait_name: "AsCompletion",
            },
        ),
        (
            "transcription",
            FeatureInfo {
                as_trait_name: "AsTranscription",
            },
        ),
        (
            "embeddings",
            FeatureInfo {
                as_trait_name: "AsEmbeddings",
            },
        ),
        (
            "image_generation",
            FeatureInfo {
                as_trait_name: "AsImageGeneration",
            },
        ),
        (
            "audio_generation",
            FeatureInfo {
                as_trait_name: "AsAudioGeneration",
            },
        ),
    ]);

    let mut impls = Vec::new();
    for (flag, feat) in known_features {
        let as_trait_ident = format_ident!("{}", feat.as_trait_name);

        if !features.iter().any(|f| f == flag) {
            impls.push(quote! {
                impl rig::client::#as_trait_ident for #ident {}
            });
        }
    }

    let output = quote! {
        #(#impls)*
    };
    output.into()
}

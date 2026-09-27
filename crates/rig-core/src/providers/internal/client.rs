//! The client half every HTTP provider shares: a configuration on a
//! transport, and the models it builds.

/// Declare `$client`, the configuration `$config` on an erased HTTP
/// transport, with the transport-handling methods every HTTP provider
/// client has: `with_http`, `config`, and the configuration's `connect` and
/// `client`. Each provider writes its own constructors and capability
/// methods beside it, through the crate-private `model`.
macro_rules! http_client {
    ($(#[$meta:meta])* $client:ident, $config:ident) => {
        $(#[$meta])*
        #[derive(Clone, Debug)]
        pub struct $client {
            config: $config,
            http: $crate::http_client::DynHttpClient,
        }

        impl $client {
            /// The same configuration, sending through `http`.
            pub fn with_http(
                self,
                http: impl $crate::http_client::HttpClientExt + 'static,
            ) -> Self {
                Self {
                    http: $crate::http_client::DynHttpClient::new(http),
                    ..self
                }
            }

            /// The configuration this client sends with.
            pub fn config(&self) -> &$config {
                &self.config
            }

            /// `wire` on this client's transport.
            pub(crate) fn model<W>(&self, wire: W) -> $crate::driver::Model<W> {
                $crate::driver::Model::new(wire, self.http.clone())
            }
        }

        impl $config {
            /// A client that sends this configuration's requests through
            /// `http`.
            pub fn connect(self, http: impl $crate::http_client::HttpClientExt + 'static) -> $client {
                $client {
                    config: self,
                    http: $crate::http_client::DynHttpClient::new(http),
                }
            }

            /// A client that sends through the shared reqwest client
            /// (`rig_reqwest::shared()`), built once per process.
            #[cfg(feature = "reqwest")]
            #[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
            pub fn client(self) -> $client {
                self.connect(rig_reqwest::shared())
            }
        }
    };
}

pub(crate) use http_client;

/// Declare a vendor's by-name constructors for the OpenAI-shaped `$dialect`:
/// `$from_env` and `$new`, each an [`OpenAI`](crate::providers::openai::OpenAI)
/// client on the shared reqwest client.
macro_rules! openai_vendor {
    ($dialect:path, $name:literal) => {
        $crate::providers::internal::client::openai_vendor!($dialect, $name, from_env, new);
    };
    ($dialect:path, $name:literal, $from_env:ident, $new:ident) => {
        #[doc = concat!(
                                                    $name,
                                                    " from the environment variables [`",
                                                    stringify!($dialect),
                                                    "`] names, on the shared reqwest client."
                                                )]
        #[cfg(feature = "reqwest")]
        #[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
        pub fn $from_env()
        -> Result<$crate::providers::openai::OpenAI, $crate::client::env::EnvError> {
            Ok($crate::providers::openai::OpenAIConfig::from_env_with(&$dialect)?.client())
        }

        #[doc = concat!($name, " with `api_key`, on the shared reqwest client.")]
        #[cfg(feature = "reqwest")]
        #[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
        pub fn $new(api_key: impl Into<$crate::wire::Secret>) -> $crate::providers::openai::OpenAI {
            $crate::providers::openai::OpenAIConfig::with_key(&$dialect, api_key).client()
        }
    };
}

pub(crate) use openai_vendor;

/// Declare a vendor's by-name constructors for the Anthropic-format
/// `$dialect`: `$from_env` and `$new`, each an
/// [`Anthropic`](crate::providers::anthropic::Anthropic) client on the
/// shared reqwest client.
macro_rules! anthropic_vendor {
    ($dialect:path, $name:literal, $from_env:ident, $new:ident) => {
        #[doc = concat!(
                    $name,
                    "'s Messages-format endpoint from the environment variables [`",
                    stringify!($dialect),
                    "`] names, on the shared reqwest client."
                )]
        #[cfg(feature = "reqwest")]
        #[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
        pub fn $from_env()
        -> Result<$crate::providers::anthropic::Anthropic, $crate::client::env::EnvError> {
            Ok($crate::providers::anthropic::AnthropicConfig::from_env_with(&$dialect)?.client())
        }

        #[doc = concat!(
                    $name,
                    "'s Messages-format endpoint with `api_key`, on the shared reqwest client."
                )]
        #[cfg(feature = "reqwest")]
        #[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
        pub fn $new(
            api_key: impl Into<$crate::wire::Secret>,
        ) -> $crate::providers::anthropic::Anthropic {
            $crate::providers::anthropic::AnthropicConfig::with_dialect(api_key, &$dialect).client()
        }
    };
}

pub(crate) use anthropic_vendor;

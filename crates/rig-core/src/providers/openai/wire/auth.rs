//! How an OpenAI-shaped request carries its credential and the caller
//! identity a gateway requires.
//!
//! ```
//! use rig_core::providers::openai::{OpenAIConfig, wire::Auth};
//! let config = OpenAIConfig::new("key").with_auth(Auth::OptionalBearer);
//! assert_eq!(config.auth, Auth::OptionalBearer);
//! ```

use serde::{Deserialize, Serialize};

use crate::client::env::{self, EnvError};
use crate::wire::Secret;

use super::{Dialect, OpenAIConfig};

/// Credential header policy, including omission of empty optional tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Auth {
    /// `Authorization: Bearer <key>`.
    Bearer,
    /// `Authorization: Bearer <key>`, omitted entirely when the key is empty.
    OptionalBearer,
    /// Azure's `api-key: <key>`.
    ApiKeyHeader,
}

/// Alternative credential environment variable and its authentication policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AuthAlternative {
    /// The variable holding this credential.
    pub api_key_env: &'static str,
    /// How it is sent.
    pub auth: Auth,
}

/// The caller identity a gateway requires on every request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Identity {
    /// The `originator` header's default value.
    pub originator: &'static str,
    /// The environment variable overriding `originator`.
    pub originator_env: &'static str,
    /// The environment variable overriding `user-agent`.
    pub user_agent_env: &'static str,
    /// Whether every request carries a fresh `session_id` header.
    pub session_ids: bool,
}

/// The identity a gateway requires on every request, resolved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CallerIdentity {
    /// The `originator` header.
    pub originator: String,
    /// The `user-agent` header.
    pub user_agent: String,
}

/// The user agent a gateway that asks for one is told: the crate, the host,
/// and who is calling.
pub(super) fn default_user_agent(originator: &str) -> String {
    format!(
        "rig/{} ({} {}; {originator})",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH,
    )
}

impl OpenAIConfig {
    /// `dialect` with the credential it accepts through its
    /// [`alternate_auth`](Dialect::alternate_auth) variable, sent with that
    /// alternative's header.
    ///
    /// Azure's account key and its Entra bearer token are both credentials
    /// for the same account but go out under different headers, so which one
    /// is held has to be recorded rather than guessed from the value.
    pub fn with_alternate_key(dialect: &Dialect, api_key: impl Into<Secret>) -> Self {
        let auth = dialect
            .alternate_auth
            .map_or(dialect.quirks.auth, |alternative| alternative.auth);
        Self {
            auth,
            ..Self::with_key(dialect, api_key)
        }
    }

    /// The credential `dialect` reads from the environment, and how it is
    /// sent. Alternative credentials require their own header policy; the
    /// primary is preferred.
    pub(crate) fn credential_from_env(dialect: &Dialect) -> Result<(String, Auth), EnvError> {
        let quirks = &dialect.quirks;
        Ok(match dialect.alternate_auth {
            Some(alternative) => match env::optional(dialect.api_key_env)? {
                Some(api_key) => (api_key, quirks.auth),
                None => match env::optional(alternative.api_key_env)? {
                    Some(api_key) => (api_key, alternative.auth),
                    None => {
                        return Err(EnvError::Invalid {
                            name: dialect.api_key_env,
                            detail: format!(
                                "either `{}` or `{}` must be set",
                                dialect.api_key_env, alternative.api_key_env
                            ),
                        });
                    }
                },
            },
            None => (env::required(dialect.api_key_env)?, quirks.auth),
        })
    }

    /// Send the credential with `auth`'s header.
    pub fn with_auth(mut self, auth: Auth) -> Self {
        self.auth = auth;
        self
    }

    /// Name the account the credential belongs to (`ChatGPT-Account-Id`).
    pub fn with_account_id(mut self, account_id: impl Into<String>) -> Self {
        self.account_id = Some(account_id.into());
        self
    }

    /// Apply the dialect's authentication to a request builder.
    pub(crate) fn authenticate(&self, builder: http::request::Builder) -> http::request::Builder {
        match self.auth {
            Auth::Bearer => {
                builder.header("Authorization", format!("Bearer {}", self.api_key.expose()))
            }
            Auth::OptionalBearer if self.api_key.is_empty() => builder,
            Auth::OptionalBearer => {
                builder.header("Authorization", format!("Bearer {}", self.api_key.expose()))
            }
            Auth::ApiKeyHeader => builder.header("api-key", self.api_key.expose()),
        }
    }

    /// Apply authentication, configured identity, account, and per-request session headers.
    pub(crate) fn headers(&self, builder: http::request::Builder) -> http::request::Builder {
        let mut builder = self.authenticate(builder);
        if let Some(identity) = &self.identity {
            builder = builder
                .header("originator", &identity.originator)
                .header(http::header::USER_AGENT, &identity.user_agent);
        }
        if self
            .dialect
            .quirks
            .identity
            .is_some_and(|identity| identity.session_ids)
        {
            // Session identity must be fresh for each request.
            builder = builder.header("session_id", crate::providers::chatgpt::session_id());
        }
        if let Some(account_id) = &self.account_id {
            builder = builder.header("ChatGPT-Account-Id", account_id);
        }
        builder
    }
}

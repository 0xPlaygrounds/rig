//! The domain policy of `RIG_HOME/fetch.toml`, with this session's
//! answers, and the rule that keeps fetches off internal addresses.

use std::net::IpAddr;
use std::path::Path;

use serde::Deserialize;

use crate::web_fetch::FetchDomains;

/// The file under `RIG_HOME` that holds the policy.
pub const CONFIG: &str = "fetch.toml";

/// `RIG_HOME/fetch.toml`: `allow = ["docs.rs", "*.github.com"]`,
/// `deny = ["*.example"]`. A pattern's `*` stands for any run of
/// characters, so `*.github.com` names every subdomain of `github.com`
/// but not `github.com` itself. Deny wins over allow. A loopback,
/// link-local or private address is reached only when its host is on
/// `allow` exactly, without a wildcard.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct FetchConfig {
    /// Hosts fetched without asking.
    pub allow: Vec<String>,
    /// Hosts never fetched.
    pub deny: Vec<String>,
}

/// What the policy says of a host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Fetch it.
    Allowed,
    /// Refuse it, and why.
    Denied(String),
    /// Ask the user, and why.
    Ask(String),
}

impl FetchConfig {
    /// The config at `path`: the default when there is none, and why when
    /// it does not read.
    pub fn read(path: &Path) -> (Self, Option<String>) {
        let Ok(text) = std::fs::read_to_string(path) else {
            return (Self::default(), None);
        };
        match toml::from_str(&text) {
            Ok(config) => (config, None),
            Err(failure) => {
                let why = format!("{}: {failure}; every domain is asked about", path.display());
                (Self::default(), Some(why))
            }
        }
    }

    /// Whether `allow` names `host` exactly.
    pub fn names_exactly(&self, host: &str) -> bool {
        self.allow
            .iter()
            .any(|pattern| pattern.eq_ignore_ascii_case(host))
    }

    /// What the policy says of `host`, with this session's answers.
    pub(crate) fn verdict(&self, host: &str, session: &FetchDomains) -> Verdict {
        let any = |patterns: &[String]| patterns.iter().any(|pattern| matches(pattern, host));
        if any(&self.deny) {
            return Verdict::Denied(format!("{host} is on the deny list of fetch.toml"));
        }
        if names_internal(host) && !self.names_exactly(host) {
            return Verdict::Denied(format!(
                "{host} is a loopback, link-local or private address; only a host on the allow \
                 list of fetch.toml by its exact name is reached there"
            ));
        }
        if any(&self.allow) || session.allowed.iter().any(|allowed| allowed == host) {
            return Verdict::Allowed;
        }
        if session.denied.iter().any(|denied| denied == host) {
            return Verdict::Denied(format!("the user denied {host} for this session"));
        }
        Verdict::Ask(format!("{host} is on neither list of fetch.toml"))
    }
}

/// Whether `host` names an internal address by itself: an IP address, or
/// `localhost`.
fn names_internal(host: &str) -> bool {
    host == "localhost"
        || host.ends_with(".localhost")
        || host.parse::<IpAddr>().is_ok_and(is_internal)
}

/// Whether `ip` is loopback, link-local, private, shared or unspecified.
pub(crate) fn is_internal(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let [first, second, ..] = v4.octets();
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                || (first == 100 && (64..128).contains(&second))
        }
        IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_unique_local()
                || v6.is_unicast_link_local()
                || v6
                    .to_ipv4_mapped()
                    .is_some_and(|v4| is_internal(IpAddr::V4(v4)))
        }
    }
}

/// Whether `host` fits `pattern`, whose `*` stands for any run of
/// characters, ignoring case.
fn matches(pattern: &str, host: &str) -> bool {
    let pattern = pattern.to_ascii_lowercase();
    let mut parts = pattern.split('*');
    let Some(first) = parts.next() else {
        return false;
    };
    let Some(mut rest) = host.strip_prefix(first) else {
        return false;
    };
    let parts: Vec<&str> = parts.collect();
    let Some((last, middle)) = parts.split_last() else {
        return rest.is_empty();
    };
    for part in middle {
        let Some(at) = rest.find(part) else {
            return false;
        };
        rest = rest.get(at + part.len()..).unwrap_or_default();
    }
    rest.ends_with(last)
}

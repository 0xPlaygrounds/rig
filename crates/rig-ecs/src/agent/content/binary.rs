//! Binary payloads shared by content hash, with source spelling retained per use.
//!
//! ```
//! use rig_ecs::agent::content::binary::BinaryAssets;
//! let mut assets = BinaryAssets::default();
//! let id = assets.insert(vec![1, 2, 3])?;
//! assert_eq!(assets.get(id)?, &[1, 2, 3]);
//! # Ok::<(), rig_ecs::agent::content::binary::BinaryError>(())
//! ```

use bevy_reflect::Reflect;
use std::collections::{BTreeMap, BTreeSet};

use base64::{
    Engine, alphabet,
    engine::{DecodePaddingMode, GeneralPurpose, GeneralPurposeConfig},
    prelude::BASE64_STANDARD,
};
use bevy_ecs::prelude::*;
use rig_core::message::DocumentSourceKind;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// SHA-256 of the decoded payload bytes, independent of source and media type.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Reflect,
)]
#[serde(try_from = "String", into = "String")]
pub struct BinaryId(pub [u8; 32]);

impl BinaryId {
    /// Compute the content identity without interpreting its media type.
    pub fn of(bytes: &[u8]) -> Self {
        Self(Sha256::digest(bytes).into())
    }
}

/// Limits checked before decoding or retaining an additional binary payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinaryLimits {
    /// Maximum decoded bytes in one asset.
    pub per_asset: usize,
    /// Maximum retained decoded bytes in the store.
    pub total: usize,
    /// Maximum distinct assets.
    pub count: usize,
}

impl Default for BinaryLimits {
    fn default() -> Self {
        Self {
            per_asset: 64 * 1024 * 1024,
            total: 256 * 1024 * 1024,
            count: 65_536,
        }
    }
}

/// Why a binary source or persisted asset cannot be used. Diagnostics contain no payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
pub enum BinaryError {
    /// A source exceeds a configured allocation bound.
    #[error("binary content exceeds its allocation limit")]
    Limit,
    /// The source is not standard-alphabet base64.
    #[error("invalid base64 binary source")]
    Base64,
    /// A handle has no payload in the store.
    #[error("unresolved binary asset")]
    Missing,
    /// Persisted bytes do not match their claimed SHA-256 identity.
    #[error("binary asset hash mismatch")]
    Corrupt,
    /// Persisted spelling metadata cannot describe these bytes.
    #[error("invalid base64 spelling metadata")]
    Spelling,
}

/// Representation at a particular use of a shared binary payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Reflect)]
pub enum BinaryEncoding {
    /// A raw byte vector in the transport DTO.
    Raw,
    /// Standard-alphabet base64. Preserve omitted padding and nonzero trailing
    /// pad bits without retaining a second copy of the entire encoded payload.
    Base64 {
        /// The original number of `=` characters (0, 1 or 2).
        padding: u8,
        /// The original last non-padding symbol, when it differs from canonical.
        last_symbol: Option<u8>,
    },
}

/// A part's source. URLs, file IDs, string data and unknown sources remain data;
/// only raw and base64 binary payloads refer to the store.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Reflect)]
pub enum PartSource {
    /// External URL or URI; never fetched by scene loading.
    Url(String),
    /// Provider file identifier.
    FileId(String),
    /// Literal string source, not interpreted as binary.
    String(String),
    /// Explicit unknown source.
    Unknown,
    /// Shared binary bytes and this occurrence's transport representation.
    Binary {
        id: BinaryId,
        encoding: BinaryEncoding,
    },
}

/// One world's binary payload store. Explicit collection keeps only caller-supplied
/// reachable handles; graph ownership, not a run's lifetime, determines reachability.
/// Host pins must be included along with graph references when collecting.
#[derive(Resource, Debug, Default)]
pub struct BinaryAssets {
    payloads: BTreeMap<BinaryId, Vec<u8>>,
    // Hash the encoded spelling to avoid decoding repeated sources. This index
    // retains only hashes, never a duplicate base64 payload, and is not persisted.
    spellings: BTreeMap<[u8; 32], BinaryId>,
    bytes: usize,
    limits: BinaryLimits,
}

impl BinaryAssets {
    /// Empty store with host-selected allocation limits.
    pub fn with_limits(limits: BinaryLimits) -> Self {
        Self {
            limits,
            ..Self::default()
        }
    }

    /// Retained decoded bytes, excluding map and index overhead.
    pub fn byte_len(&self) -> usize {
        self.bytes
    }

    /// Distinct retained payloads.
    pub fn len(&self) -> usize {
        self.payloads.len()
    }

    /// Whether no payload is retained.
    pub fn is_empty(&self) -> bool {
        self.payloads.is_empty()
    }

    /// Borrow an asset's bytes, or return [`BinaryError::Missing`].
    pub fn get(&self, id: BinaryId) -> Result<&[u8], BinaryError> {
        self.payloads
            .get(&id)
            .map(Vec::as_slice)
            .ok_or(BinaryError::Missing)
    }

    /// Retain bytes once, deduplicated by SHA-256. Return a limit error when
    /// allocation bounds are exceeded or a corruption error on a hash collision.
    pub fn insert(&mut self, bytes: Vec<u8>) -> Result<BinaryId, BinaryError> {
        if bytes.len() > self.limits.per_asset {
            return Err(BinaryError::Limit);
        }
        let id = BinaryId::of(&bytes);
        if let Some(existing) = self.payloads.get(&id) {
            return if *existing == bytes {
                Ok(id)
            } else {
                Err(BinaryError::Corrupt)
            };
        }
        let total = self
            .bytes
            .checked_add(bytes.len())
            .ok_or(BinaryError::Limit)?;
        if total > self.limits.total || self.payloads.len() >= self.limits.count {
            return Err(BinaryError::Limit);
        }
        self.bytes = total;
        self.payloads.insert(id, bytes);
        Ok(id)
    }

    /// Convert a transport source to graph data. Repeated encoded strings are
    /// decoded once while their cached spelling and asset remain retained.
    /// Returns errors for invalid base64, allocation limits, or inconsistent assets.
    pub fn intern(&mut self, source: DocumentSourceKind) -> Result<PartSource, BinaryError> {
        Ok(match source {
            DocumentSourceKind::Url(value) => PartSource::Url(value),
            DocumentSourceKind::FileId(value) => PartSource::FileId(value),
            DocumentSourceKind::String(value) => PartSource::String(value),
            DocumentSourceKind::Unknown => PartSource::Unknown,
            DocumentSourceKind::Raw(bytes) => PartSource::Binary {
                id: self.insert(bytes)?,
                encoding: BinaryEncoding::Raw,
            },
            DocumentSourceKind::Base64(value) => {
                let max_encoded = self
                    .limits
                    .per_asset
                    .checked_add(2)
                    .and_then(|n| n.checked_div(3))
                    .and_then(|n| n.checked_mul(4))
                    .ok_or(BinaryError::Limit)?;
                if value.len() > max_encoded {
                    return Err(BinaryError::Limit);
                }
                let key: [u8; 32] = Sha256::digest(value.as_bytes()).into();
                let id = if let Some(id) = self.spellings.get(&key) {
                    *id
                } else {
                    let decoder = GeneralPurpose::new(
                        &alphabet::STANDARD,
                        GeneralPurposeConfig::new()
                            .with_decode_padding_mode(DecodePaddingMode::Indifferent)
                            .with_decode_allow_trailing_bits(true),
                    );
                    let bytes = decoder.decode(&value).map_err(|_| BinaryError::Base64)?;
                    let id = self.insert(bytes)?;
                    // Bound alternative spellings as well as assets. Discarding an
                    // index entry costs a future decode, never changes semantics.
                    if self.spellings.len() < self.limits.count {
                        self.spellings.insert(key, id);
                    }
                    id
                };
                let canonical = BASE64_STANDARD.encode(self.get(id)?);
                let bare = value.trim_end_matches('=');
                let padding =
                    u8::try_from(value.len() - bare.len()).map_err(|_| BinaryError::Base64)?;
                let last_symbol = (bare.as_bytes().last()
                    != canonical.trim_end_matches('=').as_bytes().last())
                .then(|| bare.as_bytes().last().copied())
                .flatten();
                PartSource::Binary {
                    id,
                    encoding: BinaryEncoding::Base64 {
                        padding,
                        last_symbol,
                    },
                }
            }
        })
    }

    /// Validate a reference and calculate its transport size without allocating.
    pub(crate) fn resolved_len(&self, source: &PartSource) -> Result<usize, BinaryError> {
        let PartSource::Binary { id, encoding } = source else {
            return Ok(match source {
                PartSource::Url(value) | PartSource::FileId(value) | PartSource::String(value) => {
                    value.len()
                }
                _ => 0,
            });
        };
        let bytes = self.get(*id)?;
        let BinaryEncoding::Base64 {
            padding,
            last_symbol,
        } = encoding
        else {
            return Ok(bytes.len());
        };
        let remainder = bytes.len() % 3;
        let required_padding = (3 - remainder) % 3;
        if usize::from(*padding) > required_padding {
            return Err(BinaryError::Spelling);
        }
        if let Some(symbol) = last_symbol {
            let last = bytes.last().ok_or(BinaryError::Spelling)?;
            let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            let actual = alphabet
                .iter()
                .position(|value| value == symbol)
                .ok_or(BinaryError::Spelling)?;
            let (canonical, unused) = match remainder {
                1 => (usize::from(last & 3) << 4, 4),
                2 => (usize::from(last & 15) << 2, 2),
                _ => (usize::from(last & 63), 0),
            };
            if actual >> unused != canonical >> unused {
                return Err(BinaryError::Spelling);
            }
        }
        (bytes.len() / 3)
            .checked_mul(4)
            .and_then(|size| size.checked_add(if remainder == 0 { 0 } else { remainder + 1 }))
            .and_then(|size| size.checked_add(usize::from(*padding)))
            .ok_or(BinaryError::Limit)
    }

    /// Rebuild the transport source with its saved spelling, returning an error
    /// for missing bytes, invalid spelling metadata, or size overflow.
    pub fn resolve(&self, source: &PartSource) -> Result<DocumentSourceKind, BinaryError> {
        self.resolved_len(source)?;
        Ok(match source {
            PartSource::Url(value) => DocumentSourceKind::Url(value.clone()),
            PartSource::FileId(value) => DocumentSourceKind::FileId(value.clone()),
            PartSource::String(value) => DocumentSourceKind::String(value.clone()),
            PartSource::Unknown => DocumentSourceKind::Unknown,
            PartSource::Binary { id, encoding } => {
                let bytes = self.get(*id)?;
                match encoding {
                    BinaryEncoding::Raw => DocumentSourceKind::Raw(bytes.to_vec()),
                    BinaryEncoding::Base64 {
                        padding,
                        last_symbol,
                    } => {
                        let canonical = BASE64_STANDARD.encode(bytes);
                        let required_padding =
                            canonical.len() - canonical.trim_end_matches('=').len();
                        if usize::from(*padding) > required_padding {
                            return Err(BinaryError::Spelling);
                        }
                        let mut value = canonical.trim_end_matches('=').as_bytes().to_vec();
                        if let Some(symbol) = last_symbol {
                            let last = value.last_mut().ok_or(BinaryError::Spelling)?;
                            // Only unused trailing bits may differ. Compare alphabet
                            // indices, not ASCII code points.
                            let alphabet =
                                b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
                            let before = alphabet
                                .iter()
                                .position(|b| b == last)
                                .ok_or(BinaryError::Spelling)?;
                            let after = alphabet
                                .iter()
                                .position(|b| b == symbol)
                                .ok_or(BinaryError::Spelling)?;
                            let unused = match bytes.len() % 3 {
                                1 => 4,
                                2 => 2,
                                _ => 0,
                            };
                            if (before >> unused) != (after >> unused) {
                                return Err(BinaryError::Spelling);
                            }
                            *last = *symbol;
                        }
                        value.extend(std::iter::repeat_n(b'=', usize::from(*padding)));
                        DocumentSourceKind::Base64(
                            String::from_utf8(value).map_err(|_| BinaryError::Spelling)?,
                        )
                    }
                }
            }
        })
    }

    /// Drop assets unreachable from all graph references and explicit host pins.
    /// Missing roots are an error and leave the store unchanged.
    pub fn retain(&mut self, roots: impl IntoIterator<Item = BinaryId>) -> Result<(), BinaryError> {
        let roots: BTreeSet<_> = roots.into_iter().collect();
        if roots.iter().any(|id| !self.payloads.contains_key(id)) {
            return Err(BinaryError::Missing);
        }
        self.payloads.retain(|id, _| roots.contains(id));
        self.spellings.retain(|_, id| roots.contains(id));
        self.bytes = self.payloads.values().map(Vec::len).sum();
        Ok(())
    }

    /// Assets in deterministic content-hash order for scene construction.
    pub fn iter(&self) -> impl Iterator<Item = (BinaryId, &[u8])> {
        self.payloads
            .iter()
            .map(|(id, bytes)| (*id, bytes.as_slice()))
    }

    /// Validate all saved identities and limits in an isolated store. The caller
    /// installs it only after graph handles and the rest of the scene validate.
    /// Returns an error for incorrect hashes, duplicate IDs, or exceeded limits.
    pub fn from_payloads(
        payloads: impl IntoIterator<Item = (BinaryId, Vec<u8>)>,
        limits: BinaryLimits,
    ) -> Result<Self, BinaryError> {
        let mut assets = Self::with_limits(limits);
        for (id, bytes) in payloads {
            if bytes.len() > limits.per_asset {
                return Err(BinaryError::Limit);
            }
            if BinaryId::of(&bytes) != id {
                return Err(BinaryError::Corrupt);
            }
            if assets.payloads.contains_key(&id) {
                return Err(BinaryError::Corrupt);
            }
            assets.insert(bytes)?;
        }
        Ok(assets)
    }
}

/// A single scene payload. Canonical base64 stores the bytes once; graph uses
/// retain their own raw/base64 representation in BinaryEncoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryRecord {
    /// SHA-256 of decoded bytes.
    pub id: BinaryId,
    /// Canonical standard-alphabet base64.
    pub data: String,
}

impl BinaryAssets {
    /// Snapshot the retained store in content-hash order.
    pub fn snapshot(&self) -> Vec<BinaryRecord> {
        self.iter()
            .map(|(id, bytes)| BinaryRecord {
                id,
                data: BASE64_STANDARD.encode(bytes),
            })
            .collect()
    }

    /// Build an isolated merged store using the destination's limits. Invalid
    /// hashes, duplicate saved IDs, invalid base64, or allocation excess return an
    /// error without mutating the destination.
    pub fn merged(&self, records: &[BinaryRecord]) -> Result<Self, BinaryError> {
        if records.len() > self.limits.count {
            return Err(BinaryError::Limit);
        }
        let mut merged = Self::from_payloads(
            self.iter().map(|(id, bytes)| (id, bytes.to_vec())),
            self.limits,
        )?;
        let mut seen = BTreeSet::new();
        for record in records {
            let max_encoded = self
                .limits
                .per_asset
                .checked_add(2)
                .and_then(|n| n.checked_div(3))
                .and_then(|n| n.checked_mul(4))
                .ok_or(BinaryError::Limit)?;
            if record.data.len() > max_encoded {
                return Err(BinaryError::Limit);
            }
            if !seen.insert(record.id) {
                return Err(BinaryError::Corrupt);
            }
            let source = merged.intern(DocumentSourceKind::Base64(record.data.clone()))?;
            if !matches!(source, PartSource::Binary { id, .. } if id == record.id) {
                return Err(BinaryError::Corrupt);
            }
        }
        Ok(merged)
    }
}

impl From<BinaryId> for String {
    fn from(id: BinaryId) -> Self {
        let mut value = String::with_capacity(64);
        for byte in id.0 {
            for half in [byte >> 4, byte & 15] {
                value.push(char::from(if half < 10 {
                    b'0' + half
                } else {
                    b'a' + half - 10
                }));
            }
        }
        value
    }
}

impl TryFrom<String> for BinaryId {
    type Error = BinaryError;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(BinaryError::Corrupt);
        }
        let mut id = [0; 32];
        for (position, byte) in id.iter_mut().enumerate() {
            let digits = value
                .get(position * 2..position * 2 + 2)
                .ok_or(BinaryError::Corrupt)?;
            *byte = u8::from_str_radix(digits, 16).map_err(|_| BinaryError::Corrupt)?;
        }
        Ok(Self(id))
    }
}

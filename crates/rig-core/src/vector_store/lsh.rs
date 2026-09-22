//! Random-projection hashing for approximate vector-search candidates.
//!
//! ```
//! use rig_core::vector_store::lsh::LSHIndex;
//!
//! let mut index = LSHIndex::new(2, 4, 8);
//! index.insert("document", &[1.0, 0.0]);
//! assert_eq!(index.query(&[1.0, 0.0]), vec!["document"]);
//! ```

use fastrand::Rng;
use std::collections::HashMap;

#[cfg(test)]
fn lsh_rng() -> Rng {
    Rng::with_seed(0x5eed_fade_cafe_beef)
}

#[cfg(not(test))]
fn lsh_rng() -> Rng {
    Rng::new()
}

/// Locality Sensitive Hashing (LSH) with random projection.
/// Uses random hyperplanes to hash similar vectors into the same buckets for efficient
/// approximate nearest neighbor search. See <https://www.pinecone.io/learn/series/faiss/locality-sensitive-hashing-random-projection/>
/// for details on how LSH works.
#[derive(Clone, Default)]
pub struct LSH {
    hyperplanes: Vec<Vec<f32>>,
    num_tables: usize,
    num_hyperplanes: usize,
}

impl LSH {
    /// Creates random normalized projection vectors. Use at most 64 hyperplanes
    /// per table and ensure `num_tables * num_hyperplanes` fits in `usize`.
    pub fn new(dim: usize, num_tables: usize, num_hyperplanes: usize) -> Self {
        let mut rng = lsh_rng();
        let mut hyperplanes = Vec::new();

        for _ in 0..(num_tables * num_hyperplanes) {
            let mut plane = vec![0.0; dim];

            for val in plane.iter_mut() {
                *val = rng.f32() * 2.0 - 1.0;
            }

            let norm: f32 = plane.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for val in plane.iter_mut() {
                    *val /= norm;
                }
            }

            hyperplanes.push(plane);
        }

        Self {
            hyperplanes,
            num_tables,
            num_hyperplanes,
        }
    }

    /// Computes sign bits using f32 projections. Supply a valid table index
    /// and a vector matching the configured dimension; lengths are not checked.
    pub fn hash(&self, vector: &[f64], table_idx: usize) -> u64 {
        let mut hash = 0u64;
        let start = table_idx * self.num_hyperplanes;

        for (i, hyperplane) in self
            .hyperplanes
            .get(start..start + self.num_hyperplanes)
            .unwrap_or(&[])
            .iter()
            .enumerate()
        {
            let dot: f32 = vector
                .iter()
                .zip(hyperplane.iter())
                .map(|(v, h)| (*v as f32) * h)
                .sum();

            if dot >= 0.0 {
                hash |= 1u64 << i;
            }
        }

        hash
    }
}

/// LSH Index for document IDs.
/// Stores document IDs in a hashmap of hash values to document IDs.
/// This allows for efficient lookup of document IDs by hash value.
#[derive(Clone, Default)]
pub struct LSHIndex {
    lsh: LSH,
    tables: Vec<HashMap<u64, Vec<String>>>, // Hash -> document IDs
}

impl LSHIndex {
    /// Create a new LSHIndex.
    pub fn new(dim: usize, num_tables: usize, num_hyperplanes: usize) -> Self {
        let lsh = LSH::new(dim, num_tables, num_hyperplanes);
        let tables = vec![HashMap::new(); num_tables];

        Self { lsh, tables }
    }

    /// Insert a document ID with its embedding
    pub fn insert(&mut self, id: &str, embedding: &[f64]) {
        for table_idx in 0..self.lsh.num_tables {
            let hash = self.lsh.hash(embedding, table_idx);
            if let Some(table) = self.tables.get_mut(table_idx) {
                table.entry(hash).or_default().push(id.to_owned());
            }
        }
    }

    /// Returns unique IDs sharing a bucket in any table, in unspecified order.
    pub fn query(&self, embedding: &[f64]) -> Vec<String> {
        use std::collections::HashSet;

        let mut candidates = HashSet::new();

        for table_idx in 0..self.lsh.num_tables {
            let hash = self.lsh.hash(embedding, table_idx);

            if let Some(ids) = self
                .tables
                .get(table_idx)
                .and_then(|table| table.get(&hash))
            {
                candidates.extend(ids.iter().cloned());
            }
        }

        candidates.into_iter().collect()
    }

    /// Clear all tables
    pub fn clear(&mut self) {
        for table in self.tables.iter_mut() {
            table.clear();
        }
    }
}

//! Random-projection hashing for approximate vector-search candidates.
//!
//! ```
//! use rig_core::vector_store::lsh::LSHIndex;
//!
//! let mut index = LSHIndex::new(2, 4, 8);
//! index.insert("document", &[1.0, 0.0]);
//! assert_eq!(index.query(&[1.0, 0.0]), vec!["document"]);
//! ```

use std::collections::{HashMap, HashSet};

/// The hyperplanes' seed. A fixed seed makes an index, and so its candidates,
/// the same on every run.
const SEED: u64 = 0x5eed_fade_cafe_beef;

/// Locality-sensitive hashing by random projection: each table hashes a vector
/// to the side of each of its random hyperplanes it lies on, so vectors at a
/// small angle tend to share a bucket. See
/// <https://www.pinecone.io/learn/series/faiss/locality-sensitive-hashing-random-projection/>.
#[derive(Clone, Debug, Default)]
pub struct LSH {
    /// Every table's hyperplanes, one after another, each `dim` wide.
    hyperplanes: Vec<f32>,
    dim: usize,
    num_hyperplanes: usize,
}

impl LSH {
    /// Random unit hyperplanes for `num_tables` tables of `num_hyperplanes`
    /// each, `dim` wide. A table uses at most 64 hyperplanes; more are
    /// ignored.
    pub fn new(dim: usize, num_tables: usize, num_hyperplanes: usize) -> Self {
        let num_hyperplanes = num_hyperplanes.min(64);
        let mut rng = fastrand::Rng::with_seed(SEED);
        let mut hyperplanes = Vec::new();
        for _ in 0..num_tables.saturating_mul(num_hyperplanes) {
            let plane: Vec<f32> = (0..dim).map(|_| rng.f32() * 2.0 - 1.0).collect();
            let norm = plane.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm = if norm > 0.0 { norm } else { 1.0 };
            hyperplanes.extend(plane.iter().map(|x| x / norm));
        }
        Self {
            hyperplanes,
            dim,
            num_hyperplanes,
        }
    }

    /// The bucket `vector` falls in, in table `table_idx`: one bit per
    /// hyperplane, set when the vector is on its positive side. A vector of
    /// another width is compared over the shared prefix; a table index out of
    /// range hashes everything to `0`.
    pub fn hash(&self, vector: &[f64], table_idx: usize) -> u64 {
        let table = self.dim.saturating_mul(self.num_hyperplanes);
        let planes = table_idx
            .checked_mul(table)
            .and_then(|start| self.hyperplanes.get(start..start.checked_add(table)?))
            .unwrap_or_default();
        // `max(1)` keeps a zero-width index from asking for empty chunks.
        planes
            .chunks(self.dim.max(1))
            .enumerate()
            .fold(0, |hash, (bit, plane)| {
                let dot: f32 = vector.iter().zip(plane).map(|(v, h)| *v as f32 * h).sum();
                if dot >= 0.0 { hash | 1 << bit } else { hash }
            })
    }
}

/// Document IDs by bucket, one table of buckets per [`LSH`] table.
#[derive(Clone, Debug, Default)]
pub struct LSHIndex {
    lsh: LSH,
    tables: Vec<HashMap<u64, Vec<String>>>,
}

impl LSHIndex {
    /// An empty index for `dim`-wide vectors. More tables find more true
    /// neighbors; more hyperplanes per table make smaller buckets, so fewer
    /// candidates to score. Zero tables make one bucket holding every ID, so
    /// a query returns them all, as brute force would score them all.
    pub fn new(dim: usize, num_tables: usize, num_hyperplanes: usize) -> Self {
        // One table without hyperplanes hashes every vector to the same bucket.
        let (num_tables, num_hyperplanes) = match num_tables {
            0 => (1, 0),
            _ => (num_tables, num_hyperplanes),
        };
        Self {
            lsh: LSH::new(dim, num_tables, num_hyperplanes),
            tables: vec![HashMap::new(); num_tables],
        }
    }

    /// Adds `id` to the bucket `embedding` falls in, in every table.
    pub fn insert(&mut self, id: &str, embedding: &[f64]) {
        for (table_idx, table) in self.tables.iter_mut().enumerate() {
            let hash = self.lsh.hash(embedding, table_idx);
            table.entry(hash).or_default().push(id.to_owned());
        }
    }

    /// Removes `id` from the buckets `embedding` falls in, undoing
    /// [`Self::insert`] of the same pair.
    pub fn remove(&mut self, id: &str, embedding: &[f64]) {
        for (table_idx, table) in self.tables.iter_mut().enumerate() {
            let hash = self.lsh.hash(embedding, table_idx);
            if let Some(ids) = table.get_mut(&hash) {
                if let Some(at) = ids.iter().position(|stored| stored == id) {
                    ids.swap_remove(at);
                }
                if ids.is_empty() {
                    table.remove(&hash);
                }
            }
        }
    }

    /// Each ID sharing a bucket with `embedding` in any table, once, in
    /// unspecified order.
    pub fn query(&self, embedding: &[f64]) -> Vec<&str> {
        let mut candidates = HashSet::new();
        for (table_idx, table) in self.tables.iter().enumerate() {
            if let Some(ids) = table.get(&self.lsh.hash(embedding, table_idx)) {
                candidates.extend(ids.iter().map(String::as_str));
            }
        }
        candidates.into_iter().collect()
    }

    /// Removes every ID.
    pub fn clear(&mut self) {
        self.tables.iter_mut().for_each(HashMap::clear);
    }
}

#[cfg(test)]
mod tests;

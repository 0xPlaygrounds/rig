use fastrand::Rng;
use std::collections::HashMap;

/// Random Projection LSH
#[derive(Clone, Default)]
pub struct LSH {
    hyperplanes: Vec<Vec<f32>>,
    num_tables: usize,
    num_hyperplanes: usize,
}

impl LSH {
    pub fn new(dim: usize, num_tables: usize, num_hyperplanes: usize) -> Self {
        let mut rng = Rng::new();
        let mut hyperplanes = Vec::new();

        // Generate random hyperplanes for all tables
        for _ in 0..(num_tables * num_hyperplanes) {
            let mut plane = vec![0.0; dim];

            // Generate random values in [-1, 1]
            for val in plane.iter_mut() {
                *val = rng.f32() * 2.0 - 1.0;
            }

            // Normalize to unit vector
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

    /// Compute hash for a vector in a specific table
    pub fn hash(&self, vector: &[f64], table_idx: usize) -> u64 {
        let mut hash = 0u64;
        let start = table_idx * self.num_hyperplanes;

        for (i, hyperplane) in self.hyperplanes[start..start + self.num_hyperplanes]
            .iter()
            .enumerate()
        {
            // Dot product (convert f64 to f32)
            let dot: f32 = vector
                .iter()
                .zip(hyperplane.iter())
                .map(|(v, h)| (*v as f32) * h)
                .sum();

            // Set bit if positive
            if dot >= 0.0 {
                hash |= 1u64 << i;
            }
        }

        hash
    }
}

/// LSH Index for document IDs
#[derive(Clone, Default)]
pub struct LSHIndex {
    lsh: LSH,
    tables: Vec<HashMap<u64, Vec<String>>>, // Hash -> document IDs
}

impl LSHIndex {
    pub fn new(dim: usize, num_tables: usize, num_hyperplanes: usize) -> Self {
        let lsh = LSH::new(dim, num_tables, num_hyperplanes);
        let tables = vec![HashMap::new(); num_tables];

        Self { lsh, tables }
    }

    /// Insert a document ID with its embedding
    pub fn insert(&mut self, id: String, embedding: &[f64]) {
        for table_idx in 0..self.lsh.num_tables {
            let hash = self.lsh.hash(embedding, table_idx);
            self.tables[table_idx]
                .entry(hash)
                .or_default()
                .push(id.clone());
        }
    }

    /// Query for candidate document IDs
    pub fn query(&self, embedding: &[f64]) -> Vec<String> {
        use std::collections::HashSet;

        let mut candidates = HashSet::new();

        // Collect candidates from all tables
        for table_idx in 0..self.lsh.num_tables {
            let hash = self.lsh.hash(embedding, table_idx);

            if let Some(ids) = self.tables[table_idx].get(&hash) {
                candidates.extend(ids.iter().cloned());
            }
        }

        dbg!(&candidates);

        candidates.into_iter().collect()
    }

    /// Clear all tables
    pub fn clear(&mut self) {
        for table in self.tables.iter_mut() {
            table.clear();
        }
    }
}

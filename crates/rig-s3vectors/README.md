# Rig AWS S3Vectors integration
This crate integrates AWS S3Vectors into Rig, allowing you to easily use RAG with AWS' newest offering.

## Installation

```toml
[dependencies]
rig-s3vectors = "0.2.5"
rig-core = "0.36.0"
```

You can also run `cargo add rig-s3vectors rig-core` to add the latest published
versions. The root `rig` facade exposes this crate behind the `s3vectors` feature.

This integration will require that you have an AWS account, although the region does not matter.

See [`examples/s3vectors_vector_search.rs`](./examples/s3vectors_vector_search.rs)
for a vector search example.

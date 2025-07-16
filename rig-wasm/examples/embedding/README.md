## An example of how to use `rig-wasm` for embedding.

### How to use
Pull Qdrant in from Docker:
```bash
docker run -p 6333:6333 \
    -v $(pwd)/path/to/data:/qdrant/storage \
    qdrant/qdrant
```

Now run `npm run dev` and watch the magic happen!

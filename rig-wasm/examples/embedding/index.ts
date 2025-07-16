import { EmbeddingModel } from "rig-wasm/openai";
import { QdrantAdapter } from "rig-wasm/qdrant";

let key = process.env.OPENAI_API_KEY;
if (key === undefined) {
  console.log(
    `OPENAI_API_KEY not present as an environment variable. Please add it then try running this example again.`,
  );
  process.exit(1);
}

try {
  const model = new EmbeddingModel({
    apiKey: key,
    modelName: "text-embedding-3-small",
  });

  let embedding = await model.embed_text("hello world!");
  console.log(`Resulting embedding: ${embedding.vec.length}`);
  console.log(`Resulting embedding: ${embedding.document}`);

  let adapter = new QdrantAdapter("myCollection", model, {
    url: "http://127.0.0.1",
    port: 6333,
  });

  let points = [
    {
      id: "doc1",
      vector: embedding.vec,
      payload: {
        document: embedding.document,
      },
    },
  ];

  await adapter.upsertPoints(points);
} catch (e) {
  if (e instanceof Error) {
    console.log(`Error while prompting: ${e.message}`);
  }
}

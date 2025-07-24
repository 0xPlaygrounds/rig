import { EmbeddingModel } from "rig-wasm/openai";
import { QdrantAdapter } from "rig-wasm/qdrant";
import util from "util";

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
  console.log(`Resulting embedding length: ${embedding.vec.length}`);
  console.log(`Embedded text: ${embedding.document}`);
  console.log(util.inspect(embedding.vec));

  let adapter = new QdrantAdapter("meme", model, {
    url: "http://127.0.0.1:6333",
  });

  await adapter.init(1536);

  let points = [
    {
      id: 1,
      vector: embedding.vec,
      payload: {
        document: embedding.document,
      },
    },
  ];

  await adapter.insertDocuments(points);
} catch (e) {
  if (e instanceof Error) {
    console.log(`Error while embedding: ${e}`);
  }
}

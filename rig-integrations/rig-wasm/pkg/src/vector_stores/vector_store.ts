import { Embedding } from "../generated/rig_wasm";
import {
  CanEmbed,
  DocumentId,
  EmbeddedDocument,
  Metadata,
  TopNIdsResult,
  TopNResult,
  VectorSearchOpts,
  VectorStore,
} from "../types";

function cosineSim(a: number[], b: Float64Array): number {
  const arrayA = Array.from(a);
  const arrayB = Array.from(b);
  const dot = arrayA.reduce((sum, val, i) => sum + val * arrayB[i], 0);
  const normA = Math.sqrt(arrayA.reduce((sum, val) => sum + val * val, 0));
  const normB = Math.sqrt(arrayB.reduce((sum, val) => sum + val * val, 0));
  return dot / (normA * normB || 1);
}

interface StoredEntry {
  id: DocumentId;
  embedding: Embedding;
  metadata: Metadata;
}

/**
 * A basic in memory vector store.
 * Usable for small datasets.
 */
export class InMemoryVectorStore implements VectorStore {
  private store: StoredEntry[] = [];
  private model: CanEmbed;

  constructor(model: CanEmbed) {
    this.model = model;
  }

  async addDocument(
    id: DocumentId,
    embedding: Embedding,
    metadata: Metadata = {},
  ) {
    this.store.push({ id, embedding, metadata });
  }

  async topN(req: VectorSearchOpts): Promise<TopNResult[]> {
    const queryEmbedding = await this.model.embedText(req.query);
    const threshold = req.threshold ?? 0;
    return this.store
      .map((entry) => {
        const sim = cosineSim(queryEmbedding.vec, entry.embedding.vec);
        return [sim, entry.id, entry.metadata] as TopNResult;
      })
      .filter((x) => x[0] < threshold)
      .sort((a, b) => b[0] - a[0])
      .slice(0, req.samples);
  }

  async topNIds(req: VectorSearchOpts): Promise<TopNIdsResult[]> {
    const queryEmbedding = await this.model.embedText(req.query);
    const threshold = req.threshold ?? 0;
    return this.store
      .map((entry) => {
        const sim = cosineSim(queryEmbedding.vec, entry.embedding.vec);
        return [sim, entry.id] as TopNIdsResult;
      })
      .filter((x) => x[0] < threshold)
      .sort((a, b) => b[0] - a[0])
      .slice(0, req.samples);
  }
}

import { CanEmbed, VectorSearchOpts } from "../types";
import { QdrantClient } from "@qdrant/js-client-rest";

// qdrant.ts
export interface Point {
  id: string | number;
  vector: number[];
  payload?: Record<string, any>;
}

export interface SearchResult {
  id: string | number;
  score: number;
  payload?: Record<string, any>;
}

export type QdrantClientParams = {
  port?: number;
  apiKey?: string;
  https?: boolean;
  prefix?: string;
  url?: string;
  host?: string;
  timeout?: number;
  checkCompatibility?: boolean;
};

/**
 * An adapter for the Qdrant client to be able to interface with Rig.
 */
export class QdrantAdapter {
  private client: QdrantClient;
  private collectionName: string;
  private params: QdrantClientParams;
  private embeddingModel: CanEmbed;

  constructor(
    collectionName: string,
    embeddingModel: CanEmbed,
    params: QdrantClientParams,
  ) {
    this.params = params;
    this.embeddingModel = embeddingModel;
    this.collectionName = collectionName;
  }

  async loadClient() {
    if (!this.client) {
      try {
        this.client = new QdrantClient(this.params);
      } catch (err) {
        throw new Error("Failed to load Qdrant client: " + err);
      }
    }
  }

  async init(dimensions: number) {
    await this.loadClient();

    const collections = await this.client.getCollections();
    const exists = collections.collections.some(
      (c: any) => c.name === this.collectionName,
    );

    if (!exists) {
      await this.client.createCollection(this.collectionName, {
        vectors: {
          size: dimensions,
          distance: "Cosine",
        },
      });
    }
  }

  async insertDocuments(points: Point[]) {
    await this.loadClient();
    const pointsMapped = points.map((pt) => ({
      id: pt.id,
      vector: Array.from(pt.vector),
      payload: pt.payload ?? {},
    }));

    console.log(pointsMapped);

    try {
      await this.client.upsert(this.collectionName, {
        wait: true,
        points: pointsMapped,
      });
    } catch (e) {
      console.log(`Error: ${e.data.status.error}`);
    }
  }

  async topN(opts: VectorSearchOpts): Promise<SearchResult[]> {
    await this.loadClient();

    const embedding = await this.embeddingModel.embed_text(opts.query);

    const result = await this.client.search(this.collectionName, {
      vector: embedding.vec,
      limit: opts.n,
    });

    return result.map((res: any) => ({
      id: res.id,
      score: res.score,
      payload: res.payload,
    }));
  }

  async topNIds(opts: VectorSearchOpts): Promise<SearchResult[]> {
    await this.loadClient();

    const embedding = await this.embeddingModel.embed_text(opts.query);

    const result = await this.client.search(this.collectionName, {
      vector: embedding.vec,
      limit: opts.n,
    });

    return result.map((res: any) => ({
      id: res.id,
      score: res.score,
    }));
  }
}

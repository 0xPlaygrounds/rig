import { QdrantClient } from "@qdrant/js-client-rest";
import { CanEmbed, VectorSearchOpts } from "./types";

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

export class QdrantAdapter {
  private client: any;
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
        const mod = (await import("@qdrant/js-client-rest")) as {
          QdrantClient: new (params: QdrantClientParams) => any;
        };
        this.client = new mod.QdrantClient(this.params);
      } catch (err) {
        throw new Error(
          "`@qdrant/js-client-rest` is not installed. Please `npm install` it to use the Qdrant adapter.",
        );
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

  async upsertPoints(points: Point[]) {
    await this.loadClient();

    await this.client.upsert(this.collectionName, {
      points: points.map((pt) => ({
        id: pt.id,
        vector: pt.vector,
        payload: pt.payload ?? {},
      })),
    });
  }

  async top_n(opts: VectorSearchOpts): Promise<SearchResult[]> {
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

  async top_n_ids(opts: VectorSearchOpts): Promise<SearchResult[]> {
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

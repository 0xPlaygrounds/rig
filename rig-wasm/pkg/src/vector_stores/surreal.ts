import Surreal from "surrealdb";
import { CanEmbed, VectorSearchOpts } from "../types";




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

export interface SurrealDbParams {
  url: string;
  namespace: string;
  database: string;
}

export class SurrealAdapter {
  private client: Surreal;
  private embeddingModel: CanEmbed;
  private collectionName: string;
  private params: SurrealDbParams;

  constructor(embeddingModel: CanEmbed, collectionName: string, params: SurrealDbParams) {
    this.embeddingModel = embeddingModel;
    this.collectionName = collectionName;
    this.params = params;
  }

  async loadClient() {
    if (!this.client) {
      try {
        this.client = new Surreal();
        await this.client.connect(this.params.url)
        await this.client.use({
          namespace: this.params.namespace,
          database: this.params.database
        })

      } catch (err) {
        await this.client.close();
        throw new Error("Failed to load Surreal client: " + err);
      }
    }
  }

  async init(dimensions: number) {
    await this.loadClient();
    try {
      await this.client.query(`
        CREATE TABLE IF NOT EXISTS ${this.collectionName} (
          id TEXT PRIMARY KEY,
          vector vector(${dimensions}),
          payload JSONB
        )
      `);

    } catch (error) {
      throw new Error("Failed to initialize table: " + error);
    }
  }

  async insertDocuments(points: Point[]) {
    if (!points.length) return;

    await this.loadClient();

    for (const p of points) {
      await this.client.query(
        `UPDATE ${this.collectionName}:${String(p.id)} SET vector = $vector, payload = $payload;`,
        {
          vector: p.vector,
          payload: p.payload ?? {},
        }
      );
    }
  }

  async topN(opts: VectorSearchOpts): Promise<SearchResult[]> {
    await this.loadClient();
    const vector = await this.embeddingModel.embedText(opts.query);
    const samples = opts.samples;
    const result = await this.client.query<[any[]]>(`
      SELECT id, payload, vector::similarity::cosine($vector) AS score
      FROM ${this.collectionName}
      ORDER BY score DESC
      LIMIT $n;
    `, {
      vector,
      samples,
    });

    return result[0].map((r: any) => ({
      id: r.id,
      score: r.score,
      payload: r.payload,
    }));
  }
  //
  // async topNIds(opts: VectorSearchOpts): Promise<string[]> {
  //   await this.loadClient();
  //   try {
  //     const embedding = await this.embeddingModel.embedText(opts.query);
  //     const res = this.client.query(`SELECT id FROM ${this.collectionName} ORDER by vector <>`)
  //   }
  // }
}



import pg from "pg";
import pgvector from "pgvector/pg";
import { CanEmbed, VectorSearchOpts } from "../types";

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

export class PostgresAdapter {
  private embeddingModel: CanEmbed;
  private client: pg.Client;
  private table: string;

  constructor(embeddingModel: CanEmbed, collectionName: string) {
    this.embeddingModel = embeddingModel;
    this.table = collectionName;
  }

  private async loadClient() {
    try {
      if (!this.client) {
        this.client = new pg.Client({ connectionString: process.env.DATABASE_URL });
        await this.client.connect();
        await this.client.query("CREATE EXTENSION IF NOT EXISTS vector");
        await pgvector.registerTypes(this.client);
      }
    } catch (err) {
      console.error("Failed to load Postgres client:", err);
      throw err;
    }
  }

  async init(dimensions: number) {
    try {
      await this.loadClient();
      await this.client.query(`
        CREATE TABLE IF NOT EXISTS ${this.table} (
          id TEXT PRIMARY KEY,
          vector vector(${dimensions}),
          payload JSONB
        )
      `);
    } catch (err) {
      console.error("Failed to initialize table:", err);
      throw err;
    }
  }

  async insertDocuments(points: Point[]) {
    if (!points.length) return;
    await this.loadClient();
    try {
      const q = `
        INSERT INTO ${this.table}(id, vector, payload)
        VALUES ($1, $2, $3)
        ON CONFLICT (id) DO UPDATE
          SET vector = EXCLUDED.vector,
              payload = EXCLUDED.payload
      `;

      const promises = points.map(p =>
        this.client.query(q, [
          String(p.id),
          pgvector.toSql(p.vector),
          p.payload ? JSON.stringify(p.payload) : null,
        ])
      );

      await Promise.all(promises);
    } catch (err) {
      console.error("Failed to insert documents:", err);
      throw err;
    }
  }

  async topN(opts: VectorSearchOpts): Promise<SearchResult[]> {
    await this.loadClient();
    try {
      const embedding = await this.embeddingModel.embedText(opts.query);

      const q = `
        SELECT id, payload, 1 - (vector <=> $1) AS score
        FROM ${this.table}
        ORDER BY vector <=> $1
        LIMIT $2
      `;

      const res = await this.client.query(q, [pgvector.toSql(embedding), opts.samples]);

      return res.rows.map(r => ({
        id: r.id,
        score: r.score,
        payload: r.payload,
      }));
    } catch (err) {
      console.error("Failed to run topN query:", err);
      throw err;
    }
  }

  async topNIds(opts: VectorSearchOpts): Promise<string[]> {
    await this.loadClient();
    try {
      const embedding = await this.embeddingModel.embedText(opts.query);

      const q = `
        SELECT id
        FROM ${this.table}
        ORDER BY vector <=> $1
        LIMIT $2
      `;

      const res = await this.client.query(q, [pgvector.toSql(embedding), opts.samples]);
      return res.rows.map(r => r.id);
    } catch (err) {
      console.error("Failed to run topNIds query:", err);
      throw err;
    }
  }
}


import { CanEmbed, VectorSearchOpts } from "../types";
import Database from "better-sqlite3";
import type BetterSqlite from "better-sqlite3";
import * as sqliteVec from "sqlite-vec";

export interface Column {
  name: string;
  col_type: string;
  indexed?: boolean;
}

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

const COS = "cosine";

/**
 * An adapter for the Sqlite client to be able to interface with Rig.
 */
export class SqliteAdapter {
  private client!: BetterSqlite.Database;
  private collectionName: string;
  private path: string;
  private embeddingModel: CanEmbed;

  constructor(embeddingModel: CanEmbed, collectionName: string, path: string) {
    this.path = path;
    this.collectionName = collectionName;
    this.embeddingModel = embeddingModel;
  }

  async loadClient() {
    if (!this.client) {
      this.client = new Database(this.path || ":memory:");
      sqliteVec.load(this.client);
      // write-ahead logging mode, results in better concurrency
      this.client.pragma("journal_mode = WAL");
    }
  }

  async init(dimensions: number) {
    this.loadClient();
    const tableExists = this.client
      .prepare(
        `select 1 from sqlite_master where type in ('table','view','virtual table') and name = ?`
      )
      .get(this.collectionName);

    if (!tableExists) {
      const ddl = `
        create virtual table ${this.collectionName} using vec0(
          id integer primary key,
          vector float[${dimensions}] distance_metric=${COS},
          +payload text
        );
      `;
      this.client.exec(ddl);
    }
  }

  async insertDocuments(points: Point[]) {
    this.loadClient();
    if (!points.length) return;

    const insert = this.client.prepare(
      `insert into ${this.collectionName}(id, vector, payload)
       values (?, vec_f32(?), ?)`
    );
    const del = this.client.prepare(
      `delete from ${this.collectionName} where id = ?`
    );

    const tx = this.client.transaction((rows: Point[]) => {
      for (const p of rows) {
        del.run(p.id);
        insert.run(
          p.id,
          JSON.stringify(p.vector),
          p.payload ? JSON.stringify(p.payload) : null
        );
      }
    });

    tx(points);
  }

  async topN(opts: VectorSearchOpts): Promise<SearchResult[]> {
    this.loadClient();
    const thr = opts.threshold ?? 0; // similarity in [0,1]
    const embed = await this.embeddingModel.embedText(opts.query);

    const stmt = this.client.prepare(
      `select id, distance, payload
       from ${this.collectionName}
       where vector match vec_f32(?)
         and k = ?
       `
    );

    const rows = stmt
      .all(JSON.stringify(Array.from(embed.vec)), opts.samples)
      .map((r: any) => {
        const distance = r.distance as number;
        const score = 1 - distance;
        return {
          id: r.id,
          score,
          payload: r.payload ? JSON.parse(r.payload) : undefined,
        } as SearchResult;
      })
      .filter((r: SearchResult) => r.score >= thr)
      .slice(0, opts.samples);

    return rows;
  }

  async topNIds(opts: VectorSearchOpts): Promise<SearchResult[]> {
    this.loadClient();
    const embed = await this.embeddingModel.embedText(opts.query);

    const stmt = this.client.prepare(
      `select id, distance
       from ${this.collectionName}
       where vector match vec_f32(?)
         and k = ?`
    );

    const rows = stmt.all(JSON.stringify(Array.from(embed.vec)), opts.samples);
    return rows.map((r: any) => ({
      id: r.id,
      score: 1 - (r.distance as number),
    }));
  }
}


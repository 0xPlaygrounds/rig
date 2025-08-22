import sqlite3 from "sqlite3";
import { CanEmbed } from "../types";

type SqliteClientParams = {
  path: string;
};

export class SqliteAdapter {
  private client: sqlite3.Database;
  private collectionName: string;
  private params: SqliteClientParams;
  private embeddingModel: CanEmbed;

  constructor(
    collectionName: string,
    params: SqliteClientParams,
    embeddingModel: CanEmbed,
  ) {
    this.collectionName = collectionName;
    this.params = params;
    this.embeddingModel = embeddingModel;
  }

  async loadClient() {
    if (!this.client) {
      try {
        this.client = new sqlite3.Database(this.params.path);
      } catch (error) {
        throw new Error("Failed to load Sqlite client: " + error);
      }
    }
  }

  async init(dimension: number) {
    await this.loadClient();
    const collections = this.client.get(this.collectionName);
    const exists = collections.collections.some(this.collectionName);
    if (!exists) {
      await collections.exec(
        `CREATE TABLE ${this.collectionName} (col TEXT, embedding BLOB)`,
      );
    }
  }
}

import { JSONObject } from "./json";

export interface VectorStore {
  top_n: (query: string, n: number) => Promise<TopNResult[]>;
  top_n_ids: (query: string, n: number) => Promise<TopNIdsResult>;
}

export type TopNResult = [CosineSimilarity, EmbeddedDocument, Metadata];
export type TopNIdsResult = [CosineSimilarity, DocumentId];

export type Metadata = JSONObject;
export type CosineSimilarity = number;
export type EmbeddedDocument = string;
export type DocumentId = string;

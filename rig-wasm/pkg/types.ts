export type JSONPrimitive = string | number | boolean | null;
export type JSONValue = JSONPrimitive | JSONObject | JSONArray;
export interface JSONObject {
  [key: string]: JSONValue;
}
export interface JSONArray extends Array<JSONValue> {}

export interface JsToolObject {
  name: () => string;
  definition: (prompt: string) => ToolDefinition;
  call: (args: JSONObject) => Promise<JSONObject>;
}

export interface ToolDefinition {
  name: string;
  description: string;
  parameters: JSONObject;
}

export interface AgentOpts {
  apiKey: string;
  model: string;
  preamble?: string;
  context?: string[];
  temperature?: number;
  tools?: JsToolObject[];
  dynamicContext?: DynamicContextOpts;
  dynamicTools?: DynamicToolsOpts;
}

export interface DynamicContextOpts {
  sample: number;
  dynamicTools: VectorStore;
}

export interface DynamicToolsOpts {
  sample: number;
  dynamicTools: VectorStore;
}

export interface VectorStore {
  top_n: (query: string, n: number) => Promise<TopNResult[]>;
  top_n_ids: (query: string, n: number) => Promise<TopNIdsResult>;
}

export interface VectorSearchOpts {
  query: string;
  n: number;
}

export type TopNResult = [CosineSimilarity, EmbeddedDocument, Metadata];
export type TopNIdsResult = [CosineSimilarity, DocumentId];

export type Metadata = JSONObject;
export type CosineSimilarity = number;
export type EmbeddedDocument = string;
export type DocumentId = string;

export interface CanEmbed {
  embed_text: (query: string) => Promise<Embedding>;
  embed_texts: (texts: Iterable<string>) => Promise<Embedding[]>;
}

export type Embedding = {
  document: string;
  vec: number[];
};

export interface ModelOpts {
  apiKey: string;
  model: string;
  [key: string]: string | boolean | number | object;
}

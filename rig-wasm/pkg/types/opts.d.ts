export interface AgentOpts {
  apiKey: string;
  model: string;
  preamble?: string;
  context?: string[];
  temperature?: number;
  tools?: JsTool[];
  dynamicContext?: DynamicContextOpts;
  dynamicTools?: DynamicToolsOpts;
}

export interface DynamicContextOpts {
  sample: number;
  dynamicTools: JsVectorStore;
}

// TODO: Add toolset!!!
export interface DynamicToolsOpts {
  sample: number;
  dynamicTools: JsVectorStore;
}

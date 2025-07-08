import { JSONObject } from "./json";

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

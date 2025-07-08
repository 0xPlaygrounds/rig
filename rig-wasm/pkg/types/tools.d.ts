import { JSONObject } from "./json";

export interface JsToolObject {
  name: () => string;
  definition: (prompt: string) => ToolDefinition;
  call: (args: JSONObject) => any;
}

export interface ToolDefinition {
  name: string;
  description: string;
  parameters: JSONObject;
}

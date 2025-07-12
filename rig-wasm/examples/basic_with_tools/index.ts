import { Agent } from "rig-wasm/openai";

const counter = {
  counter: 5324,
  name: function () {
    return "counter";
  },
  definition: function (_prompt: string) {
    return {
      name: "counter",
      description: "a counter that can only be incremented",
      parameters: {
        $schema: "https://json-schema.org/draft/2020-12/schema",
        title: "ToolDefinition",
        type: "object",
        properties: {},
        required: [],
      },
    };
  },
  call: async function (args: any) {
    this.counter += 1;
    return { result: this.counter };
  },
};

let key = process.env.OPENAI_API_KEY;
if (key === undefined) {
  console.log(
    `OPENAI_API_KEY not present as an environment variable. Please add it then try running this example again.`,
  );
  process.exit(1);
}
let prompt = `Please increment the counter by 1 and let me know what the resulting number is.`;

try {
  const agent = new Agent({
    apiKey: key,
    model: "gpt-4o",
    tools: [counter],
  });

  console.log(`Prompt: ${prompt}`);
  let res = await agent.prompt(prompt);
  console.log(`GPT-4o: ${res}`);
} catch (e) {
  if (e instanceof Error) {
    console.log(`Error while prompting: ${e.message}`);
  }
}

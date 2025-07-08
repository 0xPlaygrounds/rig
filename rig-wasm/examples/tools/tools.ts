import { OpenAIClient, initPanicHook } from "../../pkg/index.ts";

initPanicHook();

const myTool = {
  counter: 0,

  name() {
    return "counter";
  },

  async definition(_prompt: string) {
    return {
      name: "counter",
      description: "A counter that can be incremented.",
      parameters: {
        $schema: "https://json-schema.org/draft/2020-12/schema",
        type: "object",
        properties: {},
        additionalProperties: false,
      },
    };
  },

  async call(args: any) {
    this.counter += 1;
    return { result: this.counter };
  },
};

let agent = new OpenAIClient("1234").agent("gpt-4o");

let key = process.env.OPENAI_API_KEY;
if (key === undefined) {
  console.log(
    `OPENAI_API_KEY not present as an environment variable. Please add it then try running this example again.`,
  );
  process.exit(1);
}
let prompt = `Hello world!`;

try {
  const agent = new OpenAIClient(key).agent("gpt-4o").addTool(myTool).build();
  console.log(`Prompt: ${prompt}`);
  let res = await agent.prompt(prompt);
  console.log(`GPT-4o: ${res}`);
} catch (e) {
  console.log(`Error while prompting: ${e.msg}`);
}

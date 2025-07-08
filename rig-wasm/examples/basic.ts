import { OpenAIClient } from "../pkg/index";

let key = process.env.OPENAI_API_KEY;
if (key === undefined) {
  console.log(
    `OPENAI_API_KEY not present as an environment variable. Please add it then try running this example again.`,
  );
  process.exit(1);
}
let prompt = `Hello world!`;

try {
  const agent = new OpenAIClient(key).agent("gpt-4o").build();
  console.log(`Prompt: ${prompt}`);
  let res = await agent.prompt(prompt);
  console.log(`GPT-4o: ${res}`);
} catch (e) {
  console.log(`Error while prompting: ${e.msg}`);
}

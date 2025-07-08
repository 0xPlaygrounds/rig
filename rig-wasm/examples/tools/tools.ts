export const counter = {
  counter: 2,

  name() {
    return "counter";
  },

  definition(_prompt: string) {
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

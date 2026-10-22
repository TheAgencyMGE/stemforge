import next from "eslint-config-next";

const config = [
  ...next,
  { ignores: [".next/**", "playwright-report/**", "test-results/**", "next-env.d.ts"] },
];

export default config;

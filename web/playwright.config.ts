import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  use: {
    baseURL: process.env.TARS_WEB_E2E_URL ?? "http://127.0.0.1:7438",
    trace: "retain-on-failure",
  },
  reporter: "list",
});

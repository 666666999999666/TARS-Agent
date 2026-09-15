import path from "node:path";
import fs from "node:fs";
import { expect, test } from "@playwright/test";

test("bootstrap -> Sessions -> real Core replay safely reconnects and renders text", async ({ page }) => {
  const bootstrap = process.env.TARS_WEB_BOOTSTRAP_TOKEN;
  if (!bootstrap) throw new Error("TARS_WEB_BOOTSTRAP_TOKEN is required");

  await page.goto(`/#bootstrap=${bootstrap}`);
  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByTestId("session-list")).toContainText("malicious session");
  await page.getByTestId("session-list").getByText("malicious session", { exact: false }).click();
  await expect(page.getByTestId("connection-status")).toHaveText("connected");
  await expect(page.getByTestId("event-timeline")).toContainText("session.created");
  await expect(page.getByTestId("session-list")).toContainText("<script>alert(1)</script>");
  await expect(page.locator("img")).toHaveCount(0);
  await expect(page.locator("script")).toHaveCount(1);

  await page.getByTestId("session-list").getByText("reconnect session").click();
  await expect(page.getByTestId("event-timeline")).toContainText("session.created");
  // The test Web adapter closes the first real Core subscription immediately
  // after session.created.  session.resumed can only arrive through a second
  // real event.subscribe call with the native EventSource Last-Event-ID cursor.
  await expect(page.getByTestId("event-timeline")).toContainText("session.resumed", {
    timeout: 15_000,
  });
  await expect(page.getByTestId("event-timeline").locator("pre")).toHaveCount(2);

  const cursors = await page.getByTestId("event-timeline").locator("pre").evaluateAll((items) =>
    items.map((item) => Number(JSON.parse(item.textContent ?? "{}").cursor)),
  );
  expect(cursors).toHaveLength(2);
  expect(cursors[0]).toBeGreaterThan(0);
  expect(cursors[1]).toBeGreaterThan(cursors[0]);
  const deniedWrite = await page.request.post("/api/sessions", { data: { title: "forbidden" } });
  expect([403, 404, 405]).toContain(deniedWrite.status());
  const sessionsAfterWrite = await page.request.get("/api/sessions");
  expect(JSON.stringify(await sessionsAfterWrite.json())).not.toContain("forbidden");
  const output = process.env.TARS_VISUAL_OUTPUT;
  if (output) {
    fs.mkdirSync(output, { recursive: true });
    await page.setViewportSize({ width: 1280, height: 800 });
    for (const scheme of ["light", "dark"] as const) {
      await page.emulateMedia({ colorScheme: scheme });
      await expect(page.getByRole("heading", { name: "TARS-Agent", exact: true })).toBeVisible();
      await page.evaluate(() => new Promise<void>((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
      await page.screenshot({ path: path.join(output, `web-${scheme}.png`), fullPage: true,
                              animations: "disabled" });
    }
  }
});

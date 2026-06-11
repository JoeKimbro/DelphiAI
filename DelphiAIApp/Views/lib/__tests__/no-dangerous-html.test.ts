import { describe, it, expect } from "vitest";
import { fileURLToPath } from "node:url";
import { execSync } from "node:child_process";

// Lightweight grep over the source tree (tracked files only, excludes
// node_modules + tests via the pathspec). Acts as an XSS regression gate.
describe("xss surface", () => {
  it("has no dangerouslySetInnerHTML in app/components/lib", () => {
    let hits = "";
    try {
      hits = execSync(
        `git grep -n "dangerouslySetInnerHTML" -- app components lib`,
        { cwd: fileURLToPath(new URL("../../", import.meta.url)) }
      ).toString();
    } catch {
      hits = ""; // git grep exits 1 when no matches — that's the pass case
    }
    expect(hits).toBe("");
  });
});

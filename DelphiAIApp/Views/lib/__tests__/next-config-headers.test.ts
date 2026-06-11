import { describe, it, expect } from "vitest";
import config from "@/next.config";

describe("next.config security headers", () => {
  it("sets the expected headers on all routes", async () => {
    const rules = await config.headers!();
    const all = rules.find((r) => r.source === "/(.*)");
    expect(all).toBeDefined();
    const names = all!.headers.map((h) => h.key.toLowerCase());
    expect(names).toContain("strict-transport-security");
    expect(names).toContain("x-frame-options");
    expect(names).toContain("x-content-type-options");
    expect(names).toContain("referrer-policy");
    expect(names).toContain("permissions-policy");
    expect(names).toContain("content-security-policy");
  });

  it("uses a pragmatic CSP allowing inline styles only", () => {
    return config.headers!().then((rules) => {
      const csp = rules
        .find((r) => r.source === "/(.*)")!
        .headers.find((h) => h.key.toLowerCase() === "content-security-policy")!
        .value;
      expect(csp).toContain("script-src 'self'");
      expect(csp).toContain("style-src 'self' 'unsafe-inline'");
      expect(csp).toContain("frame-ancestors 'none'");
      expect(csp).not.toContain("'unsafe-eval'");
    });
  });
});

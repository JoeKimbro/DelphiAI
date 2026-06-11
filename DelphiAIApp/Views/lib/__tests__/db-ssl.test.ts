import { describe, it, expect } from "vitest";
import { resolveSsl } from "@/lib/db";

describe("resolveSsl", () => {
  it("requires SSL for a remote DATABASE_URL", () => {
    expect(resolveSsl("postgres://u:p@ep.neon.tech/db")).toEqual({
      rejectUnauthorized: true,
    });
  });
  it("disables SSL for localhost", () => {
    expect(resolveSsl("postgres://u:p@localhost:5433/db")).toBe(false);
  });
  it("respects an explicit sslmode=disable", () => {
    expect(resolveSsl("postgres://u:p@ep.neon.tech/db?sslmode=disable")).toBe(false);
  });
});

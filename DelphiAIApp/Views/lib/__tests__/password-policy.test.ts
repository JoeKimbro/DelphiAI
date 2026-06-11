import { describe, it, expect } from "vitest";
import { passwordSchema } from "@/lib/common-passwords";

describe("passwordSchema", () => {
  it("rejects < 10 chars", () => {
    expect(passwordSchema.safeParse("short1!a").success).toBe(false);
  });
  it("rejects a common password", () => {
    expect(passwordSchema.safeParse("password123").success).toBe(false);
  });
  it("accepts a strong-enough password", () => {
    expect(passwordSchema.safeParse("Tr0ubad0ur-Sunset").success).toBe(true);
  });
});

import { describe, it, expect } from "vitest";
import { cn } from "@/lib/utils";

describe("cn", () => {
  it("merges class strings", () => expect(cn("foo", "bar")).toBe("foo bar"));
  it("dedupes conflicting Tailwind utilities (last wins)", () =>
    expect(cn("px-2", "px-4")).toBe("px-4"));
  it("filters falsy values", () =>
    expect(cn("", undefined, "baz", false as unknown as string)).toBe("baz"));
  it("returns empty string when nothing provided", () => expect(cn()).toBe(""));
  it("handles conditional class objects", () =>
    expect(cn({ "text-red": true, "text-blue": false })).toBe("text-red"));
});

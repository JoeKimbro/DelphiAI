import { describe, it, expect } from "vitest";
import {
  formatPercent,
  formatAmericanOdds,
  formatElo,
  formatEloDelta,
  slugifyFighter,
  shortDate,
  confidenceColor,
} from "@/lib/format";

describe("formatPercent", () => {
  it("returns em-dash for null", () => expect(formatPercent(null)).toBe("—"));
  it("returns em-dash for undefined", () => expect(formatPercent(undefined)).toBe("—"));
  it("returns em-dash for NaN", () => expect(formatPercent(NaN)).toBe("—"));
  it("rounds to 0 digits by default", () => expect(formatPercent(0.506)).toBe("51%"));
  it("respects explicit digits", () => expect(formatPercent(0.5, 1)).toBe("50.0%"));
  it("formats zero", () => expect(formatPercent(0)).toBe("0%"));
});

describe("formatAmericanOdds", () => {
  it("prefixes positive odds with +", () => expect(formatAmericanOdds(150)).toBe("+150"));
  it("leaves negative odds as-is", () => expect(formatAmericanOdds(-200)).toBe("-200"));
  it("returns em-dash for null", () => expect(formatAmericanOdds(null)).toBe("—"));
  it("returns em-dash for undefined", () => expect(formatAmericanOdds(undefined)).toBe("—"));
});

describe("formatElo", () => {
  it("rounds to nearest integer", () => expect(formatElo(1500.7)).toBe("1501"));
  it("returns em-dash for null", () => expect(formatElo(null)).toBe("—"));
  it("returns em-dash for undefined", () => expect(formatElo(undefined)).toBe("—"));
});

describe("formatEloDelta", () => {
  it("returns '0' for zero", () => expect(formatEloDelta(0)).toBe("0"));
  it("returns up arrow for positive", () => expect(formatEloDelta(25)).toBe("▲ +25"));
  it("returns down arrow for negative", () => expect(formatEloDelta(-25)).toBe("▼ -25"));
  it("returns empty string for null", () => expect(formatEloDelta(null)).toBe(""));
  it("rounds before formatting", () => expect(formatEloDelta(25.6)).toBe("▲ +26"));
});

describe("slugifyFighter", () => {
  it("lowercases the name", () => expect(slugifyFighter("Jon Jones")).toBe("jon-jones"));
  it("collapses spaces to hyphens", () => expect(slugifyFighter("Israel Adesanya")).toBe("israel-adesanya"));
  it("strips punctuation", () => expect(slugifyFighter("T.J. Dillashaw")).toBe("tj-dillashaw"));
});

describe("shortDate", () => {
  it("returns em-dash for null", () => expect(shortDate(null)).toBe("—"));
  it("returns em-dash for undefined", () => expect(shortDate(undefined)).toBe("—"));
  it("returns em-dash for empty string", () => expect(shortDate("")).toBe("—"));
  it("formats a valid ISO date (contains the year)", () => {
    const result = shortDate("2024-03-09");
    expect(result).toContain("2024");
    expect(result).not.toBe("—");
  });
  it("does not return em-dash for an unparseable string (falls through gracefully)", () => {
    const result = shortDate("not-a-date");
    expect(result).not.toBe("—");
    expect(typeof result).toBe("string");
  });
});

describe("confidenceColor", () => {
  it("HIGH returns gold palette", () =>
    expect(confidenceColor("HIGH")).toEqual({
      bg: "bg-gold/10",
      text: "text-gold",
      border: "border-gold/40",
    }));
  it("MED returns red palette", () =>
    expect(confidenceColor("MED")).toEqual({
      bg: "bg-red/10",
      text: "text-red",
      border: "border-red/40",
    }));
  it("LOW returns muted palette", () =>
    expect(confidenceColor("LOW")).toEqual({
      bg: "bg-surface-raised",
      text: "text-muted",
      border: "border-border",
    }));
  it("TOSS returns surface palette", () =>
    expect(confidenceColor("TOSS")).toEqual({
      bg: "bg-surface",
      text: "text-muted-2",
      border: "border-border",
    }));
});

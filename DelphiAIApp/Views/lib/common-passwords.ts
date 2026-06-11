import { z } from "zod";

// Top weak passwords (lowercased). NIST guidance: length-first + reject known-
// breached/common values, no forced symbol/complexity rules.
export const COMMON_PASSWORDS = new Set<string>([
  "password", "password1", "password123", "12345678", "123456789", "1234567890",
  "qwertyuiop", "qwerty123", "iloveyou", "admin123", "letmein123", "welcome123",
  "monkey123", "dragon123", "football1", "baseball1", "sunshine1", "princess1",
  "trustno1", "abc123456", "passw0rd", "p@ssw0rd", "changeme123", "delphiai",
]);

export const passwordSchema = z
  .string()
  .min(10, "Password must be at least 10 characters")
  .max(128)
  .refine((p) => !COMMON_PASSWORDS.has(p.toLowerCase()), {
    message: "Password is too common",
  });

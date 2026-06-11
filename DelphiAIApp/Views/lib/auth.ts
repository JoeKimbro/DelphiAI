import NextAuth, { type DefaultSession } from "next-auth";
import Credentials from "next-auth/providers/credentials";
import bcrypt from "bcryptjs";
import { z } from "zod";
import { headers } from "next/headers";
import { query } from "./db";
import { isLockedOut, recordAttempt, clientIp } from "./rate-limit";

declare module "next-auth" {
  interface Session {
    user: {
      id: string;
    } & DefaultSession["user"];
  }
}

const credentialsSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8),
});

type UserRow = {
  id: string;
  email: string;
  name: string | null;
  password_hash: string;
};

export const { handlers, auth, signIn, signOut } = NextAuth({
  trustHost: true,
  session: { strategy: "jwt" },
  pages: {
    signIn: "/auth/signin",
  },
  providers: [
    Credentials({
      name: "credentials",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" },
      },
      authorize: async (raw) => {
        const parsed = credentialsSchema.safeParse(raw);
        if (!parsed.success) return null;
        const { email, password } = parsed.data;
        const ip = clientIp(await headers());

        // Brute-force lockout (generic failure — no account-existence oracle).
        if (await isLockedOut(email, ip)) return null;

        const rows = await query<UserRow>(
          `SELECT id, email, name, password_hash FROM users WHERE email = $1 LIMIT 1`,
          [email]
        );
        const user = rows[0];
        const ok = user ? await bcrypt.compare(password, user.password_hash) : false;
        await recordAttempt(email, ip, ok);
        if (!ok || !user) return null;
        return { id: user.id, email: user.email, name: user.name ?? user.email };
      },
    }),
  ],
  callbacks: {
    jwt: ({ token, user }) => {
      if (user) {
        token.uid = user.id;
      }
      return token;
    },
    session: ({ session, token }) => {
      if (session.user && typeof token.uid === "string") {
        session.user.id = token.uid;
      }
      return session;
    },
  },
});

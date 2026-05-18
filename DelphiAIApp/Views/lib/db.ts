import { Pool } from "pg";

declare global {
  // eslint-disable-next-line no-var
  var _pgPool: Pool | undefined;
}

function makePool(): Pool {
  if (process.env.DATABASE_URL) {
    return new Pool({ connectionString: process.env.DATABASE_URL });
  }
  return new Pool({
    host: process.env.DB_HOST || "localhost",
    port: Number(process.env.DB_PORT || 5433),
    database: process.env.DB_NAME || "delphi_db",
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
  });
}

// Reuse pool across HMR reloads in dev to avoid connection leaks.
export const pool: Pool = globalThis._pgPool ?? makePool();
if (process.env.NODE_ENV !== "production") {
  globalThis._pgPool = pool;
}

export async function query<T = Record<string, unknown>>(
  text: string,
  params?: unknown[]
): Promise<T[]> {
  const res = await pool.query(text, params as never[]);
  return res.rows as T[];
}

import Link from "next/link";

export default function NotFound() {
  return (
    <div className="mx-auto flex max-w-2xl flex-col items-center justify-center px-6 py-24 text-center">
      <span className="text-[10px] font-bold uppercase tracking-widest text-red">
        Not Found
      </span>
      <h1 className="mt-2 text-4xl font-black tracking-tight text-text">
        404 — TKO Loss
      </h1>
      <p className="mt-3 max-w-md text-sm text-muted">
        We couldn&apos;t find this fighter, event, or page in the database.
      </p>
      <Link
        href="/"
        className="mt-6 inline-flex items-center rounded-md border border-red bg-red/10 px-4 py-2 text-sm font-bold text-red transition-colors hover:bg-red hover:text-white"
      >
        Return to Dashboard
      </Link>
    </div>
  );
}

import Image from "next/image";
import Link from "next/link";
import { cn } from "@/lib/utils";
import { auth } from "@/lib/auth";
import { HeaderUserMenu } from "./HeaderUserMenu";

const NAV = [
  { href: "/", label: "Dashboard" },
  { href: "/events", label: "Events" },
  { href: "/events/past", label: "Past Events" },
  { href: "/performance", label: "Performance" },
  { href: "/bets", label: "Bets" },
];

export async function Header() {
  const session = await auth();
  return (
    <header className="sticky top-0 z-40 border-b border-border bg-bg/85 backdrop-blur-md">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
        <Link href="/" className="flex items-center gap-3">
          <div className="h-10 w-10 overflow-hidden rounded-full">
            <Image
              src="/DelphiLogo.png"
              alt="DelphiAI"
              width={40}
              height={40}
              className="h-full w-full scale-110 object-cover"
              priority
            />
          </div>
          <div className="flex flex-col leading-none">
            <span className="text-lg font-black tracking-tight text-text">
              DELPHI<span className="text-red">AI</span>
            </span>
            <span className="text-[10px] font-medium uppercase tracking-widest text-muted">
              UFC Predictions
            </span>
          </div>
        </Link>

        <nav className="flex items-center gap-1">
          {NAV.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "rounded-md px-3 py-2 text-sm font-medium text-muted",
                "transition-colors hover:bg-surface hover:text-text"
              )}
            >
              {item.label}
            </Link>
          ))}
          <div className="ml-2 pl-2 border-l border-border">
            <HeaderUserMenu
              user={session?.user
                ? {
                    name: session.user.name ?? null,
                    email: session.user.email ?? null,
                  }
                : null}
            />
          </div>
        </nav>
      </div>
    </header>
  );
}

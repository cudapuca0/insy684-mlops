"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Bookings" },
  { href: "/cancellations", label: "Who cancels?" },
  { href: "/model", label: "Model" },
];

export function Nav() {
  const pathname = usePathname();
  return (
    <header className="sticky top-0 z-10 border-b border-white/5 bg-background/90 backdrop-blur">
      <nav className="mx-auto flex max-w-6xl items-center gap-1 overflow-x-auto px-4 py-3">
        <Link href="/" className="mr-4 flex shrink-0 items-center gap-2 font-bold">
          <span className="inline-block h-3 w-3 rounded-sm bg-accent" aria-hidden />
          Hotel cancellations
        </Link>
        {LINKS.map((l) => {
          const active = l.href === "/" ? pathname === "/" : pathname.startsWith(l.href);
          return (
            <Link key={l.href} href={l.href}
              className={`shrink-0 rounded-full px-4 py-1.5 text-sm transition ${active ? "bg-white text-black" : "text-muted hover:bg-card hover:text-foreground"}`}>
              {l.label}
            </Link>
          );
        })}
      </nav>
    </header>
  );
}

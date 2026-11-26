import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { Nav } from "@/components/Nav";
import "./globals.css";

const inter = Inter({ variable: "--font-inter", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Hotel cancellations — data and model",
  description: "87,000 hotel bookings in Portugal: who cancels, and how well a model can tell in advance.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${inter.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col font-sans">
        <Nav />
        <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8">{children}</main>
        <footer className="mx-auto w-full max-w-6xl px-4 py-6 text-xs text-muted">
          Data: Hotel Booking Demand (Antonio, Almeida &amp; Nunes, 2019), a city hotel and a resort hotel in Portugal, 2015–2017.
        </footer>
      </body>
    </html>
  );
}

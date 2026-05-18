import type { Metadata } from "next";
import Image from "next/image";
import { Inter, JetBrains_Mono } from "next/font/google";
import { Providers } from "@/lib/providers";
import { Header } from "@/components/layout/Header";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

const jetbrains = JetBrains_Mono({
  variable: "--font-jetbrains",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "DelphiAI — UFC Predictions",
  description:
    "XGBoost + ELO blended UFC fight predictions. Calibrated probabilities, ROI tracking, fighter analytics.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${jetbrains.variable} h-full antialiased`}
    >
      <body className="min-h-screen flex flex-col bg-bg text-text">
        <Providers>
          <Header />
          <div className="pointer-events-none fixed inset-x-0 top-24 z-30">
            <div className="relative mx-auto max-w-7xl">
              <div className="absolute left-full ml-[calc((100vw-80rem)/4-7.5rem)] h-60 w-60 overflow-hidden rounded-full">
                <Image
                  src="/DelphiLogo.png"
                  alt="DelphiAI"
                  width={240}
                  height={240}
                  className="h-full w-full scale-110 object-cover"
                  priority
                />
              </div>
            </div>
          </div>
          <main className="flex-1">{children}</main>
        </Providers>
      </body>
    </html>
  );
}

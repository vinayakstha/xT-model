import { Space_Mono } from "next/font/google";
import "./globals.css";

const spaceMono = Space_Mono({
  weight: ["400", "700"],
  subsets: ["latin"],
});

export const metadata = {
  title: "Flow of Threat — xT / PV / VAEP",
  description: "xT, PV, VAEP threat analysis dashboard",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={spaceMono.className}>{children}</body>
    </html>
  );
}

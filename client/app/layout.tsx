import "./globals.css";

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
      <body>{children}</body>
    </html>
  );
}

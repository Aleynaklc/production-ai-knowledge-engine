import type { Metadata } from 'next';

import './globals.css';

export const metadata: Metadata = {
  title: 'NovaStack Knowledge Console',
  description:
    'A transparent interface for grounded local knowledge retrieval.',
  icons: { icon: '/favicon.svg' },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

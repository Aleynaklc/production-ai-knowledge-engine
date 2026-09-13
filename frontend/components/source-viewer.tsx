'use client';

import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { downloadSource, getSource } from '@/lib/api';
import type { SourcePage } from '@/lib/api';

export interface SourceLocation {
  documentId: string;
  version: number;
  unit: number;
}

export function SourceViewer({
  source,
  onClose,
}: {
  source: SourceLocation;
  onClose: () => void;
}) {
  const [page, setPage] = useState<SourcePage | null>(null);
  const [number, setNumber] = useState(source.unit);
  const [error, setError] = useState('');
  useEffect(() => {
    let active = true;
    getSource(source.documentId, source.version, number)
      .then((result) => {
        if (active) setPage(result);
      })
      .catch((caught: Error) => {
        if (active) setError(caught.message);
      });
    return () => {
      active = false;
    };
  }, [source.documentId, source.version, number]);
  async function download() {
    if (!page) return;
    try {
      await downloadSource(source.documentId, source.version, page.filename);
    } catch (caught) {
      setError((caught as Error).message);
    }
  }
  return (
    <section
      aria-label="Document source"
      className="my-4 rounded-xl border border-mint/25 bg-panel p-5 text-ink"
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h3 className="text-lg font-medium">
          {page?.filename ?? 'Source'} · version {source.version}
        </h3>
        <Button variant="ghost" onClick={onClose}>
          Close source
        </Button>
      </div>
      {error && (
        <p role="alert" className="my-4 text-red-200">
          {error}
        </p>
      )}
      {!page && !error && (
        <p className="py-4" aria-busy="true">
          Loading source…
        </p>
      )}
      {page && (
        <>
          <p className="my-3 text-sm text-mint">
            {page.unit.kind === 'page' ? 'PDF page' : 'Section'} {number} of{' '}
            {page.unit_count}
          </p>
          <div className="max-h-[60vh] overflow-auto whitespace-pre-wrap rounded-lg bg-canvas p-4 text-base leading-7">
            {page.unit.text || 'This page contains no extractable text.'}
          </div>
          <div className="mt-4 flex flex-wrap gap-3">
            <Button
              variant="outline"
              disabled={number <= 1}
              onClick={() => {
                setPage(null);
                setError('');
                setNumber(number - 1);
              }}
            >
              Previous
            </Button>
            <Button
              variant="outline"
              disabled={number >= page.unit_count}
              onClick={() => {
                setPage(null);
                setError('');
                setNumber(number + 1);
              }}
            >
              Next
            </Button>
            <Button variant="outline" onClick={download}>
              Download original
            </Button>
          </div>
        </>
      )}
    </section>
  );
}

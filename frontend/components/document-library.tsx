'use client';

import { useEffect, useRef, useState } from 'react';
import type { SyntheticEvent } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { SourceViewer } from '@/components/source-viewer';
import type { SourceLocation } from '@/components/source-viewer';
import {
  deleteDocument,
  getDocuments,
  replaceDocument,
  retryDocument,
  uploadDocument,
} from '@/lib/api';
import type { DocumentLibraryResponse } from '@/lib/api';

export function DocumentLibrary({
  onUploaded,
  onBusyChange,
}: {
  onUploaded: () => void;
  onBusyChange: (busy: boolean) => void;
  answering: boolean;
}) {
  const [library, setLibrary] = useState<DocumentLibraryResponse | null>(null);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [replaceId, setReplaceId] = useState<string | null>(null);
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [source, setSource] = useState<SourceLocation | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const onChange = useRef(onUploaded);
  useEffect(() => {
    onChange.current = onUploaded;
  }, [onUploaded]);
  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    let lastRevision: string | null = null;
    async function poll() {
      try {
        const result = await getDocuments();
        if (!active) return;
        const revision = result.documents
          .map((item) => `${item.id}:${item.active_version}`)
          .sort()
          .join('|');
        if (lastRevision !== null && lastRevision !== revision)
          onChange.current();
        lastRevision = revision;
        setLibrary(result);
      } catch (caught) {
        if (active) setError((caught as Error).message);
      }
      if (active) timer = setTimeout(poll, 1500);
    }
    void poll();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, []);
  async function refresh() {
    setLibrary(await getDocuments());
  }
  async function submit(event: SyntheticEvent) {
    event.preventDefault();
    if (!file || !library || busy) return;
    if (
      !library.supported_extensions.some((suffix) =>
        file.name.toLowerCase().endsWith(suffix),
      )
    ) {
      setError('Choose a PDF, DOCX, Markdown, or text document.');
      return;
    }
    if (!file.size || file.size > library.max_file_bytes) {
      setError('The file is empty or exceeds the upload limit.');
      return;
    }
    setBusy(true);
    onBusyChange(true);
    setError('');
    setMessage('');
    try {
      const result = replaceId
        ? await replaceDocument(replaceId, file)
        : await uploadDocument(file);
      setMessage(
        result.duplicate
          ? 'This document is already in your workspace.'
          : 'Upload accepted. You can keep asking questions while it is processed.',
      );
      setFile(null);
      setReplaceId(null);
      if (input.current) input.current.value = '';
      await refresh();
      onUploaded();
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setBusy(false);
      onBusyChange(false);
    }
  }
  async function remove(id: string) {
    setBusy(true);
    setError('');
    try {
      await deleteDocument(id);
      setDeleteId(null);
      setSource(null);
      await refresh();
      onUploaded();
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function retry(id: string) {
    setBusy(true);
    setError('');
    try {
      await retryDocument(id);
      await refresh();
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <Card className="border-white/8 bg-panel/88 ring-0">
      <CardHeader>
        <CardTitle className="text-xl">Workspace documents</CardTitle>
        <p className="text-base text-ink-dim">
          PDF, DOCX, Markdown and text
          {library
            ? ` · up to ${Math.round(library.max_file_bytes / 1024 / 1024)} MB per file`
            : ''}
          . Only members of this workspace can access these documents.
        </p>
      </CardHeader>
      <CardContent>
        {error && (
          <p role="alert" className="mb-4 text-sm text-red-200">
            {error}
          </p>
        )}
        {message && (
          <output className="mb-4 block text-sm text-mint">{message}</output>
        )}
        {!library && !error && <p aria-busy="true">Loading documents…</p>}
        {library && library.role !== 'reader' && (
          <form onSubmit={submit} className="mb-5 space-y-3">
            <label className="block text-sm">
              {replaceId ? 'Replacement file' : 'Add a document'}
              <input
                ref={input}
                type="file"
                accept=".md,.txt,.pdf,.docx"
                disabled={busy}
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                className="mt-2 block w-full rounded-lg border border-white/10 bg-canvas p-3 text-sm file:mr-3 file:rounded file:border-0 file:bg-white/10 file:p-2 file:text-ink"
              />
            </label>
            <div className="flex flex-wrap gap-3">
              <Button type="submit" disabled={!file || busy}>
                {busy
                  ? 'Uploading…'
                  : replaceId
                    ? 'Upload new version'
                    : 'Upload document'}
              </Button>
              {replaceId && (
                <Button variant="ghost" onClick={() => setReplaceId(null)}>
                  Cancel replacement
                </Button>
              )}
            </div>
          </form>
        )}
        {library?.count === 0 && (
          <p className="py-4 text-base text-ink-dim">
            No documents yet. Add a document and wait until it is ready before
            asking a question.
          </p>
        )}
        <ul className="space-y-3">
          {library?.documents.map((item) => (
            <li key={item.id} className="rounded-lg border border-white/10 p-4">
              <div className="flex flex-wrap justify-between gap-3">
                <span className="break-all text-base font-medium">
                  {item.filename}
                </span>
                <span
                  className={`text-sm ${item.status === 'failed' ? 'text-red-200' : item.status === 'processing' ? 'text-amber-200' : 'text-mint'}`}
                >
                  {item.status === 'processing'
                    ? 'Processing…'
                    : item.status === 'failed'
                      ? 'Failed'
                      : 'Ready'}
                </span>
              </div>
              <p className="mt-2 text-sm text-ink-dim">
                Version {item.version} · {item.chunk_count} indexed passages
                {item.status !== 'ready' && item.active_version
                  ? ` · Version ${item.active_version} is still available for questions`
                  : ''}
              </p>
              {item.error && (
                <p className="mt-2 text-sm text-red-200">{item.error}</p>
              )}
              <div className="mt-3 flex flex-wrap gap-2">
                {item.active_version && (
                  <Button
                    variant="outline"
                    onClick={() =>
                      setSource({
                        documentId: item.id,
                        version: item.active_version!,
                        unit: 1,
                      })
                    }
                  >
                    View source
                  </Button>
                )}
                {library.role !== 'reader' && (
                  <>
                    <Button
                      variant="outline"
                      disabled={busy || item.status === 'processing'}
                      onClick={() => {
                        setReplaceId(item.id);
                        setFile(null);
                        if (input.current) {
                          input.current.value = '';
                          input.current.focus();
                        }
                      }}
                    >
                      Replace
                    </Button>
                    {item.status === 'failed' && (
                      <Button
                        variant="outline"
                        disabled={busy}
                        onClick={() => retry(item.id)}
                      >
                        Retry
                      </Button>
                    )}
                    <Button
                      variant="ghost"
                      disabled={busy}
                      onClick={() => setDeleteId(item.id)}
                    >
                      Delete
                    </Button>
                  </>
                )}
              </div>
              {deleteId === item.id && (
                <div className="mt-3 rounded-lg border border-red-200/20 p-3 text-sm">
                  <p>
                    Delete this document and all its versions? This cannot be
                    undone.
                  </p>
                  <div className="mt-3 flex gap-3">
                    <Button
                      disabled={busy}
                      variant="destructive"
                      onClick={() => remove(item.id)}
                    >
                      Delete permanently
                    </Button>
                    <Button variant="ghost" onClick={() => setDeleteId(null)}>
                      Cancel
                    </Button>
                  </div>
                </div>
              )}
            </li>
          ))}
        </ul>
        {source && (
          <SourceViewer
            key={`${source.documentId}:${source.version}`}
            source={source}
            onClose={() => setSource(null)}
          />
        )}
      </CardContent>
    </Card>
  );
}

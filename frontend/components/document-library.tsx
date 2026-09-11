'use client';

import { Check, FileText, LoaderCircle, RefreshCw, Upload } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import type { SyntheticEvent } from 'react';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { getDocuments, uploadDocument } from '@/lib/api';
import type { DocumentLibraryResponse } from '@/lib/api';

function fileSize(bytes: number) {
  return bytes >= 1024 * 1024
    ? `${(bytes / (1024 * 1024)).toFixed(1)} MB`
    : `${Math.max(1, Math.ceil(bytes / 1024))} KB`;
}

export function DocumentLibrary({
  onUploaded,
  onBusyChange,
  answering,
}: {
  onUploaded: () => void;
  onBusyChange: (busy: boolean) => void;
  answering: boolean;
}) {
  const [library, setLibrary] = useState<DocumentLibraryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let active = true;
    getDocuments()
      .then((result) => {
        if (active) setLibrary(result);
      })
      .catch((error: Error) => {
        if (active) setLoadError(error.message);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  async function refresh() {
    setLoading(true);
    setLoadError(null);
    try {
      setLibrary(await getDocuments());
    } catch (error) {
      setLoadError(
        error instanceof Error ? error.message : 'Could not load documents.',
      );
    } finally {
      setLoading(false);
    }
  }

  function chooseFile(selected: File | null) {
    setMessage(null);
    setUploadError(null);
    setFile(null);
    if (!selected || !library) return;
    if (
      !library.supported_extensions.some((extension) =>
        selected.name.toLowerCase().endsWith(extension),
      )
    ) {
      setUploadError('Choose a Markdown (.md) or text (.txt) document.');
      return;
    }
    if (selected.size === 0) {
      setUploadError(
        'This file is empty. Choose a document that contains text.',
      );
      return;
    }
    if (selected.size > library.max_file_bytes) {
      setUploadError(
        `Choose a file smaller than ${fileSize(library.max_file_bytes)}.`,
      );
      return;
    }
    setFile(selected);
  }

  async function submit(event: SyntheticEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file || loading || uploading || answering || !library) return;
    setUploading(true);
    onBusyChange(true);
    setUploadError(null);
    setMessage(null);
    try {
      const result = await uploadDocument(file);
      setLibrary((current) => {
        if (!current) return current;
        const documents = [
          result.document,
          ...current.documents.filter((item) => item.id !== result.document.id),
        ];
        return { ...current, documents, count: documents.length };
      });
      setMessage(
        result.duplicate
          ? `${result.document.filename} is already in the library.`
          : `${result.document.filename} is ready. Ask a question about its contents below.`,
      );
      setFile(null);
      if (input.current) input.current.value = '';
      onUploaded();
    } catch (error) {
      setUploadError(
        error instanceof Error
          ? error.message
          : 'The upload failed. Please retry.',
      );
    } finally {
      setUploading(false);
      onBusyChange(false);
    }
  }

  return (
    <Card className="border-white/8 bg-panel/88 ring-0">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between gap-3">
          <CardTitle className="flex items-center gap-2 font-display text-xl text-ink">
            <Upload className="size-5 text-mint" /> Upload a document
          </CardTitle>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={refresh}
            disabled={loading || uploading}
            aria-label="Refresh uploaded documents"
            className="text-ink-dim"
          >
            <RefreshCw className={`size-4 ${loading ? 'animate-spin' : ''}`} />
          </Button>
        </div>
        <p className="text-sm leading-6 text-ink-dim">
          Add UTF-8 Markdown or text files
          {library ? `, up to ${fileSize(library.max_file_bytes)} each` : ''}.{' '}
          Uploads are shared with everyone who can access this console.
        </p>
      </CardHeader>
      <CardContent>
        {loadError ? (
          <p
            role="alert"
            className="mb-4 rounded-lg border border-amber-200/20 bg-amber-200/5 p-3 text-sm text-amber-200"
          >
            {loadError} Use the refresh button to retry.
          </p>
        ) : null}
        <form onSubmit={submit} className="space-y-3" aria-busy={uploading}>
          <label
            htmlFor="document-file"
            className="block text-sm font-medium text-ink"
          >
            Document file
          </label>
          <input
            ref={input}
            id="document-file"
            type="file"
            accept=".md,.txt,text/plain,text/markdown"
            disabled={uploading || loading || !library}
            onChange={(event) => chooseFile(event.target.files?.[0] ?? null)}
            aria-describedby="upload-feedback"
            className="block w-full min-w-0 rounded-lg border border-white/10 bg-canvas/65 p-2 text-sm text-ink-dim file:mr-3 file:rounded-md file:border-0 file:bg-white/10 file:px-3 file:py-2 file:text-sm file:font-medium file:text-ink hover:file:bg-white/15 focus-visible:outline-2 focus-visible:outline-mint disabled:opacity-50"
          />
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="min-w-0 break-all text-sm text-ink-dim">
              {file
                ? `${file.name} · ${fileSize(file.size)}`
                : 'Choose a file to add to the knowledge base.'}
            </p>
            <Button
              type="submit"
              disabled={!file || loading || uploading || answering || !library}
              className="shrink-0 bg-mint text-canvas hover:bg-mint/85"
            >
              {uploading ? (
                <LoaderCircle className="size-4 animate-spin" />
              ) : (
                <Upload className="size-4" />
              )}
              {uploading ? 'Preparing document…' : 'Upload document'}
            </Button>
          </div>
          <div id="upload-feedback" aria-live="polite">
            {uploadError ? (
              <p role="alert" className="text-sm leading-6 text-red-200">
                {uploadError}
              </p>
            ) : null}
            {message ? (
              <output className="block text-sm leading-6 text-mint">
                {message}
              </output>
            ) : null}
          </div>
        </form>
        <div className="mt-5 border-t border-white/7 pt-4">
          <p className="mb-3 text-sm font-medium text-ink">
            Uploaded documents{library ? ` (${library.count})` : ''}
          </p>
          {loading ? (
            <output className="block text-sm text-ink-dim">
              Loading documents…
            </output>
          ) : null}
          {!loading && library?.count === 0 ? (
            <p className="text-sm text-ink-dim">
              Your uploaded documents will appear here.
            </p>
          ) : null}
          {library && library.count > 0 ? (
            <ul
              className="max-h-60 space-y-2 overflow-y-auto pr-1"
              aria-label="Uploaded documents"
            >
              {library.documents.map((document) => (
                <li
                  key={document.id}
                  className="flex min-w-0 items-start gap-3 rounded-lg border border-white/7 bg-canvas/40 p-3"
                >
                  <FileText className="mt-0.5 size-4 shrink-0 text-mint" />
                  <div className="min-w-0 flex-1">
                    <p className="break-words text-sm font-medium text-ink">
                      {document.title}
                    </p>
                    <p className="mt-1 break-all text-xs text-ink-dim">
                      {document.filename} · {fileSize(document.size_bytes)}
                    </p>
                  </div>
                  <span className="flex shrink-0 items-center gap-1 text-xs text-mint">
                    <Check className="size-3" /> Ready
                  </span>
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      </CardContent>
    </Card>
  );
}

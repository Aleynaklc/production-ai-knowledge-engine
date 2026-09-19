'use client';

import {
  Activity,
  ArrowUpRight,
  BookOpen,
  Check,
  CircleDot,
  Clock3,
  Database,
  FileText,
  KeyRound,
  LoaderCircle,
  Network,
  Search,
  Send,
  ShieldCheck,
  Sparkles,
  SquareTerminal,
  TriangleAlert,
} from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import type { ReactNode, SyntheticEvent } from 'react';

import { Badge } from '@/components/ui/badge';
import { DocumentLibrary } from '@/components/document-library';
import { WorkspaceGate } from '@/components/workspace-gate';
import { SourceViewer } from '@/components/source-viewer';
import type { SourceLocation } from '@/components/source-viewer';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { Skeleton } from '@/components/ui/skeleton';
import { Textarea } from '@/components/ui/textarea';
import {
  API_BASE_URL,
  AnswerResponse,
  RagStatus,
  SystemResponse,
  TraceStage,
  UploadedDocument,
  askKnowledgeEngine,
  getSystem,
} from '@/lib/api';

const sampleQuestions = [
  'What does the uploaded document say about access?',
  'What are the main requirements in this document?',
  'Which deadlines are mentioned in the document?',
];

const stageDetails: Record<
  TraceStage['name'],
  { label: string; icon: typeof Search; index: string }
> = {
  retrieval: { label: 'Retrieval', icon: Search, index: '01' },
  context_assembly: { label: 'Context assembly', icon: Database, index: '02' },
  generation: { label: 'Generation', icon: Sparkles, index: '03' },
  grounding: { label: 'Grounding gate', icon: ShieldCheck, index: '04' },
};

function statusConfig(status: RagStatus) {
  if (status === 'answered') {
    return {
      label: 'Answer with sources',
      className: 'border-mint/30 bg-mint/10 text-mint',
      icon: Check,
    };
  }
  if (status === 'abstained') {
    return {
      label: 'Insufficient context',
      className: 'border-amber-300/25 bg-amber-300/10 text-amber-200',
      icon: TriangleAlert,
    };
  }
  return {
    label: 'Answer rejected',
    className: 'border-red-300/25 bg-red-300/10 text-red-200',
    icon: ShieldCheck,
  };
}

function renderAnswer(text: string): ReactNode[] {
  return text.split(/(\[S\d+\])/g).map((part, index) => {
    if (/^\[S\d+\]$/.test(part)) {
      return (
        <span
          key={`${part}-${index}`}
          className="mx-0.5 inline-flex translate-y-[-1px] rounded bg-mint/12 px-1.5 py-0.5 font-mono text-[0.72em] font-semibold text-mint ring-1 ring-inset ring-mint/25"
        >
          {part.slice(1, -1)}
        </span>
      );
    }
    return <span key={`${part}-${index}`}>{part}</span>;
  });
}

function TracePanel({ response }: { response: AnswerResponse | null }) {
  const maxDuration = useMemo(
    () =>
      Math.max(
        ...(response?.trace.stages.map((stage) => stage.duration_ms) ?? [1]),
        1,
      ),
    [response],
  );

  return (
    <Card className="trace-card border-white/8 bg-panel/88 py-0 ring-0">
      <CardHeader className="border-b border-white/7 px-5 py-5 sm:px-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="eyebrow mb-2">
              <Activity className="size-3.5" /> System trace
            </div>
            <CardTitle className="font-display text-xl text-ink">
              Pipeline execution
            </CardTitle>
            <CardDescription className="mt-1 text-sm text-ink-dim">
              Every answer exposes its retrieval and grounding path.
            </CardDescription>
          </div>
          <Badge
            variant="outline"
            className="border-white/10 bg-white/[0.03] font-mono text-[10px] tracking-wide text-ink-dim"
          >
            {response ? `${response.trace.total_ms.toFixed(1)} ms` : 'WAITING'}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="px-5 py-6 sm:px-6">
        {!response ? (
          <div className="flex min-h-[370px] flex-col items-center justify-center text-center">
            <div className="trace-orbit mb-5">
              <Network className="size-6 text-mint" />
            </div>
            <p className="font-medium text-ink">No execution yet</p>
            <p className="mt-2 max-w-[260px] text-sm leading-6 text-ink-dim">
              Ask a question to inspect latency, token use, validation, and
              source lineage.
            </p>
          </div>
        ) : (
          <div>
            <div className="space-y-1">
              {response.trace.stages.map((stage, stageIndex) => {
                const details = stageDetails[stage.name];
                const StageIcon = details.icon;
                const width = Math.max(
                  (stage.duration_ms / maxDuration) * 100,
                  4,
                );
                return (
                  <div
                    key={stage.name}
                    className="relative grid grid-cols-[38px_1fr] gap-3 pb-6"
                  >
                    {stageIndex < response.trace.stages.length - 1 ? (
                      <span className="absolute bottom-0 left-[18px] top-9 w-px bg-white/8" />
                    ) : null}
                    <div className="relative z-10 flex size-9 items-center justify-center rounded-lg border border-white/8 bg-canvas text-mint">
                      <StageIcon className="size-4" />
                    </div>
                    <div className="pt-0.5">
                      <div className="flex items-center justify-between gap-4">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-[10px] text-mint/60">
                            {details.index}
                          </span>
                          <p className="text-sm font-medium text-ink">
                            {details.label}
                          </p>
                        </div>
                        <span className="font-mono text-[11px] text-ink-dim">
                          {stage.status === 'skipped'
                            ? 'SKIPPED'
                            : `${stage.duration_ms.toFixed(1)} ms`}
                        </span>
                      </div>
                      <div className="mt-2 h-1 overflow-hidden rounded-full bg-white/5">
                        <div
                          className="h-full rounded-full bg-gradient-to-r from-mint/45 to-mint transition-all"
                          style={{
                            width: `${stage.status === 'skipped' ? 0 : width}%`,
                          }}
                        />
                      </div>
                      <p className="mt-2 text-xs leading-5 text-ink-dim">
                        {stage.summary}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>

            <Separator className="my-1 bg-white/7" />
            <div className="grid grid-cols-3 gap-2 py-5">
              {[
                ['Context', response.trace.tokens.context],
                ['Input', response.trace.tokens.input],
                ['Output', response.trace.tokens.output],
              ].map(([label, value]) => (
                <div
                  key={label}
                  className="rounded-lg border border-white/7 bg-white/[0.025] p-3"
                >
                  <p className="font-mono text-lg text-ink">{value}</p>
                  <p className="mt-1 text-[10px] uppercase tracking-[0.12em] text-ink-faint">
                    {label} tokens
                  </p>
                </div>
              ))}
            </div>
            <div className="flex items-center justify-between gap-4 border-t border-white/7 pt-4 font-mono text-[10px] text-ink-faint">
              <span className="truncate">
                TRACE {response.trace_id.slice(0, 8).toUpperCase()}
              </span>
              <span>
                {response.trace.validation_issues.length} VALIDATION ISSUES
              </span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function AnswerLoading() {
  return (
    <Card className="border-white/8 bg-panel/88 ring-0">
      <CardContent className="space-y-4 py-7">
        <div className="flex items-center gap-3">
          <Skeleton className="size-7 bg-white/7" />
          <Skeleton className="h-4 w-32 bg-white/7" />
        </div>
        <Skeleton className="h-5 w-full bg-white/7" />
        <Skeleton className="h-5 w-[92%] bg-white/7" />
        <Skeleton className="h-5 w-[70%] bg-white/7" />
      </CardContent>
    </Card>
  );
}

function KnowledgeConsole() {
  const [source, setSource] = useState<SourceLocation | null>(null);
  const [question, setQuestion] = useState(sampleQuestions[0]);
  const [response, setResponse] = useState<AnswerResponse | null>(null);
  const [documents, setDocuments] = useState<UploadedDocument[]>([]);
  const [selectedDocument, setSelectedDocument] = useState('');
  const [system, setSystem] = useState<SystemResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getSystem()
      .then(setSystem)
      .catch(() => setSystem(null));
  }, []);

  async function submit(event: SyntheticEvent<HTMLFormElement, SubmitEvent>) {
    event.preventDefault();
    const normalizedQuestion = question.trim();
    if (!normalizedQuestion || isLoading) return;
    setIsLoading(true);
    setError(null);
    try {
      setResponse(
        await askKnowledgeEngine(
          normalizedQuestion,
          selectedDocument ? [selectedDocument] : [],
        ),
      );
    } catch (caught) {
      setError(
        caught instanceof Error ? caught.message : 'The answer request failed.',
      );
    } finally {
      setIsLoading(false);
    }
  }

  const baseStatus = response ? statusConfig(response.result.status) : null;
  const status = baseStatus ? { ...baseStatus, label: response?.result.extraction ? 'Source excerpts' : response?.result.calculation ? 'Calculated from sources' : baseStatus.label } : null;
  const StatusIcon = status?.icon;
  const citedEvidence = Boolean(response?.result.citations.length);
  const evidence = response
    ? citedEvidence
      ? response.result.citations
      : (response.result.retrieved_sources ?? [])
    : [];
  const answerText =
    response?.result.outcome_reason === 'generation_limit'
      ? 'The generated answer reached its length limit and could not be verified. Review the retrieved passages below, or ask about a smaller section.'
      : response?.result.outcome_reason === 'verification_failed'
        ? 'I could not verify an answer from the retrieved passages. You can inspect those passages below; they are not presented as a verified answer.'
        : (response?.result.answer ?? '');

  return (
    <main className="min-h-screen overflow-hidden bg-canvas text-ink">
      <div className="ambient-grid pointer-events-none fixed inset-0" />
      <header className="relative z-10 border-b border-white/7 bg-canvas/85 backdrop-blur-xl">
        <div className="mx-auto flex h-[68px] max-w-[1480px] items-center justify-between px-5 sm:px-8 lg:px-10">
          <div className="flex min-w-0 items-center gap-3">
            <div className="flex size-9 shrink-0 items-center justify-center rounded-xl border border-mint/25 bg-mint/10 shadow-[0_0_24px_rgba(113,241,199,0.08)]">
              <SquareTerminal className="size-[18px] text-mint" />
            </div>
            <div className="min-w-0">
              <p className="truncate font-display text-sm font-semibold tracking-tight text-ink">
                NovaStack{' '}
                <span className="font-normal text-ink-dim">
                  / Knowledge Console
                </span>
              </p>
              <p className="font-mono text-[9px] uppercase tracking-[0.2em] text-ink-faint">
                Grounded local intelligence
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="hidden items-center gap-2 rounded-full border border-white/7 bg-white/[0.025] px-3 py-1.5 sm:flex">
              <CircleDot
                className={`size-3 ${system ? 'text-mint' : 'text-amber-200'}`}
              />
              <span className="font-mono text-[10px] uppercase tracking-[0.12em] text-ink-dim">
                {system ? `API v${system.api_version}` : 'API offline'}
              </span>
            </div>
            <Badge
              variant="outline"
              className="border-white/10 bg-white/[0.03] font-mono text-[10px] text-ink-dim"
            >
              <KeyRound data-icon="inline-start" />
              PRIVATE WORKSPACE
            </Badge>
          </div>
        </div>
      </header>

      <div className="relative z-10 mx-auto max-w-[1480px] px-5 py-8 sm:px-8 sm:py-10 lg:px-10">
        <div className="mb-8 flex flex-col justify-between gap-5 lg:flex-row lg:items-end">
          <div className="max-w-3xl">
            <div className="eyebrow mb-4">
              <BookOpen className="size-3.5" /> Evidence before answers
            </div>
            <h1 className="font-display text-[clamp(1.8rem,3vw,3rem)] font-semibold leading-[1.1] tracking-[-0.035em] text-ink">
              Ask your knowledge base.
              <span className="block text-ink-faint">
                Inspect every decision.
              </span>
            </h1>
          </div>
          <p className="max-w-sm text-sm leading-6 text-ink-dim lg:pb-1">
            Local retrieval, local generation, citation validation—and a
            complete execution trace.
          </p>
        </div>

        <div className="grid items-start gap-5 xl:grid-cols-[minmax(0,1.25fr)_minmax(380px,0.75fr)]">
          <div className="space-y-5">
            <DocumentLibrary
              answering={isLoading}
              onBusyChange={setIsUploading}
              onUploaded={() => setResponse(null)}
              onDocumentsChange={(nextDocuments) => {
                setDocuments(nextDocuments);
                if (selectedDocument && !nextDocuments.some((document) => document.id === selectedDocument && document.active_version !== null)) {
                  setSelectedDocument('');
                  setResponse(null);
                }
              }}
            />
            <Card className="border-white/8 bg-panel/88 ring-0">
              <CardContent className="py-6">
                <form onSubmit={submit}>
                  <label
                    htmlFor="question-document"
                    className="mb-4 block text-sm text-ink-dim"
                  >
                    Search in
                    <select
                      id="question-document"
                      value={selectedDocument}
                      disabled={isLoading}
                      onChange={(event) => {
                        setSelectedDocument(event.target.value);
                        setResponse(null);
                      }}
                      className="mt-2 block w-full rounded-lg border border-white/15 bg-panel p-2 text-ink"
                    >
                      <option value="">All workspace documents</option>
                      {documents
                        .filter((document) => document.active_version !== null)
                        .map((document) => (
                          <option key={document.id} value={document.id}>
                            {document.filename}
                          </option>
                        ))}
                    </select>
                  </label>
                  <div className="mb-3 flex items-center justify-between gap-4">
                    <label
                      htmlFor="question"
                      className="text-sm font-medium text-ink"
                    >
                      Your question
                    </label>
                    <span className="font-mono text-[10px] text-ink-faint">
                      {question.length.toString().padStart(4, '0')} / 2000
                    </span>
                  </div>
                  <div className="relative">
                    <Textarea
                      id="question"
                      value={question}
                      onChange={(event) =>
                        setQuestion(event.target.value.slice(0, 2000))
                      }
                      onKeyDown={(event) => {
                        if (
                          (event.metaKey || event.ctrlKey) &&
                          event.key === 'Enter'
                        ) {
                          event.currentTarget.form?.requestSubmit();
                        }
                      }}
                      placeholder="Ask about NovaStack policies, APIs, incidents, or operations…"
                      className="min-h-36 resize-none border-white/9 bg-canvas/65 px-4 py-4 pr-14 text-base leading-7 text-ink placeholder:text-ink-faint focus-visible:border-mint/40 focus-visible:ring-mint/10"
                    />
                    <Button
                      type="submit"
                      size="icon-lg"
                      disabled={!question.trim() || isLoading}
                      aria-label="Ask knowledge engine"
                      className="absolute bottom-3 right-3 rounded-lg bg-mint text-canvas shadow-[0_8px_25px_rgba(113,241,199,0.16)] hover:bg-mint/85"
                    >
                      {isLoading ? (
                        <LoaderCircle className="size-4 animate-spin" />
                      ) : (
                        <Send className="size-4" />
                      )}
                    </Button>
                  </div>
                  <div className="mt-4 flex flex-wrap items-center gap-2">
                    <span className="mr-1 font-mono text-[9px] uppercase tracking-[0.16em] text-ink-faint">
                      Try
                    </span>
                    {sampleQuestions.map((sample) => (
                      <button
                        type="button"
                        key={sample}
                        onClick={() => setQuestion(sample)}
                        className="rounded-full border border-white/8 bg-white/[0.025] px-3 py-1.5 text-left text-xs text-ink-dim transition hover:border-mint/20 hover:bg-mint/[0.06] hover:text-ink"
                      >
                        {sample}
                      </button>
                    ))}
                  </div>
                </form>
              </CardContent>
            </Card>

            {error ? (
              <Card className="border-red-300/15 bg-red-300/[0.05] ring-0">
                <CardContent className="flex items-start gap-3 py-5">
                  <TriangleAlert className="mt-0.5 size-4 shrink-0 text-red-200" />
                  <div>
                    <p className="text-sm font-medium text-red-100">
                      The engine did not respond
                    </p>
                    <p className="mt-1 text-sm leading-6 text-red-100/60">
                      {error}
                    </p>
                    <p className="mt-2 font-mono text-[10px] text-red-100/40">
                      {API_BASE_URL}
                    </p>
                  </div>
                </CardContent>
              </Card>
            ) : null}

            {isLoading ? <AnswerLoading /> : null}

            {!isLoading && response && status && StatusIcon ? (
              <Card className="answer-card border-white/8 bg-panel/88 ring-0">
                <CardHeader className="border-b border-white/7 px-5 py-5 sm:px-6">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <Badge variant="outline" className={status.className}>
                      <StatusIcon data-icon="inline-start" />
                      {status.label}
                    </Badge>
                    <div className="flex items-center gap-3 font-mono text-[10px] text-ink-faint">
                      <span className="flex items-center gap-1.5">
                        <Clock3 className="size-3" />{' '}
                        {response.result.timings.total_ms.toFixed(1)} ms
                      </span>
                      <span>
                        {response.result.context_source_count} passages reviewed
                        · {response.result.citations.length} citations
                      </span>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="px-5 py-7 sm:px-7 sm:py-8">
                  <p className="answer-copy whitespace-pre-wrap font-display text-[clamp(1.15rem,2.2vw,1.55rem)] leading-[1.65] tracking-[-0.015em] text-ink">
                    {renderAnswer(answerText)}
                  </p>
                  {response.result.fallback_used ? (
                    <p className="mt-5 flex items-center gap-2 text-xs text-ink-dim">
                      <ShieldCheck className="size-3.5 text-mint" />
                      High-confidence extractive fallback used.
                    </p>
                  ) : null}
                </CardContent>
              </Card>
            ) : null}

            {!isLoading && response && evidence.length ? (
              <section aria-labelledby="sources-heading">
                {source && (
                  <SourceViewer
                    key={`${source.documentId}:${source.version}:${source.unit}`}
                    source={source}
                    onClose={() => setSource(null)}
                  />
                )}
                <div className="mb-3 flex items-center justify-between">
                  <h2 id="sources-heading" className="eyebrow">
                    <FileText className="size-3.5" />{' '}
                    {citedEvidence
                      ? 'Cited evidence'
                      : 'Retrieved passages — answer not verified'}
                  </h2>
                  <span className="font-mono text-[10px] text-ink-faint">
                    {response.result.context_token_count} CONTEXT TOKENS
                  </span>
                </div>
                <div className="grid gap-3 md:grid-cols-2">
                  {evidence.map((citation) => (
                    <Card
                      key={citation.citation_id}
                      size="sm"
                      className="source-card border-white/8 bg-panel/70 ring-0"
                    >
                      <CardContent className="py-1">
                        <div className="flex items-start justify-between gap-4">
                          <Badge className="bg-mint/10 font-mono text-[10px] text-mint">
                            {citation.citation_id}
                          </Badge>
                          <span className="font-mono text-[10px] text-ink-faint">
                            SCORE {citation.retrieval_score.toFixed(2)}
                          </span>
                        </div>
                        <h3 className="mt-4 text-sm font-medium text-ink">
                          {citation.title}
                        </h3>
                        <p className="mt-2 line-clamp-3 text-xs leading-5 text-ink-dim">
                          {citation.snippet}
                        </p>
                        <div className="mt-4 flex items-center justify-between gap-4 border-t border-white/7 pt-3">
                          <span className="truncate font-mono text-[9px] uppercase tracking-[0.1em] text-ink-faint">
                            {citation.source}
                          </span>
                          {citation.document_version && citation.source_unit ? (
                            <Button
                              variant="outline"
                              onClick={() =>
                                setSource({
                                  documentId: citation.document_id,
                                  version: citation.document_version!,
                                  unit: citation.source_unit!,
                                })
                              }
                            >
                              Open{' '}
                              {citation.source_kind === 'page'
                                ? 'page'
                                : 'section'}{' '}
                              {citation.source_unit}
                              <ArrowUpRight className="size-3.5" />
                            </Button>
                          ) : null}
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </section>
            ) : null}
          </div>

          <aside className="xl:sticky xl:top-6">
            <TracePanel response={response} />
          </aside>
        </div>

        <footer className="mt-10 flex flex-col justify-between gap-3 border-t border-white/7 pt-5 font-mono text-[9px] uppercase tracking-[0.14em] text-ink-faint sm:flex-row">
          <span>Local models · bounded traces · strict grounding</span>
          <span>
            {system?.model.split('/').at(-1) ?? 'Local model unavailable'}
          </span>
        </footer>
      </div>
    </main>
  );
}

export default function Home() {
  return (
    <WorkspaceGate>
      {(workspace) => <KnowledgeConsole key={workspace.id} />}
    </WorkspaceGate>
  );
}

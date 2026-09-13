export type RagStatus = 'answered' | 'abstained' | 'rejected';

export interface AnswerCitation {
  citation_id: string;
  chunk_id: string;
  document_id: string;
  source: string;
  title: string;
  snippet: string;
  retrieval_score: number;
  document_version?: number;
  source_unit?: number;
  source_kind?: string;
}

export interface TraceStage {
  name: 'retrieval' | 'context_assembly' | 'generation' | 'grounding';
  status: 'completed' | 'skipped';
  duration_ms: number;
  summary: string;
  metrics: Record<string, string | number | boolean>;
}

export interface TraceSource {
  citation_id: string;
  chunk_id: string;
  document_id: string;
  source: string;
  title: string;
  retrieval_score: number;
}

export interface SystemTrace {
  cache_hit: boolean;
  trace_id: string;
  request_id: string;
  created_at: string;
  question: string;
  status: RagStatus;
  total_ms: number;
  stages: TraceStage[];
  sources: TraceSource[];
  tokens: { context: number; input: number; output: number };
  validation_issues: string[];
  fallback_used: boolean;
}

export interface AnswerResult {
  cache_hit: boolean;
  question: string;
  status: RagStatus;
  answer: string;
  raw_answer: string | null;
  fallback_used: boolean;
  citations: AnswerCitation[];
  validation: {
    valid: boolean;
    abstained: boolean;
    cited_source_ids: string[];
    unknown_source_ids: string[];
    uncited_claims: string[];
    unsupported_claims: string[];
    issues: string[];
  };
  context_source_count: number;
  context_token_count: number;
  timings: {
    retrieval_ms: number;
    context_ms: number;
    generation_ms: number;
    grounding_ms: number;
    total_ms: number;
  };
  input_tokens: number;
  output_tokens: number;
}

export interface AnswerResponse {
  request_id: string;
  trace_id: string;
  result: AnswerResult;
  trace: SystemTrace;
}

export interface SystemResponse {
  service: string;
  environment: string;
  api_version: string;
  inference: 'local';
  model: string;
  retrieval: string;
  api_key_required: false;
  trace_retention: number;
}

export interface UploadedDocument {
  id: string;
  filename: string;
  title: string;
  size_bytes: number;
  created_at: string;
  chunk_count: number;
  chunk_strategy: 'fixed' | 'recursive';
  chunk_size_tokens: number;
  overlap_tokens: number;
  status: 'processing' | 'ready' | 'failed';
  version: number;
  active_version: number | null;
  error: string | null;
}

export interface DocumentLibraryResponse {
  documents: UploadedDocument[];
  count: number;
  max_file_bytes: number;
  supported_extensions: string[];
  scope: 'workspace';
  role: 'owner' | 'editor' | 'reader';
}

export interface DocumentUploadResponse {
  document: UploadedDocument;
  duplicate: boolean;
}

interface ErrorResponse {
  error?: { code?: string; message?: string; request_id?: string };
}

export const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000'
).replace(/\/$/, '');

let workspaceId = '';
export function selectWorkspace(id: string) {
  workspaceId = id;
}

export async function apiFetch<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  let response: Response;
  try {
    const headers = new Headers(init?.headers);
    headers.set('X-Workspace-ID', workspaceId);
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      credentials: 'include',
      headers,
    });
  } catch {
    throw new Error(
      'The knowledge service is unreachable. Check your connection and try again.',
    );
  }
  if (!response.ok) {
    const payload = (await response.json().catch(() => ({}))) as ErrorResponse;
    throw new Error(
      payload.error?.message ??
        `Request failed with status ${response.status}.`,
    );
  }
  return (await response.json()) as T;
}

export function getSystem(): Promise<SystemResponse> {
  return apiFetch<SystemResponse>('/api/v1/system');
}

export function askKnowledgeEngine(question: string): Promise<AnswerResponse> {
  return apiFetch<AnswerResponse>('/api/v1/answers', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  });
}

export function getDocuments(): Promise<DocumentLibraryResponse> {
  return apiFetch<DocumentLibraryResponse>('/api/v1/documents');
}

export function uploadDocument(file: File): Promise<DocumentUploadResponse> {
  const parameters = new URLSearchParams({ filename: file.name });
  return apiFetch<DocumentUploadResponse>(`/api/v1/documents?${parameters}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/octet-stream' },
    body: file,
  });
}

export interface Workspace {
  id: string;
  name: string;
  role: 'owner' | 'editor' | 'reader';
}
export interface Session {
  user: { id: string; email: string };
  workspaces: Workspace[];
}
export const getSession = () => apiFetch<Session>('/api/v1/auth/me');
export function signIn(
  email: string,
  password: string,
  workspaceName?: string,
) {
  return apiFetch<Session>(
    `/api/v1/auth/${workspaceName === undefined ? 'login' : 'register'}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, workspace_name: workspaceName }),
    },
  );
}
export const signOut = () =>
  apiFetch('/api/v1/auth/logout', { method: 'POST' });
export const deleteDocument = (id: string) =>
  apiFetch(`/api/v1/documents/${encodeURIComponent(id)}`, { method: 'DELETE' });
export const retryDocument = (id: string) =>
  apiFetch(`/api/v1/documents/${encodeURIComponent(id)}/retry`, {
    method: 'POST',
  });
export function replaceDocument(id: string, file: File) {
  return apiFetch<DocumentUploadResponse>(
    `/api/v1/documents/${encodeURIComponent(id)}?${new URLSearchParams({ filename: file.name })}`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/octet-stream' },
      body: file,
    },
  );
}
export interface SourcePage {
  filename: string;
  version: number;
  unit: { number: number; kind: string; text: string };
  unit_count: number;
}
export function getSource(id: string, version: number, unit: number) {
  return apiFetch<SourcePage>(
    `/api/v1/documents/${encodeURIComponent(id)}/source?${new URLSearchParams({ version: String(version), unit: String(unit) })}`,
  );
}
export async function downloadSource(
  id: string,
  version: number,
  filename: string,
) {
  const response = await fetch(
    `${API_BASE_URL}/api/v1/documents/${encodeURIComponent(id)}/download?version=${version}`,
    {
      credentials: 'include',
      headers: { 'X-Workspace-ID': workspaceId },
    },
  );
  if (!response.ok)
    throw new Error('The original document could not be downloaded.');
  const url = URL.createObjectURL(await response.blob());
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

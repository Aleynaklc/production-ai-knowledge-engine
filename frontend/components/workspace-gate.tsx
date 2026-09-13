'use client';

import { useEffect, useState } from 'react';
import type { ReactNode, SyntheticEvent } from 'react';
import { Button } from '@/components/ui/button';
import {
  apiFetch,
  getSession,
  selectWorkspace,
  signIn,
  signOut,
} from '@/lib/api';
import type { Session, Workspace } from '@/lib/api';

const inputStyle =
  'w-full rounded-lg border border-white/15 bg-canvas p-3 text-base text-ink focus:outline-mint';

function Members({ workspace }: { workspace: Workspace }) {
  const [members, setMembers] = useState<
    { id: string; email: string; role: string }[]
  >([]);
  const [email, setEmail] = useState('');
  const [role, setRole] = useState('reader');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  async function refresh() {
    const result = await apiFetch<{ members: typeof members }>(
      '/api/v1/workspace/members',
    );
    setMembers(result.members);
  }
  useEffect(() => {
    let active = true;
    apiFetch<{ members: typeof members }>('/api/v1/workspace/members')
      .then((result) => {
        if (active) setMembers(result.members);
      })
      .catch((caught: Error) => {
        if (active) setError(caught.message);
      });
    return () => {
      active = false;
    };
  }, [workspace.id]);
  async function save(event: SyntheticEvent) {
    event.preventDefault();
    setBusy(true);
    setError('');
    try {
      await apiFetch('/api/v1/workspace/members', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, role }),
      });
      setEmail('');
      await refresh();
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function remove(id: string) {
    setBusy(true);
    setError('');
    try {
      await apiFetch(`/api/v1/workspace/members/${id}`, { method: 'DELETE' });
      await refresh();
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <details className="mx-auto max-w-[1480px] px-5 pb-4 text-sm text-ink">
      <summary className="cursor-pointer py-2">
        Manage workspace members
      </summary>
      <p className="my-3 text-ink-dim">
        Add an existing account. Readers can ask questions and view sources;
        editors can also manage documents.
      </p>
      <form onSubmit={save} className="flex flex-wrap items-end gap-3">
        <label className="min-w-48 flex-1">
          Email
          <input
            type="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className={inputStyle}
          />
        </label>
        <label>
          Role
          <select
            className={inputStyle}
            value={role}
            onChange={(e) => setRole(e.target.value)}
          >
            <option value="reader">Reader</option>
            <option value="editor">Editor</option>
          </select>
        </label>
        <Button type="submit" disabled={busy}>
          Save member
        </Button>
      </form>
      {error && (
        <p role="alert" className="my-3 text-red-200">
          {error}
        </p>
      )}
      <ul className="mt-4 space-y-2">
        {members.map((member) => (
          <li
            key={member.id}
            className="flex flex-wrap items-center justify-between gap-3 border-t border-white/10 py-2"
          >
            <span>
              {member.email} · {member.role}
            </span>
            {member.role !== 'owner' && (
              <Button
                disabled={busy}
                variant="ghost"
                onClick={() => remove(member.id)}
              >
                Remove access
              </Button>
            )}
          </li>
        ))}
      </ul>
    </details>
  );
}

export function WorkspaceGate({
  children,
}: {
  children: (workspace: Workspace) => ReactNode;
}) {
  const [session, setSession] = useState<Session | null>(null);
  const [selected, setSelected] = useState('');
  const [loading, setLoading] = useState(true);
  const [registering, setRegistering] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [error, setError] = useState('');

  function accept(result: Session) {
    const id = result.workspaces[0]?.id ?? '';
    selectWorkspace(id);
    setSelected(id);
    setSession(result);
    setPassword('');
  }
  useEffect(() => {
    let active = true;
    getSession()
      .then((result) => {
        if (active) accept(result);
      })
      .catch(() => {})
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);
  async function submit(event: SyntheticEvent) {
    event.preventDefault();
    setError('');
    setLoading(true);
    try {
      accept(await signIn(email, password, registering ? name : undefined));
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setLoading(false);
    }
  }
  async function logout() {
    try {
      await signOut();
      selectWorkspace('');
      setSession(null);
      setSelected('');
      setError('');
    } catch (caught) {
      setError((caught as Error).message);
    }
  }
  if (loading && !session)
    return (
      <main className="min-h-screen bg-canvas p-10 text-ink" aria-busy="true">
        Opening your workspace…
      </main>
    );
  if (!session)
    return (
      <main className="flex min-h-screen items-center justify-center bg-canvas p-5 text-ink">
        <section className="w-full max-w-md rounded-2xl border border-white/10 bg-panel p-7 shadow-xl">
          <p className="mb-3 text-sm text-mint">NovaStack Knowledge Console</p>
          <h1 className="font-display text-3xl">
            {registering ? 'Create your workspace' : 'Sign in'}
          </h1>
          <p className="my-4 text-base text-ink-dim">
            Your documents and answers stay within your workspace.
          </p>
          <form onSubmit={submit} className="space-y-4">
            <label className="block text-sm">
              Email
              <input
                className={inputStyle}
                type="email"
                autoComplete="email"
                maxLength={254}
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </label>
            <label className="block text-sm">
              Password
              <input
                className={inputStyle}
                type="password"
                autoComplete={registering ? 'new-password' : 'current-password'}
                minLength={registering ? 12 : 1}
                maxLength={128}
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </label>
            {registering && (
              <>
                <p className="text-sm text-ink-dim">
                  Use at least 12 characters.
                </p>
                <label className="block text-sm">
                  Workspace name
                  <input
                    className={inputStyle}
                    maxLength={100}
                    required
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                  />
                </label>
              </>
            )}
            {error && (
              <p role="alert" className="text-sm text-red-200">
                {error}
              </p>
            )}
            <Button className="h-11 w-full" type="submit" disabled={loading}>
              {registering ? 'Create account' : 'Sign in'}
            </Button>
          </form>
          <Button
            variant="link"
            className="mt-4"
            onClick={() => {
              setRegistering(!registering);
              setError('');
            }}
          >
            {registering
              ? 'Already have an account? Sign in'
              : 'Create an account'}
          </Button>
        </section>
      </main>
    );
  const workspace = session.workspaces.find((item) => item.id === selected);
  return (
    <div className="min-h-screen bg-canvas text-ink">
      <div className="border-b border-white/10 bg-panel">
        <div className="mx-auto flex max-w-[1480px] flex-wrap items-center justify-between gap-4 p-5">
          <label className="flex items-center gap-3 text-sm">
            Workspace
            <select
              className="max-w-64 rounded-lg border border-white/15 bg-canvas p-2 text-base"
              value={selected}
              onChange={(e) => {
                selectWorkspace(e.target.value);
                setSelected(e.target.value);
              }}
            >
              {session.workspaces.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name} · {item.role}
                </option>
              ))}
            </select>
          </label>
          <div className="flex flex-wrap items-center gap-3 text-sm">
            <span>{session.user.email}</span>
            <Button variant="outline" onClick={logout}>
              Sign out
            </Button>
          </div>
        </div>
        {error && (
          <p role="alert" className="px-5 pb-3 text-red-200">
            {error}
          </p>
        )}
        {workspace?.role === 'owner' && (
          <Members key={workspace.id} workspace={workspace} />
        )}
      </div>
      {workspace ? (
        <div key={workspace.id}>{children(workspace)}</div>
      ) : (
        <p className="p-8">
          No workspace access. Ask a workspace owner to add your account, then
          sign in again.
        </p>
      )}
    </div>
  );
}

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type Json = Record<string, unknown>;
type Session = {
  session_id: string;
  title: string;
  status: string;
  active_run_id?: string | null;
  updated_at: string;
};

function pathSession(): string | null {
  const match = window.location.pathname.match(/^\/sessions\/([^/]+)$/);
  return match ? decodeURIComponent(match[1]) : null;
}

function pathRun(): string | null {
  const match = window.location.pathname.match(/^\/runs\/([^/]+)$/);
  return match ? decodeURIComponent(match[1]) : null;
}

async function exchangeBootstrap(): Promise<void> {
  const params = new URLSearchParams(window.location.hash.slice(1));
  const token = params.get("bootstrap");
  if (!token) return;
  history.replaceState(null, "", window.location.pathname + window.location.search);
  const response = await fetch("/api/bootstrap", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "same-origin",
    body: JSON.stringify({ token }),
  });
  if (!response.ok) throw new Error("Bootstrap token is invalid or expired");
}

async function getJson(path: string): Promise<Json> {
  const response = await fetch(path, { credentials: "same-origin" });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return (await response.json()) as Json;
}

export function App() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selected, setSelected] = useState<string | null>(pathSession());
  const [detail, setDetail] = useState<Json | null>(null);
  const [runDetail, setRunDetail] = useState<Json | null>(null);
  const [events, setEvents] = useState<Json[]>([]);
  const [connection, setConnection] = useState("idle");
  const [error, setError] = useState<string | null>(null);
  const lastCursor = useRef(0);

  const loadSessions = useCallback(async () => {
    const data = await getJson("/api/sessions");
    setSessions((data.sessions as Session[]) ?? []);
  }, []);

  useEffect(() => {
    exchangeBootstrap().then(loadSessions).catch((reason: unknown) => {
      setError(reason instanceof Error ? reason.message : String(reason));
    });
  }, [loadSessions]);

  useEffect(() => {
    if (!selected) { setDetail(null); return; }
    getJson(`/api/sessions/${encodeURIComponent(selected)}`)
      .then(setDetail)
      .catch((reason: unknown) => setError(String(reason)));
  }, [selected]);

  useEffect(() => {
    const runId = pathRun();
    if (!runId) return;
    getJson(`/api/runs/${encodeURIComponent(runId)}`)
      .then(setRunDetail)
      .catch((reason: unknown) => setError(String(reason)));
  }, []);

  useEffect(() => {
    if (!selected) return;
    setEvents([]);
    lastCursor.current = 0;
    setConnection("connecting");
    const source = new EventSource(`/api/events?session_id=${encodeURIComponent(selected)}`);
    source.onopen = () => setConnection("connected");
    source.addEventListener("event", (raw) => {
      const envelope = JSON.parse((raw as MessageEvent).data) as Json;
      const cursor = Number(envelope.cursor ?? 0);
      if (!Number.isInteger(cursor) || cursor <= 0) return;
      // Delivery is at-least-once. Reject duplicates and stale/out-of-order
      // envelopes before they enter React state, including cursors that have
      // already fallen out of the 500-item render window.
      if (cursor <= lastCursor.current) return;
      lastCursor.current = cursor;
      setEvents((current) => {
        return [...current, envelope]
          .sort((left, right) => Number(left.cursor) - Number(right.cursor))
          .slice(-500);
      });
    });
    source.addEventListener("overflow", (raw) => {
      setConnection(`overflow: ${(raw as MessageEvent).data}`);
    });
    source.addEventListener("core.disconnected", () => {
      setConnection("core disconnected");
    });
    source.addEventListener("compatibility.error", (raw) => {
      setConnection(`compatibility error: ${(raw as MessageEvent).data}`);
    });
    source.onerror = () => setConnection("reconnecting");
    return () => source.close();
  }, [selected]);

  const latestRun = useMemo(() => detail?.latest_run as Json | undefined, [detail]);

  function chooseSession(sessionId: string) {
    history.pushState(null, "", `/sessions/${encodeURIComponent(sessionId)}`);
    setRunDetail(null);
    setSelected(sessionId);
  }

  return (
    <main className="mx-auto min-h-screen max-w-7xl p-5 md:p-8">
      <header className="mb-7 flex items-end justify-between border-b border-slate-800 pb-5">
        <div><p className="text-xs tracking-[.3em] text-cyan-400">LOCAL AGENT RUNTIME</p><h1 className="text-3xl font-semibold">TARS-Agent</h1></div>
        <span className="rounded-full border border-slate-700 px-3 py-1 text-xs text-slate-400">read-only</span>
      </header>
      {error && <div role="alert" className="mb-5 rounded border border-red-800 bg-red-950/50 p-3 text-red-200">{error}</div>}
      <div className="grid gap-5 lg:grid-cols-[340px_1fr]">
        <section className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
          <h2 className="mb-4 font-medium">Sessions</h2>
          <div className="space-y-2" data-testid="session-list">
            {sessions.map((session) => (
              <button key={session.session_id} onClick={() => chooseSession(session.session_id)} className={`w-full rounded-lg border p-3 text-left ${selected === session.session_id ? "border-cyan-600 bg-cyan-950/30" : "border-slate-800 bg-slate-950/50"}`}>
                <span className="block truncate font-medium">{session.title || session.session_id}</span>
                <span className="mt-1 flex justify-between text-xs text-slate-500"><code>{session.session_id}</code><span>{session.status}</span></span>
              </button>
            ))}
          </div>
        </section>
        <div className="space-y-5">
          <section className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
            <div className="flex justify-between"><h2 className="font-medium">Run status</h2><span data-testid="connection-status" className="text-xs text-cyan-400">{connection}</span></div>
            {selected || runDetail ? <pre data-testid="run-status" className="mt-4 overflow-auto whitespace-pre-wrap text-sm text-slate-300">{JSON.stringify(runDetail ?? latestRun ?? detail, null, 2)}</pre> : <p className="mt-4 text-slate-500">Select a Session.</p>}
          </section>
          <section className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
            <h2 className="mb-4 font-medium">Event timeline</h2>
            <div data-testid="event-timeline" className="max-h-[52vh] space-y-2 overflow-auto">
              {events.map((envelope) => <pre key={String(envelope.cursor)} className="whitespace-pre-wrap break-words rounded border border-slate-800 bg-slate-950 p-3 text-xs text-slate-300">{JSON.stringify(envelope, null, 2)}</pre>)}
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}

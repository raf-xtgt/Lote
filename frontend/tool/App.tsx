import { useEffect, useState } from "react";

type Health = { status: string; checks: Record<string, boolean> };

export function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/health")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setHealth)
      .catch((e: Error) => setError(e.message));
  }, []);

  return (
    <main style={{ fontFamily: "system-ui, sans-serif", padding: 24 }}>
      <h1>Lote</h1>
      <p>Walking skeleton (setup 01).</p>
      <p>
        API health: <strong>{error ? `error: ${error}` : (health?.status ?? "checking...")}</strong>
      </p>
      {health && <pre>{JSON.stringify(health.checks, null, 2)}</pre>}
      <p>
        <a href="/player/">Open the player</a>
      </p>
    </main>
  );
}

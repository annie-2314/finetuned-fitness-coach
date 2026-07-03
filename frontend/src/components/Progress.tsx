import { useEffect, useState } from "react";
import api from "../api";

interface Prog { total_sessions: number; avg_rpe: number; top_lifts: Record<string, number>; }

export default function Progress({ refresh }: { refresh: number }) {
  const [p, setP] = useState<Prog | null>(null);
  useEffect(() => { api.get("/logs/progress").then((r) => setP(r.data)).catch(() => {}); }, [refresh]);
  if (!p) return null;

  return (
    <div className="card p-5 space-y-4">
      <h3 className="font-display font-semibold">Progress</h3>
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-surface2/60 border border-hair rounded-xl px-4 py-3">
          <div className="stat text-2xl font-bold text-teal">{p.total_sessions}</div>
          <div className="text-muted text-xs">sessions logged</div>
        </div>
        <div className="bg-surface2/60 border border-hair rounded-xl px-4 py-3">
          <div className="stat text-2xl font-bold">{p.avg_rpe || "—"}</div>
          <div className="text-muted text-xs">avg effort (RPE)</div>
        </div>
      </div>
      {Object.keys(p.top_lifts).length > 0 && (
        <div className="text-sm">
          <div className="text-muted text-xs uppercase tracking-wide mb-1">Best lifts</div>
          {Object.entries(p.top_lifts).map(([k, v]) => (
            <div key={k} className="flex justify-between border-b border-hair/70 py-1.5">
              <span className="truncate mr-2">{k}</span><span className="stat text-teal">{v} kg</span>
            </div>
          ))}
        </div>
      )}
      {p.total_sessions === 0 && (
        <p className="text-muted text-sm">No sessions yet — log a set and your stats appear here.</p>
      )}
    </div>
  );
}

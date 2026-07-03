import { useEffect, useState } from "react";
import api from "../api";

interface Prog { total_sessions: number; avg_rpe: number; top_lifts: Record<string, number>; }

export default function Progress({ refresh }: { refresh: number }) {
  const [p, setP] = useState<Prog | null>(null);
  useEffect(() => { api.get("/logs/progress").then((r) => setP(r.data)).catch(() => {}); }, [refresh]);
  if (!p) return null;
  return (
    <div className="bg-white rounded-xl shadow p-5 space-y-3">
      <h3 className="font-semibold">📊 Progress</h3>
      <div className="flex gap-3 text-sm">
        <span className="bg-slate-100 rounded-lg px-3 py-1">{p.total_sessions} sessions</span>
        <span className="bg-slate-100 rounded-lg px-3 py-1">avg RPE {p.avg_rpe}</span>
      </div>
      {Object.keys(p.top_lifts).length > 0 && (
        <div className="text-sm">
          <div className="text-slate-500 mb-1">Best lifts:</div>
          {Object.entries(p.top_lifts).map(([k, v]) => (
            <div key={k} className="flex justify-between border-b py-1"><span>{k}</span><span>{v} kg</span></div>
          ))}
        </div>
      )}
    </div>
  );
}

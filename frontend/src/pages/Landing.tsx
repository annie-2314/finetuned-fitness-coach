import { useNavigate } from "react-router-dom";
import { isLoggedIn } from "../api";

function MockPlan() {
  return (
    <div className="card p-5 w-full max-w-sm rotate-1 hover:rotate-0 transition-transform duration-300">
      <div className="flex items-baseline justify-between mb-3">
        <span className="font-display font-semibold">Monday</span>
        <span className="text-grad text-sm font-medium">Full body</span>
      </div>
      <div className="flex flex-wrap gap-2 mb-4 text-xs">
        <span className="chip"><span className="stat text-teal mr-1">08:00</span> Breakfast</span>
        <span className="chip border-energyTo/40 bg-energyFrom/10"><span className="stat text-energyTo mr-1">18:00</span> Workout</span>
        <span className="chip"><span className="stat text-muted mr-1">22:30</span> Sleep</span>
      </div>
      {[["Goblet Squat", "3 × 10"], ["Push Up", "3 × 12"], ["Dumbbell Row", "3 × 10"]].map(([n, s], i) => (
        <div key={i} className="flex items-center gap-3 bg-surface2/60 border border-hair rounded-lg p-2.5 mb-2">
          <div className="w-9 h-9 rounded-md grad-energy grid place-items-center text-white text-sm">🏋️</div>
          <div className="min-w-0"><div className="text-sm font-medium truncate">{n}</div>
            <div className="stat text-xs text-muted">{s} · 90s rest</div></div>
        </div>
      ))}
      <div className="flex gap-2 mt-3 text-xs">
        <span className="chip"><span className="stat text-teal">2,070</span> kcal</span>
        <span className="chip"><span className="stat">150g</span> protein</span>
      </div>
    </div>
  );
}

function Feature({ icon, title, body }: { icon: string; title: string; body: string }) {
  return (
    <div className="card p-6">
      <div className="grad-energy w-11 h-11 rounded-xl grid place-items-center text-white text-xl mb-4">{icon}</div>
      <h3 className="font-display font-semibold text-lg">{title}</h3>
      <p className="text-muted text-sm mt-1.5 leading-relaxed">{body}</p>
    </div>
  );
}

export default function Landing() {
  const nav = useNavigate();
  const go = (mode: "signup" | "login") =>
    isLoggedIn() ? nav("/dashboard") : nav("/login", { state: { mode } });

  return (
    <div className="min-h-screen">
      {/* Nav */}
      <header className="max-w-6xl mx-auto px-5 sm:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="grad-energy w-9 h-9 rounded-xl grid place-items-center text-white shadow-glow">⚡</div>
          <span className="font-display text-lg font-bold tracking-tight">Coach<span className="text-grad">AI</span></span>
        </div>
        <div className="flex gap-2">
          <button onClick={() => go("login")} className="btn-ghost text-sm px-4 py-1.5">Log in</button>
          <button onClick={() => go("signup")} className="btn-energy text-sm px-4 py-1.5">Get started</button>
        </div>
      </header>

      {/* Hero */}
      <section className="max-w-6xl mx-auto px-5 sm:px-8 pt-10 sm:pt-16 pb-16
                          grid lg:grid-cols-2 gap-10 items-center">
        <div>
          <span className="chip text-xs text-muted">⚡ Personal AI coaching</span>
          <h1 className="font-display text-4xl sm:text-5xl font-bold leading-[1.05] mt-4">
            The coach that <span className="text-grad">adapts</span> to you — every week.
          </h1>
          <p className="text-muted text-lg mt-5 max-w-md leading-relaxed">
            Get a personalized workout and nutrition plan built on your goal, equipment, and injuries —
            then watch it reshape itself around how your sessions actually felt.
          </p>
          <div className="flex flex-wrap gap-3 mt-7">
            <button onClick={() => go("signup")} className="btn-energy px-6 py-3">Get started free</button>
            <button onClick={() => go("login")} className="btn-ghost px-6 py-3 font-medium">I have an account</button>
          </div>
          <p className="text-muted/80 text-sm mt-5">
            Powered by a fine-tuned model · <span className="stat text-teal">97%</span> schema-valid plans on held-out tests
          </p>
        </div>
        <div className="flex justify-center lg:justify-end">
          <MockPlan />
        </div>
      </section>

      {/* Features */}
      <section className="max-w-6xl mx-auto px-5 sm:px-8 pb-16">
        <div className="grid md:grid-cols-3 gap-5">
          <Feature icon="🎯" title="Built around you"
            body="Goal, gear, diet, and any injuries go in — a complete weekly plan with meal and sleep timing comes out in seconds." />
          <Feature icon="🔄" title="Adapts to your logs"
            body="Log how each session felt. Too hard, too easy, or a niggle? Next week's plan adjusts the load and swaps risky moves." />
          <Feature icon="🛡️" title="Grounded, not guessed"
            body="Real exercises with demo images, macro targets, and injury-aware programming — no made-up movements or numbers." />
        </div>
      </section>

      {/* How it works */}
      <section className="max-w-6xl mx-auto px-5 sm:px-8 pb-20">
        <div className="card p-8">
          <h2 className="font-display text-2xl font-bold mb-6">How it works</h2>
          <div className="grid sm:grid-cols-3 gap-6">
            {[["Set your profile", "Age, goal, equipment, injuries, days per week."],
              ["Get your plan", "A full week of workouts + nutrition, tailored and safe."],
              ["Log & adapt", "Track sessions; your coach evolves the plan weekly."]].map(([t, b], i) => (
              <div key={i}>
                <div className="stat text-grad text-3xl font-bold">{String(i + 1).padStart(2, "0")}</div>
                <h3 className="font-display font-semibold mt-2">{t}</h3>
                <p className="text-muted text-sm mt-1">{b}</p>
              </div>
            ))}
          </div>
          <div className="mt-8">
            <button onClick={() => go("signup")} className="btn-energy px-6 py-3">Build my first plan</button>
          </div>
        </div>
      </section>

      <footer className="max-w-6xl mx-auto px-5 sm:px-8 py-8 border-t border-hair text-muted/70 text-sm">
        CoachAI · General fitness guidance, not medical advice.
      </footer>
    </div>
  );
}

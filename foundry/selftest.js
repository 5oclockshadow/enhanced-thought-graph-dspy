/* Headless self-test: proves the engine writes & runs real code and that
 * evolution improves MEASURED fitness on every task. Exit 1 on any failure. */
const FB = require("./foundry-builder.js");

let failures = 0;
function check(cond, msg) { if (!cond) { console.log("  ✗ " + msg); failures++; } else { console.log("  ✓ " + msg); } }

console.log("Graph Foundry · Universal Builder — self-test\n");
for (const name of Object.keys(FB.TASKS)) {
  console.log(name + ":");
  const r = FB.runRound(name, { seed: 7, popSize: 60, gens: 24 });
  check(r.finalBest >= 0 && r.finalBest <= 1, `fitness in [0,1] (got ${r.finalBest.toFixed(3)})`);
  check(r.finalBest > r.initialMean + 0.02, `improved over random mean (${r.initialMean.toFixed(3)} → ${r.finalBest.toFixed(3)})`);
  check(r.evals >= 60 * 24, `ran real evaluations (${r.evals})`);
  check(typeof r.spec.source === "string" && r.spec.source.length > 10, "emitted executable source");
  check(Array.isArray(r.spec.genome) && r.spec.genome.length === FB.TASKS[name].genomeSize, "genome serialised in spec");
  console.log("");
}

// Goal routing
check(FB.routeGoal("help me classify spam vs not spam") === "linear_classifier", "routes 'classify' → linear_classifier");
check(FB.routeGoal("plan a delivery route across cities") === "route_planner", "routes 'route' → route_planner");

// Determinism: same seed → same fitness
const a = FB.runRound("symbolic_regressor", { seed: 42, gens: 12 }).finalBest;
const b = FB.runRound("symbolic_regressor", { seed: 42, gens: 12 }).finalBest;
check(a === b, "deterministic for a fixed seed");

console.log(failures ? `\nFAILED (${failures})` : "\nALL PASSED");
process.exit(failures ? 1 : 0);

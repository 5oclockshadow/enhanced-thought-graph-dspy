/* Validate a model-proposed task decoder before the engine trusts it.
 *
 * Reads a task spec (JSON with JS function strings) on stdin, constructs the
 * task, and runs a short evolutionary search on it. A spec is ACCEPTED only if
 * it decodes to real executable source, scores inside [0,1], and measurably
 * improves over the random baseline — otherwise it is rejected. This is the
 * safety gate for the self-extending skill set: generated decoders that don't
 * actually work never reach the engine.
 *
 * Runs as an ephemeral, time-limited subprocess with no network/args.
 */
const FB = require("./foundry-builder.js");

let input = "";
process.stdin.on("data", (d) => (input += d));
process.stdin.on("end", () => {
  let spec;
  try { spec = JSON.parse(input); } catch (e) { return done({ ok: false, error: "spec is not valid JSON" }); }
  for (const k of ["name", "genomeSize", "box", "decode", "cases", "score"]) {
    if (!(k in spec)) return done({ ok: false, error: "missing field: " + k });
  }
  let task;
  try {
    task = {
      name: String(spec.name).replace(/[^a-z0-9_]/gi, "_"),
      label: spec.label || spec.name, keywords: spec.keywords || [],
      genomeSize: spec.genomeSize | 0, box: spec.box,
      decode: (0, eval)("(" + spec.decode + ")"),
      cases: (0, eval)("(" + spec.cases + ")"),
      score: (0, eval)("(" + spec.score + ")")
    };
  } catch (e) { return done({ ok: false, error: "could not evaluate functions: " + e.message }); }

  try {
    const rng = FB.mulberry32(7);
    const cases = task.cases(rng);
    const probe = task.decode(new Array(task.genomeSize).fill((task.box[0] + task.box[1]) / 2));
    if (!probe || typeof probe.src !== "string" || typeof probe.fn !== "function")
      return done({ ok: false, error: "decode must return {src:string, fn:function}" });
    const ev = FB.createEvolve(task, cases, { seed: 7, popSize: 40, gens: 20 });
    const init = ev.snapshot();
    ev.runAll();
    const be = ev.bestEver.fitness;
    const ok = isFinite(be) && be >= 0 && be <= 1.0001 && be > init.mean + 0.01;
    done({ ok, name: task.name, label: task.label, genomeSize: task.genomeSize,
      initialMean: round(init.mean), finalBest: round(be),
      error: ok ? null : "search did not improve measured fitness" });
  } catch (e) { done({ ok: false, error: "execution failed: " + e.message }); }
});

function round(x) { return Math.round(x * 1000) / 1000; }
function done(o) { process.stdout.write(JSON.stringify(o)); process.exit(o.ok ? 0 : 1); }

// Safety: never hang.
setTimeout(() => done({ ok: false, error: "validation timed out" }), 8000);

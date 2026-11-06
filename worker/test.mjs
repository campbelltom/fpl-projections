import test from "node:test";
import assert from "node:assert/strict";
import worker, { route, cors } from "./src/index.js";

const env = { ALLOWED_ORIGINS: "https://fpl.campbelltom.co.uk, http://localhost:8000" };
const req = (path, o = {}) => new Request("https://w.dev" + path, { headers: { Origin: "https://fpl.campbelltom.co.uk", ...(o.headers || {}) }, method: o.method || "GET" });

test("route allows only the listed read-only paths", () => {
  assert.equal(route("/fpl/bootstrap-static/", "").ttl, 600);
  assert.equal(route("/fpl/event/10/live/", "").ttl, 20);
  assert.equal(route("/fpl/fixtures/", "?event=10").url, "https://fantasy.premierleague.com/api/fixtures/?event=10");
  assert.equal(route("/fpl/entry/155989/event/10/picks/", "").ttl, 60);
  assert.equal(route("/fpl/leagues-classic/123/standings/", "").ttl, 120);
  for (const bad of ["/fpl/me/", "/fpl/entry/1/transfers/", "/fpl/../x", "/other", "/fpl/event/10/live", "/fpl/entry/abc/"])
    assert.equal(route(bad, ""), null, bad);
  assert.equal(route("/fpl/fixtures/", ""), null);
  assert.equal(route("/fpl/fixtures/", "?event=1;drop"), null);
  assert.equal(route("/fpl/bootstrap-static/", "?x=1"), null);
});

test("cors only for our own origins", () => {
  assert.ok(cors("https://fpl.campbelltom.co.uk", env));
  assert.ok(cors("http://localhost:8000", env));
  assert.equal(cors("https://evil.example", env), null);
  assert.equal(cors("", env), null);
});

test("proxies, adds CORS, and refuses strangers and other methods", async () => {
  globalThis.fetch = async (url, init) => { assert.ok(String(url).startsWith("https://fantasy.premierleague.com/api/")); assert.equal(init.cf.cacheTtl, 20); return new Response('{"ok":1}', { status: 200 }); };
  const ok = await worker.fetch(req("/fpl/event/10/live/"), env);
  assert.equal(ok.status, 200);
  assert.equal(ok.headers.get("Access-Control-Allow-Origin"), "https://fpl.campbelltom.co.uk");
  assert.equal((await ok.json()).ok, 1);
  assert.equal((await worker.fetch(req("/fpl/event/10/live/", { headers: { Origin: "https://evil.example" } }), env)).status, 403);
  assert.equal((await worker.fetch(req("/fpl/event/10/live/", { method: "POST" }), env)).status, 405);
  assert.equal((await worker.fetch(req("/fpl/secret/"), env)).status, 404);
  assert.equal((await worker.fetch(req("/fpl/event/10/live/", { method: "OPTIONS" }), env)).status, 204);
});

test("upstream errors are passed on and never cached", async () => {
  globalThis.fetch = async () => new Response("nope", { status: 503 });
  const r = await worker.fetch(req("/fpl/bootstrap-static/"), env);
  assert.equal(r.status, 503);
  assert.equal(r.headers.get("Cache-Control"), "no-store");
  globalThis.fetch = async () => { throw new Error("down"); };
  assert.equal((await worker.fetch(req("/fpl/bootstrap-static/"), env)).status, 502);
});

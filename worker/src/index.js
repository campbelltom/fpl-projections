/**
 * Campbell FPL proxy.
 *
 * Browsers cannot read the official Fantasy API directly (it sends no CORS permission header), so this Worker sits in between:
 * the page asks the Worker, the Worker asks fantasy.premierleague.com, and the answer goes back with the permission header added.
 * Each answer is also kept at Cloudflare's edge for a short time, so many page loads cost the FPL servers one request.
 *
 * It is deliberately narrow: GET only, only a handful of read-only FPL paths, and only for requests from our own site.
 */
const FPL = "https://fantasy.premierleague.com/api/";

// path (after /fpl/) -> seconds to keep the answer. Anything not matching is refused.
const ROUTES = [
  [/^bootstrap-static\/$/, 600],
  [/^fixtures\/$/, 20],
  [/^event\/\d{1,2}\/live\/$/, 20],
  [/^entry\/\d{1,9}\/$/, 600],
  [/^entry\/\d{1,9}\/event\/\d{1,2}\/picks\/$/, 60],
  [/^leagues-classic\/\d{1,9}\/standings\/$/, 120],
];

export function cors(origin, env) {
  const ok = (env.ALLOWED_ORIGINS || "").split(",").map((s) => s.trim()).filter(Boolean);
  return ok.includes(origin) ? { "Access-Control-Allow-Origin": origin, Vary: "Origin" } : null;
}

export function route(pathname, search) {
  if (!pathname.startsWith("/fpl/")) return null;
  const path = pathname.slice(5);
  const hit = ROUTES.find(([re]) => re.test(path));
  if (!hit) return null;
  if (path === "fixtures/" && !/^\?event=\d{1,2}$/.test(search)) return null;
  if (path !== "fixtures/" && search) return null;
  return { url: FPL + path + search, ttl: hit[1] };
}

const json = (obj, status, headers = {}) =>
  new Response(JSON.stringify(obj), { status, headers: { "Content-Type": "application/json", ...headers } });

export default {
  async fetch(request, env) {
    const origin = request.headers.get("Origin") || "";
    const allow = cors(origin, env);
    if (request.method === "OPTIONS") {
      return new Response(null, { status: allow ? 204 : 403, headers: { ...(allow || {}), "Access-Control-Allow-Methods": "GET", "Access-Control-Max-Age": "86400" } });
    }
    if (request.method !== "GET") return json({ error: "GET only" }, 405);
    if (!allow) return json({ error: "origin not allowed" }, 403);
    const u = new URL(request.url);
    const r = route(u.pathname, u.search);
    if (!r) return json({ error: "not found" }, 404, allow);

    let up;
    try {
      up = await fetch(r.url, { headers: { "User-Agent": "campbell-fpl-proxy/1.0" }, cf: { cacheEverything: true, cacheTtl: r.ttl } });
    } catch (e) {
      console.log(JSON.stringify({ msg: "upstream failed", url: r.url, err: String(e) }));
      return json({ error: "upstream unreachable" }, 502, allow);
    }
    if (!up.ok) console.log(JSON.stringify({ msg: "upstream status", url: r.url, status: up.status }));
    const out = new Response(up.body, { status: up.status, headers: { "Content-Type": "application/json", ...allow } });
    // the browser may reuse it for a few seconds too; errors are never kept
    out.headers.set("Cache-Control", up.ok ? `public, max-age=${Math.min(r.ttl, 15)}` : "no-store");
    return out;
  },
};

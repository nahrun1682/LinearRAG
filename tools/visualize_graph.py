"""Extract a small, readable subgraph from a LinearRAG GraphML and render it as an
interactive HTML (pyvis) so you can see *what kind* of graph was built.

The LinearRAG graph has two node types:
  - passage nodes  (name starts with "passage-")  : a text chunk
  - entity  nodes  (name starts with "entity-")    : an entity mentioned in chunks
and two edge types:
  - passage <-> entity      : the entity appears in that passage
  - passage <-> passage     : the two chunks are adjacent in the source corpus

The key idea of LinearRAG is "semantic bridging": a single entity can connect
several passages, enabling multi-hop reasoning. This script centers the view on a
"bridge" entity (one linked to multiple passages) and expands a capped neighborhood.

Usage:
  python tools/visualize_graph.py \
      --graphml import/test10/LinearRAG.graphml \
      --seed "lothair ii" \
      --max-nodes 80 \
      --out import/test10/graph_view.html
"""
import argparse
import json
import logging
import igraph as ig
from pyvis.network import Network

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

PASSAGE_COLOR = "#4C8BF5"   # blue
ENTITY_COLOR = "#F5A623"    # orange
BRIDGE_COLOR = "#E0457B"    # pink/red: entity linked to >=2 passages
SEED_COLOR = "#2ECC71"      # green: the seed node
SENTENCE_COLOR = "#9B59B6"  # purple: a sentence (entity<->sentence<->entity bridge)


def node_type(name):
    return "passage" if name.startswith("passage-") else "entity"


def short(text, n=30):
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[:n] + "…"


def pick_seed(g, deg, seed):
    """Return the vertex index of the seed entity (by substring) or an auto-picked
    mid-degree bridge entity (degree 2..6) that is specific rather than a giant hub."""
    if seed:
        cands = [v.index for v in g.vs
                 if node_type(v["name"]) == "entity" and seed.lower() in (v["content"] or "").lower()]
        if cands:
            return max(cands, key=lambda i: deg[i])
        log.warning("[warn] seed '%s' not found; auto-picking a bridge entity", seed)
    ent = [v.index for v in g.vs if node_type(v["name"]) == "entity"]
    mid = [i for i in ent if 2 <= deg[i] <= 6]
    pool = mid or ent
    return max(pool, key=lambda i: deg[i])


def build_subgraph(g, deg, seed_idx, max_nodes, per_passage_entities):
    """seed entity -> its passages -> (adjacent passages) + a few other entities each."""
    keep = {seed_idx}
    passages = []
    for nb in g.neighbors(seed_idx):
        if node_type(g.vs[nb]["name"]) == "passage":
            passages.append(nb)
    passages.sort(key=lambda i: deg[i], reverse=True)
    for p in passages:
        if len(keep) >= max_nodes:
            break
        keep.add(p)
        # add adjacent passages (passage<->passage edges)
        for nb in g.neighbors(p):
            if node_type(g.vs[nb]["name"]) == "passage" and len(keep) < max_nodes:
                keep.add(nb)
        # add a few other entities of this passage, preferring bridges (degree>=2)
        ents = [nb for nb in g.neighbors(p) if node_type(g.vs[nb]["name"]) == "entity"]
        ents.sort(key=lambda i: deg[i], reverse=True)
        added = 0
        for e in ents:
            if added >= per_passage_entities or len(keep) >= max_nodes:
                break
            if e not in keep:
                keep.add(e)
                added += 1
    return keep


def add_sentence_layer(net, g, keep, seed_idx, ner_path, max_sentences):
    """Add purple sentence nodes from ner_results.json. A sentence connects to the
    kept entities it mentions, so 'entity <-> sentence <-> entity' bridges are visible.
    Sentences that mention more kept entities (real bridges) are added first."""
    with open(ner_path) as f:
        sentence_to_entities = json.load(f)["sentence_to_entities"]
    kept_entity_text_to_idx = {
        g.vs[i]["content"]: i for i in keep if node_type(g.vs[i]["name"]) == "entity"
    }
    seed_text = g.vs[seed_idx]["content"]
    scored = []
    for sent, ents in sentence_to_entities.items():
        hits = [e for e in ents if e in kept_entity_text_to_idx]
        if not hits:
            continue
        # prioritise: contains seed, then number of kept entities it bridges
        scored.append(((seed_text in ents, len(hits)), sent, hits))
    scored.sort(key=lambda x: x[0], reverse=True)

    added = 0
    for _, sent, hits in scored:
        if added >= max_sentences:
            break
        sid = f"S{added}"
        net.add_node(sid, label=short(sent, 26), title=f"[sentence]\n{sent[:300]}",
                     color=SENTENCE_COLOR, size=10, shape="diamond")
        for e in hits:
            net.add_edge(sid, kept_entity_text_to_idx[e],
                         color="#F1C40F", width=1, title="entity in sentence")
        added += 1
    return added


def render(g, deg, keep, seed_idx, out_path, ner_path=None, max_sentences=25):
    net = Network(height="800px", width="100%", bgcolor="#1e1e1e",
                  font_color="white", directed=False, notebook=False,
                  cdn_resources="in_line")
    net.barnes_hut(gravity=-8000, spring_length=120)
    for i in keep:
        v = g.vs[i]
        t = node_type(v["name"])
        d = deg[i]
        if i == seed_idx:
            color = SEED_COLOR
        elif t == "passage":
            color = PASSAGE_COLOR
        else:
            color = BRIDGE_COLOR if d >= 2 else ENTITY_COLOR
        label = short(v["content"], 28) if t == "entity" else short(v["content"], 22)
        title = f"[{t}] degree={d}\n{(v['content'] or '')[:300]}"
        size = 12 + min(d, 30) * 1.2 + (10 if i == seed_idx else 0)
        net.add_node(i, label=label, title=title, color=color, size=size,
                     shape="dot" if t == "entity" else "square")
    kept = set(keep)
    for e in g.es:
        s, t = e.source, e.target
        if s in kept and t in kept:
            both_pass = node_type(g.vs[s]["name"]) == "passage" and node_type(g.vs[t]["name"]) == "passage"
            net.add_edge(s, t,
                         color="#2ECC71" if both_pass else "#888888",
                         width=3 if both_pass else 1,
                         dashes=both_pass,
                         title="adjacent passages" if both_pass else "entity in passage")
    n_sent = add_sentence_layer(net, g, keep, seed_idx, ner_path, max_sentences) if ner_path else 0
    net.write_html(out_path, notebook=False)
    return n_sent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphml", default="import/test10/LinearRAG.graphml")
    ap.add_argument("--seed", default=None, help="entity text to center on (substring match)")
    ap.add_argument("--max-nodes", type=int, default=80)
    ap.add_argument("--per-passage-entities", type=int, default=6)
    ap.add_argument("--with-sentences", action="store_true",
                    help="also add sentence nodes (entity<->sentence<->entity layer)")
    ap.add_argument("--ner", default="import/test10/ner_results.json",
                    help="ner_results.json path (used with --with-sentences)")
    ap.add_argument("--max-sentences", type=int, default=25)
    ap.add_argument("--out", default="import/test10/graph_view.html")
    args = ap.parse_args()

    g = ig.Graph.Read_GraphML(args.graphml)
    deg = g.degree()
    seed_idx = pick_seed(g, deg, args.seed)
    keep = build_subgraph(g, deg, seed_idx, args.max_nodes, args.per_passage_entities)
    ner_path = args.ner if args.with_sentences else None
    n_sent = render(g, deg, keep, seed_idx, args.out, ner_path, args.max_sentences)

    n_pass = sum(1 for i in keep if node_type(g.vs[i]["name"]) == "passage")
    n_ent = len(keep) - n_pass
    log.info("seed: [%s] %r (degree %d)",
             node_type(g.vs[seed_idx]["name"]), g.vs[seed_idx]["content"], deg[seed_idx])
    log.info("subgraph: %d nodes (%d passages + %d entities + %d sentences)",
             len(keep) + n_sent, n_pass, n_ent, n_sent)
    log.info("written: %s", args.out)


if __name__ == "__main__":
    main()

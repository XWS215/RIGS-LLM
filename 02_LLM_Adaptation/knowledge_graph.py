import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import networkx as nx
except Exception:
    nx = None

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(HERE, "Data", "triples.jsonl")
DEFAULT_OUTPUT = os.path.join(HERE, "outputs", "graph")
RELATION_TYPES = [
    "triggers",
    "constrains",
    "precedes",
    "interacts_with",
    "leads_to",
    "requires",
]
ENTITY_TYPES = ["actor", "environment", "behavior", "mechanism", "condition", "unknown"]
NORMALIZE_PATTERN = re.compile(r"[\s　]+")


def fail(message):
    sys.stderr.write("[error] " + str(message) + "\n")
    sys.exit(1)


def info(message):
    sys.stdout.write("[info] " + str(message) + "\n")
    sys.stdout.flush()


def warn(message):
    sys.stderr.write("[warn] " + str(message) + "\n")
    sys.stderr.flush()


def require_networkx():
    if nx is None:
        fail("未安装 networkx，无法进行图分析")


def normalize_entity(name):
    if name is None:
        return ""
    text = str(name).strip()
    text = NORMALIZE_PATTERN.sub("", text)
    text = text.replace("，", "").replace(",", "")
    text = text.replace("。", "").replace(".", "")
    text = text.replace("（", "(").replace("）", ")")
    return text.lower()


def canonical_relation(predicate):
    value = str(predicate).strip().lower()
    aliases = {
        "trigger": "triggers",
        "causes": "triggers",
        "constraint": "constrains",
        "limits": "constrains",
        "before": "precedes",
        "interact": "interacts_with",
        "interacts": "interacts_with",
        "lead_to": "leads_to",
        "causes_failure": "leads_to",
        "require": "requires",
        "depends_on": "requires",
    }
    if value in RELATION_TYPES:
        return value
    if value in aliases:
        return aliases[value]
    return ""


@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    evidence: str = ""
    weight: float = 1.0
    source: str = ""
    mechanism: str = ""

    def to_dict(self):
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "evidence": self.evidence,
            "weight": self.weight,
            "source": self.source,
            "mechanism": self.mechanism,
        }


def triple_from_dict(item):
    subject = str(item.get("subject", "")).strip()
    predicate = canonical_relation(item.get("predicate", ""))
    obj = str(item.get("object", "")).strip()
    if len(subject) == 0 or len(obj) == 0:
        return None
    if len(predicate) == 0:
        return None
    return Triple(
        subject=subject,
        predicate=predicate,
        object=obj,
        evidence=str(item.get("evidence", "")).strip(),
        weight=float(item.get("weight", 1.0)),
        source=str(item.get("source", "")).strip(),
        mechanism=str(item.get("mechanism", "")).strip(),
    )


def load_triples(path):
    triples = []
    if not os.path.isfile(path):
        fail("三元组文件不存在：%s" % path)
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception as err:
                    fail("%s:%d JSON 解析失败：%s" % (path, line_no, err))
                triple = triple_from_dict(item)
                if triple is None:
                    warn("%s:%d 三元组不完整，已跳过" % (path, line_no))
                    continue
                triples.append(triple)
        return triples
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "triples" in payload:
        payload = payload["triples"]
    if not isinstance(payload, list):
        fail("三元组文件格式不正确：%s" % path)
    for item in payload:
        triple = triple_from_dict(item)
        if triple is None:
            continue
        triples.append(triple)
    return triples


class RiskKnowledgeGraph:
    def __init__(self):
        require_networkx()
        self.graph = nx.MultiDiGraph()
        self.aliases = {}
        self.entity_types = {}
        self.evidence = defaultdict(list)

    def node_count(self):
        return self.graph.number_of_nodes()

    def edge_count(self):
        return self.graph.number_of_edges()

    def register_entity(self, name, entity_type="unknown"):
        if entity_type not in ENTITY_TYPES:
            entity_type = "unknown"
        key = normalize_entity(name)
        if len(key) == 0:
            return None
        if key in self.aliases:
            canonical = self.aliases[key]
        else:
            canonical = str(name).strip()
            self.aliases[key] = canonical
            self.aliases[normalize_entity(canonical)] = canonical
        if canonical not in self.graph:
            self.graph.add_node(canonical, entity_type=entity_type, mentions=0)
            self.entity_types[canonical] = entity_type
        self.graph.nodes[canonical]["mentions"] = int(self.graph.nodes[canonical].get("mentions", 0)) + 1
        if entity_type != "unknown":
            self.graph.nodes[canonical]["entity_type"] = entity_type
            self.entity_types[canonical] = entity_type
        return canonical

    def resolve(self, name):
        key = normalize_entity(name)
        if key in self.aliases:
            return self.aliases[key]
        return str(name).strip()

    def add_triple(self, triple, entity_type_subject="unknown", entity_type_object="unknown"):
        subject = self.register_entity(triple.subject, entity_type_subject)
        obj = self.register_entity(triple.object, entity_type_object)
        if subject is None or obj is None:
            return False
        if subject == obj:
            return False
        key = None
        for edge_key in self.graph[subject][obj] if self.graph.has_edge(subject, obj) else []:
            data = self.graph[subject][obj][edge_key]
            if data.get("predicate") == triple.predicate:
                key = edge_key
                break
        if key is not None:
            self.graph[subject][obj][key]["weight"] += triple.weight
            if triple.evidence:
                existing = self.graph[subject][obj][key].get("evidence", "")
                if triple.evidence not in existing:
                    self.graph[subject][obj][key]["evidence"] = (existing + " | " + triple.evidence).strip(" |")
            return False
        self.graph.add_edge(
            subject,
            obj,
            key=triple.predicate,
            predicate=triple.predicate,
            weight=triple.weight,
            evidence=triple.evidence,
            source=triple.source,
            mechanism=triple.mechanism,
        )
        if triple.evidence:
            self.evidence[subject].append(triple.evidence)
        return True

    def add_triples(self, triples):
        added = 0
        for triple in triples:
            if self.add_triple(triple):
                added += 1
        info("加入 %d 条新关系，当前节点 %d，边 %d"
             % (added, self.node_count(), self.edge_count()))
        return added

    def merge_similar_entities(self, min_similarity=0.9):
        require_networkx()
        keys = sorted(self.aliases.keys())
        merged = 0
        for index in range(len(keys)):
            left = keys[index]
            if left not in self.aliases:
                continue
            for right_index in range(index + 1, len(keys)):
                right = keys[right_index]
                if right not in self.aliases:
                    continue
                if left == right:
                    continue
                similarity = sequence_similarity(left, right)
                if similarity < min_similarity:
                    continue
                source = self.aliases[left]
                target = self.aliases[right]
                if source == target:
                    continue
                if source in self.graph and target in self.graph:
                    self.graph = nx.contracted_nodes(self.graph, target, source, self_loops=False)
                    merged += 1
        info("合并 %d 组相似实体" % merged)
        return merged

    def prune(self, min_weight=1.0, min_mentions=1):
        remove_edges = []
        for subject, obj, key, data in self.graph.edges(keys=True, data=True):
            if float(data.get("weight", 0.0)) < min_weight:
                remove_edges.append((subject, obj, key))
        for subject, obj, key in remove_edges:
            self.graph.remove_edge(subject, obj, key)
        isolated = [node for node, degree in self.graph.degree() if degree < min_mentions - 1]
        for node in isolated:
            if self.graph.degree(node) == 0:
                self.graph.remove_node(node)
        info("剪枝：删除 %d 条弱边，%d 个孤立节点" % (len(remove_edges), len(isolated)))

    def mechanisms(self):
        names = set()
        for subject, obj, key, data in self.graph.edges(keys=True, data=True):
            mechanism = data.get("mechanism", "")
            if mechanism:
                names.add(mechanism)
        return sorted(names)

    def mechanism_subgraph(self, mechanism):
        sub = nx.MultiDiGraph()
        for subject, obj, key, data in self.graph.edges(keys=True, data=True):
            if data.get("mechanism") == mechanism:
                if subject not in sub:
                    sub.add_node(subject, **self.graph.nodes[subject])
                if obj not in sub:
                    sub.add_node(obj, **self.graph.nodes[obj])
                sub.add_edge(subject, obj, key=key, **data)
        return sub

    def degree_table(self):
        rows = []
        for node in self.graph.nodes():
            rows.append({
                "entity": node,
                "in_degree": int(self.graph.in_degree(node)),
                "out_degree": int(self.graph.out_degree(node)),
                "degree": int(self.graph.degree(node)),
                "mentions": int(self.graph.nodes[node].get("mentions", 0)),
            })
        rows.sort(key=lambda row: (-row["degree"], row["entity"]))
        return rows

    def centrality_table(self):
        undirected = nx.Graph(self.graph)
        if undirected.number_of_nodes() == 0:
            return []
        degree_centrality = nx.degree_centrality(undirected)
        try:
            pagerank = nx.pagerank(self.graph, weight="weight")
        except Exception:
            pagerank = {node: 0.0 for node in undirected.nodes()}
        try:
            betweenness = nx.betweenness_centrality(undirected, weight="weight")
        except Exception:
            betweenness = {node: 0.0 for node in undirected.nodes()}
        rows = []
        for node in undirected.nodes():
            rows.append({
                "entity": node,
                "degree_centrality": degree_centrality.get(node, 0.0),
                "pagerank": pagerank.get(node, 0.0),
                "betweenness": betweenness.get(node, 0.0),
            })
        rows.sort(key=lambda row: -row["pagerank"])
        return rows

    def communities(self):
        undirected = nx.Graph(self.graph)
        groups = []
        if undirected.number_of_nodes() == 0:
            return groups
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            communities = greedy_modularity_communities(undirected, weight="weight")
            for index, community in enumerate(communities):
                members = sorted(community)
                groups.append({"community": index, "size": len(members), "members": members})
        except Exception as err:
            warn("社区划分失败：%s" % err)
        return groups

    def paths(self, source, target, cutoff=6, top_k=20):
        require_networkx()
        if source not in self.graph or target not in self.graph:
            return []
        simple_graph = nx.DiGraph()
        for subject, obj, key, data in self.graph.edges(keys=True, data=True):
            weight = float(data.get("weight", 1.0))
            if simple_graph.has_edge(subject, obj):
                if weight > simple_graph[subject][obj]["weight"]:
                    simple_graph[subject][obj]["weight"] = weight
            else:
                simple_graph.add_edge(subject, obj, weight=weight, predicate=data.get("predicate"))
        found = []
        try:
            generator = nx.all_simple_paths(simple_graph, source, target, cutoff=cutoff)
        except Exception:
            return []
        for path in generator:
            score = 0.0
            relations = []
            for index in range(len(path) - 1):
                data = simple_graph[path[index]][path[index + 1]]
                score += float(data.get("weight", 1.0))
                relations.append(data.get("predicate", ""))
            found.append({
                "path": list(path),
                "relations": relations,
                "score": score,
                "length": len(path) - 1,
            })
            if len(found) >= top_k * 5:
                break
        found.sort(key=lambda item: (-item["score"], item["length"]))
        return found[:top_k]

    def stats(self):
        require_networkx()
        undirected = nx.Graph(self.graph)
        components = list(nx.connected_components(undirected)) if undirected.number_of_nodes() else []
        component_sizes = sorted([len(item) for item in components], reverse=True)
        node_types = Counter(self.graph.nodes[node].get("entity_type", "unknown") for node in self.graph.nodes())
        relation_types = Counter(data.get("predicate", "") for _, _, _, data in self.graph.edges(keys=True, data=True))
        density = 0.0
        if self.node_count() > 1:
            density = nx.density(undirected)
        isolated = 0
        for node in self.graph.nodes():
            if self.graph.degree(node) == 0:
                isolated += 1
        return {
            "nodes": self.node_count(),
            "edges": self.edge_count(),
            "density": density,
            "components": len(components),
            "largest_component": component_sizes[0] if component_sizes else 0,
            "component_sizes": component_sizes,
            "isolated_nodes": isolated,
            "entity_types": dict(node_types),
            "relation_types": dict(relation_types),
            "mechanisms": len(self.mechanisms()),
        }

    def to_dict(self):
        nodes = []
        for node in self.graph.nodes():
            nodes.append({
                "id": node,
                "entity_type": self.graph.nodes[node].get("entity_type", "unknown"),
                "mentions": int(self.graph.nodes[node].get("mentions", 0)),
            })
        edges = []
        for subject, obj, key, data in self.graph.edges(keys=True, data=True):
            edges.append({
                "subject": subject,
                "object": obj,
                "predicate": data.get("predicate", ""),
                "weight": float(data.get("weight", 1.0)),
                "evidence": data.get("evidence", ""),
                "mechanism": data.get("mechanism", ""),
            })
        return {"nodes": nodes, "edges": edges}

    def from_dict(self, payload):
        for node in payload.get("nodes", []):
            self.register_entity(node.get("id", ""), node.get("entity_type", "unknown"))
        for edge in payload.get("edges", []):
            triple = triple_from_dict(edge)
            if triple is None:
                continue
            self.add_triple(triple)
        return self

    def save_json(self, path):
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
        info("图谱已保存到 %s" % path)
        return path

    def load_json(self, path):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return self.from_dict(payload)

    def export_graphml(self, path):
        require_networkx()
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        view = nx.MultiDiGraph()
        for node in self.graph.nodes():
            view.add_node(node, **self.graph.nodes[node])
        for subject, obj, key, data in self.graph.edges(keys=True, data=True):
            view.add_edge(subject, obj, **data)
        nx.write_graphml(view, path)
        info("GraphML 已导出到 %s" % path)
        return path

    def export_dot(self, path):
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        lines = ["digraph RiskGraph {"]
        lines.append("  rankdir=LR;")
        for node in self.graph.nodes():
            label = node.replace("\"", "'")
            lines.append("  \"%s\" [label=\"%s\"];" % (label, label))
        for subject, obj, key, data in self.graph.edges(keys=True, data=True):
            label = str(data.get("predicate", "")).replace("\"", "'")
            lines.append("  \"%s\" -> \"%s\" [label=\"%s\"];" % (subject.replace("\"", "'"), obj.replace("\"", "'"), label))
        lines.append("}")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        info("DOT 已导出到 %s" % path)
        return path


def sequence_similarity(left, right):
    if left == right:
        return 1.0
    if len(left) == 0 or len(right) == 0:
        return 0.0
    short, long = (left, right) if len(left) <= len(right) else (right, left)
    if short in long:
        return float(len(short)) / float(len(long))
    matches = 0
    for char in set(short):
        if char in long:
            matches += 1
    return float(matches) / float(len(set(short) | set(long)))


class RiskPathFinder:
    def __init__(self, graph):
        self.graph = graph

    def candidate_sources(self, top_k=10):
        rows = []
        for node in self.graph.graph.nodes():
            out_degree = self.graph.graph.out_degree(node)
            in_degree = self.graph.graph.in_degree(node)
            if out_degree > 0 and in_degree == 0:
                rows.append((node, out_degree))
        rows.sort(key=lambda item: -item[1])
        return [item[0] for item in rows[:top_k]]

    def candidate_targets(self, top_k=10):
        rows = []
        for node in self.graph.graph.nodes():
            out_degree = self.graph.graph.out_degree(node)
            in_degree = self.graph.graph.in_degree(node)
            if in_degree > 0 and out_degree == 0:
                rows.append((node, in_degree))
        rows.sort(key=lambda item: -item[1])
        return [item[0] for item in rows[:top_k]]

    def enumerate_paths(self, cutoff=6, top_k=20):
        results = []
        for source in self.candidate_sources():
            for target in self.candidate_targets():
                if source == target:
                    continue
                for item in self.graph.paths(source, target, cutoff=cutoff, top_k=top_k):
                    item["source"] = source
                    item["target"] = target
                    results.append(item)
        results.sort(key=lambda item: (-item["score"], item["length"]))
        return results[:top_k]

    def coverage(self, mechanisms):
        covered = set(self.graph.mechanisms())
        required = set(mechanisms)
        if len(required) == 0:
            return 0.0, [], []
        hit = sorted(required & covered)
        missed = sorted(required - covered)
        return float(len(hit)) / float(len(required)), hit, missed


def build_graph(triples, merge=True):
    graph = RiskKnowledgeGraph()
    graph.add_triples(triples)
    if merge:
        graph.merge_similar_entities(min_similarity=0.92)
    return graph


def build_parser():
    parser = argparse.ArgumentParser(description="风险知识图谱构建与图分析")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--task", type=str, default="build")
    parser.add_argument("--source", type=str, default="")
    parser.add_argument("--target", type=str, default="")
    parser.add_argument("--cutoff", type=int, default=6)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_weight", type=float, default=1.0)
    parser.add_argument("--no_merge", action="store_true")
    parser.add_argument("--mechanisms", type=str, default="")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    triples = load_triples(args.input)
    info("读入 %d 条三元组" % len(triples))
    graph = build_graph(triples, merge=not args.no_merge)
    graph.prune(min_weight=args.min_weight)
    info("图谱规模：节点 %d，边 %d" % (graph.node_count(), graph.edge_count()))
    if args.task == "build":
        path = os.path.join(args.output_dir, "graph.json")
        graph.save_json(path)
        graph.export_dot(os.path.join(args.output_dir, "graph.dot"))
        graph.export_graphml(os.path.join(args.output_dir, "graph.graphml"))
        info(json.dumps(graph.stats(), ensure_ascii=False, indent=2))
        return
    if args.task == "stats":
        payload = graph.stats()
        payload["top_nodes"] = graph.degree_table()[:10]
        payload["centrality"] = graph.centrality_table()[:10]
        payload["communities"] = graph.communities()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if args.task == "paths":
        finder = RiskPathFinder(graph)
        if args.source and args.target:
            results = graph.paths(args.source, args.target, cutoff=args.cutoff, top_k=args.top_k)
        else:
            results = finder.enumerate_paths(cutoff=args.cutoff, top_k=args.top_k)
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return
    if args.task == "coverage":
        mechanisms = [item.strip() for item in args.mechanisms.split(",") if item.strip()]
        finder = RiskPathFinder(graph)
        ratio, hit, missed = finder.coverage(mechanisms)
        print(json.dumps({
            "coverage": ratio,
            "hit": hit,
            "missed": missed,
        }, ensure_ascii=False, indent=2))
        return
    if args.task == "export":
        graph.save_json(os.path.join(args.output_dir, "graph.json"))
        graph.export_dot(os.path.join(args.output_dir, "graph.dot"))
        graph.export_graphml(os.path.join(args.output_dir, "graph.graphml"))
        return
    fail("未知任务：%s" % args.task)


if __name__ == "__main__":
    main()

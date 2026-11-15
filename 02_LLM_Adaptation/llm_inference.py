import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROMPT_DIR = os.path.join(HERE, "prompt")
DEFAULT_DATA_DIR = os.path.join(HERE, "Data")
DEFAULT_OUTPUT_DIR = os.path.join(HERE, "outputs", "generation")
PLACEHOLDER_PATTERN = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def fail(message):
    sys.stderr.write("[error] " + str(message) + "\n")
    sys.exit(1)


def info(message):
    sys.stdout.write("[info] " + str(message) + "\n")
    sys.stdout.flush()


def warn(message):
    sys.stderr.write("[warn] " + str(message) + "\n")
    sys.stderr.flush()


@dataclass
class PromptTemplate:
    name: str
    path: str
    text: str
    placeholders: List[str] = field(default_factory=list)

    def render(self, **values):
        missing = []
        for key in self.placeholders:
            if key not in values:
                missing.append(key)
        if missing:
            fail("提示词 %s 缺少参数：%s" % (self.name, ", ".join(missing)))
        rendered = self.text
        for key in self.placeholders:
            rendered = rendered.replace("{" + key + "}", str(values[key]))
        return rendered


class PromptLibrary:
    def __init__(self, prompt_dir):
        self.prompt_dir = prompt_dir
        self.templates = {}
        self.loaded_at = 0.0
        self.reload()

    def reload(self):
        if not os.path.isdir(self.prompt_dir):
            fail("提示词目录不存在：%s" % self.prompt_dir)
        files = []
        for name in sorted(os.listdir(self.prompt_dir)):
            if name.endswith(".txt") or name.endswith(".md"):
                files.append(os.path.join(self.prompt_dir, name))
        if not files:
            fail("提示词目录下没有 .txt/.md 文件：%s" % self.prompt_dir)
        templates = {}
        for path in files:
            name = os.path.splitext(os.path.basename(path))[0]
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
            if len(text.strip()) == 0:
                warn("提示词文件为空：%s" % path)
                continue
            placeholders = []
            for match in PLACEHOLDER_PATTERN.finditer(text):
                key = match.group(1)
                if key not in placeholders:
                    placeholders.append(key)
            templates[name] = PromptTemplate(
                name=name,
                path=path,
                text=text,
                placeholders=placeholders,
            )
        self.templates = templates
        self.loaded_at = time.time()
        return self.templates

    def get(self, name):
        if name in self.templates:
            return self.templates[name]
        if name + ".txt" in self.templates:
            return self.templates[name + ".txt"]
        fail("找不到提示词 %s，可用：%s"
             % (name, ", ".join(sorted(self.templates.keys()))))

    def render(self, name, **values):
        return self.get(name).render(**values)

    def names(self):
        return sorted(self.templates.keys())

    def describe(self):
        rows = []
        for name in self.names():
            template = self.templates[name]
            rows.append({
                "name": name,
                "path": template.path,
                "placeholders": list(template.placeholders),
                "chars": len(template.text),
            })
        return rows


@dataclass
class LLMConfig:
    model: str = "qwen3.5-27b"
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1024
    timeout: float = 120.0
    retries: int = 3
    retry_wait: float = 2.0
    system_prompt: str = ""


@dataclass
class ChatResult:
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency: float = 0.0
    attempts: int = 1
    raw: Optional[Dict[str, Any]] = None


def estimate_tokens(text):
    if not text:
        return 0
    chinese = 0
    other = 0
    for char in text:
        if ord(char) > 127:
            chinese += 1
        else:
            other += 1
    return int(chinese * 0.75 + other * 0.25) + 1


def strip_code_fence(text):
    if text is None:
        return ""
    match = JSON_BLOCK_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def repair_json_text(text):
    text = strip_code_fence(text)
    if len(text) == 0:
        return text
    text = text.replace("“", "\"").replace("”", "\"")
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("，", ",")
    text = text.replace("：", ":")
    text = re.sub(r",\s*([}\]])", r"\1", text)
    first_obj = text.find("{")
    first_arr = text.find("[")
    if first_arr == -1 or (first_obj != -1 and first_obj < first_arr):
        start = first_obj
        end = text.rfind("}")
    else:
        start = first_arr
        end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start:end + 1]


def parse_json_payload(text):
    cleaned = strip_code_fence(text)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    repaired = repair_json_text(text)
    try:
        return json.loads(repaired)
    except Exception as err:
        fail("模型输出无法解析为 JSON：%s\n原文：%s" % (err, text[:500]))


class LLMClient:
    def __init__(self, config):
        self.config = config
        self.client = None
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.failures = 0
        self._init_client()

    def _init_client(self):
        if OpenAI is None:
            fail("未安装 openai 库，无法调用模型")
        if self.config.api_key:
            os.environ.setdefault("OPENAI_API_KEY", self.config.api_key)
        kwargs = {}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        self.client = OpenAI(**kwargs)

    def build_messages(self, user_content, system_content=None):
        messages = []
        system_text = system_content if system_content else self.config.system_prompt
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": user_content})
        return messages

    def chat(self, user_content, system_content=None, overrides=None):
        overrides = overrides or {}
        messages = self.build_messages(user_content, system_content)
        temperature = overrides.get("temperature", self.config.temperature)
        top_p = overrides.get("top_p", self.config.top_p)
        max_tokens = overrides.get("max_tokens", self.config.max_tokens)
        last_error = None
        for attempt in range(1, self.config.retries + 1):
            started = time.time()
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    timeout=self.config.timeout,
                )
                latency = time.time() - started
                content = response.choices[0].message.content
                usage = getattr(response, "usage", None)
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
                if prompt_tokens == 0:
                    prompt_tokens = estimate_tokens(str(user_content))
                if completion_tokens == 0:
                    completion_tokens = estimate_tokens(content)
                if total_tokens == 0:
                    total_tokens = prompt_tokens + completion_tokens
                self.calls += 1
                self.prompt_tokens += prompt_tokens
                self.completion_tokens += completion_tokens
                self.total_tokens += total_tokens
                return ChatResult(
                    content=content,
                    model=self.config.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    latency=latency,
                    attempts=attempt,
                    raw=None,
                )
            except Exception as err:
                last_error = err
                self.failures += 1
                wait = self.config.retry_wait * attempt
                warn("第 %d 次调用失败：%s，%.1f 秒后重试" % (attempt, err, wait))
                time.sleep(wait)
        fail("模型调用连续失败 %d 次：%s" % (self.config.retries, last_error))

    def chat_json(self, user_content, system_content=None, overrides=None):
        result = self.chat(user_content, system_content, overrides)
        payload = parse_json_payload(result.content)
        return payload, result

    def batch_chat(self, requests, overrides=None, pause=0.0):
        outputs = []
        total = len(requests)
        for index, request in enumerate(requests):
            if isinstance(request, dict):
                user_content = request.get("user")
                system_content = request.get("system")
                payload, result = self.chat_json(user_content, system_content, overrides)
            else:
                payload, result = self.chat_json(request, None, overrides)
            outputs.append({"index": index, "payload": payload, "usage": result})
            if (index + 1) % 10 == 0 or index + 1 == total:
                info("批量调用进度 %d / %d" % (index + 1, total))
            if pause > 0:
                time.sleep(pause)
        return outputs

    def usage_summary(self):
        return {
            "calls": self.calls,
            "failures": self.failures,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


def require_fields(payload, fields, context):
    if not isinstance(payload, dict):
        fail("%s：期望 JSON 对象，实际得到 %s" % (context, type(payload).__name__))
    missing = []
    for key in fields:
        if key not in payload or payload[key] is None:
            missing.append(key)
    if missing:
        fail("%s：缺少字段 %s" % (context, ", ".join(missing)))
    return payload


def validate_requirement_payload(payload):
    require_fields(payload, ["goal", "risk_factors", "environment", "actors"], "需求解析")
    if not isinstance(payload["risk_factors"], list) or len(payload["risk_factors"]) == 0:
        fail("需求解析：risk_factors 必须是非空列表")
    normalized = []
    for item in payload["risk_factors"]:
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            reason = str(item.get("reason", "")).strip()
        else:
            name = str(item).strip()
            reason = ""
        if len(name) == 0:
            continue
        normalized.append({"name": name, "reason": reason})
    if len(normalized) == 0:
        fail("需求解析：没有有效的风险机制")
    payload["risk_factors"] = normalized
    return payload


def validate_triples_payload(payload):
    if not isinstance(payload, list):
        fail("三元组抽取：期望 JSON 数组")
    allowed = {"triggers", "constrains", "precedes", "interacts_with", "leads_to", "requires"}
    triples = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            warn("三元组第 %d 项不是对象，已跳过" % index)
            continue
        subject = str(item.get("subject", "")).strip()
        predicate = str(item.get("predicate", "")).strip()
        obj = str(item.get("object", "")).strip()
        evidence = str(item.get("evidence", "")).strip()
        if len(subject) == 0 or len(obj) == 0:
            warn("三元组第 %d 项缺少主体或客体，已跳过" % index)
            continue
        if predicate not in allowed:
            warn("三元组第 %d 项关系 %s 不在允许集合内，已跳过" % (index, predicate))
            continue
        triples.append({
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "evidence": evidence,
        })
    return triples


def validate_scenario_payload(payload):
    require_fields(payload, ["scenario", "risk_chain", "actors", "mechanism"], "场景生成")
    text = str(payload["scenario"]).strip()
    if len(text) < 20:
        fail("场景生成：场景描述过短（%d 字）" % len(text))
    if not isinstance(payload["risk_chain"], list) or len(payload["risk_chain"]) == 0:
        fail("场景生成：risk_chain 必须是非空列表")
    payload["scenario"] = text
    return payload


def validate_evaluation_payload(payload):
    require_fields(payload, ["plausibility", "criticality", "testability"], "场景评价")
    for key in ("plausibility", "criticality", "testability"):
        block = payload[key]
        if not isinstance(block, dict):
            fail("场景评价：%s 必须是对象" % key)
        score = block.get("score")
        if score is None:
            fail("场景评价：%s 缺少 score" % key)
        score = float(score)
        if score < 1 or score > 5:
            fail("场景评价：%s 的 score 超出 1—5" % key)
        block["score"] = score
        block.setdefault("evidence", "")
        block.setdefault("uncertainty", 0.0)
    payload.setdefault("conflicts", [])
    payload.setdefault("missing_evidence", [])
    return payload


def make_output_path(output_dir, requirement, function, suffix):
    os.makedirs(output_dir, exist_ok=True)
    safe = re.sub(r"[^0-9A-Za-z_一-龥]+", "_", requirement)[:40]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, "%s_%s_%s_%s.json" % (function, safe, stamp, suffix))


def append_jsonl(path, record):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path):
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


class ScenarioPipeline:
    def __init__(self, client, library, output_dir):
        self.client = client
        self.library = library
        self.output_dir = output_dir

    def parse_requirement(self, requirement, function):
        prompt = self.library.render(
            "requirement_parse",
            requirement=requirement,
            function=function,
        )
        payload, result = self.client.chat_json(prompt)
        payload = validate_requirement_payload(payload)
        info("需求解析完成，得到 %d 个风险机制，用时 %.1f 秒"
             % (len(payload["risk_factors"]), result.latency))
        return payload

    def extract_triples(self, scenario):
        prompt = self.library.render("triple_extraction", scenario=scenario)
        payload, result = self.client.chat_json(prompt)
        triples = validate_triples_payload(payload)
        info("三元组抽取完成，得到 %d 条关系，用时 %.1f 秒"
             % (len(triples), result.latency))
        return triples

    def generate_scenario(self, requirement, function, mechanism, operation,
                          constraints="", reference=""):
        prompt = self.library.render(
            "scenario_generation",
            requirement=requirement,
            function=function,
            mechanism=mechanism,
            operation=operation,
            constraints=constraints,
            reference=reference,
        )
        payload, result = self.client.chat_json(prompt)
        payload = validate_scenario_payload(payload)
        payload["function"] = function
        payload["operation"] = operation
        payload["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        payload["usage"] = {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency": result.latency,
        }
        return payload

    def evaluate_scenario(self, scenario):
        prompt = self.library.render("scenario_evaluation", scenario=scenario)
        payload, result = self.client.chat_json(prompt)
        payload = validate_evaluation_payload(payload)
        payload["usage"] = {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency": result.latency,
        }
        return payload

    def evolve_scenario(self, scenario, operation, mechanism,
                        preserve="", modify=""):
        prompt = self.library.render(
            "evolution_operation",
            scenario=scenario,
            operation=operation,
            mechanism=mechanism,
            preserve=preserve,
            modify=modify,
        )
        payload, result = self.client.chat_json(prompt)
        if not isinstance(payload, dict) or "scenario" not in payload:
            fail("场景演化：输出缺少 scenario 字段")
        payload["scenario"] = str(payload["scenario"]).strip()
        payload["operation"] = operation
        payload["mechanism"] = mechanism
        payload["usage"] = {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency": result.latency,
        }
        return payload

    def run_generation(self, requirement, function, count, operation="risk_condition",
                       constraints="", reference="", batch_file=None):
        parsed = self.parse_requirement(requirement, function)
        mechanisms = parsed["risk_factors"]
        records = []
        for index in range(count):
            mechanism = mechanisms[index % len(mechanisms)]["name"]
            record = self.generate_scenario(
                requirement=requirement,
                function=function,
                mechanism=mechanism,
                operation=operation,
                constraints=constraints,
                reference=reference,
            )
            record["requirement"] = requirement
            record["mechanism"] = mechanism
            records.append(record)
            info("已生成 %d / %d：%s" % (index + 1, count, mechanism))
        if batch_file is None:
            batch_file = make_output_path(self.output_dir, requirement, function, "scenarios")
        for record in records:
            append_jsonl(batch_file, record)
        info("生成结果已写入 %s" % batch_file)
        return {"requirements": parsed, "records": records, "path": batch_file}


def build_llm_config(args):
    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", "")
    return LLMConfig(
        model=args.model,
        base_url=base_url,
        api_key=api_key,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        retries=args.retries,
        retry_wait=args.retry_wait,
        system_prompt=args.system_prompt,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="大模型调用与功能场景生成")
    parser.add_argument("--prompt_dir", type=str, default=DEFAULT_PROMPT_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=str, default="qwen3.5-27b")
    parser.add_argument("--base_url", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry_wait", type=float, default=2.0)
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--task", type=str, default="generate")
    parser.add_argument("--requirement", type=str, default="")
    parser.add_argument("--function", type=str, default="AEB")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--operation", type=str, default="risk_condition")
    parser.add_argument("--constraints", type=str, default="")
    parser.add_argument("--reference", type=str, default="")
    parser.add_argument("--scenario", type=str, default="")
    parser.add_argument("--list_prompts", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    library = PromptLibrary(args.prompt_dir)
    info("提示词目录：%s，共 %d 个模板" % (args.prompt_dir, len(library.names())))
    if args.list_prompts:
        for row in library.describe():
            info("%-24s 参数 %s，%d 字"
                 % (row["name"], ",".join(row["placeholders"]), row["chars"]))
        return
    if args.task == "generate":
        if len(args.requirement.strip()) == 0:
            fail("生成任务需要 --requirement")
        config = build_llm_config(args)
        client = LLMClient(config)
        pipeline = ScenarioPipeline(client, library, args.output_dir)
        result = pipeline.run_generation(
            requirement=args.requirement,
            function=args.function,
            count=args.count,
            operation=args.operation,
            constraints=args.constraints,
            reference=args.reference,
        )
        info("调用统计：%s" % json.dumps(client.usage_summary(), ensure_ascii=False))
        return
    if args.task == "extract":
        if len(args.scenario.strip()) == 0:
            fail("抽取任务需要 --scenario")
        config = build_llm_config(args)
        client = LLMClient(config)
        pipeline = ScenarioPipeline(client, library, args.output_dir)
        triples = pipeline.extract_triples(args.scenario)
        path = make_output_path(args.output_dir, "triples", args.function, "triples")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(triples, handle, ensure_ascii=False, indent=2)
        info("三元组已写入 %s" % path)
        return
    if args.task == "evaluate":
        if len(args.scenario.strip()) == 0:
            fail("评价任务需要 --scenario")
        config = build_llm_config(args)
        client = LLMClient(config)
        pipeline = ScenarioPipeline(client, library, args.output_dir)
        payload = pipeline.evaluate_scenario(args.scenario)
        path = make_output_path(args.output_dir, "evaluation", args.function, "eval")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        info("评价结果已写入 %s" % path)
        return
    if args.task == "evolve":
        if len(args.scenario.strip()) == 0:
            fail("演化任务需要 --scenario")
        config = build_llm_config(args)
        client = LLMClient(config)
        pipeline = ScenarioPipeline(client, library, args.output_dir)
        payload = pipeline.evolve_scenario(
            scenario=args.scenario,
            operation=args.operation,
            mechanism=args.constraints,
            preserve=args.reference,
        )
        path = make_output_path(args.output_dir, "evolve", args.function, "evolve")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        info("演化结果已写入 %s" % path)
        return
    fail("未知任务：%s" % args.task)


if __name__ == "__main__":
    main()

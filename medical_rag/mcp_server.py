from __future__ import annotations

import json
import sys
from .bridge import _positive_int
from .library import Library


def _citation(item: dict) -> str:
    book = item.get("book") or "未知教材"
    page = item.get("page", "?")
    section = item.get("section") or "未标注章节"
    return f"依据《{book}》原书第 {page} 页（{section}）"


def _public_evidence(item: object) -> str:
    if not isinstance(item, dict):
        return str(item)
    return f"{_citation(item)}\n{item.get('text') or ''}"


def _public_text(name: str, data: object) -> str:
    if name == "search_medical_books":
        items = data if isinstance(data, list) else []
        return "\n\n".join(_public_evidence(item) for item in items) if items else "本地教材知识库没有检索到证据。"
    if name == "get_book_section":
        if data is None:
            return "未找到该证据块（chunk_id 可能已失效，请重新调用 search_medical_books）"
        return _public_evidence(data)
    if name == "answer_medical_question" and isinstance(data, dict):
        lines = [f"状态：{data.get('status', '未知')}"]
        guidance = data.get("answer_guidance")
        if guidance:
            lines.append(f"作答提示：{guidance}")
        lines.extend(["", "证据："])
        evidence = data.get("evidence")
        if isinstance(evidence, list) and evidence:
            for item in evidence:
                lines.extend([_public_evidence(item), ""])
        else:
            lines.append("本地教材知识库没有检索到证据。")
        return "\n".join(lines).strip()
    return json.dumps(data, ensure_ascii=False, indent=2)


def _out(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


class LibraryUnavailable(RuntimeError):
    """本地书库打不开（索引缺失 / 只读 / 路径错）：回 JSON-RPC error，但不退出进程。"""


def main() -> None:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    # 书库延迟到首次 tools/call 再打开：打不开时只是每条请求回 error，
    # 而不是让整个 MCP 进程在启动阶段就带着 traceback 退出（客户端只看到 EOF）。
    state: dict = {"library": None, "error": ""}

    def get_library() -> Library:
        if state["library"] is not None:
            return state["library"]
        if state["error"]:
            raise LibraryUnavailable(state["error"])
        try:
            state["library"] = Library()
        except Exception as exc:  # noqa: BLE001 - 任何打开失败都要变成可应答的错误
            state["error"] = f"{type(exc).__name__}: {exc}"
            raise LibraryUnavailable(state["error"]) from exc
        return state["library"]

    for raw in sys.stdin:
        if not raw.strip():
            continue
        request = {}
        try:
            request = json.loads(raw)
            if not isinstance(request, dict):
                # JSON-RPC batch 数组（[1,2] 之类）暂时不支持：给一条错误而不是崩掉
                raise ValueError("请求必须是 JSON 对象（暂不支持 batch 数组）")
            request_id = request.get("id")
            method = request.get("method")
            params = request.get("params") or {}
            if method == "initialize":
                _out({"jsonrpc":"2.0","id":request_id,"result":{
                    "protocolVersion":"2024-11-05",
                    "capabilities":{"tools":{}},
                    "serverInfo":{"name":"medical-books","version":"0.2.0"},
                    "instructions":"Use answer_medical_question for exam questions, or search_medical_books for focused lookup. Answer only from returned evidence, cite 《书名》、原书页码和章节, and say when the local library did not provide evidence. Do not invent textbook support. Pages are calibrated printed book pages (原书页码), so users can verify them directly in the physical book. Pass book (id, title or abbreviation like 组胚) to limit a search to one textbook; answer_medical_question detects a textbook named in the question automatically and reports it as book_scope."
                }})
            elif method in {"notifications/initialized", "notifications/cancelled"}:
                continue
            elif method == "ping":
                _out({"jsonrpc":"2.0","id":request_id,"result":{}})
            elif method == "tools/list":
                _out({"jsonrpc":"2.0","id":request_id,"result":{"tools":[
                    {"name":"list_books","description":"列出本地已建立索引的医学书籍及索引状态。","inputSchema":{"type":"object","properties":{}}},
                    {"name":"search_medical_books","description":"只在本地医学教材中检索证据。每条结果含书名、原书页码、章节和原文；内部定位标识仅供工具链调用。无结果时不得声称教材支持某个答案。可用 book 限定教材。","inputSchema":{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer","minimum":1,"maximum":20},"book":{"type":"string","description":"可选：限定教材（id、书名或缩写，如 组胚）；不填则全库检索"}},"required":["query"]}},
                    {"name":"get_book_section","description":"根据 search_medical_books 返回的 chunk_id 获取原始证据块。","inputSchema":{"type":"object","properties":{"chunk_id":{"type":"string"}},"required":["chunk_id"]}},
                    {"name":"answer_medical_question","description":"分析医学题目、识别题型并进行多轮本地教材检索，返回带页码的证据包。题干点名教材（如“组胚里…”）时会自动限定范围；也可用 book 显式限定。调用方必须基于 evidence 生成最终答案；本工具不会凭空补充教材没有的结论。","inputSchema":{"type":"object","properties":{"question":{"type":"string"},"limit":{"type":"integer","minimum":1,"maximum":20},"book":{"type":"string","description":"可选：限定教材（id、书名或缩写）；不填则自动识别题干点名的教材，识别不到就全库检索"}},"required":["question"]}}
                ]}})
            elif method == "tools/call":
                name = params.get("name")
                args = params.get("arguments") or {}
                if not isinstance(args, dict):
                    args = {}
                try:
                    lib = get_library()
                    if name == "list_books":
                        data = lib.list_books()
                    elif name == "search_medical_books":
                        # 复用 bridge 的兜底：LLM 客户端会把 limit 传成 null/字符串，
                        # 裸值直接进 Library 会变成一条无信息量的 TypeError 工具错误
                        data = lib.search(
                            args.get("query", ""),
                            _positive_int(args.get("limit"), 5),
                            book=args.get("book"),
                        )
                    elif name == "get_book_section":
                        data = lib.get_chunk(args.get("chunk_id", ""))
                    elif name == "answer_medical_question":
                        data = lib.answer_question(
                            args.get("question", ""),
                            _positive_int(args.get("limit"), 6),
                            book=args.get("book"),
                        )
                    else:
                        raise ValueError(f"未知工具：{name}")
                except LibraryUnavailable as exc:
                    _out({"jsonrpc":"2.0","id":request_id,"error":{"code":-32000,"message":f"本地书库不可用：{exc}"}})
                    continue
                except Exception as exc:  # noqa: BLE001 - 工具内部异常用 tool result 表达
                    _out({"jsonrpc":"2.0","id":request_id,"result":{
                        "content":[{"type":"text","text":f"工具执行失败：{type(exc).__name__}: {exc}"}],
                        "isError": True
                    }})
                    continue
                text = _public_text(name, data)
                _out({"jsonrpc":"2.0","id":request_id,"result":{"content":[{"type":"text","text":text}],"structuredContent":data}})
            else:
                _out({"jsonrpc":"2.0","id":request_id,"error":{"code":-32601,"message":f"未知方法：{method}"}})
        except Exception as exc:
            request_id = request.get("id") if isinstance(request, dict) else None
            _out({"jsonrpc":"2.0","id":request_id,"error":{"code":-32000,"message":str(exc)}})


if __name__ == "__main__":
    main()

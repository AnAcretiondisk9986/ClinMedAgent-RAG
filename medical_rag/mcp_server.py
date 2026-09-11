from __future__ import annotations

import json
import sys
from .library import Library


def _out(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> None:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    library = Library()
    for raw in sys.stdin:
        if not raw.strip():
            continue
        request = {}
        try:
            request = json.loads(raw)
            request_id = request.get("id")
            method = request.get("method")
            params = request.get("params") or {}
            if method == "initialize":
                _out({"jsonrpc":"2.0","id":request_id,"result":{
                    "protocolVersion":"2024-11-05",
                    "capabilities":{"tools":{}},
                    "serverInfo":{"name":"medical-books","version":"0.2.0"},
                    "instructions":"Use answer_medical_question for exam questions, or search_medical_books for focused lookup. Answer only from returned evidence, cite book/page/section/chunk_id, and say when the local library did not provide evidence. Do not invent textbook support. Pages are calibrated printed book pages (原书页码), so users can verify them directly in the physical book. Pass book (id, title or abbreviation like 组胚) to limit a search to one textbook; answer_medical_question detects a textbook named in the question automatically and reports it as book_scope."
                }})
            elif method in {"notifications/initialized", "notifications/cancelled"}:
                continue
            elif method == "ping":
                _out({"jsonrpc":"2.0","id":request_id,"result":{}})
            elif method == "tools/list":
                _out({"jsonrpc":"2.0","id":request_id,"result":{"tools":[
                    {"name":"list_books","description":"列出本地已建立索引的医学书籍及索引状态。","inputSchema":{"type":"object","properties":{}}},
                    {"name":"search_medical_books","description":"只在本地医学教材中检索证据。每条结果含书名、页码、章节、原文和稳定 chunk_id。无结果时不得声称教材支持某个答案。可用 book 限定教材。","inputSchema":{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer","minimum":1,"maximum":20},"book":{"type":"string","description":"可选：限定教材（id、书名或缩写，如 组胚）；不填则全库检索"}},"required":["query"]}},
                    {"name":"get_book_section","description":"根据 search_medical_books 返回的 chunk_id 获取原始证据块。","inputSchema":{"type":"object","properties":{"chunk_id":{"type":"string"}},"required":["chunk_id"]}},
                    {"name":"answer_medical_question","description":"分析医学题目、识别题型并进行多轮本地教材检索，返回带页码的证据包。题干点名教材（如“组胚里…”）时会自动限定范围；也可用 book 显式限定。调用方必须基于 evidence 生成最终答案；本工具不会凭空补充教材没有的结论。","inputSchema":{"type":"object","properties":{"question":{"type":"string"},"limit":{"type":"integer","minimum":1,"maximum":20},"book":{"type":"string","description":"可选：限定教材（id、书名或缩写）；不填则自动识别题干点名的教材，识别不到就全库检索"}},"required":["question"]}}
                ]}})
            elif method == "tools/call":
                name = params.get("name")
                args = params.get("arguments") or {}
                if name == "list_books":
                    data = library.list_books()
                elif name == "search_medical_books":
                    data = library.search(args.get("query", ""), args.get("limit", 5), book=args.get("book"))
                elif name == "get_book_section":
                    data = library.get_chunk(args.get("chunk_id", ""))
                elif name == "answer_medical_question":
                    data = library.answer_question(args.get("question", ""), args.get("limit", 6), book=args.get("book"))
                else:
                    raise ValueError(f"未知工具：{name}")
                text = json.dumps(data, ensure_ascii=False, indent=2)
                _out({"jsonrpc":"2.0","id":request_id,"result":{"content":[{"type":"text","text":text}],"structuredContent":data}})
            else:
                _out({"jsonrpc":"2.0","id":request_id,"error":{"code":-32601,"message":f"未知方法：{method}"}})
        except Exception as exc:
            _out({"jsonrpc":"2.0","id":request.get("id"),"error":{"code":-32000,"message":str(exc)}})


if __name__ == "__main__":
    main()

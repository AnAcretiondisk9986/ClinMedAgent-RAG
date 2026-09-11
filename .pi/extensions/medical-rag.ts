/**
 * Medical RAG extension for pi
 *
 * Exposes the local medical textbook library (`medical_rag`) as native pi
 * tools. Pi has no built-in MCP support, so this extension talks to the Python
 * package through a small JSON bridge (`python -m medical_rag.bridge`) instead
 * of the MCP stdio server used by Codex.
 *
 * Project root is discovered by walking up from the session cwd until a
 * `medical_rag/bridge.py` is found. Override with:
 *   MEDICAL_RAG_ROOT    absolute path to the project containing medical_rag/
 *   MEDICAL_RAG_PYTHON  Python executable to use (default: python/python3)
 *   MEDICAL_RAG_DB      path to a specific library.sqlite3
 */

import { spawn, type ChildProcess } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

const BRIDGE_MODULE = "medical_rag.bridge";
const DEFAULT_TIMEOUT_MS = 120_000;

interface BridgeResponse {
	ok: boolean;
	data?: unknown;
	error?: string;
}

interface PythonCandidate {
	command: string;
	prefixArgs: string[];
}

let cachedPython: PythonCandidate | undefined;

function pythonCandidates(): PythonCandidate[] {
	const configured = process.env.MEDICAL_RAG_PYTHON?.trim();
	if (configured) return [{ command: configured, prefixArgs: [] }];
	return process.platform === "win32"
		? [
				{ command: "python", prefixArgs: [] },
				{ command: "py", prefixArgs: ["-3"] },
			]
		: [
				{ command: "python3", prefixArgs: [] },
				{ command: "python", prefixArgs: [] },
			];
}

function findProjectRoot(cwd: string): string | undefined {
	const configured = process.env.MEDICAL_RAG_ROOT?.trim();
	if (configured && existsSync(join(configured, "medical_rag", "bridge.py"))) {
		return resolve(configured);
	}
	let current = resolve(cwd);
	for (;;) {
		if (existsSync(join(current, "medical_rag", "bridge.py"))) return current;
		const parent = dirname(current);
		if (parent === current) return undefined;
		current = parent;
	}
}

function runBridge(root: string, request: Record<string, unknown>, signal?: AbortSignal): Promise<unknown> {
	return new Promise((resolvePromise, rejectPromise) => {
		let child: ChildProcess | undefined;
		let stdout = "";
		let stderr = "";
		let finished = false;
		let candidateIndex = 0;
		let timer: NodeJS.Timeout | undefined;

		const cleanup = () => {
			if (timer) clearTimeout(timer);
			signal?.removeEventListener("abort", onAbort);
		};
		const succeed = (value: unknown) => {
			if (finished) return;
			finished = true;
			cleanup();
			resolvePromise(value);
		};
		const fail = (error: Error) => {
			if (finished) return;
			finished = true;
			cleanup();
			rejectPromise(error);
		};
		const onAbort = () => {
			child?.kill();
			fail(new Error("medical_rag bridge 调用已取消"));
		};

		const attempt = () => {
			if (finished) return;
			const candidates = cachedPython ? [cachedPython] : pythonCandidates();
			if (!cachedPython && candidateIndex >= candidates.length) {
				fail(new Error("未找到可用的 Python 解释器，请设置环境变量 MEDICAL_RAG_PYTHON"));
				return;
			}
			const candidate = cachedPython ?? candidates[candidateIndex++];
			stdout = "";
			stderr = "";
			child = spawn(candidate.command, [...candidate.prefixArgs, "-m", BRIDGE_MODULE], {
				cwd: root,
				env: { ...process.env, PYTHONIOENCODING: "utf-8", PYTHONUTF8: "1" },
				stdio: ["pipe", "pipe", "pipe"],
				windowsHide: true,
			});
			child.on("error", (error: NodeJS.ErrnoException) => {
				if (error.code === "ENOENT" && !cachedPython) {
					attempt();
					return;
				}
				fail(new Error(`无法启动 Python（${candidate.command}）：${error.message}`));
			});
			child.stdout?.on("data", (chunk: Buffer) => {
				stdout += chunk.toString("utf8");
			});
			child.stderr?.on("data", (chunk: Buffer) => {
				stderr += chunk.toString("utf8");
			});
			child.on("close", (code) => {
				if (finished) return;
				const line = stdout.trim().split(/\r?\n/).filter(Boolean).pop();
				if (!line) {
					const detail = stderr.trim() ? `：${stderr.trim().slice(-500)}` : `（退出码 ${code}）`;
					fail(new Error(`medical_rag bridge 没有返回数据${detail}`));
					return;
				}
				try {
					cachedPython = candidate;
					succeed(JSON.parse(line));
				} catch {
					fail(new Error(`medical_rag bridge 返回了非法 JSON：${line.slice(0, 300)}`));
				}
			});
			child.stdin?.end(`${JSON.stringify(request)}\n`);
		};

		signal?.addEventListener("abort", onAbort, { once: true });
		timer = setTimeout(() => {
			child?.kill();
			fail(new Error("medical_rag bridge 调用超时"));
		}, DEFAULT_TIMEOUT_MS);
		attempt();
	});
}

async function callBridge(
	ctx: ExtensionContext,
	request: Record<string, unknown>,
	signal?: AbortSignal,
): Promise<unknown> {
	const root = findProjectRoot(ctx.cwd);
	if (!root) {
		throw new Error(
			"未找到 medical_rag 项目根目录（应包含 medical_rag/bridge.py）。请设置环境变量 MEDICAL_RAG_ROOT 指向该项目。",
		);
	}
	const db = process.env.MEDICAL_RAG_DB?.trim();
	const payload = db ? { ...request, db } : request;
	const response = (await runBridge(root, payload, signal)) as BridgeResponse;
	if (!response || typeof response !== "object") {
		throw new Error("medical_rag bridge 返回格式异常");
	}
	if (!response.ok) {
		throw new Error(response.error || "medical_rag bridge 调用失败");
	}
	return response.data;
}

interface EvidenceItem {
	book: string;
	page: number;
	section: string;
	text: string;
	chunk_id: string;
	score?: number;
	retrieval_score?: number;
	context?: EvidenceItem[];
}

function formatEvidence(item: EvidenceItem, index?: number): string {
	const prefix = index === undefined ? "" : `[${index}] `;
	const scores = [
		item.score !== undefined ? `score=${item.score}` : undefined,
		item.retrieval_score !== undefined ? `retrieval_score=${item.retrieval_score}` : undefined,
	]
		.filter(Boolean)
		.join(" · ");
	return `${prefix}《${item.book}》第${item.page}页 · ${item.section}${scores ? ` · ${scores}` : ""}\n${item.text}\nchunk_id=${item.chunk_id}`;
}

function formatEvidenceList(items: EvidenceItem[]): string {
	if (!items.length) return "本地教材知识库没有检索到证据。";
	return items.map((item, index) => formatEvidence(item, index + 1)).join("\n\n");
}

function formatAnswer(data: Record<string, unknown>): string {
	const plan = (data.plan ?? {}) as Record<string, unknown>;
	const evidence = (data.evidence ?? []) as EvidenceItem[];
	const lines: string[] = [];
	lines.push(`状态：${data.status ?? "未知"}`);
	lines.push(`题型：${plan.question_type ?? "未知"}`);
	lines.push(`核心概念：${Array.isArray(plan.concepts) && plan.concepts.length ? plan.concepts.join("、") : "未识别"}`);
	const scope = (data.book_scope ?? null) as Record<string, unknown> | null;
	if (scope && scope.title) {
		lines.push(`检索范围：《${scope.title}》${scope.mention ? `（题干点名“${scope.mention}”）` : "（显式指定）"}`);
	}
	if (Array.isArray(plan.queries) && plan.queries.length) {
		lines.push(`检索式：${plan.queries.map((query) => JSON.stringify(query)).join(" → ")}`);
	}
	if (data.answer_guidance) lines.push(`作答提示：${data.answer_guidance}`);
	if (data.citation_rule) lines.push(`引用规则：${data.citation_rule}`);
	lines.push("");
	lines.push(evidence.length ? "证据：" : "证据：本地教材知识库没有检索到证据。");
	for (const [index, item] of evidence.entries()) {
		lines.push("");
		lines.push(formatEvidence(item, index + 1));
		if (item.context?.length) {
			lines.push("相邻上下文：");
			for (const context of item.context) {
				if (context.chunk_id === item.chunk_id) continue;
				lines.push(`  - 《${context.book}》第${context.page}页 · ${context.section}：${context.text.slice(0, 220)}`);
			}
		}
	}
	return lines.join("\n");
}

function formatBooks(books: Record<string, unknown>[]): string {
	if (!books.length) {
		return "本地书库为空。请先运行 `python -m medical_rag.cli ingest-text res/系统解剖学/processed_v3` 或 `ingest <pdf>` 建立索引。";
	}
	return books
		.map(
			(book) =>
				`《${book.title}》· ${book.pages} 页 · 可提取文字 ${book.extractable_pages} 页 · 图片页 ${book.image_only_pages} 页\n  ${book.path}`,
		)
		.join("\n");
}

export default function medicalRagExtension(pi: ExtensionAPI) {
	const guidelines = [
		"Use medical_answer_question for Chinese medical/anatomy exam questions and medical_search_books for focused lookups before answering from memory.",
		"Pass book (id, title or abbreviation like 组胚) to limit a search to one textbook; medical_answer_question automatically scopes to a textbook named in the question and reports it as book_scope.",
		"Cite 《书名》、原书页码、章节和 chunk_id from medical_* tool results. Pages are calibrated printed book pages (原书页码) and can be checked directly in the physical book. If a medical_* tool returns no evidence, say the local library has no supporting evidence instead of inventing textbook support.",
	];

	pi.registerTool({
		name: "medical_list_books",
		label: "Medical: List Books",
		description: "列出本地医学教材知识库中已建立索引的书籍及索引状态。",
		promptSnippet: "List indexed medical textbooks in the local library",
		promptGuidelines: guidelines,
		parameters: Type.Object({}),
		async execute(_toolCallId, _params, signal, _onUpdate, ctx) {
			const books = (await callBridge(ctx, { action: "list_books" }, signal)) as Record<string, unknown>[];
			return {
				content: [{ type: "text", text: formatBooks(books) }],
				details: { books },
			};
		},
	});

	pi.registerTool({
		name: "medical_search_books",
		label: "Medical: Search Books",
		description:
			"只在本地医学教材中检索证据。每条结果含书名、页码、章节、原文和稳定 chunk_id；无结果时不得声称教材支持某个答案。可用 book 限定教材。",
		promptSnippet: "Search local medical textbooks for citation-first evidence",
		promptGuidelines: guidelines,
		parameters: Type.Object({
			query: Type.String({ description: "医学检索词或问题，例如“肱骨的形态特点”" }),
			limit: Type.Optional(Type.Integer({ minimum: 1, maximum: 20, description: "返回证据条数，默认 5" })),
			book: Type.Optional(Type.String({ description: "可选：限定教材（id、书名或缩写，如 组胚）；不填则全库检索" })),
		}),
		async execute(_toolCallId, params, signal, _onUpdate, ctx) {
			const items = (await callBridge(
				ctx,
				{ action: "search", query: params.query, limit: params.limit ?? 5, book: params.book },
				signal,
			)) as EvidenceItem[];
			return {
				content: [{ type: "text", text: formatEvidenceList(items) }],
				details: { query: params.query, results: items },
			};
		},
	});

	pi.registerTool({
		name: "medical_get_evidence",
		label: "Medical: Get Evidence",
		description: "根据 medical_search_books 或 medical_answer_question 返回的 chunk_id 获取原始证据块。",
		promptSnippet: "Fetch a full evidence chunk by chunk_id",
		promptGuidelines: guidelines,
		parameters: Type.Object({
			chunk_id: Type.String({ description: "稳定的证据块 ID，来自 medical_* 工具的返回结果" }),
		}),
		async execute(_toolCallId, params, signal, _onUpdate, ctx) {
			const item = (await callBridge(ctx, { action: "get_chunk", chunk_id: params.chunk_id }, signal)) as EvidenceItem;
			return {
				content: [{ type: "text", text: formatEvidence(item) }],
				details: { chunk: item },
			};
		},
	});

	pi.registerTool({
		name: "medical_answer_question",
		label: "Medical: Answer Question",
		description:
			"分析医学题目、识别题型并进行多轮本地教材检索，返回带页码的证据包。题干点名教材（如“组胚里…”）时会自动限定范围，也可用 book 显式限定。调用方必须基于 evidence 生成最终答案；本工具不会凭空补充教材没有的结论。",
		promptSnippet: "Plan a medical exam question and return a citation-first evidence pack",
		promptGuidelines: guidelines,
		parameters: Type.Object({
			question: Type.String({ description: "完整医学题目，例如“肩关节由哪些结构组成？有什么特点？”" }),
			limit: Type.Optional(Type.Integer({ minimum: 1, maximum: 20, description: "返回证据条数，默认 6" })),
			book: Type.Optional(Type.String({ description: "可选：限定教材（id、书名或缩写）；不填则自动识别题干点名的教材" })),
		}),
		async execute(_toolCallId, params, signal, _onUpdate, ctx) {
			const data = (await callBridge(
				ctx,
				{ action: "answer", question: params.question, limit: params.limit ?? 6, book: params.book },
				signal,
			)) as Record<string, unknown>;
			return {
				content: [{ type: "text", text: formatAnswer(data) }],
				details: data,
			};
		},
	});

	pi.registerCommand("medical-rag", {
		description: "检查本地医学教材知识库；也可直接检索：/medical-rag 肩关节的组成",
		handler: async (args, ctx) => {
			const query = args.trim();
			if (!query) {
				try {
					const data = (await callBridge(ctx, { action: "ping" })) as Record<string, unknown>;
					ctx.ui.notify(
						`medical_rag 可用 · 数据库：${data.database} · 已索引 ${data.books} 本书：${(data.titles as string[] | undefined)?.join("、") || "无"}`,
						"info",
					);
				} catch (error) {
					ctx.ui.notify(`medical_rag 不可用：${error instanceof Error ? error.message : String(error)}`, "error");
				}
				return;
			}
			try {
				const items = (await callBridge(ctx, { action: "search", query, limit: 5 })) as EvidenceItem[];
				pi.sendMessage(
					{
						customType: "medical-rag",
						content: `本地医学教材检索「${query}」：\n\n${formatEvidenceList(items)}`,
						display: true,
					},
					{ triggerTurn: false },
				);
			} catch (error) {
				ctx.ui.notify(`检索失败：${error instanceof Error ? error.message : String(error)}`, "error");
			}
		},
	});
}

---
name: medical-rag
description: 本地医学教材检索与题目证据整理。当用户提出医学、解剖学或教材题目（选择题、填空、名词解释、简答、比较、原因题），或要求查教材原文、页码、引用、chunk_id 时使用。优先调用 pi 原生工具 medical_answer_question / medical_search_books；扩展不可用时改用 python -m medical_rag.cli 或 python -m medical_rag.bridge。
---

# 本地医学教材检索（medical-rag）

本技能让 Agent 基于本地医学教材索引回答问题。核心原则是**引用优先**：

- 只根据检索到的教材证据作答，证据不足就明确说“本地教材未检索到依据”，**不得凭空补充教材没有的结论**。
- 每条引用必须包含：`《书名》`、原书页码、章节、`chunk_id`。
- 索引文本经过版面重构（xy-cut 分栏修复、段落重建、标题层级、图注分离）与定向 OCR 纠错，**页码是校准后的原书页码，可直接对应纸质书**（2026-09 v3 重建，原书页 = PDF 页 − 12）。
- 表格页已还原为 HTML 表格（含 `rowspan`），列结构可直接读取，不再是平铺行序。
- OCR 文本仍可能有零星错别字；关键结论应结合返回的上下文或原书页面复核。
- 索引保存在项目下的 `.medical_rag/library.sqlite3`，PDF 不会被复制或上传。
- **多教材**：书库有多本教材时按相关性全库排序；题干点名教材（含“组胚”“系解”这类缩写）会自动收窄到该书，也可用 `book` / `--book` 显式限定（id、书名或缩写）。

## 方式一：pi 原生工具（扩展已加载时首选）

本项目的 `.pi/extensions/medical-rag.ts` 把书库注册成四个 pi 工具：

| 工具 | 用途 |
|------|------|
| `medical_answer_question` | 分析题目（题型/概念/多轮检索），返回带页码的证据包。回答医学题先用它；题干点名教材时自动限定范围（见返回的 `book_scope`）。 |
| `medical_search_books` | 定向检索，返回书名、页码、章节、原文、`chunk_id`；可用 `book` 参数限定教材。 |
| `medical_get_evidence` | 用 `chunk_id` 取回完整证据块。 |
| `medical_list_books` | 列出已索引书籍和索引状态。 |

推荐流程：

1. 题目类问题调用 `medical_answer_question`，根据返回的 `status`、`plan`、`evidence`、`answer_guidance` 作答；题干里出现《书名》或“组胚 / 系解”等缩写时会自动限定教材，返回的 `book_scope` 标明范围，答复时应把范围写进说明；
2. 需要更多细节时，用 `medical_search_books` 补充检索，或对关键 `chunk_id` 调用 `medical_get_evidence`；
3. 最终答案逐点标注引用；`status=no_evidence` 时说明知识库没有依据。

扩展不可用（项目未信任、在其他目录运行、Python 缺失）时，用下面的 CLI。

## 方式二：命令行（不依赖扩展）

在项目根目录执行（即包含 `medical_rag/` 的目录）：

```bash
# 查看书库
python -m medical_rag.cli list

# 定向检索
python -m medical_rag.cli search "肱骨的形态特点" --limit 5

# 限定教材检索：--book 支持索引 id、书名或缩写（组胚 / 系解）
python -m medical_rag.cli search "上皮组织" --book 组胚 --limit 5

# 题目分析 + 多轮检索证据包（JSON）
python -m medical_rag.cli answer "肩关节由哪些结构组成？有什么特点？" --json
python -m medical_rag.cli answer "上皮组织的分类" --book "组织学与胚胎学"
```

输出中的 `score` 只用于排序，不是置信度；引用以 `chunk_id` 为准。

## 方式三：JSON bridge（给其他程序/脚本调用）

`python -m medical_rag.bridge` 从 stdin 读一行 JSON、向 stdout 写一行 JSON：

```bash
echo '{"action":"search","query":"骨的构造","limit":3}' | python -m medical_rag.bridge
echo '{"action":"search","query":"上皮组织","book":"组胚","limit":3}' | python -m medical_rag.bridge
echo '{"action":"answer","question":"请简述骨的构造"}' | python -m medical_rag.bridge
echo '{"action":"get_chunk","chunk_id":"f5f7bdab45a736bba09f"}' | python -m medical_rag.bridge
```

`action` 可选：`ping`、`list_books`、`search`、`get_chunk`、`answer`。
可选字段：`book`（限定教材：id、书名或缩写，如 组胚）、`db`（指定另一个 `library.sqlite3`）。响应为 `{"ok": true, "data": ...}` 或 `{"ok": false, "error": "..."}`。

## 方式四：本地网站（教材工作台）

```bash
python -m medical_rag.webapp        # 默认 http://127.0.0.1:17173
```

面向人看的书库面板：教材状态、封面（PDF 第一页）、逐页预览与页码跳转、该页识别文本与 `chunk_id`、导入教材、一键处理并实时显示各阶段进度和日志。顶部搜索框左侧有「全部教材 / 指定教材」范围下拉，结果标题显示当前范围。导入时会自动检测文字层：文字层 PDF 直接走 text → index，图片型才跑 OCR；导入弹窗的路径一栏也可点「检测文字层」先看一眼。Agent 回答题目仍然优先用 `medical_answer_question`。

## 建立/更新索引

《系统解剖学 第5版》已用 v3 流水线重建索引（2026-09）；日常重新索引只需：

```bash
python -m medical_rag.cli ingest-text res/系统解剖学/processed_v3 --title "系统解剖学 第5版（OCR v3）"
```

完整重建流水线（图片型 PDF，300 DPI，需要两套 Python 环境）：

```bash
# 1. 逐页识别：rapidocr 3.9.2 + PP-OCRv6（GPU）→ text_v3/{boxes,pages}
.venv-ocr/Scripts/python.exe tools/tools_ocr_v3.py "res/系统解剖学/PDF/系统解剖学 第5版.pdf" res/系统解剖学/text_v3

# 2. 版面检测：找出含表格的页 → text_v3/layout.json（含表格框与每页渲染尺寸）
.venv-ocr312/Scripts/python.exe tools/tools_layout_v3.py "res/系统解剖学/PDF/系统解剖学 第5版.pdf" res/系统解剖学/text_v3/layout.json --dpi 300 --device gpu

# 3. 表格结构：裁出表格区域后只跑表格管线（8G 显存友好）→ text_v3/tables/
.venv-ocr312/Scripts/python.exe tools/tools_table_v3_gpu.py "res/系统解剖学/PDF/系统解剖学 第5版.pdf" res/系统解剖学/text_v3

# 4. 结构化：正文 xy-cut + 表格页追加 HTML 表格 → processed_v3/
#    默认处理 res/系统解剖学；其他教材用 --book-dir res/<书名>，章节读 <book-dir>/chapters.json，缺省自动检测
.venv-ocr/Scripts/python.exe tools/tools_structure_v3.py

# 5.（可选）定向 OCR 纠错：聘→腭、挠→桡、於→于、內→内 等
.venv-ocr/Scripts/python.exe tools/tools_fix_ocr_v3.py

# 6. 重建索引
python -m medical_rag.cli ingest-text res/系统解剖学/processed_v3 --title "系统解剖学 第5版（OCR v3）"
```

新增书籍（有文字层的 PDF，如人卫电子教材）：

```bash
# 先检测（只读，不产出文件；输出人读信息 + 一行 JSON）
python -m medical_rag.pdftext "D:\医学教材\组胚.pdf" res/组胚 --detect-only

# 判定为文字层后直接解析（章节、页眉剔除、页脚页码校准、标题层级）
python -m medical_rag.pdftext "D:\医学教材\组胚.pdf" res/组胚
python -m medical_rag.cli ingest-text res/组胚/processed_v3 --title "组织学与胚胎学 第10版"
```

输出目录与 OCR 流水线一致（`processed_v3/{cleaned,structured,quality.json}`），页码引用行为相同；检测结果按 PDF mtime 缓存在 `.medical_rag/pdf_text/`。旧式的纯文字 ingest（无章节与页码校准）仍可用：`python -m medical_rag.cli ingest "D:\医学教材\生理学.pdf"`。

已知局限：4 个空白页（PDF 4/12/166/252）无文字；表格单元格偶有错字，可用 `tools/tools_fix_ocr_v3.py` 的词典继续补充。

## 环境变量

| 变量 | 作用 |
|------|------|
| `MEDICAL_RAG_ROOT` | 指定包含 `medical_rag/` 的项目根目录（扩展在别的 cwd 运行时用） |
| `MEDICAL_RAG_PYTHON` | 指定 Python 解释器（默认 `python` / `python3`） |
| `MEDICAL_RAG_DB` | 指定要查询的 `library.sqlite3` 路径 |

## 回答格式建议

```
依据《系统解剖学 第5版（OCR v3）》原书第 41 页（02-关节学 · 第二节 自由上肢骨连结，chunk_id=...）：
<教材原文要点>

……

说明：以上仅依据本地教材索引；未检索到的部分不作推断。页码为校准后的原书页码，可直接翻书复核。
```

# 本地医学教材知识库（Codex 雏形）

这是一个**本地、可追溯、面向多 Agent 平台**的医学教材检索层。当前已适配 Codex（MCP）与 pi（原生扩展 + Skill），底层不绑定某个 Agent：

- **核心库**：SQLite + 页码级文本块，未来可替换向量索引而不改变工具接口。
- **本地网站**：`python -m medical_rag.webapp` 打开教材工作台，可视化书库状态、封面/逐页预览、导入与处理进度。
- **Codex 接口**：MCP stdio server，暴露 `list_books`、`search_medical_books`、`get_book_section`、`answer_medical_question`。
- **pi 接口**：原生扩展（`.pi/extensions/medical-rag.ts`）+ Skill（`.pi/skills/medical-rag/SKILL.md`）+ JSON bridge，不依赖 MCP。
- **多书籍**：每执行一次 `ingest` 就新增/重建一本书，书籍元数据和路径独立保存。
- **引用优先**：每个检索结果包含书名、页码、章节、原文和稳定 `chunk_id`；没有证据时工具返回空列表。
- **本地优先**：PDF 不会被复制或上传，索引保存到项目下的 `.medical_rag/library.sqlite3`。

## 当前状态

已完成《系统解剖学 第5版》的 OCR 和**版面级结构化**：原书 356 页正文按 18 个章节、**校准后的原书页码**和 **939 个文本块**建立了索引。原始 PDF 是图片型 PDF，应使用 `ingest-text` 导入 `res\系统解剖学\processed_v3` 中的结构化 Markdown。

结构化流水线（v3，2026-09 重建）：

1. `tools_ocr_v3.py` 用 **rapidocr 3.9.2 + PP-OCRv6**（CUDA）以 300 DPI 逐页识别，导出每个文本框的坐标到 `res\系统解剖学\text_v3\boxes\`；全 368 页平均置信度 **0.977**。
2. `tools_layout_v3.py` 用版面模型（PP-DocLayout）扫描全书，找出含表格的页（24 页），输出 `text_v3\layout.json`（表格框 + 每页渲染尺寸）。
3. `tools_table_v3_gpu.py` **裁出表格区域后只跑表格管线**（`TableRecognitionPipelineV2`，关闭版面检测），显存占用约 1.3GB，24 个表格页约 22 秒；表格还原为带 `rowspan` 的 HTML。
4. `tools_structure_v3.py` 基于坐标做版面重建：xy-cut 递归分栏、图内标签分离、段落重建、标题层级、页眉剔除、页脚页码校准；**表格框内的文本框会先从正文剔除**，再在页尾追加 HTML 表格，避免重复收录。
5. `tools_fix_ocr_v3.py` 做定向纠错（`聘→腭`、`挠→桡`、`於→于`、`內→内` 等），只做本教材语境下不可能有歧义的替换。
6. **原书页码 = PDF 页 − 12**（347/348 个页脚页码检测点一致）。检索结果中的页码可直接对应纸质书。
7. `ingest_markdown_tree` 把 `###`/`####` 标题写入证据的 section 元数据（如 `04-消化系统 · 三、腭`），并按标题对齐切块边界。

### 环境与依赖

流水线分两套解释器：OCR 需要 Python 3.14，而 PaddlePaddle 的 GPU 轮子不覆盖 3.14，
所以版面/表格单独用 Python 3.12。两者都由 `pipeline.py` 按阶段分别调用，可用
`MEDICAL_RAG_OCR_PYTHON` / `MEDICAL_RAG_PADDLE_PYTHON` 覆盖。

| 用途 | 目录 | Python | 关键依赖（本项目实测可用版本） |
| --- | --- | --- | --- |
| 核心 / CLI / 网页 / 文字层 | 任意 | >= 3.10（实测 3.14.3） | `PyMuPDF 1.28.2` |
| OCR、纠错、结构化 | `.venv-ocr` | 3.14.3 | `rapidocr 3.9.2`、`onnxruntime-gpu 1.26.0`（CUDA provider 可用）、`opencv-python 5.0.0.93`、`numpy 2.5.3`、`pillow 12.3.0`、`shapely 2.1.2`、`pyclipper 1.4.0` |
| 版面 / 表格 | `.venv-ocr312` | 3.12.10 | `paddlepaddle-gpu 3.2.2`（自报 CUDA 12.9 / cuDNN 9.9.0）、`paddleocr 3.7.0`、`paddlex 3.7.2`、`numpy 2.3.5` |

实测 GPU：NVIDIA RTX 5070 Laptop（驱动 596.49）。换机器后 CUDA/cuDNN 不匹配时
paddle 会回退 CPU（版面与表格阶段会非常慢），可用 `--device cpu` 显式走 CPU；
`onnxruntime-gpu` 可换成 CPU 版 `onnxruntime`。

安装与自检：

```powershell
pip install -r requirements-core.txt                       # 核心
python -m venv .venv-ocr
.venv-ocr\Scripts\pip install -r requirements-core.txt -r requirements-ocr.txt
py -3.12 -m venv .venv-ocr312
.venv-ocr312\Scripts\pip install -r requirements-core.txt -r requirements-paddle.txt

medical-rag doctor                 # 环境自检（解释器 / 依赖 / 索引库 / 工作区）
medical-rag doctor --deep          # 额外启动子解释器验证依赖可导入
medical-rag doctor --repair        # 重建 FTS 索引，修复历史遗留的一致性问题
```

`doctor` 有任何 error 项时退出码为 1，可直接用于脚本或 CI。网页端对应接口为
`GET /api/health`（异常时返回 HTTP 503，需访问令牌）。

已知局限：4 个空白页（PDF 4/12/166/252）无文字；表格单元格偶有错字，可用 `tools_fix_ocr_v3.py` 的词典继续补充；图形编号连字符、半角标点尚未统一。

## 命令行

```powershell
# 导入/重新索引一本书
python -m medical_rag.cli ingest "C:\Users\AnAcretiondisk\Downloads\系统解剖学 第5版 (廖华,姚伯春,孙俊,高艳 等) (z-library.sk, 1lib.sk, z-lib.sk).pdf"

# 查看书库
python -m medical_rag.cli list

# 检索
python -m medical_rag.cli search "肱骨的形态特点"

# 分析题目并整理教材证据（最终答案由 Codex/Agent 根据 evidence 生成）
python -m medical_rag.cli answer "请简述骨的构造"
python -m medical_rag.cli answer "肩关节由哪些结构组成？有什么特点？" --json
```

添加其他书籍只需重复 `ingest`：

```powershell
python -m medical_rag.cli ingest "D:\医学教材\生理学.pdf"
```

### 多教材检索

书库中有多本教材时，检索默认在全库按相关性排序；题干点名教材会自动收窄，也支持显式限定：

```powershell
python -m medical_rag.cli search "上皮组织" --book 组胚      # --book 支持 id、书名或缩写（组胚 / 系解）
python -m medical_rag.cli answer "组胚里上皮组织如何分类"     # 题干点名 → 自动限定《组织学与胚胎学》
```

`answer` 的 JSON 输出会带 `book_scope`（`source=question` 表示题干点名，`explicit` 表示显式指定）；题干同时点名多本或不点名时保持全库检索。MCP、pi 扩展（`book` 参数）与网页搜索范围同样支持。

## 本地网站（教材工作台）

```powershell
python -m medical_rag.webapp            # 默认 http://127.0.0.1:17173（冷门端口）
python -m medical_rag.webapp --port 18080 --no-browser
```

Windows 下也可以直接双击项目根目录的 `start_web.bat`：脚本会自动定位可用 Python（PATH 里的 `python` → `py -3` → 项目自带 `.venv-ocr`）并检查 PyMuPDF，缺依赖时给出中文提示；参数会原样透传，例如 `start_web.bat --port 18080`。

页面功能：

- **教材状态**：每本教材显示 PDF / OCR / 版面 / 表格 / 结构化 / 索引 六个阶段的完成情况与证据块数量；
- **封面与预览**：封面取 PDF 第一页，逐页预览图由 PyMuPDF 渲染并缓存到 `.medical_rag/web_cache/`；
- **页码跳转**：原书页与 PDF 页两个输入框联动（原书页 = PDF 页 − 页码偏移，偏移取自 `quality.json`），左侧还显示该页识别文本与 `chunk_id`；
- **放大与全屏**：预览工具栏支持缩放（适应 / 125% – 400%），放大后可滚动或按住鼠标拖拽平移，自动换成更高清渲染；支持全屏预览，双击图片在适应与 200% 间切换，快捷键 `+` / `-` / `0` / `F`；
- **快捷导入**：支持拖拽上传（大文件流式写入，实时进度）或填写本地 PDF 路径复制到 `res/<书名>/`；
- **一键处理**："自动"模式只跑尚未完成的阶段，也可以手动指定阶段、强制重跑 OCR、选择 DPI 与 GPU/CPU；
- **过程进度**：右下角任务面板显示每个阶段的进度条、当前页码/章节和原始日志，支持取消任务；
- **证据检索**：顶部搜索框直接查本地索引，左侧范围下拉可限定「全部教材 / 指定教材」，结果标题显示当前范围；点击结果跳转到对应教材的对应页。

后端为 Python 标准库 HTTP 服务（无额外依赖），处理时自动调用 `.venv-ocr` 与 `.venv-ocr312`；环境变量 `MEDICAL_RAG_WEB_PORT` 可指定默认端口，`MEDICAL_RAG_OCR_PYTHON` / `MEDICAL_RAG_PADDLE_PYTHON` 可覆盖解释器。API 一览见 `medical_rag/webapp.py` 顶部文档字符串。

### 监听地址与访问令牌

默认只监听 `127.0.0.1`，**不启用鉴权**（仅本机可访问，等同于本机权限）。

> ⚠️ **`--host 0.0.0.0` 的风险**：该服务没有任何账号体系，一旦监听非本机地址，
> 同一局域网内的任何人都可以读取全部 PDF 与教材文本、上传任意文件、触发 OCR/GPU
> 任务并查看任务日志。因此**默认禁止**监听非本机地址，必须显式加
> `--allow-remote` 才会启动，且此时会自动启用访问令牌。

启用令牌后，服务启动时会打印带令牌的地址，直接点开即可（令牌会种成
`SameSite=Strict` 的 Cookie，所以封面、预览图和 PDF 都能正常加载）。
非浏览器调用可改用 `X-Auth-Token` 请求头或 `?token=` 查询参数：

```bash
python -m medical_rag.webapp --host 0.0.0.0 --allow-remote          # 自动生成令牌
python -m medical_rag.webapp --host 0.0.0.0 --allow-remote --token 你的令牌
MEDICAL_RAG_TOKEN=你的令牌 python -m medical_rag.webapp --host 0.0.0.0 --allow-remote
curl -H "X-Auth-Token: 你的令牌" http://192.168.1.10:17173/api/books
```

即使用令牌，也不要把端口映射到公网。

> 新版 `tools_structure_v3.py` 支持任意教材：`--book-dir res/生理学`，章节优先读 `<book-dir>/chapters.json`，没有就自动检测章标题（跳过目录页），检测不到则按单章输出。

## 文字层 PDF（跳过 OCR）

导入时（网页上传 / 本地路径 / CLI）会**自动检测 PDF 是否带文字层**，结果按 PDF 的 mtime 缓存在 `.medical_rag/pdf_text/`：

- **文字层 PDF**（如人卫电子教材）：自动走「文字层解析 → 建立索引」，通常几秒钟完成，完全不调用 GPU；
- **图片型 / 混合型 PDF**：走原有 OCR 流水线（OCR → 版面 → 表格 → 纠错 → 结构化 → 索引）。

判定依据是文字层覆盖率与平均字数（至少一半页面有 ≥30 字、且平均每页 ≥40 字视为文字层；只有水印/页码的 PDF 仍按图片型处理）。书库列表和详情页会显示「文字层 / 图片型 / 混合型」，处理弹窗会按类型给出对应阶段；导入弹窗的路径一栏还可以点「检测文字层」先看一眼再导入。

命令行单独跑：

```powershell
# 只检测，不产出任何文件（输出人读信息 + 一行 JSON）
python -m medical_rag.pdftext "D:\医学教材\组胚.pdf" res\组胚 --detect-only

# 文字层解析：章节识别、页眉剔除、页脚页码校准、标题层级 → processed_v3
python -m medical_rag.pdftext "D:\医学教材\组胚.pdf" res\组胚
python -m medical_rag.cli ingest-text res\组胚\processed_v3 --title "组织学与胚胎学 第10版"
```

输出目录与 OCR 流水线完全一致（`processed_v3/{cleaned,structured,quality.json}`），因此页码引用、检索和预览行为相同。

## Codex MCP 配置

在支持 MCP 的 Codex 客户端配置中加入一个本地 stdio server（使用绝对路径）：

```toml
[mcp_servers.medical-books]
command = "python"
args = ["-m", "medical_rag.mcp_server"]
cwd = "F:\\AI Agent检索知识库"
```

配置后重新打开 Codex 会话，让它加载该 server。工具的角色是“只返回教材证据”，不是替 Agent 生成医学结论。推荐的回答策略是：先调用 `answer_medical_question` 进行题型识别和多轮检索，再基于返回的 evidence 回答，并明确标注书名、原书页码、章节和 chunk_id；无命中则说明本地教材未检索到依据。`answer_medical_question` 只整理证据，不凭空生成医学结论。

## pi 适配（原生扩展 + Skill）

pi 不内置 MCP，推荐用「CLI + Skill」或原生扩展。本仓库已同时提供两种方式，与 Codex 共用同一套 `Library`：

- **原生工具**（项目被 pi 信任后自动加载）：
  - `medical_answer_question`：分析题目并返回带页码的证据包（回答医学题首选）；
  - `medical_search_books`：定向检索，返回书名、页码、章节、原文和 `chunk_id`；
  - `medical_get_evidence`：按 `chunk_id` 取回完整证据块；
  - `medical_list_books`：查看已索引书籍。
  - 另有 `/medical-rag [关键词]` 命令：不带参数显示书库状态，带参数直接把检索结果注入会话。
- **Skill**：`/skill:medical-rag` 或让 Agent 自动加载，文档化 CLI/工具用法和引用规则。
- **JSON bridge**：`python -m medical_rag.bridge`，从 stdin 读一行 JSON、向 stdout 写一行 JSON，供任意 Agent/脚本调用（扩展底层就用它）。

```powershell
# bridge 调用示例
echo '{"action":"search","query":"骨的构造","limit":3}' | python -m medical_rag.bridge
echo '{"action":"answer","question":"请简述骨的构造"}' | python -m medical_rag.bridge
```

扩展在别的目录运行时可用环境变量定位项目：`MEDICAL_RAG_ROOT`（包含 `medical_rag/` 的根目录）、`MEDICAL_RAG_PYTHON`（Python 解释器）、`MEDICAL_RAG_DB`（指定数据库）。

### 环境变量一览

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `MEDICAL_RAG_ROOT` | 自动定位 | 项目根目录（含 `medical_rag/`） |
| `MEDICAL_RAG_DB` | `<root>/.medical_rag/library.sqlite3` | 索引库路径 |
| `MEDICAL_RAG_PYTHON` | 当前解释器 | 主解释器（文字层解析、索引） |
| `MEDICAL_RAG_OCR_PYTHON` | `.venv-ocr` | OCR / 纠错 / 结构化使用的解释器 |
| `MEDICAL_RAG_PADDLE_PYTHON` | `.venv-ocr312` | 版面 / 表格使用的解释器 |
| `MEDICAL_RAG_WEB_PORT` | `17173` | 网页默认端口 |
| `MEDICAL_RAG_TOKEN` | 空 | 访问令牌；监听非本机地址时必填（缺省自动生成并打印） |
| `MEDICAL_RAG_MAX_UPLOAD` | `2147483648`（2 GiB） | 单次上传字节上限 |
| `MEDICAL_RAG_EMBEDDING` | `lexical` | 向量后端：`lexical` 或 `ollama:<模型>`；不可用时自动回退词法后端 |
| `MEDICAL_RAG_RERANKER` | `feature` | 重排器；未知取值回退特征式重排 |
| `MEDICAL_RAG_ALIASES` | `<root>/.medical_rag/aliases.json` | 教材缩写白名单文件（覆盖/追加内置项） |

教材缩写默认识别 `系解`、`组胚` 等内置别名（见 `medical_rag/qa.py` 的 `DEFAULT_BOOK_ALIASES`）。
自己新增的教材可以写一份 `aliases.json` 补充，避免缩写被当普通词而无法限定范围：

```json
{ "影像": ["医学影像学"], "口组": "口腔组织病理学" }
```

> 缩写只在作为独立引用时生效（`组胚里…`、`《组胚》`、`请问组胚…`）。
> `生理功能`、`组织的分类` 这类普通医学词不会被当成书名，题干没点名教材时保持全库检索。

### 检索实现与已知限制

- 中文检索靠 `chunks_fts_ngram`（字符二元组 FTS5）。`chunks_fts` 用的 unicode61
  会把一整段连续中文当成单个 token，因此**不能**用它做中文子串检索；
- 排序信号包括：完整题干原样出现、相邻概念、概念集中度、章节/段落标题匹配，
  并对目录页（点线引导）、图注、过短页眉降权；每条结果都带 `match_reason`；
- `answer_question` 会做证据去重（完全相同或互为子串者只留一条）与同页限流（最多 2 条），
  并对同书相邻页标注 `adjacent_pages`。**相邻页不会被合并成一条**：引用必须保留
  具体页码才能校验，且 `context`（radius=1）已带回前后页内容；
- 向量通道只对词法候选的前 200 条**重排**，不做独立全库召回——没有 ANN 索引
  （sqlite-vec / faiss 属新依赖）时，独立向量扫描会让延迟回到线性增长；
- 默认的 `lexical` 向量后端是**词法级**表示，不做同义词泛化（`心梗` 与 `心肌梗死`
  仍不相似）。接真语义模型请实现 `medical_rag.embeddings.EmbeddingBackend` 并用
  `MEDICAL_RAG_EMBEDDING` 选中；Ollama 需要以 `--embeddings` 启动并已 pull 一个
  embedding 模型（如 `bge-m3`），否则会自动退回词法后端。

## 题目解答流程

当前已增加题目分析层：

1. 自动识别选择题、填空题、名词解释、比较题、原因题和简答题；
2. 提取医学核心概念，生成完整题干、概念组合和单概念等多组检索查询；
3. 合并去重检索结果，优先保留匹配完整题干的证据；
4. 返回原书页码、章节、chunk_id 及相邻上下文；
5. 由 Codex/其他 Agent 根据证据生成最终答案，并在证据不足时拒答或说明不确定性。

## 下一步路线

1. 建立一组带标准页码的医学题目回归测试集。
2. 增加医学词典、同义词扩展和更可靠的中文分词。
3. 增加本地向量索引和可选 rerank，提高语义检索质量。
4. 表格单元格错字的持续修正，以及图形编号连字符、半角标点归一化。
5. 保持 MCP、CLI、HTTP API 共用同一套 `Library` 接口，以适配其他 Agent 平台。

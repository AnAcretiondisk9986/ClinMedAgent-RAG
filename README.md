# ClinMedAgent-RAG

本地化、可追溯的医学教材检索与问答系统。项目将 PDF 教材解析为按页组织的证据块，使用 SQLite/FTS5 建立本地索引，并通过 CLI、Web、MCP 和 JSON Bridge 为人和 Agent 提供统一的检索接口。

> **定位**：检索与证据定位工具，不替代医生诊断、处方或临床决策。所有医学结论都应回到原教材和可靠临床指南核对。

## 特性

- **本地优先**：教材、索引和处理产物默认保存在本机，不上传 PDF。
- **引用优先**：检索结果返回教材、原书页码、PDF 页码、章节和 `chunk_id`。
- **文字层直通**：自动识别文字层 PDF，跳过 OCR，直接解析并建索引。
- **完整 OCR 流水线**：支持 OCR、版面检测、表格还原、定向纠错和结构化。
- **中文检索优化**：SQLite FTS5 字符二元组索引，适配中文教材，避免全表扫描。
- **候选重排**：支持标题匹配、章节集中度、内容类型降权和可选向量重排。
- **多种接入方式**：本地 Web 工作台、命令行、MCP Server、JSON Lines Bridge。
- **安全默认值**：默认仅监听 loopback；远程监听需要显式开启并使用访问令牌。
- **可诊断可维护**：提供 `doctor` 环境检查、索引修复和 SQLite 压缩命令。

## 系统要求

- Windows 10/11
- Python 3.10+
- 核心依赖：PyMuPDF
- OCR、版面和表格处理为可选依赖，建议使用独立虚拟环境
- NVIDIA GPU 仅在需要 GPU OCR/版面处理时使用

## 安装

### 仅安装核心功能

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements-core.txt
```

或安装项目本身：

```powershell
python -m pip install .
```

核心功能包括：文字层 PDF 解析、SQLite 索引、检索、问答、Web、MCP 和 Bridge。

### 安装 OCR 环境

OCR 依赖建议单独安装：

```powershell
python -m venv .venv-ocr
.venv-ocr\Scripts\activate
python -m pip install -r requirements-core.txt -r requirements-ocr.txt
```

### 安装 Paddle 版面 / 表格环境

Paddle 相关依赖与 Python、CUDA/cuDNN 强相关，建议使用 Python 3.12：

```powershell
py -3.12 -m venv .venv-ocr312
.venv-ocr312\Scripts\activate
python -m pip install -r requirements-core.txt -r requirements-paddle.txt
```

不同机器的 CUDA、驱动和 GPU provider 可能不同，请以本机环境为准。

## 快速开始

### 启动 Web 工作台

```powershell
python -m medical_rag.webapp
```

默认地址：

```text
http://127.0.0.1:17173/
```

Windows 也可以运行：

```powershell
scripts\start_web.bat
```

常用参数：

```powershell
python -m medical_rag.webapp --port 18080
python -m medical_rag.webapp --root F:\MedBooks
python -m medical_rag.webapp --no-browser
```

默认只允许本机访问。若确需局域网访问，必须显式使用：

```powershell
python -m medical_rag.webapp --host 0.0.0.0 --allow-remote
```

远程监听会启用访问令牌。不要将服务直接暴露到公网。

### 导入教材

可以通过 Web 工作台上传 PDF，也可以把教材放入工作区：

```text
res/
└── 教材名称/
    ├── PDF/
    │   └── textbook.pdf
    └── processed_v3/
```

教材状态、文字层检测、处理进度和索引结果可在 Web 工作台查看。

### 文字层 PDF 解析

只检测：

```powershell
python -m medical_rag.pdftext "D:\医学教材\组胚.pdf" res\组胚 --detect-only
```

解析文字层 PDF：

```powershell
python -m medical_rag.pdftext "D:\医学教材\组胚.pdf" res\组胚
```

将已生成的结构化 Markdown 建立索引：

```powershell
python -m medical_rag.cli ingest-text res\组胚\processed_v3 --title "组织学与胚胎学"
```

## 命令行接口

查看帮助：

```powershell
medical-rag --help
medical-rag doctor --help
```

环境与工作区检查：

```powershell
medical-rag doctor
medical-rag doctor --json
medical-rag doctor --deep
```

修复全文索引：

```powershell
medical-rag doctor --repair
```

压缩索引数据库：

```powershell
medical-rag doctor --compact
```

`--repair` 和 `--compact` 会修改索引数据库，请确保数据库目录可写，并在重要数据上提前备份。

## Agent 接入

### MCP Server

在支持 MCP 的客户端配置本地 stdio Server：

```toml
[mcp_servers.medical-books]
command = "python"
args = ["-m", "medical_rag.mcp_server"]
cwd = "F:\\AI Agent检索知识库"
```

主要工具：

- `list_books`：列出教材
- `search_medical_books`：检索教材证据
- `get_book_section`：读取指定章节或证据块
- `answer_medical_question`：基于本地教材生成带引用的回答

### JSON Lines Bridge

启动：

```powershell
medical-rag-bridge
```

每行输入一个 JSON 请求，每行输出一个 JSON 响应。支持的 action：

- `ping`
- `list_books`
- `search`
- `get_chunk`
- `answer`

示例：

```json
{"action":"search","query":"肱骨的形态特点","limit":5}
```

## 数据与目录

运行时数据默认位于项目目录：

```text
.medical_rag/
├── library.sqlite3       # 本地索引数据库
├── pdf_text/             # 文字层检测缓存
└── web_cache/             # Web 预览缓存

.build/                   # 渲染、测试和构建缓存
res/                      # 教材 PDF 与流水线产物
```

上述本地数据默认不纳入 Git。请勿将包含教材 PDF 或敏感资料的目录提交到公共仓库。

## 检索实现概览

1. 规范化查询和教材文本；
2. 使用 FTS5 字符二元组索引召回候选；
3. 根据标题匹配、章节集中度和内容类型计算排序特征；
4. 可选使用 lexical embedding 或 Ollama embedding 进行候选重排；
5. 返回教材名称、原书页码、PDF 页码、章节、证据文本和 `chunk_id`。

默认 embedding 后端为零依赖的 lexical bigram。它可以改善词法相似度排序，但不具备真正的同义词语义泛化能力。

## 测试与质量检查

```powershell
python -m pytest -q
```

JavaScript 前端语法检查：

```powershell
node --check medical_rag/webui/app.js
node --check medical_rag/webui/workspace.js
```

环境诊断：

```powershell
medical-rag doctor --json
```

OCR、Paddle 和 GPU 相关测试应在对应虚拟环境中执行。

## 已知限制

- OCR、版面和表格识别速度取决于本机 GPU、CUDA/cuDNN 和模型缓存。
- 图片型 PDF 必须经过 OCR，处理时间通常明显长于文字层 PDF。
- 默认 SQLite 索引适合本地单机工作区，不是多租户数据库服务。
- lexical embedding 不会自动理解医学同义词；Ollama 等语义后端需要另行配置。
- 系统只提供教材证据检索，不保证医学内容完整、最新或适用于具体患者。

## 许可证

本项目采用 MIT License，详见 [LICENSE](LICENSE)。

## 项目链接

- Repository: <https://github.com/AnAcretiondisk9986/ClinMedAgent-RAG>
- Issues: <https://github.com/AnAcretiondisk9986/ClinMedAgent-RAG/issues>

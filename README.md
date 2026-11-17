# ClinMedAgent-RAG

本地化、可追溯的医学教材检索与问答系统。项目将 PDF 教材解析为按页组织的证据块，使用 SQLite/FTS5 建立本地索引，并通过 CLI、Web、MCP 和 JSON Bridge 为人和 Agent 提供统一的检索接口。

> **定位**：检索与证据定位工具，不替代医生诊断、处方或临床决策。所有医学结论都应回到原教材和可靠临床指南核对。

## 特性

- **本地优先**：教材、索引和处理产物默认保存在本机，不上传 PDF。
- **引用优先**：检索结果返回教材、原书页码、章节和 `chunk_id`；页码按页脚校准为原书页码，可以直接对纸质书核对。
- **文字层直通**：自动识别文字层 PDF，跳过 OCR，直接解析并建索引。
- **完整 OCR 流水线**：支持 OCR、版面检测、表格还原、定向纠错和结构化。
- **中文检索优化**：SQLite FTS5 字符二元组索引，适配中文教材，避免全表扫描。
- **候选重排**：支持标题匹配、章节集中度、内容类型降权和可选向量重排。
- **多种接入方式**：本地 Web 工作台、命令行、MCP Server、JSON Lines Bridge。
- **安全默认值**：默认仅监听 loopback；远程监听需要显式开启并使用访问令牌；请求会校验 Host 与同源 Origin，变更类接口还要求 `application/json`。
- **可整体移动**：索引里只存相对项目根目录的路径，项目目录改名、换盘、复制一份到别处后，书库与检索都不需要重做（详见[文件说明](文件说明_FILE-MAP.md)）。
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
python -m pip install -r 依赖-核心_requirements-core.txt
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
python -m pip install -r 依赖-核心_requirements-core.txt -r 依赖-OCR识别_requirements-ocr.txt
```

### 安装 Paddle 版面 / 表格环境

Paddle 相关依赖与 Python、CUDA/cuDNN 强相关，建议使用 Python 3.12：

```powershell
py -3.12 -m venv .venv-ocr312
.venv-ocr312\Scripts\activate
python -m pip install -r 依赖-核心_requirements-core.txt -r 依赖-版面表格_requirements-paddle.txt
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

Windows 也可以直接双击或运行：

```powershell
.\启动网页工作台_Start-Web.bat
```

（内部真正启动服务的是 `scripts\start_web.bat`，根目录这个文件只负责找 Python 并转发参数。）

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

自检是只读的：用只读连接读取索引库，不建库、不写库；索引库被占用或位于只读介质上时，这一项记为警告而不是错误。

修复全文索引：

```powershell
medical-rag doctor --repair
```

压缩索引数据库：

```powershell
medical-rag doctor --compact
```

`--repair` 和 `--compact` 会修改索引数据库，请确保数据库目录可写，并在重要数据上提前备份。
索引里记录的是相对项目根目录的路径，项目目录被移动/改名后打开库会自动迁移，通常不需要手工改库。

## 环境变量

都可以在启动前用 PowerShell 设置（例如 `$env:MEDICAL_RAG_DB="D:\索引\library.sqlite3"`），不设置时取默认值。

| 变量 | 作用 | 默认 |
| --- | --- | --- |
| `MEDICAL_RAG_ROOT` | 项目根目录；索引里的相对路径按它还原 | 按索引库位置推断 |
| `MEDICAL_RAG_DB` | 索引库路径（未显式指定 `--db` 时使用；MCP 与 Bridge 同样生效） | `<项目根>/.medical_rag/library.sqlite3` |
| `MEDICAL_RAG_EMBEDDING` | 向量后端：`lexical` 或 `ollama:<模型>`；不可用时自动回退词法后端 | `lexical` |
| `MEDICAL_RAG_RERANKER` | 候选重排器，目前只有 `feature` | `feature` |
| `MEDICAL_RAG_ALIASES` | 自定义教材缩写表 JSON 的路径 | 索引库同目录下的 `aliases.json`（存在才生效） |
| `MEDICAL_RAG_PYTHON` | 文字层解析、结构化、建立索引三个阶段使用的 Python | 当前解释器 |
| `MEDICAL_RAG_OCR_PYTHON` | OCR 与定向纠错阶段使用的 Python | 自动探测 `.venv-ocr` |
| `MEDICAL_RAG_PADDLE_PYTHON` | 版面/表格阶段使用的 Python | 自动探测 `.venv-ocr312` |
| `MEDICAL_RAG_STAGE_TIMEOUT_<阶段>` | 覆盖单个流水线阶段的超时秒数，`<=0` 表示不限时；阶段名用 `TEXT`/`OCR`/`LAYOUT`/`TABLES`/`FIX`/`STRUCTURE`/`INDEX` | 见下方默认值 |
| `MEDICAL_RAG_WEB_PORT` | Web 工作台的默认端口 | `17173` |
| `MEDICAL_RAG_TOKEN` | Web 访问令牌 | 本机监听为空；监听非本机地址时自动生成 |
| `MEDICAL_RAG_MAX_UPLOAD` | 单次上传的字节上限 | 2 GB |
| `MEDICAL_RAG_WEB_CACHE_MAX_MB` | 页面预览图缓存上限（MB） | `256` |

各阶段超时默认值（秒）：文字层解析 1800、OCR 按页数估算（12 秒/页，最少 600）、版面 1200、表格 1200、纠错 900、结构化 1200、建立索引 900。

## Agent 接入

### MCP Server

在支持 MCP 的客户端配置本地 stdio Server：

```toml
[mcp_servers.medical-books]
command = "python"
args = ["-m", "medical_rag.mcp_server"]
cwd = "D:\\你的项目目录\\ClinMedAgent-RAG"
```

`cwd` 要写成你本机项目实际所在目录；项目移动过就把这里一起改。

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
├── library.sqlite3       # 本地索引数据库（教材路径按“相对项目根目录”保存）
├── pdf_text/             # 文字层检测缓存
└── web_cache/             # Web 预览缓存

.build/                   # 渲染、测试和构建缓存（可随时删除）
res/                      # 教材 PDF 与流水线产物（索引指向这里）
output/                   # 生成好的 PDF 等交付物（可选，不参与运行）
```

上述目录中 `.build/`、`res/`、`.medical_rag/` 已被 `.gitignore` 排除；`output/` 不在忽略列表里，提交前请自行确认。请勿将包含教材 PDF 或敏感资料的目录提交到公共仓库。

逐项用途（中英文对照）见 [文件说明_FILE-MAP.md](文件说明_FILE-MAP.md)。

## 移动 / 备份 / 换机

本项目按"整个目录可以随便搬"设计：

1. **同一台电脑换目录、换盘、改名**：直接复制或移动整个文件夹即可。索引库只记相对路径，
   第一次打开时会自动把老绝对路径迁移过来，书卡、按书检索、页码正文、预览缓存都不受影响。
2. **复制一份做备份/试验**：两份副本各自用自己的 `res/` 与 `.medical_rag/`，互不干扰（不会再指向原来那份）。
3. **换电脑**：`res/`、`.medical_rag/`、源码可以直接带走；但 `.venv-ocr` 与 `.venv-ocr312`
   里的 `python.exe` 只是"引导器"，它依赖写死的基础 Python（如 `C:\Python314`、当前用户目录下的 Python 3.12），
   换机后必然失效——用 `scripts\重建虚拟环境_Rebuild-Venvs.bat` 重建即可（约 10 GB，重建需联网下载依赖）。
4. **路径长度**：`res/` 下已有接近 200 字符的长路径，新位置建议控制在整个项目根目录不超过 60 个字符，
   避免撞 Windows 260 字符上限。
5. **改名后检查**：`python -m medical_rag.cli doctor` 的「索引路径」与「索引时效性」两项应为 ok；
   MCP 客户端配置里的 `cwd`、快捷方式里的路径也要跟着改。

移动之后如果 `doctor` 报「N 本教材的索引路径找不到」，说明 `res/` 下对应目录被删或改名了，
按提示重建该教材索引即可。

## 检索实现概览

1. 规范化查询和教材文本；
2. 使用 FTS5 字符二元组索引召回候选；
3. 根据标题匹配、章节集中度和内容类型计算排序特征；
4. 可选使用 lexical embedding 或 Ollama embedding 进行候选重排；
5. 返回教材名称、原书页码、章节、证据文本和 `chunk_id`。

默认 embedding 后端为零依赖的 lexical bigram。它可以改善词法相似度排序，但不具备真正的同义词语义泛化能力。

## 测试与质量检查

```powershell
python -m pytest -q
```

其中一个专门的回归测试会真的把假项目目录搬到新路径，验证索引、按书检索与重建索引仍然正常：

```powershell
python -m pytest tests/test_relocation.py -q
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
- Web 工作台的变更类接口只接受 `application/json`（`text/plain` 等返回 415；不带 `Content-Type` 的 curl 和脚本仍可用）；JSON 请求体上限 8 MB，单次上传默认上限 2 GB。
- Web 工作台对所有请求都校验 `Host`，变更请求还要求 `Origin` 同源，不符返回 403；绑定 `0.0.0.0` 等通配地址时做不了 Host 枚举，改由访问令牌把关。
- 页面预览图按 900/1200/1600/2000/2600 五档宽度渲染（请求宽度向上取档），缓存按修改时间淘汰、默认上限 256 MB，同一页只有固定几档尺寸的图。
- lexical embedding 不会自动理解医学同义词；Ollama 等语义后端需要另行配置。
- 系统只提供教材证据检索，不保证医学内容完整、最新或适用于具体患者。

## 许可证

本项目采用 MIT License，详见 [LICENSE](LICENSE)。

## 项目链接

- Repository: <https://github.com/AnAcretiondisk9986/ClinMedAgent-RAG>
- Issues: <https://github.com/AnAcretiondisk9986/ClinMedAgent-RAG/issues>

# 文件说明 / FILE MAP

这份文件解释项目目录里每一项**是做什么的、能不能移动、能不能删除、要不要入库**。
This file explains what every item is for, and whether it can be moved, deleted, or committed.

命名约定 / Naming conventions：

- 用户直接双击或执行的脚本、依赖清单，用「中文用途_English-name」命名，一眼能看懂用途；
- 代码模块保持英文（`medical_rag/`、`tools/*.py`）——它们被 `import` 和子进程按文件名引用，
  改成中文会直接断掉，所以这些文件的用途写在本文档里；
- 教材、缓存、索引等数据目录保持英文短名（`res/`、`.medical_rag/`），避免路径变长和编码问题。

---

## 一、根目录文件（可执行 / 可读的功能文件）

| 名称 Name | 中文用途 | English purpose | 入库 | 说明 |
| --- | --- | --- | --- | --- |
| `启动网页工作台_Start-Web.bat` | 双击即可启动本地网页工作台（端口 17173） | Start the local web workbench | ✅ | 只负责找 Python 并转发参数；真正启动逻辑在 `scripts\start_web.bat` |
| `依赖-核心_requirements-core.txt` | 核心依赖清单（PyMuPDF），检索/网页/MCP 必需 | Core dependencies (PyMuPDF) | ✅ | `pip install -r` 用 |
| `依赖-OCR识别_requirements-ocr.txt` | OCR 流水线依赖（rapidocr、onnxruntime-gpu 等） | OCR pipeline dependencies | ✅ | 装在 `.venv-ocr`（Python 3.14） |
| `依赖-版面表格_requirements-paddle.txt` | 版面/表格流水线依赖（paddlepaddle-gpu 等） | Layout / table pipeline dependencies | ✅ | 装在 `.venv-ocr312`（Python 3.12） |
| `README.md` | 项目说明：安装、启动、接入方式 | Project readme | ✅ | 面向 GitHub 的正式说明 |
| `AGENTS.md` | 给 AI Agent 的协作约定（答案引用格式等） | Agent collaboration rules | ✅ | 不影响程序运行 |
| `pyproject.toml` | 打包 / 依赖声明 / pytest 配置 | Packaging & test config | ✅ | `pip install .` 与测试用 |
| `LICENSE` | MIT 许可证 | MIT license | ✅ | |
| `.gitignore` / `.gitattributes` | Git 忽略规则 / 换行与文本属性 | Git ignore & attributes | ✅ | |
| `文件说明_FILE-MAP.md` | 本文件：目录逐项用途说明 | This file map | ✅ | 移动、清理前先看这里 |

## 二、根目录文件夹

| 名称 Name | 中文用途 | English purpose | 入库 | 移动/删除 |
| --- | --- | --- | --- | --- |
| `medical_rag/` | 核心 Python 包：检索、索引、网页、MCP、Bridge、doctor | Main Python package | ✅ | **必须随项目**；被 `import`，不能改名 |
| `res/` | 教材资源：原版 PDF + 流水线产物 + 索引指向的数据 | Textbook resources | ❌ 本地数据 | 可整体移动，但内部结构不能改（见第四节） |
| `.medical_rag/` | 索引数据库与缓存（`library.sqlite3`、`pdf_text/`、`web_cache/`） | Local index & caches | ❌ 本地数据 | 可移动；删掉后需重建索引（检索暂时为空） |
| `tests/` | 自动化测试（含“移动目录后索引仍可用”的回归测试） | Automated tests | ✅ | **保留**；`test_relocation.py` 专门验证可移动性 |
| `tools/` | 流水线子进程脚本（OCR/版面/表格/纠错/结构化），以及它们共用的 `tools_common.py` | Pipeline subprocess scripts | ✅ | **保留**；文件名被 `pipeline.py` 引用，改名要同步改代码 |
| `scripts/` | 内部启动脚本与虚拟环境重建脚本 | Internal launchers & venv rebuild | ✅ | **保留** |
| `docs/` | 项目文档（如《系统解剖学》标题清单） | Project docs | ✅ | 可保留 |
| `.pi/` | Agent 平台（Pi / Codex）的技能与扩展定义 | Agent platform skill/extension | ✅ | 保留；与网页/检索运行无关 |
| `output/` | 生成好的交付物（名词解释、练习卷 PDF 等） | Generated deliverables (PDFs) | ❌ 可选 | 可删可移；不参与运行 |
| `.venv-ocr/` | OCR 虚拟环境（Python 3.14，约 4.3 GB） | OCR virtualenv | ❌ | **机器专属**：同机移动可用；换电脑请重建（见第五节） |
| `.venv-ocr312/` | 版面/表格虚拟环境（Python 3.12，约 5.2 GB） | Layout/table virtualenv | ❌ | 同上 |
| `.build/` | 渲染缓存、pytest 缓存等构建产物 | Build & test caches | ❌ | **可随时删除**，会自动重建 |
| `.git/` | Git 版本库（含一个外部 Codex 工作树注册） | Git repository | — | 保留；移动仓库本身不影响主仓库 |

## 三、`tools/` 流水线脚本与共用工具（脚本被 `medical_rag/pipeline.py` 以子进程调用）

| 文件 File | 中文用途 | English purpose | 运行环境 |
| --- | --- | --- | --- |
| `tools_common.py` | OCR/版面/表格/纠错这几个脚本共用的产物读写工具：原子写文本、单页 JSON 完整性校验、带 PDF 指纹的页图缓存 | Shared helpers for the OCR/layout/table/fix scripts: atomic write, page-JSON validation, render cache | 不单独运行，被下列脚本 `import` |
| `tools_ocr_v3.py` | 逐页 OCR 识别，输出 `text_v3/boxes/page-*.json` | Page OCR → boxes | `.venv-ocr` |
| `tools_layout_v3.py` | 版面分析（PP-StructureV3），输出 `text_v3/layout.json` | Layout analysis | `.venv-ocr312` |
| `tools_table_v3_gpu.py` | 表格识别，输出 `text_v3/tables/page-*.md` | Table recognition | `.venv-ocr312` |
| `tools_fix_ocr_v3.py` | 按医学词典对 OCR 结果做定向纠错 | Targeted OCR correction | `.venv-ocr` |
| `tools_structure_v3.py` | 把 boxes/表格/版面合成带页码的结构化 Markdown（`processed_v3/structured/`） | Structure Markdown from boxes | `.venv-ocr` |
| `tools_structure_v2.py` | 旧版结构化脚本（存档；v3 会引用其中部分函数） | Legacy structure script (archive) | `.venv-ocr` |

> 不要单独改名这些 `.py`：`pipeline.py`、`test_pipeline.py` 和脚本之间的 `import` 都按文件名引用（`tools_common.py` 也是被 `import` 的）。

## 四、`res/` 教材目录结构（索引指向这里，结构不要改）

```text
res/
└── 教材名称/
    ├── PDF/                     # 原版 PDF（一本教材一个主文件）
    ├── processed_v3/            # 最终产物：structured/*.md + quality.json（建索引用这里）
    └── text_v3/                 # 中间产物：boxes/、tables/、layout.json、manifest.json
```

- 目录名即书名来源；改名等于换教材，需要重新建索引。
- 索引库只记录相对路径（如 `res/系统解剖学/processed_v3`），整份 `res/` 跟着项目一起搬不会失效。

## 五、换电脑 / 移动后要做什么

| 情况 | 需要做的事 |
| --- | --- |
| 同一台电脑换目录、换盘、改名 | 直接搬；`doctor` 的「索引路径」应显示 `N/N 本用相对路径保存`。MCP 配置里的 `cwd` 记得改 |
| 复制一份做备份 | 直接复制；两份副本各自使用自己的 `res/` 与 `.medical_rag/` |
| 换电脑 / 换 Windows 用户 | 源码 + `res/` + `.medical_rag/` 直接带走；两个 `.venv-*` 不要带，双击 `scripts\重建虚拟环境_Rebuild-Venvs.bat` 重建 |
| 只想保留检索功能 | 只需 `medical_rag/`、`res/`、`.medical_rag/` + 一个装了 PyMuPDF 的 Python |

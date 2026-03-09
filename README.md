# Intelligent Document - 智能语音文档系统

[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 基于 **Qwen3-ASR-1.7B** + **Qwen3-1.7B** 双模型的智能语音文档生成系统

---

## 📋 系统简介

Intelligent Document 是一套面向企业文档处理的智能语音文档生成系统。系统采用先进的语音识别（ASR）和大语言模型（LLM）技术，实现从实时录音到标准文档记录的全流程自动化处理。

### 核心能力

- 🎙️ **实时语音识别** - 基于 Qwen3-ASR-1.7B 的高准确率中文语音识别
- 🤖 **智能文档生成** - 利用 Qwen3-1.7B 提取关键信息并生成规范文档
- 🔄 **引擎动态管理** - ASR 与 LLM 引擎互斥运行，优化 GPU 资源利用
- 📝 **分段处理算法** - 智能处理万字级长文本，适配有限上下文窗口
- 🖥️ **现代化 Web 界面** - 实时录音、转写查看、文档编辑一站式操作

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         前端界面 (Web UI)                         │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ 录音控制台   │  │ 语音识别面板  │  │   文档生成工作台      │   │
│  └─────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      后端服务 (FastAPI)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐   │
│  │  WebSocket   │  │ 文档处理引擎  │  │    引擎管理器        │   │
│  │  实时音频流   │  │ 分段处理算法  │  │ ASR/LLM 动态调度    │   │
│  └──────────────┘  └──────────────┘  └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
       ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
       │ Qwen3-ASR  │ │  Qwen3-1.7B  │ │  SQLite DB   │
       │  -1.7B     │ │     LLM      │ │              │
       │ 语音识别引擎 │ │  文档生成引擎 │ │  数据持久化   │
       └──────────────┘ └──────────────┘ └──────────────┘
```

### 数据流

1. **录音识别阶段** - 用户通过 Web 界面录音，ASR 引擎实时转写为文本
2. **人工审核阶段** - 用户在界面上查看、编辑识别结果，确保准确性
3. **文档生成阶段** - LLM 引擎分析文本，提取身份信息、地址、经过、动机等要素
4. **编辑导出阶段** - 生成标准格式文档，支持在线编辑和导出

---

## 🤖 模型配置

### ASR 模型：Qwen3-ASR-1.7B

| 属性 | 配置 |
|------|------|
| 模型名称 | Qwen3-ASR-1.7B |
| 用途 | 语音识别转写 |
| 参数规模 | 1.7B |
| 服务端口 | 8001 |
| 运行方式 | Docker 容器动态启动 |

### LLM 模型：Qwen3-1.7B

| 属性 | 配置 |
|------|------|
| 模型名称 | Qwen3-1.7B |
| 用途 | 文档内容分析与生成 |
| 参数规模 | 1.7B |
| 上下文长度 | 2048 tokens |
| 服务端口 | 8002 |
| 运行方式 | Docker 容器动态启动 |

### 资源优化策略

- **引擎互斥运行**：ASR 与 LLM 引擎不会同时占用 GPU，最大化资源利用
- **动态启停**：引擎按需启动，处理完成后自动停止释放资源
- **分段处理**：万字长文本智能分段，适配有限上下文窗口

---

## 🚀 快速开始

### 环境要求

- **操作系统**: Windows 10/11 或 Linux
- **GPU**: NVIDIA GPU (推荐 RTX 5060 Ti 8GB 或更高)
- **软件**: Docker Desktop 4.0+
- **显存**: 最低 8GB VRAM

### 1. 克隆仓库

```bash
git clone https://github.com/yourusername/intelligent-document.git
cd intelligent-document
```

### 2. 下载模型

```bash
# 创建模型目录
mkdir -p models

# 下载 ASR 模型
git clone https://huggingface.co/Qwen/Qwen3-ASR-1.7B models/Qwen3-ASR-1.7B

# 下载 LLM 模型
git clone https://huggingface.co/Qwen/Qwen3-1.7B models/Qwen3-1.7B
```

### 3. 启动系统

**Windows:**
```batch
start.bat
```

**Linux/macOS:**
```bash
bash start.sh
```

或直接运行 Docker：
```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d
```

### 4. 访问系统

打开浏览器访问：**http://localhost:8080**

---

## 📁 项目结构

```
intelligent-document/
├── 📄 README.md                 # 项目说明文档
├── 📄 docker-compose.yml        # Docker 编排配置
├── 📄 Dockerfile               # 后端服务镜像构建
├── 📄 Dockerfile.asr           # ASR 服务镜像构建
├── 📄 start.bat / start.sh     # 启动脚本
├── 📄 stop.bat / stop.sh       # 停止脚本
│
├── 📁 backend/                 # 后端服务
│   ├── 📄 main.py             # FastAPI 主应用
│   ├── 📄 database.py         # SQLite 数据库操作
│   ├── 📄 engine_manager.py   # AI 引擎管理器
│   ├── 📄 requirements.txt    # Python 依赖
│   │
│   ├── 📁 services/           # 业务服务
│   │   └── 📄 document_processor.py  # 文档处理引擎
│   │
│   └── 📁 static/             # 前端静态资源
│       ├── 📄 index.html      # 主页面
│       ├── 📁 css/            # 样式文件
│       └── 📁 js/             # JavaScript 脚本
│
├── 📁 models/                  # 模型目录 (需下载)
│   ├── 📁 Qwen3-ASR-1.7B/    # 语音识别模型
│   └── 📁 Qwen3-1.7B/        # 大语言模型
│
├── 📁 data/                    # 数据存储目录
│   └── 📄 intelligent_document.db  # SQLite 数据库
│
├── 📁 plans/                   # 设计文档
│   └── 📄 document_system_architecture.md  # 架构设计
│
└── 📁 test/                    # 测试脚本
    └── 📄 test_api.py         # API 测试
```

---

## 🔧 配置说明

### 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `ASR_API_URL` | http://localhost:8001 | ASR 服务地址 |
| `ASR_MODEL_NAME` | qwen3-asr | ASR 模型名称 |
| `LLM_API_URL` | http://localhost:8002 | LLM 服务地址 |
| `LLM_MODEL_NAME` | qwen3-1.7b | LLM 模型名称 |
| `LLM_MAX_MODEL_LEN` | 2048 | LLM 最大上下文长度 |
| `DATABASE_PATH` | /app/data/intelligent_document.db | 数据库路径 |

### 服务端口

| 服务 | 端口 | 说明 |
|------|------|------|
| Web 界面 | 8080 | 主应用访问端口 |
| ASR API | 8001 | 语音识别服务 (动态启动) |
| LLM API | 8002 | 大模型服务 (动态启动) |

---

## 🎯 使用指南

### 1. 实时录音识别

1. 访问 Web 界面
2. 点击"启动 ASR 引擎"按钮
3. 等待引擎就绪
4. 点击"开始录音"进行实时转写
5. 录音完成后点击"停止"

### 2. 生成智能文档

1. 在记录列表中选择一条记录
2. 点击"生成文档"按钮
3. 系统自动启动 LLM 引擎
4. 等待分段处理和文档生成
5. 查看、编辑生成的文档内容

### 3. 导出文档

- 支持导出为 Word 文档 (.docx)
- 支持导出为 PDF 文档
- 支持导出为纯文本 (.txt)

---

## 🛠️ 开发指南

### 本地开发环境

```bash
# 创建虚拟环境
python -m venv venv

# 激活环境
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 安装依赖
cd backend
pip install -r requirements.txt

# 启动开发服务器
uvicorn main:app --reload --port 8080
```

### 运行测试

```bash
cd test
python test_api.py
```

---

## 📚 技术栈

- **后端**: Python 3.10+, FastAPI, WebSocket
- **前端**: HTML5, CSS3, JavaScript (原生)
- **数据库**: SQLite
- **AI 推理**: vLLM, Transformers
- **容器化**: Docker, Docker Compose
- **模型**: Qwen3-ASR-1.7B, Qwen3-1.7B

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交变更 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

- [Qwen](https://huggingface.co/Qwen) - 提供优秀的开源模型
- [FastAPI](https://fastapi.tiangolo.com/) - 高性能 Web 框架
- [vLLM](https://github.com/vllm-project/vllm) - 高效的 LLM 推理引擎

---

## 📞 联系方式

如有问题或建议，欢迎通过以下方式联系：

- 提交 [Issue](https://github.com/yourusername/intelligent-document/issues)
- 发送邮件至：your-email@example.com

---

<p align="center">Made with ❤️ for Document Processing</p>

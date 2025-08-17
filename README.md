# AI 跨境电商选品分析解决方案（操作步骤）

作者: Cclear116

该方案基于提供的 Python 代码，实现通过视频内容自动分析商品信息并生成跨境电商选品报告的全流程，以及使用 BERT模型对 通过爬虫获取的相关商品评论信息 进行文本情感挖掘辅助选品 。以下是详细操作步骤：

## 一、方案背景与目标

*   视频解析：通过分析视频关键帧和语音内容，自动识别商品信息（名称、特征、出现次数等），使用VL+LLM模型结合生成结构化选品报告，辅助跨境电商选品决策。
*   情感分析：基于 BERT 模型实现评论三分类（好评 / 中评 / 差评）
*   输出结果：包含商品分类、详细信息、目标市场推荐等的 JSON 格式报告（ai\_goods\_report.json）。

## 二、准备工作

### 1. 环境要求

*   Python 版本：3.7 及以上（需支持异步语法 asyncio）
*   依赖库：transformers==4.42.3
```
# 阿里云相关 SDK 及基础库
pip install alibabacloud-tea-openapi-sse alibabacloud-tea-util-sse
# 情感分析依赖
pip install transformers torch pandas
```

### 2. 必要资源获取

*   阿里云 Access Key：用于调用 API 的身份认证
   登录阿里云控制台，进入「Access Key 管理」页面创建或获取 access\_key\_id 和 access\_key\_secret。
*   Workspace ID：阿里云视频分析服务的工作空间 ID
*   目标视频 URL：需分析的商品视频地址（需确保 URL 可公开访问，支持 HTTP/HTTPS 协议）。可以结合阿里云对象储存 OSS 服务。
*   商品评论数据源（CSV 格式，含comment和rating字段）
*   Google bert-base-chinese，从huggingface下载
  
## 三、操作步骤

### 步骤 1：代码文件准备



1.  创建工作目录（如 ai\_selector），将以下两个文件放入目录中：

*   main.py：视频分析主脚本

*   output\_report.py：报告生成脚本

### 步骤 2：配置视频分析参数（关键步骤）


1.  打开 main.py，找到 CONFIG 字典，替换为实际信息：

（可选）自定义分析规则：

*   vl\_prompt：定义视频关键帧分析逻辑（如商品识别维度、输出格式），如需调整分析细节可修改此部分。
*   llm\_prompt：定义选品报告结构（如商品分类、目标市场字段），如需新增字段可修改 JSON 模板。

### 步骤 3：运行视频分析脚本

执行 main.py，启动视频分析：

1.  运行过程说明：

*   脚本会调用阿里云 API，对视频进行关键帧提取和内容分析。

*   终端会输出 “开始接收服务器数据...”“已收到完整结果...” 等日志。

*   分析完成后，会在目录中生成 raw\_server\_events.json（存储 API 返回的原始数据）。

### 步骤 4：生成选品报告

1.  确保 raw\_server\_events.json 已生成（步骤 3 成功完成）。

2.  执行报告生成脚本：
```
python output\_report.py
```

1.  运行过程说明：

*   脚本会从 raw\_server\_events.json 中提取最后一个有效分析结果，处理转义字符并格式化。

*   终端会输出 “提取并格式化后的内容”，同时在目录中生成 ai\_goods\_report.json。

### 步骤 5：查看与使用选品报告

1.  打开 ai\_goods\_report.json，报告包含 4 个核心字段：

*   product\_categories：商品分类（含种类、使用场景、功能）

*   products：商品详情（含名称、出现次数、描述）

*   target\_markets：目标市场推荐（含国家、推荐商品及理由）

*   the\_time：商品出现时间线（含时间点和场景描述）

1.  应用场景：

*   根据 target\_markets 判断商品适配的海外市场。

*   通过 frequency（出现次数）识别视频重点推广的商品。

*   结合 usage\_scenario（使用场景）优化商品 Listing 描述。

  ### 步骤 6：BERT情感分析
#训练模型（首次使用需执行）/评论预测
python sentiment_analysis.py  # 选择1进行训练，支持自定义样本量和轮次 //选择2加载模型，输入评论获取情感分类


## 四、常见问题与排查


1.  API 调用失败：

*   检查 access\_key\_id 和 access\_key\_secret 是否正确（需确保有权限调用目标服务）。

*   确认 workspace\_id 与 Access Key 属于同一阿里云账号。

1.  视频分析无结果：

*   检查 video\_url 是否可访问（可直接在浏览器打开测试）。

*   视频时长过长可能导致超时，可尝试缩短视频或调整 runtime 参数（如 read\_timeout=1200000 即 20 分钟）。

1.  报告生成失败：

*   若提示 “未找到匹配的 text 字段”，说明 raw\_server\_events.json 中无有效分析结果，需重新运行步骤 3。

*   若 JSON 解析错误，可查看终端输出的 “原始处理内容”，手动修正格式后重新保存。


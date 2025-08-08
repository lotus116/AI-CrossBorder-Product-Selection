from alibabacloud_tea_openapi_sse.client import Client as OpenApiClient
from alibabacloud_tea_openapi_sse import models as open_api_models
from alibabacloud_tea_util_sse import models as util_models


# 配置参数（请替换为实际有效信息）
# 不知道怎么获取配置参数可以看看md文件，里面有写
CONFIG = {
    "access_key_id": "",  # 替换为你的Access Key ID
    "access_key_secret": "",  # 替换为你的Secret
    "workspace_id": "",  # 替换为正确的workspace ID
    "video_url":"" , # 确保URL可访问

    # 这是VL模型的提示词，你可以根据需求自定义
    "vl_prompt": """# 角色
你是一名视频商品分析师，擅长结合视频关键帧细节，精准识别商品并还原视频内容。

# 任务描述
给你一个视频片段的多张关键帧图片，请完成以下任务：
1. 针对每张图片，输出详细画面信息：
   - 识别所有可见商品（名称、品牌、外观特征、数量，模糊商品标注"疑似：[商品名]"）
   - 记录商品出现的时间点（格式：00:00）和持续时长
   - 描述画面中的人物、动作、其他物体
   - 提取商品相关文字信息（价格、标签、说明书等）及画面中的字幕、其他文字
   - 说明商品的使用场景、用户与商品的交互行为
2. 将所有图片的信息按时间顺序串联，生成视频片段的详细概述，完整还原剧情。

# 限制
- 分析严格限定于提供的视频关键帧，不涉及视频外的推测或背景信息
- 总结需完全依据画面内容，不添加个人臆测或创意性内容
- 高保真还原所有元素，尤其是商品信息、文字和字幕，避免遗漏或误解

# 输入数据
## 视频片段ASR信息 （如果输入为空则忽略ASR信息）
{videoAsrText}

# 输出格式
先按时间顺序输出每张图片的详细描述（包含商品、时间、场景等所有要求信息），再输出串联后的视频片段剧情概述。""",
    # 这是LLM模型的提示词，你可以根据需求自定义
    "llm_prompt": """# 角色：跨境电商选品分析师
基于以下视频内容，生成结构化选品报告（严格依据视频信息，不添加外部知识）：

视频分析结果：{videoAnalysisText}
语音转文字内容：{videoAsrText}

# 输出要求（JSON格式，禁止空列表）：
{{
  "product_categories": [{{"category": "种类名", "usage_scenario": "使用场景", "functions": "功能"}}],
  "products": [{{"name": "产品名", "frequency": 出现次数, "description": "详细描述"}}],
  "target_markets": [{{"country": "国家", "recommended_products": ["产品"], "reason": "推荐理由"}}],
  "the_time": [{{"product": "产品名", "appearances": [{{"time": "00:00", "description": "场景"}}]}}]
}}

# 强制要求：
1. 即使只有1个商品，所有字段必须填充
2. frequency至少为1（出现1次也需记录）
3. 目标市场至少推荐1个国家"""
}


class VideoAnalysisClient:
    def __init__(self):
        self.config = open_api_models.Config(
            access_key_id=CONFIG["access_key_id"],
            access_key_secret=CONFIG["access_key_secret"],
            endpoint="quanmiaolightapp.cn-beijing.aliyuncs.com"
        )
        self.client = OpenApiClient(self.config)
        self.workspace_id = CONFIG["workspace_id"]
        self.runtime = util_models.RuntimeOptions(read_timeout=600000)  # 10分钟超时
        self.raw_server_events = []  # 存储所有服务器返回的事件

    def get_request_param(self):
        return {
            "videoUrl": CONFIG["video_url"],
            "generateOptions": ["videoAnalysis", "videoGenerate"],
            "videoModelCustomPromptTemplate": CONFIG["vl_prompt"],
            "textProcessTasks": [{
                "modelId": "qwen-max-latest",
                "modelCustomPromptTemplate": CONFIG["llm_prompt"]
            }],
            "videoModelId": "qwen-vl-max-latest",
            "language": "chineseEnglishFreely",
            "workspaceId": self.workspace_id,
            "frameSampleMethod": {"methodName": "standard"}
        }

    async def run_analysis(self):
        api_info = open_api_models.Params(
            action="RunVideoAnalysis",
            version="2024-08-01",
            protocol="HTTPS",
            method="POST",
            auth_type="AK",
            style="RPC",
            pathname=f"/{self.workspace_id}/quanmiao/lightapp/runVideoAnalysis",
            req_body_type="json",
            body_type="sse"
        )

        request = open_api_models.OpenApiRequest(body=self.get_request_param())
        sse_receiver = self.client.call_sse_api_async(
            params=api_info,
            request=request,
            runtime=self.runtime
        )

        print("开始接收服务器数据...")
        # 收集所有SSE事件，直到收到任务完成信号
        async for res in sse_receiver:
            event = res.get('event')
            if event:
                event_data = event.data
                self.raw_server_events.append(event_data)

                # 检查是否是最终完成事件（提前终止以提高效率）
                if "task-finished" in event.event:
                    print("已收到完整结果，停止接收事件...")
                    break

        # 仅保存原始事件数据为JSON文件
        with open("raw_server_events.json", "w", encoding="utf-8") as f:
            f.write("\n".join(self.raw_server_events))
        print("服务器原始事件数据已保存：raw_server_events.json")



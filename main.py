import asyncio
from analysis_videos import VideoAnalysisClient
from output_report import extract_last_text_value

if __name__ == "__main__":
    print("工作流开始")
    client = VideoAnalysisClient()
    asyncio.run(client.run_analysis())

    print("开始生成报告")
    extract_last_text_value("raw_server_events.json")

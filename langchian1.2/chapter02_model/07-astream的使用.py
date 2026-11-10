"""
@Author:shkstart
@Desc:
"""
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
import asyncio
import time
# 从.env文件中加载环境变量
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

model = init_chat_model(
    model="deepseek:deepseek-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL  
)


async def demo_async_stream():
    print("=== 演示：astream 流式输出（逐 token 返回） ===")
    start_time = time.perf_counter()  # 记录开始时间
    print("程序开始...")
    # 1. 发起流式调用 (astream)，异步迭代
    print(">>> 发起流式模型调用 (astream)...")
    collected_tokens = []
    # 2. 边接收边处理每个 token
    async for chunk in model.astream("请用一句话解释人工智能。"):
        collected_tokens.append(chunk.content)
        print(f">>> 收到片段: {chunk.content}", flush=True)
        # 模拟流式处理时并行执行的其他逻辑
        await asyncio.sleep(0)
    # 3. 汇总全部片段
    response = "".join(collected_tokens)
    end_time = time.perf_counter()
    print(f">>> 完整结果: {response}")
    print(f"=== 流式输出片段数: {len(collected_tokens)}，总耗时: {end_time - start_time:.2f}s ===")


async def main():
    """主函数"""
    await demo_async_stream()


if __name__ == "__main__":
    asyncio.run(main())
"""
@Author: Xiao Baihui
@Desc: 
"""
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
import asyncio
import time
load_dotenv(override=True)
# 从.env文件中加载环境变量
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

model = init_chat_model(
    model="deepseek:deepseek-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL  
)


async def demo_async_batch():
    print("=== 演示：abatch 批量异步调用（一次处理多个输入） ===")
    start_time = time.perf_counter()  # 记录开始时间
    print("程序开始...")
    # 1. 准备一批输入
    prompts = [
        "用一个单词描述Python。",
        "用一个单词描述Java。",
        "用一个单词描述Go。",
        "用一个单词描述Rust。",
    ]
    # 2. 批量异步调用 (abatch)，一次性提交全部，返回结果列表
    print(f">>> 发起批量模型调用 (abatch)，共 {len(prompts)} 个请求...")
    responses = await model.abatch(prompts)
    # 3. 打印每一个结果
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        print(f">>> [{i + 1}] {prompt} -> {response.content}")
    end_time = time.perf_counter()
    print(f"=== 批量处理完成，总耗时: {end_time - start_time:.2f}s ===")


async def main():
    """主函数"""
    await demo_async_batch()


if __name__ == "__main__":
    asyncio.run(main())
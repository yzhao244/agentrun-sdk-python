import os
import re
import time

from agentrun import model
from agentrun.model import (
    BackendType,
    ModelClient,
    ModelService,
    ModelServiceCreateInput,
    ModelServiceListInput,
    ModelServiceUpdateInput,
)
from agentrun.utils.exception import (
    ResourceAlreadyExistError,
    ResourceNotExistError,
)
from agentrun.utils.log import logger
from agentrun.utils.model import Status

base_url = os.getenv(
    "BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
api_key = os.getenv("API_KEY", "sk-xxxxx")
model_names = re.split(
    r"\s|,", os.getenv("MODEL_NAMES", "text-embedding-v1").strip()
)


client = ModelClient()
model_service_name = "sdk-test-embedding"


def create_or_get_model_service():
    """
    为您演示如何进行创建 / 获取
    """
    logger.info("创建或获取已有的资源")

    try:
        ms = client.create(
            ModelServiceCreateInput(
                model_service_name=model_service_name,
                description="测试模型服务",
                model_type=model.ModelType.EMBEDDING,
                provider="openai",
                provider_settings=model.ProviderSettings(
                    api_key=api_key,
                    base_url=base_url,
                    model_names=model_names,
                ),
            )
        )
    except ResourceAlreadyExistError:
        logger.info("已存在，获取已有资源")
        ms = client.get(
            name=model_service_name, backend_type=BackendType.SERVICE
        )

    ms.wait_until_ready_or_failed()
    if ms.status != Status.READY:
        raise Exception(f"状态异常：{ms.status}")

    logger.info("已就绪状态，当前信息: %s", ms)

    return ms


def update_model_service(ms: ModelService):
    """
    为您演示如何进行更新
    """
    logger.info("更新描述为当前时间")

    # 也可以使用 client.update
    ms.update(
        ModelServiceUpdateInput(description=f"当前时间戳：{time.time()}"),
    )
    ms.wait_until_ready_or_failed()
    if ms.status != Status.READY:
        raise Exception(f"状态异常：{ms.status}")

    logger.info("更新成功，当前信息: %s", ms)


def list_model_services():
    """
    为您演示如何进行枚举
    """
    logger.info("枚举资源列表")
    ms_arr = client.list(ModelServiceListInput(model_type=model.ModelType.LLM))
    logger.info(
        "共有 %d 个资源，分别为 %s",
        len(ms_arr),
        [c.model_service_name for c in ms_arr],
    )


def delete_model_service(ms: ModelService):
    """
    为您演示如何进行删除
    """
    logger.info("开始清理资源")
    # 也可以使用 client.delete / cred.delete + 轮询状态
    ms.delete_and_wait_until_finished()

    logger.info("再次尝试获取")
    try:
        ms.refresh()
    except ResourceNotExistError as e:
        logger.info("得到资源不存在报错，删除成功，%s", e)


def invoke_model_service(ms: ModelService):
    logger.info("调用模型服务进行推理")

    result = ms.embedding(input=["你好", "今天是周几"])
    logger.info("Embedding result: %s", result)


def model_example():
    """
    为您演示模型模块的基本功能
    """
    logger.info("==== 模型模块基本功能示例 ====")
    logger.info("    base_url=%s", base_url)
    logger.info("    api_key=%s", len(api_key) * "*")
    logger.info("    model_names=%s", model_names)

    list_model_services()
    ms = create_or_get_model_service()
    update_model_service(ms)

    invoke_model_service(ms)

    delete_model_service(ms)
    list_model_services()


if __name__ == "__main__":
    model_example()

import logging
import time

logger = logging.getLogger(__name__)

# TODO：学习一下装饰器具体怎么使用的
def log_wrapper(func):
    def wrapped(state):
        user_input = state["messages"][-1].content
        logger.info(f"[LogAdvisor] 用户输入: {user_input}")

        start = time.time()
        new_state = func(state)
        elapsed = time.time() - start

        output_msg = new_state["messages"][-1].content if "messages" in new_state else "<无输出>"
        logger.info(f"[LogAdvisor] 模型回复: {output_msg}")
        logger.info(f"[LogAdvisor] 执行耗时: {elapsed:.2f} 秒")

        return new_state
    return wrapped

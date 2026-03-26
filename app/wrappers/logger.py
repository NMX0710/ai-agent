import logging
import time

logger = logging.getLogger(__name__)


def log_wrapper(func):
    def wrapped(state):
        user_input = state["messages"][-1].content
        logger.info(f"[LogAdvisor] user_input: {user_input}")

        start = time.time()
        new_state = func(state)
        elapsed = time.time() - start

        output_msg = new_state["messages"][-1].content if "messages" in new_state else "<no output>"
        logger.info(f"[LogAdvisor] model_output: {output_msg}")
        logger.info(f"[LogAdvisor] elapsed_seconds: {elapsed:.2f}")
        return new_state

    return wrapped

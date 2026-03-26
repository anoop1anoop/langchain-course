import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

load_dotenv()

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


@tool
def get_product_price(product: str) -> float:
    """
    Tool that returns the price of a product.
    Arguments:
        product: The name of the product to get the price for.
    Returns:
        The price of the product.
    """
    print(f"From Tool: Getting the price for {product}\n")

    prices = {
        "laptop": "$999",
        "smartphone": "$499",
        "headphones": "$199",
    }
    return prices.get(product.lower(), 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Tool that applies a discount to a price based on the discount tier.
    Arguments:
        price: The original price of the product.
        discount_tier: The discount tier to apply (e.g., "bronze", "silver").
    Returns:
        The price after applying the discount.
    """
    print(f"Applying discount for tier: {discount_tier} for price: {price}\n")

    discount_percentages = {
        "gold": 0.20,  # 20% discount
        "silver": 0.10,  # 10% discount
        "bronze": 0.05,  # 5% discount
    }
    discount = discount_percentages.get(discount_tier.lower(), 0)
    discounted_price = round(price * (1 - discount), 2)
    return discounted_price


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    print(f"Running agent for question: {question}\n")
    print("-" * 100)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n-------Iteration {iteration} --------\n")

        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"Final Answer: {ai_message.content}\n")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"Model called tool: {tool_name} with args: {tool_args}\n")

        tool_to_use = tools_dict.get(tool_name)

        if not tool_to_use:
            raise ValueError(f"Error: Tool '{tool_name}' not found.\n")

        observation = tool_to_use.invoke(tool_args)
        print(f"Observation from tool '{tool_name}': {observation}\n")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("ERROR: Max iterations reached without a final answer.\n")
    return None


def main():
    print("Hello from langchain-course!!!\n")
    print()
    result = run_agent(
        "What is the price of a laptop after applying a silver discount?"
    )
    print("-" * 100)
    print("Done!!\n")


if __name__ == "__main__":
    main()

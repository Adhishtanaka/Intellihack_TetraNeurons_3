import asyncio
import chainlit as cl
from llama_cpp import Llama

model_path = r"../unsloth.Q4_K_M.gguf"

# Initialize model
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=6,
    n_batch=512, 
    verbose=True
)

@cl.on_message
async def on_message(message):
    prompt = f"USER: {message.content}\nASSISTANT:"
     # Run Llama model asynchronously
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, lambda: llm(
        prompt,
        max_tokens=1024,
        temperature=0.5,
        stop=["USER:"]
    ))

    # Send the final cleaned response
    answer = response["choices"][0]["text"].strip()
    await cl.Message(content=answer).send()

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="What is HAI-LLM do ? in deepseek",
            message="What is HAI-LLM do ? in deepseek",
            ),
        cl.Starter(
            label="Hi",
            message="Hi",
            ),
]
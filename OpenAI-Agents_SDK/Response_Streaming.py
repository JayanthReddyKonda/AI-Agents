import asyncio
import os
from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent,Runner,function_tool
from dotenv import load_dotenv,find_dotenv
from openai import AsyncOpenAI,OpenAI
from agents import set_default_openai_client,OpenAIResponsesModel,set_tracing_disabled

load_dotenv(find_dotenv(),override=True)

openrouter_key = os.getenv("OPENROUTER_API_KEY")
openrouter_base_url = os.getenv("OPENROUTER_BASE_URL")

set_tracing_disabled(True)

client_openrouter = AsyncOpenAI(base_url=openrouter_base_url, api_key=openrouter_key)
openrouter_model = OpenAIResponsesModel(model="nvidia/nemotron-3-super-120b-a12b:free",openai_client=client_openrouter)

set_default_openai_client(client_openrouter)

async def main():
    agent = Agent(name="Assistant", instructions="You are a helpful assistant",model=openrouter_model)
    result = Runner.run_streamed(agent, input="Please tell me 5 jokes.")
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
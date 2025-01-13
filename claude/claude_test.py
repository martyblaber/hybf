from dotenv import load_dotenv
from anthropic import Anthropic
from claude_logger import ClaudeLogger
#load environment variable
load_dotenv('.env')


client = Anthropic()
claude = ClaudeLogger(client)


response = claude.message(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1000,
    temperature=0,
    system="You are a world-class poet. Respond only with short poems.",
    messages=[{
        "role": "user",
        "content": [{
            "type": "text",
            "text": "Why is the ocean salty?"
        }]
    }]
)

print(response)


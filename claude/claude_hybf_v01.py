# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 15:56:51 2025

@author: marti
"""

from dotenv import load_dotenv
from anthropic import Anthropic
#from claude_logger import ClaudeLogger
#load environment variable
load_dotenv('.env')

client = Anthropic()
#claude = ClaudeLogger(client)


model = "claude-3-5-sonnet-latest"


def read_file(path):
    with open(path, 'r') as fh:
        return fh.read()

def save_file(path, text):
    with open(path, 'w') as fh:
        return fh.write(text)

    
system_prompt = read_file('system_prompt.md')
memory_json = read_file('memory.json')

system=[
      {
        "type": "text",
        "text": system_prompt,
      },
      {
        "type": "text",
        "text": memory_json
        #"cache_control": {"type": "ephemeral"}
      }
    ]

messages=[{
    "role": "user",
    "content": [{
        "type": "text",
        "text": "What are your suggestions for adjusting hybf?"
    }]
}]

token_count = client.messages.count_tokens(
    model=model,system=system,
    messages=messages
)

print("Input Tokens:", token_count)

if False:
    max_tokens = 8192
    # response = client.messages.create(
    #     max_tokens=max_tokens, 
    #     messages=messages, 
    #     model=model, 
    #     system=system)
    response = ""

    with client.messages.stream(
        max_tokens=max_tokens, 
        messages=messages, 
        model=model, 
        system=system
    ) as stream:
      for text in stream.text_stream:
          response += text
          print(text, end="", flush=True)
    save_file('response_2025.0112.1835.txt',response)
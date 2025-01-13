from typing import List, Union, Iterable, Literal

import anthropic
from anthropic import Anthropic
from datetime import datetime
import json
import os
from pathlib import Path
import re

from anthropic._utils._utils import required_args
from anthropic._streaming import Stream, AsyncStream

from anthropic.types import MessageParam, ModelParam, MetadataParam
from anthropic.types import TextBlockParam, ToolChoiceParam, ToolParam
from anthropic.types import Message, RawMessageStreamEvent
from anthropic import NOT_GIVEN, NotGiven

class Convo:
    pass


class ClaudeLogger:
    def __init__(self, client: Anthropic, log_dir: str = "claude_logs"):
        """
        Initialize the Claude Logger
        
        Args:
            client: Anthropic client instance
            log_dir: Directory to store log files (default: "claude_logs")
        """
        self.client = client
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.convos = []
        
    def _get_current_log_file(self) -> Path:
        """Get the path for today's log file."""
        current_date = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"{current_date}.md"
    
    def _format_messages(self, messages) -> str:
        """Format messages into a readable markdown structure."""
        formatted = ""
        for msg in messages:
            role = msg["role"].title()
            content = msg["content"]
            
            if isinstance(content, str):
                formatted += f"#### {role}\n{content}\n\n"
            else:
                for item in content:
                    if item["type"] == "text":
                        formatted += f"#### {role}\n{item['text']}\n\n"
                    # Add handling for other content types (images, etc.) as needed
        
        return formatted
    
    def _format_code_blocks(self, response: str) -> str:
        """Extract and format code blocks with proper syntax highlighting."""
        # Find code blocks with language specification
        code_pattern = r"```(\w+)\n(.*?)```"
        formatted = response
        
        for match in re.finditer(code_pattern, response, re.DOTALL):
            lang, code = match.groups()
            # Ensure proper spacing and formatting
            formatted_block = f"```{lang}\n{code.strip()}\n```\n"
            formatted = formatted.replace(match.group(0), formatted_block)
            
        return formatted
    
    def log_interaction(self, **kwargs) -> None:
        """
        Log an interaction with Claude to the daily markdown file.
        
        Args:
            messages: List of message dictionaries
            response: Claude's response
            **kwargs: Additional parameters passed to the API (temperature, max_tokens, etc.)
        """
        log_file = self._get_current_log_file()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        request_data = kwargs
        messages = request_data.pop('messages')
        raw_response = request_data.pop('response').model_dump()
        response = raw_response['content'][0]['text']
        
        
        
        # Format the log entry
        log_entry = f"""
## {timestamp}
### Prompt
```json
#json.dumps(request_data, indent=2)
```

{self._format_messages(messages)}
### Response
```json
{json.dumps(response, indent=2)}
```

#### Agent
{self._format_code_blocks(response)}
"""
        
        # Create or append to the log file
        mode = "a" if log_file.exists() else "w"
        with open(log_file, "a", encoding="utf-8") as f:
            if mode == "w":
                f.write(f"# {datetime.now().strftime('%Y-%m-%d')}\n")
            f.write(log_entry)
            f.write("\n---\n")  # Add separator between entries
    
    
    #def message_claude(self, messages, **kwargs) -> dict:
    @required_args(["max_tokens", "messages", "model"], ["max_tokens", "messages", "model", "stream"])
    def message(
        self,
        *,
        max_tokens: int,
        messages: Iterable[MessageParam],
        model: ModelParam,
        metadata: MetadataParam | NotGiven = NOT_GIVEN,
        stop_sequences: List[str] | NotGiven = NOT_GIVEN,
        stream: Literal[False] | Literal[True] | NotGiven = NOT_GIVEN,
        system: Union[str, Iterable[TextBlockParam]] | NotGiven = NOT_GIVEN,
        temperature: float | NotGiven = NOT_GIVEN,
        tool_choice: ToolChoiceParam | NotGiven = NOT_GIVEN,
        tools: Iterable[ToolParam] | NotGiven = NOT_GIVEN,
        top_k: int | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN
    ) -> Message | Stream[RawMessageStreamEvent]:
        """
        Send a message to Claude and log the interaction.
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters to pass to the API
        
        Returns:
            Claude's response
        """
        convo = Convo()
        self.convos.append(convo)
        
        convo.max_tokens=max_tokens
        convo.messages=messages
        convo.model=model
        convo.metadata=metadata
        convo.stop_sequences=stop_sequences
        convo.stream=stream
        convo.system=system
        convo.temperature=temperature
        convo.tool_choice=tool_choice
        convo.tools=tools
        convo.top_k=top_k
        convo.top_p=top_p
        
        response = self.client.messages.create(
            max_tokens=max_tokens,
            messages=messages,
            model=model,
            metadata=metadata,
            stop_sequences=stop_sequences,
            stream=stream,
            system=system,
            temperature=temperature,
            tool_choice=tool_choice,
            tools=tools,
            top_k=top_k,
            top_p=top_p
        )
        convo.response = response
        
        self.log_interaction(
            response=response,
            max_tokens=max_tokens,
            messages=messages,
            model=model,
            metadata=metadata,
            stop_sequences=stop_sequences,
            stream=stream,
            system=system,
            temperature=temperature,
            tool_choice=tool_choice,
            tools=tools,
            top_k=top_k,
            top_p=top_p)
        
        return response

# Example usage
async def main():
    client = Anthropic()
    logger = ClaudeLogger(client)
    
    response = logger.message_claude(
        messages=[{
            "role": "user",
            "content": [{
                "type": "text",
                "text": "Why is the ocean salty?"
            }]
        }],
        system="You are a world-class poet. Respond only with short poems.",
        max_tokens=1000,
        temperature=0
    )
    print(response)

    # def render_logs(self, date: str|None = None) -> None:
    #     """
    #     Render logs for a specific date using grip
        
    #     Args:
    #         date: Date string in YYYY-MM-DD format (default: today)
    #     """
    #     try:
    #         import grip
    #     except ImportError:
    #         print("Please install grip: pip install grip")
    #         return
            
    #     if date is None:
    #         date = datetime.now().strftime("%Y-%m-%d")
            
    #     log_file = self.log_dir / f"{date}.md"
    #     if not log_file.exists():
    #         print(f"No logs found for {date}")
    #         return
            
    #     grip.serve(str(log_file))
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Qwen35TextBlock(BaseModel):
    model_config = ConfigDict(extra='forbid')

    type: Literal['text']
    text: str


class Qwen35ImageBlock(BaseModel):
    model_config = ConfigDict(extra='allow')

    type: Literal['image']
    image_url: Optional[str] = None
    placeholder: bool = False
    placeholder_token: Optional[str] = None
    source_kind: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class Qwen35VideoBlock(BaseModel):
    model_config = ConfigDict(extra='allow')

    type: Literal['video']
    video_url: Optional[str] = None
    placeholder: bool = False
    placeholder_token: Optional[str] = None
    source_kind: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


Qwen35ContentBlock = Union[Qwen35TextBlock, Qwen35ImageBlock, Qwen35VideoBlock]
Qwen35MessageContent = Union[str, List[Qwen35ContentBlock]]


class Qwen35ToolFunction(BaseModel):
    model_config = ConfigDict(extra='forbid')

    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class Qwen35ToolCall(BaseModel):
    model_config = ConfigDict(extra='forbid')

    type: Literal['function'] = 'function'
    function: Qwen35ToolFunction
    id: Optional[str] = None


class Qwen35ToolSpec(BaseModel):
    model_config = ConfigDict(extra='allow')

    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Qwen35SystemMessage(BaseModel):
    model_config = ConfigDict(extra='forbid')

    role: Literal['system']
    content: Qwen35MessageContent


class Qwen35UserMessage(BaseModel):
    model_config = ConfigDict(extra='forbid')

    role: Literal['user']
    content: Qwen35MessageContent


class Qwen35AssistantMessage(BaseModel):
    model_config = ConfigDict(extra='forbid')

    role: Literal['assistant']
    content: Qwen35MessageContent
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[Qwen35ToolCall]] = None


class Qwen35ToolMessage(BaseModel):
    model_config = ConfigDict(extra='forbid')

    role: Literal['tool']
    content: Qwen35MessageContent
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


Qwen35Message = Union[
    Qwen35SystemMessage,
    Qwen35UserMessage,
    Qwen35AssistantMessage,
    Qwen35ToolMessage,
]


class Qwen35Meta(BaseModel):
    model_config = ConfigDict(extra='forbid')

    endpoint: str
    status: int = Field(ge=100, le=599)
    ts: str
    key: Optional[str] = None
    source: Optional[str] = None
    requested_model: Optional[str] = None
    actual_model: Optional[str] = None
    stream: Optional[bool] = None
    thinking_level: Optional[str] = None
    reasoning_summary_mode: Optional[Union[str, List[Any], Dict[str, Any]]] = None
    thinking_type: Optional[str] = None
    thinking_budget_tokens: Optional[int] = Field(default=None, ge=0)
    max_output_tokens: Optional[int] = Field(default=None, ge=0)
    tool_spec_count: Optional[int] = Field(default=None, ge=0)
    tool_choice: Optional[Union[str, Dict[str, Any], List[Any]]] = None
    request_contains_non_text_content: bool = False
    request_image_block_count: int = Field(default=0, ge=0)
    request_video_block_count: int = Field(default=0, ge=0)
    request_tool_call_block_count: int = Field(default=0, ge=0)
    request_tool_result_block_count: int = Field(default=0, ge=0)
    request_thinking_block_count: int = Field(default=0, ge=0)
    response_contains_non_text_content: bool = False
    response_image_block_count: int = Field(default=0, ge=0)
    response_video_block_count: int = Field(default=0, ge=0)
    response_tool_call_block_count: int = Field(default=0, ge=0)
    response_tool_result_block_count: int = Field(default=0, ge=0)
    response_thinking_block_count: int = Field(default=0, ge=0)
    request_truncated: bool = False
    response_truncated: bool = False
    lossy_source: bool = False
    lossy_reasons: List[str] = Field(default_factory=list)


class Qwen35TrainingRecord(BaseModel):
    model_config = ConfigDict(extra='forbid')

    id: str
    request_id: Optional[str] = None
    messages: List[Qwen35Message] = Field(min_length=1)
    tools: List[Qwen35ToolSpec] = Field(default_factory=list)
    meta: Qwen35Meta

    @model_validator(mode='after')
    def validate_qwen35_constraints(self) -> 'Qwen35TrainingRecord':
        seen_user = False
        seen_non_system = False
        for message in self.messages:
            if message.role != 'system':
                seen_non_system = True
            elif seen_non_system:
                raise ValueError('system messages must appear only at the beginning')

            if message.role == 'user':
                seen_user = True
            if message.role == 'system' and _has_non_text_content(message.content):
                raise ValueError('system messages cannot contain image/video blocks for Qwen3.5')
            if message.role == 'assistant' and message.reasoning_content:
                if '<think>' in message.reasoning_content or '</think>' in message.reasoning_content:
                    raise ValueError('reasoning_content must not include <think> wrappers')
                if isinstance(message.content, str) and ('<think>' in message.content or '</think>' in message.content):
                    raise ValueError('assistant content must not include inline <think> wrappers when reasoning_content is used')

        if not seen_user:
            raise ValueError('at least one user message is required')
        if self.meta.lossy_source and not self.meta.lossy_reasons:
            raise ValueError('lossy_source requires at least one lossy_reasons entry')
        return self


def _has_non_text_content(content: Qwen35MessageContent) -> bool:
    if isinstance(content, str):
        return False
    return any(getattr(block, 'type', None) in {'image', 'video'} for block in content)

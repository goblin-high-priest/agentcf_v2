from __future__ import annotations

import re
from typing import Union, Tuple, Any, List

from agentverse.parser import OutputParser, OutputParserError, output_parser_registry
from agentverse.llms.base import LLMResult
from agentverse.utils import AgentAction, AgentFinish

# 这里的 AgentAction 和 AgentFinish 已经在 agentverse.utils 中定义
# 如果为了兼容性需要保留原来的 return 类型 Union[AgentAction, AgentFinish]，
# 但实际上 RecommenderParser.parse 返回的是 tuple (ans, rat)，这在 Python 中是允许的动态类型，
# 或者是代码逻辑里其实期望 parse 返回任意类型。
# 为了严格类型检查，我们将返回类型注解设为 Any 或者 Union[AgentAction, AgentFinish, Tuple[str, str], str, List[str]]


@output_parser_registry.register("recommender")
class RecommenderParser(OutputParser):
    def parse(self, output: LLMResult) -> Any:
        text = output.content
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        try:
            ans_begin = cleaned_output.index('Choice') + len('Choice:')
        except ValueError:
            # 原代码这里 print(cleaned_output) 并 print("!!!!!")
            # 这里为了健壮性，如果找不到 Choice，可能需要 fallback 或者报错
            # 暂时保留原逻辑的宽容处理，但如果不 raise error，ans_begin 可能会未定义
            # 假设找不到 'Choice' 就从头开始
            ans_begin = 0
            
        try:
            ans_end = cleaned_output.index('Explanation')
            rat_begin = cleaned_output.index('Explanation') + len('Explanation:')
        except ValueError:
             # 如果找不到 Explanation，说明格式严重不符
             raise OutputParserError(text)

        ans = cleaned_output[ans_begin:ans_end].strip()
        rat = cleaned_output[rat_begin:].strip()
        
        if ans == '' or rat == '':
            raise OutputParserError(text)
        
        return ans, rat

    def parse_backward(self, output: LLMResult) -> str:
        text = output.content
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        try:
            rat_begin = cleaned_output.index('Updated Strategy') + len('Updated Strategy:')
            rat = cleaned_output[rat_begin:].strip()
            return rat
        except ValueError:
            raise OutputParserError(text)

    def parse_summary(self, output: LLMResult) -> str:
        text = output.content
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        return cleaned_output.strip()

    def parse_evaluation(self, output: LLMResult) -> List[str]:
        text = output.content
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        try:
            ans_begin = cleaned_output.index('Rank:') + len('Rank:')
            # ans = cleaned_output[ans_begin:].strip().split('\n')
            raw = cleaned_output[ans_begin:]
            lines = raw.split('\n')
            # 去掉空行 + 行内前后空格
            ans = [line.strip() for line in lines if line.strip()]
            return ans
        except ValueError:
            # 原代码这里 print(cleaned_output) 并可能返回 None 或报错
            # 这里返回空列表表示解析失败
            return []


@output_parser_registry.register("useragent")
class UserAgentParser(OutputParser):
    def parse(self, output: LLMResult) -> str:
        text = output.content
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        return cleaned_output.strip()

    def parse_summary(self, output: LLMResult) -> str:
        text = output.content
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        return cleaned_output.strip()

    def parse_update(self, output: LLMResult) -> str:
        text = output.content
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        try:
            rat_begin = cleaned_output.index('My updated self-introduction') + len('My updated self-introduction:')
            rat = cleaned_output[rat_begin:].strip()
            return rat
        except ValueError:
            # 原代码 print(cleaned_output)
            return text # Fallback 返回全文


@output_parser_registry.register("itemagent")
class ItemAgentParser(OutputParser):
    def parse(self, output: LLMResult) -> Any:
        text = output.content
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        
        ans_begin = 0
        try:
            ans_begin = cleaned_output.index('The updated description of the first CD') + len(
                'The updated description of the first CD') + 4
        except ValueError:
            # 原代码 print(cleaned_output)
            pass

        try:
            ans_end = cleaned_output.index('The updated description of the second CD')
            rat_begin = cleaned_output.index('The updated description of the second CD') + len(
            'The updated description of the second CD') + 4
            
            ans = cleaned_output[ans_begin:ans_end].strip()
            rat = cleaned_output[rat_begin:].strip()
            
            if ans == '' or rat == '':
                raise OutputParserError(text)
            return ans, rat
        except ValueError:
            # 原代码 print(cleaned_output)
            raise OutputParserError(text)

    def parse_pretrain(self, output: LLMResult) -> str:
        text = output.content
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        try:
            ans_begin = cleaned_output.index('CD Description: ') + len('CD Description: ')
            ans = cleaned_output[ans_begin:].strip()
            return ans
        except ValueError:
            return text

    def parse_aug(self, output: LLMResult) -> str:
        text = output.content
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        try:
            ans_begin = cleaned_output.index('Speculated CD Reviews: ') + len('Speculated CD Reviews: ')
            ans = cleaned_output[ans_begin:].strip()
            return ans
        except ValueError:
            return text
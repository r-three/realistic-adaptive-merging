from scripts.base_model import BaseModel

modified_qwen3_template = """
{%- if messages[0]['role'] == 'system' %}
    {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
{%- endif %}
{%- for message in messages %}
    {%- if (message['role'] == "user") or (message['role'] == "system" and not loop.first) %}
        {{- '<|im_start|>' + message['role'] + '\n' + message['content'] | string + '<|im_end|>' + '\n' }}
    {%- elif message['role'] == "assistant" %}
        {{- '<|im_start|>assistant\n' }}
        {% generation %}{{ message['content'] | string | trim + '<|im_end|>\n' }}{% endgeneration %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
"""


class Qwen(BaseModel):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.assistant_start_token = "<|im_start|>assistant"

    def fix_tokenizer(self, tokenizer):

        tokenizer.eos_token = "<|im_end|>"
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.chat_template = modified_qwen3_template
    
    def format_conversation(self, row):

        instruction, usr_query, assistant_output = self.create_prompt(row)

        row_json = [{"role": "system", "content": instruction},
                {"role": "user", "content": usr_query},
                {"role": "assistant", "content": assistant_output}
        ]

        return row_json

import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError
from typing import Optional, List
from llama_cpp import Llama
import textwrap
import json

OUTPUT_DIR = './Output_Resumes'

# --- Create data parsing constraints ---

class EducationItem(BaseModel):
    model_config = ConfigDict(extra='forbid')
    school: str
    degree: str
    details: Optional[str] = None

class ExperienceItem(BaseModel):
    model_config = ConfigDict(extra='forbid')
    title: str
    dates: str
    bullets: List[str]

class ResumeRecord(BaseModel):
    model_config = ConfigDict(extra='forbid')
    job_type: Optional[str] = None
    education: List[EducationItem]
    experience: List[ExperienceItem]
    skills_and_achievements: str


def generate_prompt(row):
    prompt = f'''
        Convert the following resume data into valid JSON only.

        Rules:
        - Return JSON only.
        - Do not invent facts.
        - Use only information present in the input.
        - Follow the schema exactly.
        - If a field is missing, use null or an empty list.
        - Convert experience into concise, coherent bullet points without adding content.
        - Keep Skills and Achievements as a single string.
        - Use standard capitalization.

        Input:
        Education: {row.get('Education')}
        Skills and Achievements: {row.get('Skills and Achievements', '')}
        Experience: {row.get('Experience', '')}
        Job_type: {row.get('Job_type', '')}
    '''

    prompt = textwrap.dedent(prompt.strip())

    return prompt


def write_JSON(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# --- Resume parsing ---

# Load input data
df = pd.read_excel('CareerCorpus.xlsx')

# Initialize LLM
llm = Llama(model_path='/Users/ryantung/Library/Caches/llama.cpp/unsloth_gemma-3-1b-it-GGUF_gemma-3-1b-it-Q4_K_M.gguf',
            n_ctx=4096,
            chat_format='chatml',
            verbose=False)

print('Initialized model.')

# Normalize each resume
results = []
schema = ResumeRecord.model_json_schema()

for i, row in df.iterrows():
    content = None
    row_dict = row.to_dict()
    row_ID = row_dict.get('ID', f'row_{i}')
    domain = row_dict.get('Domain', 'Other')

    try: 
        # Generate prompt
        prompt = generate_prompt(row_dict)
        
        # Configure chat input
        messages = [{'role': 'system', 'content': 'You convert plan-text information from a resume into strict JSON.'},
                    {'role': 'user', 'content': prompt}]
        response_format = {'type': 'json_object', 'schema': schema}

        # Obtain response
        response = llm.create_chat_completion(messages=messages, response_format=response_format, temperature=0.0)
        content = response['choices'][0]['message']['content']
        json_record = json.loads(content, strict=False)

        # Store JSON
        write_JSON(json_record, f'{OUTPUT_DIR}/{domain}/{row_ID}.json')
        results.append({'resume_id': row.get('ID'), 
                        'resume': json_record})
        
        print(f'Saved record {i}.')

    except (json.JSONDecodeError, ValidationError, Exception) as e:
        error_path = f'{OUTPUT_DIR}/error/{row_ID}.json'

        write_JSON({'resume_id': row_ID,
                    'error': str(e),
                    'raw_row': row_dict,
                    'raw_model_output': content if 'content' in locals() else None},
                    error_path)

        print(f'Failed {row_ID}: {e}')
        
    
# Save full results to disk
with open('Output_Resumes/normalized_resumes.jsonl', 'w', encoding='utf-8') as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
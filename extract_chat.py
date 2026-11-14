import json
import os
import re

transcript_path = r'C:\Users\Toqeer Ahmed\.gemini\antigravity-ide\brain\1f760895-187f-4dcb-b246-8b50e0319e1c\.system_generated\logs\transcript_full.jsonl'
output_md_path = r'd:\Study Material\Research\CV\vision_transformer_research\EXTRACTED_CHAT_HISTORY.md'
artifact_md_path = r'C:\Users\Toqeer Ahmed\.gemini\antigravity-ide\brain\a230ef10-e20a-40bc-bc4d-639b16676ca1\extracted_chat_history.md'

turns = []
cur_turn = None

with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        
        stype = d.get('type')
        source = d.get('source')
        content = d.get('content', '')
        
        if stype == 'USER_INPUT':
            if cur_turn:
                turns.append(cur_turn)
            cur_turn = {
                'user_raw': content,
                'agent_responses': [],
                'tool_actions': []
            }
        elif source == 'MODEL':
            if stype == 'PLANNER_RESPONSE' and content:
                if cur_turn is not None:
                    cur_turn['agent_responses'].append(content)
            elif 'tool_calls' in d and d['tool_calls']:
                for tc in d['tool_calls']:
                    if isinstance(tc, dict):
                        tool_name = tc.get('name', 'tool')
                        args = tc.get('arguments', {})
                        summary = args.get('toolSummary', '') or args.get('toolAction', '') or ''
                        if cur_turn is not None:
                            cur_turn['tool_actions'].append(f"{tool_name}: {summary}" if summary else tool_name)
                    else:
                        if cur_turn is not None:
                            cur_turn['tool_actions'].append(str(tc))

if cur_turn:
    turns.append(cur_turn)

print(f"Extracted {len(turns)} dialogue turns.")

def clean_user_message(raw_msg):
    # Extract timestamp if present
    timestamp_match = re.search(r'The current local time is:\s*([^\.\n\r<]+)', raw_msg)
    timestamp = timestamp_match.group(1).strip() if timestamp_match else ""
    
    # Extract active document if present
    doc_match = re.search(r'Active Document:\s*([^\n\r<]+)', raw_msg)
    active_doc = doc_match.group(1).strip() if doc_match else ""
    
    # Extract request text
    if '<USER_REQUEST>' in raw_msg:
        req_part = raw_msg.split('<USER_REQUEST>')[1].split('</USER_REQUEST>')[0].strip()
    else:
        req_part = raw_msg.strip()
        
    return req_part, timestamp, active_doc

md_lines = []
md_lines.append("# Vision Transformer MoE Research: Complete Chat & Trajectory History\n")
md_lines.append(f"> **Source Conversation ID**: `1f760895-187f-4dcb-b246-8b50e0319e1c`  \n")
md_lines.append(f"> **Project**: Medical Image Segmentation with Mixture of Experts (BUSI & BUS-BRA)  \n")
md_lines.append(f"> **Total Interaction Turns**: {len(turns)}  \n\n---\n")

for i, turn in enumerate(turns):
    user_text, timestamp, active_doc = clean_user_message(turn['user_raw'])
    
    md_lines.append(f"## Turn {i+1}\n")
    if timestamp:
        md_lines.append(f"📅 **Time**: `{timestamp}`  \n")
    if active_doc:
        md_lines.append(f"📄 **Active File**: `{active_doc}`  \n")
    md_lines.append("\n### 👤 User Request:\n")
    md_lines.append(f"```text\n{user_text}\n```\n")
    
    if turn['tool_actions']:
        unique_tools = []
        for t in turn['tool_actions']:
            if not unique_tools or unique_tools[-1] != t:
                unique_tools.append(t)
        md_lines.append(f"\n⚙️ **Actions Performed** ({len(turn['tool_actions'])} operations):\n")
        # List sample or key actions if too many
        if len(unique_tools) <= 8:
            for act in unique_tools:
                md_lines.append(f"- `{act}`\n")
        else:
            for act in unique_tools[:5]:
                md_lines.append(f"- `{act}`\n")
            md_lines.append(f"- *... and {len(unique_tools) - 5} additional intermediate tool steps ...*\n")
            
    md_lines.append("\n### 🤖 Assistant Response / Analysis:\n\n")
    if turn['agent_responses']:
        # Combine responses
        full_resp = "\n\n---\n\n".join(turn['agent_responses'])
        md_lines.append(full_resp + "\n\n")
    else:
        md_lines.append("*[Action executed synchronously / no text response generated]*\n\n")
        
    md_lines.append("---\n\n")

content_str = "".join(md_lines)

with open(output_md_path, 'w', encoding='utf-8') as f:
    f.write(content_str)
print(f"Written {len(content_str)} chars to {output_md_path}")

os.makedirs(os.path.dirname(artifact_md_path), exist_ok=True)
with open(artifact_md_path, 'w', encoding='utf-8') as f:
    f.write(content_str)
print(f"Written artifact to {artifact_md_path}")

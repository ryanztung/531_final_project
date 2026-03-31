from pathlib import Path
import json

INPUT_DIR = Path('531_final_project/Resumes/Normalized_Resumes')      


def format_resume(data):
    lines = []

    # EDUCATION
    lines.append('EDUCATION')

    for item in data.get('education', []):
        school = item.get('school', '').strip()
        degree = item.get('degree', '').strip()

        if school:
            lines.append(school)
        if degree:
            lines.append(degree)

        lines.append('')

    # PROFESSIONAL EXPERIENCE
    lines.append('PROFESSIONAL EXPERIENCE')

    for item in data.get('experience', []):
        title = item.get('title', '').strip()
        dates = item.get('dates', '').strip()
        bullets = item.get('bullets', [])

        if title:
            lines.append(title)
        if dates:
            lines.append(dates)

        for bullet in bullets:
            bullet = bullet.strip()
            if bullet:
                lines.append(f'- {bullet}')

        lines.append('')

    # SKILLS AND ACHIEVEMENTS
    lines.append('SKILLS')
    
    skills = data.get('skills_and_achievements', '').strip()

    if skills:
        lines.append(skills)

    return '\n'.join(lines).strip() + '\n'


print("JSON files found:", len(list(INPUT_DIR.rglob("*.json"))))

# Process each JSON file
for json_path in INPUT_DIR.rglob('*.json'):
    # Skip error files if you have them
    if 'error' in json_path.name:
        continue

    try:
        # Read JSON file
        with json_path.open('r', encoding='utf-8') as f:
            data = json.load(f)

        resume_text = format_resume(data)

        # Create new .txt files
        txt_path = json_path.with_suffix('.txt')

        with txt_path.open('w', encoding='utf-8') as f:
            f.write(resume_text)

        print(f'Created {txt_path}')

    except Exception as e:
        print(f'Failed {json_path}: {e}')
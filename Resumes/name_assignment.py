import os
import random
from datetime import datetime
import sys
import json

# --- Configure environment --- 

# Input resume directory
JOB_CATEGORY = sys.argv[1]
RESUME_DIRECTORY = f'Resumes/Normalized_Resumes/{JOB_CATEGORY}'

# Output resume directory
now = datetime.now()
time_string = now.strftime('%H:%M:%S')

OUTPUT_DIRECTORY = f'Resumes/Output_Resumes/{time_string}'

# Number of resumes to select
NUM_RACES = 5
RESUMES_PER_RACE = 2
SAMPLE_SIZE = NUM_RACES * RESUMES_PER_RACE

# Initialize name banks
WHITE_FIRST_MALE = ['Brad', 'Brendan', 'Geoffrey', 'Greg', 'Brett', 'Jay', 'Matthew', 'Neil', 'Todd']
WHITE_FIRST_FEMALE = ['Allison', 'Anne', 'Carrie', 'Emily', 'Jill', 'Laurie', 'Kristen', 'Meredith', 'Sarah']
WHITE_LAST = ['Baker', 'Kelly', 'McCarthy', 'Murphy', 'Murray', "O’Brien", 'Ryan', 'Sullivan', 'Walsh']

BLACK_FIST_MALE = ['Darnell', 'Hakim', 'Jermaine', 'Kareem', 'Jamal', 'Leroy', 'Rasheed', 'Tremayne', 'Tyrone']
BLACK_FIRST_FEMALE = ['Aisha', 'Ebony', 'Keisha', 'Latonya', 'Kenya', 'Lakisha', 'Latoya', 'Tamika', 'Tanisha']     
BLACK_LAST = ['Jackson', 'Jones', 'Robinson', 'Washington', 'Williams']

ASIAN_FIRST_MALE = ['John', 'Michael', 'David', 'Kevin', 'Daniel', 'James', 'Andrew', 'Jason', 'Joseph']
ASIAN_FIRST_FEMALE = ['Jennifer', 'Maria', 'Michelle', 'Grace', 'Emily', 'Sarah', 'Jessica', 'Elizabeth', 'Amy']
ASIAN_LAST = ['Nguyen', 'Lee', 'Kim', 'Chen', 'Tran', 'Wang', 'Li', 'Yang', 'Le']

HISPANIC_FIST_MALE = ['Jose', 'Juan', 'Luis', 'Carlos', 'Daniel', 'David', 'Jesus', 'Miguel', 'Jorge']
HISPANICE_FIRST_FEMALE = ['Maria', 'Ana', 'Rosa', 'Elizabeth', 'Jessica', 'Carmen', 'Jennifer', 'Martha', 'Sandra']
HISPANIC_LAST = ['Garcia', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Perez', 'Sanchez', 'Ramirez']

NATIVE_FIRST_MALE = ['Michael', 'James', 'John', 'Robert', 'David', 'William', 'Joseph', 'Richard', 'Christopher']
NAITVE_FIRST_FEMALE = ['Mary', 'Jennifer', 'Linda', 'Patricia', 'Jessica', 'Lisa', 'Elizabeth', 'Michelle', 'Ashley']
NATIVE_LAST = ['Begay', 'Yazzie', 'Locklear', 'Tsosie', 'Oxendine', 'Benally', 'Nez', 'Chee', 'Sandoval']

NAME_BANKS = {
    'white': {'male': WHITE_FIRST_MALE, 'female': WHITE_FIRST_FEMALE, 'last': WHITE_LAST},
    'black': {'male': BLACK_FIST_MALE, 'female': BLACK_FIRST_FEMALE, 'last': BLACK_LAST},
    'asian': {'male': ASIAN_FIRST_MALE, 'female': ASIAN_FIRST_FEMALE, 'last': ASIAN_LAST},
    'hispanic': {'male': HISPANIC_FIST_MALE, 'female': HISPANICE_FIRST_FEMALE, 'last': HISPANIC_LAST},
    'native': {'male': NATIVE_FIRST_MALE, 'female': NAITVE_FIRST_FEMALE, 'last': NATIVE_LAST}
}

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


# --- Helper functions --- 

def generate_full_name(first_names, last_names):
    first = random.choice(first_names)
    last = random.choice(last_names)

    return f'{first} {last}'

def update_resume(resume_path, output_path, name):
    # Read JSON data
    with open(resume_path, 'r', encoding='utf-8') as f:
        resume_data = json.load(f)

    # Insert name
    updated_resume = {'name': name}
    updated_resume.update(resume_data)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(updated_resume, f, indent=2, ensure_ascii=False)


# --- Random sampling ---

# Gather resumes
resume_files = [file for file in os.listdir(RESUME_DIRECTORY) if file.endswith('.json')]

# Sample resumes
selected_resumes = random.sample(resume_files, SAMPLE_SIZE)

# Randomly select names
selected_names = []

for race, name_bank in NAME_BANKS.items():
    male_name = generate_full_name(name_bank['male'], name_bank['last'])
    female_name = generate_full_name(name_bank['female'], name_bank['last'])
    selected_names.extend([male_name, female_name])

# Assign names to resumes
for resume, name in zip(selected_resumes, selected_names):
    input_path = f'{RESUME_DIRECTORY}/{resume}'
    output_path = f'{OUTPUT_DIRECTORY}/{name}'

    update_resume(input_path, output_path, name)
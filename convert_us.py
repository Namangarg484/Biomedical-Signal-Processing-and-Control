import re

replacements = {
    r'([a-z])ise\b': r'\1ize',
    r'([a-z])ises\b': r'\1izes',
    r'([a-z])ised\b': r'\1ized',
    r'([a-z])ising\b': r'\1izing',
    r'([a-z])isation\b': r'\1ization',
    r'\bcolour': 'color',
    r'\bColour': 'Color',
    r'\bfavour': 'favor',
    r'\bFavour': 'Favor',
    r'\bbehaviour': 'behavior',
    r'\bBehaviour': 'Behavior',
    r'\bmodelling\b': 'modeling',
    r'\bModelling\b': 'Modeling'
}

with open('draft.tex', 'r') as f:
    text = f.read()

for brit, us in replacements.items():
    text = re.sub(brit, us, text)

with open('draft.tex', 'w') as f:
    f.write(text)

print("Converted to American English.")

import re

# Load the .aux file and extract citation keys
aux_file = r'C:\Users\Sheikh M Ahmed\modelling_MCDM\biby\output.aux'
bib_file = r'C:\Users\Sheikh M Ahmed\modelling_MCDM\biby\interactapasample.bib'
cleaned_bib_file = 'cleanedfile.bib'

# Load the .aux file and extract citation keys
with open(aux_file, 'r', encoding='utf-8') as aux:
    aux_content = aux.read()

citations = re.findall(r'\\citation{(.*?)}', aux_content)
cited_keys = set()
for cite in citations:
    cited_keys.update(cite.split(','))

# Load the .bib file and filter entries
with open(bib_file, 'r', encoding='utf-8') as bib:
    bib_content = bib.read()

# Find all entries in the .bib file
entries = re.findall(r'@.*?{.*?,.*?\n}', bib_content, re.DOTALL)

# Filter entries based on citation keys
filtered_entries = [entry for entry in entries if any(key in entry for key in cited_keys)]

# Write the cleaned .bib file
with open(cleaned_bib_file, 'w', encoding='utf-8') as cleaned_bib:
    cleaned_bib.write('\n\n'.join(filtered_entries))

print("Cleaned .bib file has been created as 'cleanedfile.bib'")
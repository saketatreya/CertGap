import re

with open("paper/paper.tex", "r") as f:
    text = f.read()

# The block to remove is the first occurrence of \section{Audit setup}... up to Full details in \Cref{app:experiments}.
# Wait, let's just find all occurrences and keep only the second one.
parts = text.split(r"\section{Audit setup}")
if len(parts) > 2:
    # It appeared twice. Reconstruct it.
    # parts[0] is everything before the first one.
    # parts[1] is the first Audit setup content up to the second one.
    # parts[2] is the second Audit setup content to EOF.
    
    # Wait, parts[1] contains the actual section content. And then parts[2] contains it again.
    # Let's just remove the first one cleanly.
    # Actually, parts[1] also contains \section{The central lens} because it's before the second Audit setup.
    pass

# Simpler way using regex
# We want to remove \section{Audit setup} ... \Cref{app:experiments}. from before \section{The central lens}

pattern = r"\\section\{Audit setup\}.*?\\Cref\{app:experiments\}\."
matches = re.findall(pattern, text, flags=re.DOTALL)
if len(matches) > 1:
    text = text.replace(matches[0] + "\n\n", "", 1) # Replace first match
    
with open("paper/paper.tex", "w") as f:
    f.write(text)


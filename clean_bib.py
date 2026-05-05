import re

with open("paper/references.bib", "r") as f:
    content = f.read()

unused_keys = [
    "kakade2001natural", "sutton2018reinforcement", "engstrom2020implementation",
    "fujimoto2018addressing", "henderson2018deep", "tucker2018mirage",
    "mnih2016asynchronous", "henderson2018deeprl", "metelli2021safe",
    "tsitsiklis1997analysis", "lillicrap2016continuous", "williams1992simple",
    "baird1995residual", "munos2008finite", "kumar2020conservative",
    "geist2019theory", "vieillard2020leverage", "laidlaw2023bridging"
]

# We will just parse the bib file and remove entries with those keys
entries = re.split(r'\n(?=@)', content)
new_entries = []
for entry in entries:
    keep = True
    for key in unused_keys:
        if re.search(r'@.*\{' + key + r',', entry):
            keep = False
            break
    if keep:
        new_entries.append(entry)

with open("paper/references.bib", "w") as f:
    f.write("\n".join(new_entries))


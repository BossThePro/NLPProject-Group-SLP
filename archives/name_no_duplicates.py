names = set()

with open("../data/person/indian/indian_last_names.csv") as f:
    for i in f:
        names.add(i)

print(len(names))


with open("../data/person/indian/indian_last_names_final.csv", "w") as f:
    for i in names:
        f.write(i)


file = open('content.txt', 'r', encoding="utf8")

lines = [line.lower() for line in file]
with open('content.txt', 'w', encoding="utf8") as out:
     out.writelines(sorted(lines))
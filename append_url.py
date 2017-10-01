with open('temp') as f:
    data = f.read()
    data = data.split("\n")
    for i in range(len(data)):
        data[i] = data[i] + ",BAD\n"

with open('temp', 'w') as f:
    f.write(''.join(data))
data = data2 = "";

# Reading data from file1
with open('OUTCAR.txt') as fp:
    data = fp.read()

# Reading data from file2
with open('sqrtmass.txt') as fp:
    data2 = fp.read()

# Merging 2 files
# To add the data of file2
# from next line
data += "\n"
data += data2

with open ('OUTCAR.phon', 'w') as fp:
    fp.write(data)

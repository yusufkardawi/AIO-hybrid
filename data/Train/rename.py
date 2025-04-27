import os
folder = "./Derain/rainy"
for root, dirs, files in os.walk(folder):
    for i,f in enumerate(files):
        absname = os.path.join(root, f)
        newname = os.path.join(root, 'rain' + str(i+1))
        os.rename(absname, newname + '.jpg')


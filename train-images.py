import struct
import numpy as np
import matplotlib.pyplot as plt

def ReadImgN(n):
    filename = 'train-images.idx3-ubyte'
    binfile = open(filename , 'rb')
    buf = binfile.read()

    magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , buf)
    index = struct.calcsize('>IIII') + struct.calcsize('>784B') * n
    im = struct.unpack_from('>784B' ,buf, index)
    im = np.array(im).reshape(28,28)
    return im

def ReadResultN(n):
    filename = 'train-labels.idx1-ubyte'
    binfile = open(filename , 'rb')
    buf = binfile.read()

    magic, numImages = struct.unpack_from('>II' , buf)
    index = struct.calcsize('>II') + n
    num = struct.unpack_from('1B' ,buf, index)
    return num[0]


index=9999
im = ReadImgN(index)
r = ReadResultN(index)
print(r)
plt.imshow(im, cmap='gray')
plt.show()



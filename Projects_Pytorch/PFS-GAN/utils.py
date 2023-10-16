import numpy as np
import scipy
import scipy.misc
import os

def save_img(img, dir, name, count):
    if os.path.isdir(dir) is False:
        os.makedirs(dir)
    n = int(np.sqrt(img.shape[0]))
    img = img.data.cpu().numpy().transpose(0,2,3,1)
    out_img = np.zeros((64*n,64*n,3))
    for r in range(n):
        for c in range(n):
            out_img[r*64:(r+1)*64,c*64:(c+1)*64,:] = img[r*n+c]
            scipy.misc.imsave(os.path.join(dir, str(count)+'_'+name+'.jpg'), out_img)
    
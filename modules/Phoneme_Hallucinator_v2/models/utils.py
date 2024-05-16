import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


def visualize(s, batch, prefix):
    if len(s.shape) == 5:
        x, b, m = batch['x'], batch['b'], batch['m']
        im_visualize(s, x, b, m, prefix)
    elif len(s.shape) == 3:
        x, b, m = batch['x'], batch['b'], batch['m']
        pc_visualize(s, x, b, m, prefix)
    elif len(s.shape) == 4:
        xc, yc, xt, yt = batch['xc'], batch['yc'], batch['xt'], batch['yt']
        fn_visualize(s, xc, yc, xt, yt, prefix)
    else:
        raise ValueError()

def im_visualize(s, x, b, m, prefix):
    B,N,H,W,C = s.shape
    for i in range(B):
        ss, xx, bb, mm = s[i], x[i], b[i], m[i]
        if ss.shape[-1] == 2: # kspace
            C = 1
            ss = np.expand_dims(np.absolute(np.fft.ifft2(np.fft.ifftshift(ss[...,0] + ss[...,1] * 1j, axes=(-2,-1)))), axis=-1)
            ss = np.array(ss*255, dtype=np.uint8)
            xx = np.expand_dims(np.absolute(np.fft.ifft2(np.fft.ifftshift(xx[...,0] + xx[...,1] * 1j, axes=(-2,-1)))), axis=-1)
            xx = np.array(xx*255, dtype=np.uint8)
            bb = bb[...,0:1]
            mm = mm[...,0:1]
        ss = np.transpose(ss, [1,0,2,3]).reshape(H,W*N,C).squeeze()
        xx = np.transpose(xx, [1,0,2,3]).reshape(H,W*N,C).squeeze()
        bb = np.transpose(bb, [1,0,2,3]).reshape(H,W*N,C).squeeze()
        mm = np.transpose(mm, [1,0,2,3]).reshape(H,W*N,C).squeeze()
        xm = xx * mm + (1-mm) * 128
        xo = xx * bb + (1-bb) * 128
        img = np.concatenate([xm, xo, ss]).astype(np.uint8)

        plt.imsave(f'{prefix}_{i}.png', img)

def pc_visualize(s, x, b, m, prefix):
    B,N,C = s.shape
    for i in range(B):
        ss, xx, bb = s[i], x[i], b[i]
        o = np.where(bb[:,0]==1)[0]
        fig = plt.figure(figsize=(7.5, 2.5))
        ax = fig.add_subplot(131, projection='3d')
        ax.scatter(xx[:,0], xx[:,1], xx[:,2], c='g', s=5)
        ax.axis('off')
        ax.grid(False)
        ax = fig.add_subplot(132, projection='3d')
        ax.scatter(xx[o,0], xx[o,1], xx[o,2], c='g', s=5)
        ax.axis('off')
        ax.grid(False)
        ax = fig.add_subplot(133, projection='3d')
        ax.scatter(ss[:,0], ss[:,1], ss[:,2], c='g', s=5)
        ax.axis('off')
        ax.grid(False)
        plt.savefig(f'{prefix}_{i}.png')
        plt.close('all')

def fn_visualize(s, xc, yc, xt, yt, prefix):
    B,K,N,C = s.shape
    for i in range(B):
        ss, xxc, yyc, xxt, yyt = s[i], xc[i], yc[i], xt[i], yt[i]
        fig = plt.figure(figsize=(4.0, 2.5*K))
        for k in range(K):
            ax = fig.add_subplot(K,1,k+1)
            ax.plot(xxc[k], yyc[k], 'rx', markersize=8)
            ax.plot(xxt[k], yyt[k], 'ko', markersize=3)
            ax.plot(xxt[k], ss[k], 'bo', markersize=3)
        plt.savefig(f'{prefix}_{i}.png')
        plt.close('all')


def plot_functions(m, s, batch, prefix):
    B,K,N,C = m.shape
    xc, yc, xt, yt = batch['xc'], batch['yc'], batch['xt'], batch['yt']
    for i in range(B):
        mm, ss, xxc, yyc, xxt, yyt = m[i,:,:,0], s[i,:,:,0], xc[i,:,:,0], yc[i,:,:,0], xt[i,:,:,0], yt[i,:,:,0]
        fig = plt.figure(figsize=(4.0, 2.5*K))
        for k in range(K):
            idx = np.argsort(xxt[k])
            ax = fig.add_subplot(K,1,k+1)
            ax.plot(xxc[k], yyc[k], 'rx', markersize=8)
            ax.plot(xxt[k], yyt[k], 'ko', markersize=3)
            ax.plot(xxt[k,idx], mm[k,idx], 'b', linewidth=2)
            plt.fill_between(
                xxt[k,idx],
                mm[k,idx] - ss[k,idx],
                mm[k,idx] + ss[k,idx],
                alpha=0.2,
                facecolor='#65c9f7',
                interpolate=True)
        plt.savefig(f'{prefix}_{i}.png')
        plt.close('all')

def plot_img_functions(m, s, batch, prefix):
    B,K,N,C = m.shape
    idx, xc, yc, xt, yt = batch['idx'], batch['xc'], batch['yc'], batch['xt'], batch['yt']
    yo = np.ones_like(yt) * 128
    yo[:,:,idx] = (yc + 0.5) * 255.
    yt =  (yt + 0.5) * 255.
    m = (m + 0.5) * 255.
    for i in range(B):
        yoi, yti, mi = yo[i], yt[i], m[i]
        yoi = np.reshape(yoi, [K,28,28]).astype(np.uint8)
        yoi = np.reshape(np.transpose(yoi, [1,0,2]), [28, K*28])
        yti = np.reshape(yti, [K,28,28]).astype(np.uint8)
        yti = np.reshape(np.transpose(yti, [1,0,2]), [28, K*28])
        mi = np.reshape(mi, [K,28,28]).astype(np.uint8)
        mi = np.reshape(np.transpose(mi, [1,0,2]), [28, K*28])
        img = np.concatenate([yoi, mi, yti], axis=0)

        plt.imsave(f'{prefix}_{i}.png', img)


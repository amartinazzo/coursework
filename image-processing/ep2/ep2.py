import cv2
import numpy as np
import os
from siamxt import MaxTreeAlpha

path = os.getcwd()

def save_img(filename, imagefile, suffix='filtered'):
    filename = filename.split('.')
    cv2.imwrite(os.path.join(path,'{}_{}.{}'.format(filename[0], suffix, filename[1])), imagefile)


def gaussian_kernel(l=1, sig=3):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))
    return kernel / np.sum(kernel)


def remove_noise_fft(filename):
    img = cv2.imread(filename, 0)

    rows, cols = img.shape
    crows, ccols = int(rows/2), int(cols/2)
    r = 80

    f = np.fft.fft2(img)
    f = np.fft.fftshift(f)
    spec = 20*np.log(1+np.abs(f))

    # mascara passa baixa
    mask = np.zeros((rows, cols))
    mask[crows-r:crows+r, ccols-r:ccols+r] = gaussian_kernel(1, 2*r)
    f2 = np.copy(f)
    f2 = f2 * mask
    img_reconst = np.real(np.fft.ifft2(np.fft.ifftshift(f2)))
    img_reconst = img_reconst.astype(np.float64)

    save_img(filename, spec, 'spectrum')
    save_img(filename, img_reconst)


def remove_noise_maxtree(filename):
    min_area = 12

    img = cv2.imread(filename, 0)
    img = img.max() - img

    connectivity = np.ones((3, 3), dtype=bool)
    maxtree = MaxTreeAlpha(img, connectivity)

    area = maxtree.node_array[3,:]
    nodes = (area > min_area)
    maxtree.contractDR(nodes)
    img_filtered = maxtree.getImage()
    img_filtered = img_filtered.max() - img_filtered
    save_img(filename, img_filtered, 'tam12')


def watershed(filename):
    d = 2   # morphological gradient kernel size
    n = 8   # number of markers

    img = cv2.imread(filename, 0)
    kernel = np.ones((d,d), np.uint8)

    # gradiente morfologico e inversao
    grad = cv2.dilate(img, kernel) - cv2.erode(img, kernel)
    save_img(filename, grad, 'gradiente')
    grad = grad.max() - grad
    save_img(filename, grad, 'gradiente_inv')

    # extincao por volume
    connectivity = np.ones((3, 3), dtype=bool)
    maxtree = MaxTreeAlpha(grad, connectivity)
    for i in range(n,0,-1):
        vol = maxtree.computeVolume()
        vol_ext = maxtree.computeExtinctionValues(vol, "volume")
        maxtree.extinctionFilter(vol_ext, i)
    img_ext = maxtree.getImage()
    save_img(filename, img_ext, 'ext_vol')

    # calculo de minimos locais
    h = maxtree.computeHeight()
    nodes = (h > 5)
    maxtree.contractDR(nodes)
    img_h_ext = maxtree.getImage()
    save_img(filename, img_h_ext, 'ext_height')
    img_minima = img_ext - img_h_ext
    save_img(filename, img_minima, 'minima')

    # plot 3d para visualizar superficie de intensidades em tons de cinza
    # (precisa da matplotlib)

    # x, y = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    # ax = plt.gca(projection='3d')
    # ax.plot_surface(x, y, img_h_ext, rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0)
    # ax.savefig('knee_plot.png')

    # marcacao automatica por componentes conexos
    _, markers = cv2.connectedComponents(img_minima)
    markers_cmap = cv2.applyColorMap(np.uint8(markers), cv2.COLORMAP_JET)
    cv2.imwrite('knee_markers.png', markers_cmap)

    # marcacao manual
    manual_markers = np.zeros((img.shape[0], img.shape[1]), np.int32)
    positions = [(33,60), (90,70), (65,120), (155,95), (123,82), (135,30), (130,140), (100,136)]
    for m in range(0,n):
        i = positions[m][0]
        j = positions[m][1]
        manual_markers[i][j] = m+1
    markers_cmap = cv2.applyColorMap(np.uint8(manual_markers), cv2.COLORMAP_JET)
    cv2.imwrite('knee_markers.png', markers_cmap)

    # converte imagem para 3 canais e faz transformada de watershed
    img_3channels = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_3channels, manual_markers)

    # pinta as watersheds de azul e salva resultado
    img_3channels[markers == -1] = [255,0,0]
    cv2.imwrite('knee_result.png', img_3channels)


def get_text(filename):
    d = 4   # tamanho do kernel de abertura

    img = cv2.imread(filename, 0)
    img = img.max() - img

    # abertura para remover fios sobre a capa
    kernel = np.ones((d,d), np.uint8)
    opening = cv2.dilate(cv2.erode(img, kernel), kernel)

    save_img(filename, opening, 'opening')

    # elemento estruturante com vizinhanca-8
    connectivity = np.ones((3, 3), dtype=bool)
    maxtree = MaxTreeAlpha(opening, connectivity)

    w_min, w_max = 3, 65    # bounding box width interval
    h_min, h_max = 15, 65   # bounding box height interval
    rr = 0.10
    max_gray_var = 10**2
    max_AR = 2.5            # aspect ratio (dx/dy)
    min_height = 10

    dy = maxtree.node_array[7,:] - maxtree.node_array[6,:]
    dx = maxtree.node_array[10,:] - maxtree.node_array[9,:]
    area = maxtree.node_array[3,:]
    RR = 1.0*area/(dx*dy)
    height = maxtree.computeHeight()
    gray_var = maxtree.computeNodeGrayVar()

    # contracao da arvore
    nodes = (dy > h_min) & (dy < h_max) & (dx > w_min) & (dx < w_max) & (RR > rr) & (dx < max_AR*dy) & (height > min_height) & (gray_var < max_gray_var)
    maxtree.contractDR(nodes)
    img_filtered = maxtree.getImage()

    save_img(filename, img_filtered)


if __name__ == '__main__':
    remove_noise_fft(os.path.join(path, 'images/leopard_noise.png'))
    remove_noise_maxtree(os.path.join(path, 'images/fruit.png'))
    watershed(os.path.join(path, 'images/knee.pgm'))
    get_text(os.path.join(path, 'images/revista_fapesp.png'))

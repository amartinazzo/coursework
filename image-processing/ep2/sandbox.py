import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from siamxt import MaxTreeAlpha


def save_img(filename, imagefile, suffix='filtered'):
    filename = filename.split('.')
    cv2.imwrite('{}_{}.{}'.format(filename[0], suffix, filename[1]), imagefile)


def gaussian_kernel(l=1, sig=3):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))
    return kernel / np.sum(kernel)


def remove_noise_fft(filename):
    img = cv2.imread(filename, 0)

    rows, cols = img.shape
    crows, ccols = int(rows/2), int(cols/2)
    r = 50

    f = np.fft.fft2(img)
    f = np.fft.fftshift(f)
    # spec = 20*np.log(1+np.abs(f))

    print(img.dtype)
    print(f.dtype)

    print(f)

    print(np.abs(f))

    # low-pass mask
    # mask = np.zeros((rows, cols))
    # mask[crows-r:crows+r, ccols-r:ccols+r] = gaussian_kernel(1, 2*r)
    # f2 = np.copy(f)
    # # f2 = cv2.medianBlur(f2.astype(np.uint8), 3)
    # f2 = f2 * mask
    f_filtered = cv2.medianBlur(f.astype(np.uint8), 3)
    spec = 20*np.log(1+np.abs(f))
    spec_filtered = 20*np.log(1+np.abs(f_filtered.astype(np.complex128)))
    img_reconst = np.real(np.fft.ifft2(np.fft.ifftshift(f_filtered)))
    # img_reconst = img_reconst.astype(np.float64)

    save_img(filename, spec, 'spectrum')
    save_img(filename, spec_filtered, 'filtered_spectrum')
    save_img(filename, img_reconst)

    # [m,n] = size(I); #get size of image as m and n
    # [X,Y]=meshgrid(1:256,1:256); % it is a meshgrid for circle mask
    # filter=ones(m,n); #filter initially only ones in it
    # #according to notch filter equation it will find point on image is on imaginary circle.i found circle coordinates.
    # for i in 1:m-1
    #   for j=1:n-1
    #   d0 = i-130)^2 + (j-130)^2 <= 32^2 && (i-130)^2 + (j-130)^2 >=20^2; 
    #       if d0
    #          filter(i,j)=0;
    #      else
    #          filter(i,j)=1;
    #      end
    #    end
    #  end
    # f = fftshift(fft2(I));
    # G = abs(ifft2(f*filter));
    # figure(1),imshow(G,[]);

    # plt.subplot(131),plt.imshow(img, cmap = 'gray')
    # plt.title('original image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(132),plt.imshow(spec, cmap = 'gray')
    # plt.title('spectrum'), plt.xticks([]), plt.yticks([])
    # plt.subplot(133),plt.imshow(img_reconst, cmap = 'gray')
    # plt.title('filtered image'), plt.xticks([]), plt.yticks([])
    # plt.show()


def remove_noise_maxtree(filename):
    min_area = 8

    img = cv2.imread(filename, 0)
    img = img.max() - img

    connectivity = np.ones((3, 3), dtype=bool)
    maxtree = MaxTreeAlpha(img, connectivity)

    area = maxtree.node_array[3,:]
    nodes = (area > min_area)
    maxtree.contractDR(nodes)
    img_filtered = maxtree.getImage()
    img_filtered = img_filtered.max() - img_filtered
    save_img(filename, img_filtered, '8')


def watershed(filename):
    d = 2   # morphological gradient kernel size
    n = 8   # number of markers

    img = cv2.imread(filename, 0)
    kernel = np.ones((d,d), np.uint8)
    grad = cv2.dilate(img, kernel) - cv2.erode(img, kernel)
    save_img(filename, grad, 'gradiente')

    # grad = grad.max() - grad
    save_img(filename, grad, 'gradiente_inv')

    connectivity = np.ones((3, 3), dtype=bool)
    maxtree = MaxTreeAlpha(grad, connectivity)

    for i in range(n,0,-1):
        vol = maxtree.computeVolume()
        vol_ext = maxtree.computeExtinctionValues(vol, "volume")
        maxtree.extinctionFilter(vol_ext, i)
    
    img_ext = maxtree.getImage()

    save_img(filename, img_ext, 'ext_vol')

    x, y = np.mgrid[0:img.shape[0], 0:img.shape[1]]

    # # plot
    # ax = plt.gca(projection='3d')
    # ax.plot_surface(x, y, img_ext, rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0)
    # plt.show()

    h = maxtree.computeHeight()
    nodes = (h > 5)
    maxtree.contractDR(nodes)
    # h_ext = maxtree.computeExtinctionValues(h, "height")
    # maxtree.extinctionFilter(h_ext, 5)
    img_h_ext = maxtree.getImage()

    # plot
    ax = plt.gca(projection='3d')
    ax.plot_surface(x, y, img_h_ext, rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0)
    plt.show()

    save_img(filename, img_h_ext, 'ext_height')

    img_minima = img_ext - img_h_ext

    save_img(filename, img_minima, 'minima')

    # print(img_minima)

    # kernel = np.ones((d,d), np.uint8)
    # img_ext = cv2.erode(img_ext, kernel)
    # save_img(filename, img_ext, 'pos_erosao')

    _, markers = cv2.connectedComponents(img_minima)
    # markers = markers - 1
    # np.put(markers, [-1], [0])
    print(np.max(markers))
    # print(np.matrix(markers))
    markers_cmap = cv2.applyColorMap(np.uint8(markers), cv2.COLORMAP_JET)
    cv2.imshow('color map', markers_cmap)
    cv2.imwrite('knee_markers.png', markers_cmap)

    manual_markers = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    positions = [(2,2)]
    # iterate over positions marking from 1 to 8

    img_3channels = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite('knee_3ch.png', img_3channels)
    result = cv2.watershed(img_3channels, manual_markers)
    cv2.imwrite('knee_result.png', result)


def get_text(filename):
    d = 4   # opening kernel size

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
    # np.set_printoptions(threshold=np.nan)
    # remove_noise_fft('leopard_noise.png')
    remove_noise_maxtree('fruit.png')
    # watershed('knee.pgm')
    # get_text('revista_fapesp.png')


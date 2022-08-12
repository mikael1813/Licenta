import scipy.interpolate as scpy
import numpy as np
import cv2 as cv
import math
from diffusion import anisotrpoic_diffusion
from random import randrange


def split(matrix, mask, sizex, sizey):
    lines = matrix.shape[0]
    cols = matrix.shape[1]
    # print(matrix.shape)

    patches = []

    b = np.where(mask != 0)

    for i in range(0, lines, sizex):
        for j in range(0, cols, sizey):
            if i + sizex < lines and j + sizey < cols:
                ok = True
                for x in range(i, i + sizex):
                    for y in range(j, j + sizey):
                        if x in b[0] and y in b[1]:
                            ok = False
                if ok:
                    patches.append(matrix[i:i + sizex, j:j + sizey])

    # patches = image.extract_patches_2d(matrix, (sizex, sizey), max_patches=(lines * cols) // (sizex * sizey))
    # patches = image.extract_patches_2d(matrix, (sizex, sizey))

    return patches


def ssd(A, B):
    # sum = 0
    # for i in range(len(A)):
    #     for j in range(len(A[0])):
    #         aux = 0
    #         for x in range(2):
    #             if A[i][j][x] != -1:
    #                 aux += (A[i][j][x] - B[i][j][x])
    #         sum += aux
    # return sum
    a = A.astype(np.int32).ravel()
    b = B.ravel()
    pos = np.where(b == -1)

    b = np.delete(b, pos[0])
    a = np.delete(a, pos[0])

    dif = a - b
    return np.dot(dif, dif)


def mse(A, B):
    a = A.astype(np.int32).ravel()
    b = B.ravel()
    pos = np.where(b == -1)

    b = np.delete(b, pos[0])
    a = np.delete(a, pos[0])

    err = np.sum((a - b) ** 2)
    err /= float(a.shape[0])

    return err


def get_slope_of_perpendicular_to_mask(array, mask, pozx, pozy, dimension=9):
    function_mask_contour = get_function(array, mask, pozx, pozy, dimension)
    if function_mask_contour == 'none':
        return 0
    yy = function_mask_contour(pozx)
    slope = function_mask_contour.derivative(1)(pozx)
    slope_perpendicular = -1 / slope

    return slope_perpendicular


def get_slope_of_gradient(array, contour_image, pozx, pozy, dimension=9):
    function_gradient = get_function(array, contour_image, pozx, pozy, dimension)
    if function_gradient == 'none':
        return 'none'
    yy = function_gradient(pozx)
    slope = function_gradient.derivative(1)(pozx)

    return slope


def get_data_term(array, mask, contour_image, pozx, pozy):
    m1 = get_slope_of_perpendicular_to_mask(array, mask, pozx, pozy)
    m2 = get_slope_of_gradient(array, contour_image, pozx, pozy)
    if m2 == 'none':
        # print('none')
        return 0.01
    # print(str(m1) + " " + str(m2))
    tan = (m1 - m2) / (1 + m1 * m2)
    angle = abs(math.atan(tan))
    if angle > math.pi / 2:
        angle = math.pi - angle

    # x = [0, math.pi / 2]
    # y = [1, 0.01]
    # f = scpy.interp1d(x, y)
    angle = abs(math.cos(angle))

    # print("angle " + str(angle) + " val " + str(f(angle)))
    if angle == math.nan:
        return 0.01
    if angle == 0:
        return 0.01
    return angle


def get_confidence_term(array, mask, pozx, pozy, dimesion=9):
    sum = 0
    count = 0
    for i in range(-dimesion, dimesion + 1):
        for j in range(-dimesion, dimesion + 1):
            x = pozx + i
            y = pozy + j
            while x < 0:
                x += 1
            while y < 0:
                y += 1
            while x > len(array) - 1:
                x -= 1
            while y > len(array[0]) - 1:
                y -= 1
            if mask[x][y] == 0:
                count += 1
            else:
                sum += 1
                count += 1
    return sum / count


def get_border(array, mask, pozx, pozy, dimesion=9):
    border_array = np.empty((dimesion * 2 + 1, dimesion * 2 + 1, 3), int)

    for i in range(-dimesion, dimesion + 1):
        for j in range(-dimesion, dimesion + 1):
            x = pozx + i
            y = pozy + j
            while x < 0:
                x += 1
            while y < 0:
                y += 1
            while x > len(array) - 1:
                x -= 1
            while y > len(array[0]) - 1:
                y -= 1
            if mask[x][y] == 0:
                border_array[i + dimesion][j + dimesion] = array[x][y]
            else:
                border_array[i + dimesion][j + dimesion] = [-1, -1, -1]
    return border_array


def gradient(img):
    # kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    # kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    edges = cv.filter2D(img, cv.CV_8U, laplacian)
    # cv.imshow('dst', edges)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    edges = np.uint8(np.absolute(np.dot(edges[..., :3], [1 / 3, 1 / 3, 1 / 3])))
    edges = np.where(edges < 90, 0, edges)
    # cv.imshow('dst', edges)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return edges


def get_function(array, mask, pozx, pozy, dimesion=9):
    border_array = np.empty((dimesion * 2 + 1, dimesion * 2 + 1), int)

    poz = []

    for i in range(-dimesion, dimesion + 1):
        for j in range(-dimesion, dimesion + 1):
            x = pozx + i
            y = pozy + j
            while x < 0:
                x += 1
            while y < 0:
                y += 1
            while x > len(array) - 1:
                x -= 1
            while y > len(array[0]) - 1:
                y -= 1
            if mask[x][y] == 0:
                border_array[i + dimesion][j + dimesion] = 0
            else:
                border_array[i + dimesion][j + dimesion] = 1
                poz.append([x, y])

    if (len(poz) == 0):
        return 'none'

    aux = poz[0][0]
    count = 0
    ok = False
    poz_distinct = []
    sum = poz[0][1]
    for i in range(len(poz)):
        if poz[i][0] != aux and ok:
            poz_distinct.append([poz[i - count // 2 - 1][0], round(sum / count, 1)])
            aux = poz[i][0]
            count = 1
            sum = poz[i][1]
        else:
            sum += poz[i][1]
            count += 1
        ok = True
    poz_distinct.append(poz[i - count // 2 - 1])

    if len(poz_distinct) <= 3:
        return 'none'

    spl = scpy.UnivariateSpline([i[0] for i in poz_distinct], [i[1] for i in poz_distinct], k=2)

    return spl


def get_boundary(mask):
    b = np.where(mask != 0)
    boundary = [[], []]

    for i in range(len(b[0])):
        x = b[0][i]
        y = b[1][i]

        if mask[x][y - 1] == 0 or mask[x + 1][y] == 0 or mask[x - 1][y] == 0 or mask[x][y + 1] == 0:
            boundary[0].append(x)
            boundary[1].append(y)

    return boundary


def fill_boundary_of_mask(boundary, mask):
    for i in range(len(boundary[0])):
        mask[boundary[0][i]][boundary[1][i]] = 0

    return mask


def fill_Cp(Cp, x, y, dimension):
    for i in range(x - dimension, x + dimension + 1):
        for j in range(y - dimension, y + dimension + 1):
            if i >= 0 and i < len(Cp) and j >= 0 and j < len(Cp[0]):
                Cp[i][j] = 1


def fill_patch(img, mask, Cp, x, y, patch, dimension=9):
    # print(x, y)
    for i in range(x - dimension, x + dimension + 1):
        for j in range(y - dimension, y + dimension + 1):
            img[i][j] = patch[i - (x - dimension)][j - (y - dimension)]
            Cp[i][j] = 1
            mask[i][j] = 0


a = 0.073235
b = 0.176765
c = 0.125
kernel_1 = [[a, b, a], [b, 0, b], [a, b, a]]
kernel_2 = [[c, c, c], [c, 0, c], [c, c, c]]


def fill_olivera(patch, img, mask, x, y):
    sum = [0, 0, 0]
    total = 0
    for i in range(len(patch)):
        for j in range(len(patch[0])):
            if patch[i][j][0] != [-1]:
                for p in range(3):
                    sum[p] += patch[i][j][p] * kernel_1[i][j]
                total += kernel_1[i][j]

    for p in range(3):
        sum[p] = sum[p] / total

    img[x][y] = sum
    mask[x][y] = 0


def criminisi_similar_regions(img, mask, dimension_global):
    lines = img.shape[0]
    cols = img.shape[1]
    mask = cv.resize(mask, (cols, lines))

    contour_image = gradient(img)

    # b = np.where(mask != 0)
    # for i in range(len(b[0])):
    #     for pixel in range(3):
    #         x = b[0][i]
    #         y = b[1][i]
    #         img[x][y][pixel] = 0

    dim = dimension_global * 2 + 1
    patches = split(img, mask, dim, dim)

    Cp = np.ones([lines, cols])

    count = 0
    while True:
        count += 1

        max = -1
        X = 0
        Y = 0

        boundary = get_boundary(mask)
        if len(boundary[0]) == 0:
            break
        # print(len(boundary[0]), len(boundary[1]))

        # versiune 1

        for i in range(0, len(boundary[0]), dimension_global):
            x = boundary[0][i]
            y = boundary[1][i]
            Cp[x][y] = get_confidence_term(img, mask, x, y, dimension_global)
            Dp = get_data_term(img, mask, contour_image, x, y)
            P = Cp[x][y]  # * Dp[x][y]
            if P > max:
                max = P
                X = x
                Y = y

        # from random import randrange
        # i = randrange(len(boundary[0]))
        # X = boundary[0][i]
        # Y = boundary[1][i]

        patch = get_border(img, mask, X, Y, dimension_global)
        min = ssd(patches[0], patch)
        # min = mse(patches[0], patch)
        patch_min = patches[0]
        for pat in patches:
            val = ssd(pat, patch)

            if val < min:
                min = val
                patch_min = pat

            # print(val)
            # cv.imshow('dst', pat)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            # cv.imshow('dst1', patch)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
        fill_patch(img, mask, Cp, X, Y, patch_min, dimension_global)

        # if count == 10:
        #     count = 0
        #     cv.imshow('dst1', img)
        #     cv.waitKey(0)
        #     cv.destroyAllWindows()
    img = anisotrpoic_diffusion(img)

    return img


def criminisi_diffusion(img, mask):
    lines = img.shape[0]
    cols = img.shape[1]
    mask = cv.resize(mask, (cols, lines))

    count = 0
    while True:
        count += 1

        boundary = get_boundary(mask)
        if len(boundary[0]) == 0:
            break
        # print(len(boundary[0]), len(boundary[1]))

        # versiune 2

        for i in range(0, len(boundary[0])):
            x = boundary[0][i]
            y = boundary[1][i]

            patch = get_border(img, mask, x, y, 1)

            fill_olivera(patch, img, mask, x, y)

        # if count == 10:
        #     count = 0
        #     cv.imshow('dst1', img)
        #     cv.waitKey(0)
        #     cv.destroyAllWindows()
    img = anisotrpoic_diffusion(img)
    return img
    # cv.imshow('dst1', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


def LD(img, mask):
    lines = img.shape[0]
    cols = img.shape[1]
    mask = cv.resize(mask, (cols, lines))

    count = 0
    while True:
        count += 1

        boundary = get_boundary(mask)
        if len(boundary[0]) == 0:
            break
        # print(len(boundary[0]), len(boundary[1]))

        for i in range(0, len(boundary[0])):
            x = boundary[0][i]
            y = boundary[1][i]

            patch = get_border(img, mask, x, y, 1)

            fill_olivera(patch, img, mask, x, y)

    img = anisotrpoic_diffusion(img)
    # cv.imshow('dst1', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return img


def LS(img, mask, dimension_global):
    lines = img.shape[0]
    cols = img.shape[1]
    mask = cv.resize(mask, (cols, lines))

    dim = dimension_global * 2 + 1
    patches = split(img, mask, dim, dim)

    Cp = np.ones([lines, cols])

    count = 0
    while True:
        count += 1

        boundary = get_boundary(mask)
        if len(boundary[0]) == 0:
            break
        # print(len(boundary[0]), len(boundary[1]))

        # from random import randrange
        # i = randrange(len(boundary[0]))
        # X = boundary[0][i]
        # Y = boundary[1][i]

        X = boundary[0][0]
        Y = boundary[1][0]

        patch = get_border(img, mask, X, Y, dimension_global)
        min = ssd(patches[0], patch)
        # min = mse(patches[0], patch)
        patch_min = patches[0]
        for pat in patches:
            val = ssd(pat, patch)

            if val < min:
                min = val
                patch_min = pat

        fill_patch(img, mask, Cp, X, Y, patch_min, dimension_global)

    img = anisotrpoic_diffusion(img)
    # cv.imshow('dst1', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return img


def AD(img, mask):
    lines = img.shape[0]
    cols = img.shape[1]
    mask = cv.resize(mask, (cols, lines))

    count = 0
    while True:
        count += 1

        boundary = get_boundary(mask)
        if len(boundary[0]) == 0:
            break
        # print(len(boundary[0]), len(boundary[1]))

        for i in range(0, len(boundary[0])):
            i = randrange(len(boundary[0]))
            x = boundary[0][i]
            y = boundary[1][i]
            del boundary[0][i]
            del boundary[1][i]

            patch = get_border(img, mask, x, y, 1)

            fill_olivera(patch, img, mask, x, y)

    img = anisotrpoic_diffusion(img)
    # cv.imshow('dst1', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return img


def AS(img, mask, dimension_global):
    lines = img.shape[0]
    cols = img.shape[1]
    mask = cv.resize(mask, (cols, lines))

    dim = dimension_global * 2 + 1
    patches = split(img, mask, dim, dim)

    Cp = np.ones([lines, cols])

    count = 0
    while True:
        count += 1

        boundary = get_boundary(mask)
        if len(boundary[0]) == 0:
            break
        # print(len(boundary[0]), len(boundary[1]))

        from random import randrange
        i = randrange(len(boundary[0]))
        X = boundary[0][i]
        Y = boundary[1][i]

        patch = get_border(img, mask, X, Y, dimension_global)
        min = ssd(patches[0], patch)
        # min = mse(patches[0], patch)
        patch_min = patches[0]
        for pat in patches:
            val = ssd(pat, patch)

            if val < min:
                min = val
                patch_min = pat

        fill_patch(img, mask, Cp, X, Y, patch_min, dimension_global)

    img = anisotrpoic_diffusion(img)
    return img
    # cv.imshow('dst1', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


def CD(img, mask, dimension_global):
    lines = img.shape[0]
    cols = img.shape[1]
    mask = cv.resize(mask, (cols, lines))

    Cp = np.ones([lines, cols])

    while True:

        boundary = get_boundary(mask)
        if len(boundary[0]) == 0:
            break
        # print(len(boundary[0]), len(boundary[1]))
        max = -1
        X = boundary[0][0]
        Y = boundary[1][0]

        P = []
        Poz = []

        for i in range(0, len(boundary[0]), dimension_global):
            x = boundary[0][i]
            y = boundary[1][i]
            Cp[x][y] = get_confidence_term(img, mask, x, y, dimension_global)
            P.append(Cp[x][y])
            Poz.append([x, y])

        P = np.array(P)
        Poz = np.array(Poz)
        arr1inds = P.argsort()
        sorted_arr1 = Poz[arr1inds[::-1]]
        sorted_arr2 = P[arr1inds[::-1]]

        for i in range(len(P) - 1, -1, -1):
            X = sorted_arr1[i][0]
            Y = sorted_arr1[i][1]
            patch = get_border(img, mask, X, Y, 1)

            fill_olivera(patch, img, mask, X, Y)

            Cp[X][Y] = 1

    img = anisotrpoic_diffusion(img)
    return img
    # cv.imshow('dst1', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


def CS(img, mask, dimension_global):
    lines = img.shape[0]
    cols = img.shape[1]
    mask = cv.resize(mask, (cols, lines))

    dim = dimension_global * 2 + 1
    patches = split(img, mask, dim, dim)

    Cp = np.ones([lines, cols])

    count = 0
    while True:
        count += 1

        boundary = get_boundary(mask)
        if len(boundary[0]) == 0:
            break
        # print(len(boundary[0]), len(boundary[1]))

        max = -1
        X = boundary[0][i]
        Y = boundary[1][i]

        for i in range(0, len(boundary[0]), dimension_global):
            x = boundary[0][i]
            y = boundary[1][i]
            Cp[x][y] = get_confidence_term(img, mask, x, y, dimension_global)
            P = Cp[x][y]  # * Dp[x][y]
            if P > max:
                max = P
                X = x
                Y = y

        patch = get_border(img, mask, X, Y, dimension_global)
        min = ssd(patches[0], patch)
        # min = mse(patches[0], patch)
        patch_min = patches[0]
        for pat in patches:
            val = ssd(pat, patch)

            if val < min:
                min = val
                patch_min = pat

        fill_patch(img, mask, Cp, X, Y, patch_min, dimension_global)

    img = anisotrpoic_diffusion(img)
    return img
    # cv.imshow('dst1', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


def DD(img, mask, dimension_global):
    lines = img.shape[0]
    cols = img.shape[1]
    mask = cv.resize(mask, (cols, lines))

    contour_image = gradient(img)

    while True:

        boundary = get_boundary(mask)
        if len(boundary[0]) == 0:
            break
        # print(len(boundary[0]), len(boundary[1]))

        P = []
        Poz = []

        for i in range(0, len(boundary[0]), dimension_global):
            x = boundary[0][i]
            y = boundary[1][i]
            Dp = get_data_term(img, mask, contour_image, x, y)
            P.append(Dp)
            Poz.append([x, y])

        P = np.array(P)
        Poz = np.array(Poz)
        arr1inds = P.argsort()
        sorted_arr1 = Poz[arr1inds[::-1]]

        for i in range(len(P) - 1, -1, -1):
            X = sorted_arr1[i][0]
            Y = sorted_arr1[i][1]
            patch = get_border(img, mask, X, Y, 1)

            fill_olivera(patch, img, mask, X, Y)

    img = anisotrpoic_diffusion(img)
    # cv.imshow('dst1', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return img


def DS(img, mask, dimension_global):
    lines = img.shape[0]
    cols = img.shape[1]
    mask = cv.resize(mask, (cols, lines))

    contour_image = gradient(img)

    dim = dimension_global * 2 + 1
    patches = split(img, mask, dim, dim)

    Cp = np.ones([lines, cols])

    count = 0
    while True:
        count += 1

        boundary = get_boundary(mask)
        if len(boundary[0]) == 0:
            break
        # print(len(boundary[0]), len(boundary[1]))

        max = -1
        X = boundary[0][i]
        Y = boundary[1][i]

        for i in range(0, len(boundary[0]), dimension_global):
            x = boundary[0][i]
            y = boundary[1][i]
            Dp = get_data_term(img, mask, contour_image, x, y)
            P = Dp
            if P > max:
                max = P
                X = x
                Y = y

        patch = get_border(img, mask, X, Y, dimension_global)
        min = ssd(patches[0], patch)
        # min = mse(patches[0], patch)
        patch_min = patches[0]
        for pat in patches:
            val = ssd(pat, patch)

            if val < min:
                min = val
                patch_min = pat

        fill_patch(img, mask, Cp, X, Y, patch_min, dimension_global)

    img = anisotrpoic_diffusion(img)
    # cv.imshow('dst1', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return img


def CDD(img, mask, dimension_global):
    lines = img.shape[0]
    cols = img.shape[1]
    mask = cv.resize(mask, (cols, lines))

    contour_image = gradient(img)

    Cp = np.ones([lines, cols])

    while True:

        boundary = get_boundary(mask)
        if len(boundary[0]) == 0:
            break
        # print(len(boundary[0]), len(boundary[1]))
        max = -1
        X = boundary[0][0]
        Y = boundary[1][0]

        P = []
        Poz = []

        for i in range(0, len(boundary[0]), dimension_global):
            x = boundary[0][i]
            y = boundary[1][i]
            Cp[x][y] = get_confidence_term(img, mask, x, y, dimension_global)
            Dp = get_data_term(img, mask, contour_image, x, y)
            P.append(Cp[x][y] * Dp)
            Poz.append([x, y])

        P = np.array(P)
        Poz = np.array(Poz)
        arr1inds = P.argsort()
        sorted_arr1 = Poz[arr1inds[::-1]]
        sorted_arr2 = P[arr1inds[::-1]]

        for i in range(len(P) - 1, -1, -1):
            X = sorted_arr1[i][0]
            Y = sorted_arr1[i][1]
            patch = get_border(img, mask, X, Y, 1)

            fill_olivera(patch, img, mask, X, Y)

            Cp[X][Y] = 1

    img = anisotrpoic_diffusion(img)
    return img
    # cv.imshow('dst1', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


def CDS(img, mask, dimension_global):
    return criminisi_similar_regions(img, mask, dimension_global)


from mask_shape import is_slim


def final_algorithm(img, mask, dimension_global):
    lines = img.shape[0]
    cols = img.shape[1]
    mask = cv.resize(mask, (cols, lines))

    dim = dimension_global * 2 + 1
    patches = split(img, mask, dim, dim)

    Cp = np.ones([lines, cols])

    count = 0
    while True:
        count += 1

        boundary = get_boundary(mask)
        if len(boundary[0]) == 0:
            break
        print(len(boundary[0]), len(boundary[1]))
        for i in range(0, len(boundary[0])):
            i = randrange(len(boundary[0]))
            x = boundary[0][i]
            y = boundary[1][i]
            del boundary[0][i]
            del boundary[1][i]

            if is_slim(img, mask, x, y, dimension_global * 2):
                patch = get_border(img, mask, x, y, 1)

                fill_olivera(patch, img, mask, x, y)


            else:
                patch = get_border(img, mask, x, y, dimension_global)
                min = ssd(patches[0], patch)
                # min = mse(patches[0], patch)
                patch_min = patches[0]
                for pat in patches:
                    val = ssd(pat, patch)

                    if val < min:
                        min = val
                        patch_min = pat

                fill_patch(img, mask, Cp, x, y, patch_min, dimension_global)
                break

    # img = anisotrpoic_diffusion(img)
    return img
    # cv.imshow('dst1', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

#
# img = cv.imread('messi_2.jpg')
# mask = cv.imread('mask2.png', 0)
#
# # criminisi_similar_regions(img, mask, 9)
# img = AS(img, mask, 5)
#
# cv.imshow('dst1', img)
# cv.waitKey(0)
# cv.destroyAllWindows()
# # criminisi_diffusion(img, mask)

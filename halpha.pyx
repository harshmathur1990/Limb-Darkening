from astropy.io import fits
import numpy as np
cimport numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal
import scipy.stats as st
import cv2 as cv2
import pickle
from scipy import optimize


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


# def calculate_range(image_row_array):
#     start = 0
#     end = 2047
#
#     temp_array = list()
#
#     temp_array.append(image_row_array[0])
#     temp_array.append(image_row_array[1])
#
#     mean = np.mean(temp_array)
#     std = np.std(temp_array, ddof=1)
#
#     i = start + 2
#
#     while (mean-(3 * std))<=image_row_array[i]<=(mean+(3 * std)):
#         temp_array.append(image_row_array[i])
#         i = i + 1
#         mean = np.mean(temp_array)
#         std = np.std(temp_array, ddof=1)
#
#     start = i
#
#     temp_array = list()
#
#     temp_array.append(image_row_array[2047])
#     temp_array.append(image_row_array[2046])
#
#     mean = np.mean(temp_array)
#     std = np.std(temp_array, ddof=1)
#
#     i = end - 2
#
#     while (mean - (3 * std)) <= image_row_array[i] <= (mean + (3 * std)):
#         temp_array.append(image_row_array[i])
#         i = i - 1
#         mean = np.mean(temp_array)
#         std = np.std(temp_array, ddof=1)
#
#     end = i
#
#     return start, end


def get_rms_image(a, window_size):
    a2 = np.power(a, 2)
    window = np.ones(shape=(window_size, window_size)) / float(window_size)
    return np.sqrt(scipy.signal.convolve2d(a2, window, 'same'))


def get_initial_guesses_for_center(rms_image):
    mean = np.mean(rms_image)
    num_x = 0.0
    num_y = 0.0
    den_x = 0.0
    den_y = 0.0
    for i in range(0, rms_image.shape[0]):
        for j in range(0, rms_image.shape[1]):
            if rms_image[i][j] > mean:
                num_x += (i * rms_image[i][j])
                den_x += rms_image[i][j]
                num_y += (j * rms_image[i][j])
                den_y += rms_image[i][j]

    return num_x/den_x, num_y/den_y


def get_actual_gradient_image(image_data, reverse=True, direction=1):
    cent_x, cent_y = get_initial_guesses_for_center(get_rms_image(image_data, 5))

    gradient_image = np.zeros(shape=image_data.shape, dtype=float)

    gradient_image[int(cent_x)][int(cent_y)] = 0.0

    def get_data(image_data, i, j):
        if 0<=i<image_data.shape[0] and 0<=j<image_data.shape[1]:
            return image_data[i][j]
        return 0.0

    # f = open('values_halpha.csv', 'w+')
    # headers = ['target_x', 'target_y', 'direction', 'x', 'y', 'value']

    # f.write(','.join(headers) + '\n')

    for x in range(0, image_data.shape[0]):
        for y in range(0, image_data.shape[1]):
            if int(cent_x) == x and int(cent_y) == y:
                continue

            if -7<=np.arctan((y-cent_y)/(x-cent_x)) * 180/np.pi<=7:
                # the target is with slope infinite
                outgoing_sum = 0.0
                for i in range(y-2, y+5):
                    for j in range(x-6, x+1):
                        data = [
                            x,
                            y,
                            'outgoing',
                            i,
                            j,
                            get_data(image_data, i, j)
                        ]
                        # f.write(','.join(str(x) for x in data) + '\n')
                        outgoing_sum += get_data(image_data, i, j)

                incoming_sum = 0.0
                for i in range(y-2, y+5):
                    for j in range(x, x+7):
                        data = [
                            x,
                            y,
                            'incoming',
                            i,
                            j,
                            get_data(image_data, i, j)
                        ]
                        # f.write(','.join(str(x) for x in data) + '\n')
                        incoming_sum += get_data(image_data, i, j)

                if not reverse:
                    gradient_image[x][y] = (incoming_sum - outgoing_sum) / 35
                else:
                    if x < cent_x:
                        gradient_image[x][y] = direction * (incoming_sum - outgoing_sum) / 35
                    else:
                        gradient_image[x][y] = direction * (outgoing_sum - incoming_sum ) / 35

            elif x == cent_x:
                # slope is zero
                outgoing_sum = 0.0

                for i in range(x-2, x+3):
                    for j in range(y, y+7):
                        data = [
                            x,
                            y,
                            'outgoing',
                            i,
                            j,
                            get_data(image_data, i, j)
                        ]
                        # f.write(','.join(str(x) for x in data) + '\n')
                        outgoing_sum += get_data(image_data, i, j)

                incoming_sum = 0.0

                for i in range(x-2, x+3):
                    for j in range(y-6, y+1):
                        data = [
                            x,
                            y,
                            'incoming',
                            i,
                            j,
                            get_data(image_data, i, j)
                        ]
                        # f.write(','.join(str(x) for x in data) + '\n')
                        incoming_sum += get_data(image_data, i, j)

                gradient_image[x][y] = (incoming_sum - outgoing_sum) / 35

            else:
                slope = float(x-cent_x)/float(y-cent_y)

                perpendicular_slope = -1/slope

                outgoing_sum = 0.0
                for i in range(y-2, y+3):
                    _x = int((perpendicular_slope * i) + (x - (perpendicular_slope * y)))

                    for j in range (i, i+7):
                        _y = int( (slope * j) + (_x - (slope * i)) )
                        data = [
                            x,
                            y,
                            'outgoing',
                            _y,
                            j,
                            get_data(image_data, _y, j)
                        ]
                        # f.write(','.join(str(x) for x in data) + '\n')
                        outgoing_sum += get_data(image_data, _y, j)

                incoming_sum = 0.0
                for i in range(y - 2, y + 3):
                    _x = int((perpendicular_slope * i) + (x - (perpendicular_slope * y)))

                    for j in range(i-6, i + 1):
                        _y = int((slope * j) + (_x - (slope * i)))
                        data = [
                            x,
                            y,
                            'incoming',
                            _y,
                            j,
                            get_data(image_data, _y, j)
                        ]
                        # f.write(','.join(str(x) for x in data) + '\n')
                        incoming_sum += get_data(image_data, _y, j)

                if not reverse:
                    gradient_image[x][y] = (incoming_sum - outgoing_sum) / 35
                else:
                    if y < cent_y:
                        gradient_image[x][y] = direction * (outgoing_sum - incoming_sum) / 35
                    else:
                        gradient_image[x][y] = direction * (incoming_sum - outgoing_sum) / 35

    gradient_image = gradient_image - np.min(gradient_image) + 1

    # f.close()
    return gradient_image, cent_x, cent_y


def get_flat_fielded_corrected_image(base_path, image, dark_image, flat_image):
    image_data = fits.getdata(base_path + image)
    dark_image_data = fits.getdata(base_path + dark_image)
    flat_image_data = fits.getdata(base_path + flat_image)

    if len(image_data.shape) == 3:
        image_data = image_data[0]
    if len(dark_image_data.shape) == 3:
        dark_image_data = dark_image_data[0]
    if len(flat_image_data.shape) == 3:
        flat_image_data = flat_image_data[0]

    flt_dark = flat_image_data - dark_image_data

    sum = 0.0
    for i in range(0, image_data.shape[0]):
        for j in range(0, image_data.shape[1]):
            sum += flt_dark[i][j]

    image_average_flt_dark = sum / (image_data.shape[0] * image_data.shape[1])
    image_data = (image_data - dark_image_data) / (flat_image_data - dark_image_data)

    image_data = image_data * image_average_flt_dark

    image_data = image_data - np.min(image_data) + 1

    return image_data


def get_guess_radii(gradient_image, cent_x, cent_y):
    gima = scipy.signal.medfilt(gradient_image[int(cent_x)+20], 11)
    maxima_array = np.r_[True, gima[1:] < gima[:-1]] & np.r_[gima[:-1] < gima[1:], True]

    maxima_points = np.where(maxima_array==True)

    return abs(maxima_points[0][-1]-maxima_points[0][0])


def save_gradient_image_with_guessed_center(base_path, image, dark_image, flat_image):

    image_data = get_flat_fielded_corrected_image(base_path, image, dark_image, flat_image)

    gradient_image, cent_x, cent_y = get_actual_gradient_image(image_data)

    f = open('gradient_images/'+image+'_gradient', 'wb')

    pickle.dump((gradient_image, cent_x, cent_y), f)

    f.close()

    plt.imsave('gradient_images/'+image+'_gradient.png', gradient_image, cmap='gray', format='png')

    plt.close()

    return gradient_image, cent_x, cent_y


def get_accurate_center_and_radius(points, cent_x, cent_y):
    def calcR(xc,yc):
        distance_list = list()
        for point in points:
            distance_list.append(np.sqrt(((point[0]-xc)**2)+((point[1]-yc)**2)))
        return np.array(distance_list)

    def f_2(c):
        Ri = calcR(*c)
        return Ri - Ri.mean()

    center_estimate = cent_x, cent_y

    center_accurate, ier = optimize.leastsq(f_2, center_estimate)

    accurate_x, accurate_y = center_accurate

    radius_array = calcR(*center_accurate)

    radius_mean = np.mean(radius_array)

    return accurate_x, accurate_y, radius_mean


def get_points_on_the_circle(gradient_image, cent_x, cent_y):
    hist = np.histogram(gradient_image)

    threshold = hist[1][-2]

    points = np.where(gradient_image > threshold)

    points_tuple = list()

    for x,y in zip(points[0], points[1]):
        points_tuple.append((x,y))

    return points_tuple


def get_limb_darkening_corrected_image(base_path, image, dark_image, flat_image, accurate_x, accurate_y, radius_mean):

    image_data = get_flat_fielded_corrected_image(base_path, image, dark_image, flat_image)

    small_image = cv2.resize(image_data, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

    flat_image = scipy.signal.medfilt(small_image, 105)

    large_flat_image = cv2.resize(flat_image, dsize=(image_data.shape[0], image_data.shape[1]), interpolation=cv2.INTER_CUBIC)

    large_flat_image = large_flat_image - np.min(large_flat_image) + 1

    corrected_image = np.zeros(shape=image_data.shape)

    for i in range(image_data.shape[0]):
        for j in range(image_data.shape[1]):
            if (i-accurate_x)**2 + (j-accurate_y)**2 - ((0.97**2) * radius_mean**2) <=0:
                corrected_image[i][j] = image_data[i][j]/large_flat_image[i][j]

    plt.imsave('output_images/'+image+'.png', corrected_image, cmap='gray', format='png')

    plt.close()

    plt.plot(corrected_image[int(accurate_x)])

    plt.savefig('profiles/'+image+'_corrected.png', format='png')

    plt.close()

    plt.plot(image_data[int(accurate_x)])

    plt.savefig('profiles/'+image+'_original.png', format='png')

    plt.close()

    return corrected_image


def limb_darkening_corrected_flow(base_path, image, dark_image, flat_image):
    gradient_image, cent_x, cent_y = save_gradient_image_with_guessed_center(
        base_path, image, dark_image, flat_image
    )

    points = get_points_on_the_circle(gradient_image, cent_x, cent_y)

    accurate_x, accurate_y, radius_mean = get_accurate_center_and_radius(points, cent_x, cent_y)

    corrected_image = get_limb_darkening_corrected_image(
        base_path,
        image,
        dark_image,
        flat_image,
        accurate_x,
        accurate_y,
        radius_mean
    )

    return corrected_image


if __name__ == '__main__':
    # image_list = ['Ha_20170803_100841830.fits', 'Ha_20170803_100942120.fits', 'Ha_20170803_101042190.fits','Ha_20170803_101142270.fits','Ha_20170803_101242340.fits','Ha_20170803_101342400.fits','Ha_20170803_101442480.fits','Ha_20170803_101542570.fits','Ha_20170803_101642630.fits','Ha_20170803_101742710.fits','Ha_20170803_101842760.fits','Ha_20170803_101942820.fits','Ha_20170803_102042950.fits','Ha_20170803_102143000.fits','Ha_20170803_102243080.fits','Ha_20170803_102343190.fits','Ha_20170803_102443220.fits','Ha_20170803_103034690.fits','Ha_20170803_103134770.fits','Ha_20170803_103234850.fits','Ha_20170803_103334940.fits']
    base_path = '/Users/harshmathur/Documents/H alpha20170803/'
    # base_path = '/Users/harshmathur/Documents/warm data/2017/'
    # base_path = '/Users/harshmathur/Documents/warm data/'
    image = 'Ha_20170803_100841830.fits'
    dark_image = 'Dark_20170803_133401800.fits'
    flat_image = 'Flat_20170803_112801100.fits'
    # image_list = ['20170908T080116_GBand.fits']
    # dark_image = 'MasterDark_20170908T112418_GBand.fits'
    # flat_image = 'MasterFlat_20170908T080726_GBand.fits'
    # image_list = ['20180527T114644_CaK.fits']
    # flat_image = 'MasterFlat_20180527T102637_CaK.fits'
    # dark_image = 'MasterDark_20180527T112709_CaK.fits'
    limb_darkening_corrected_flow(
        base_path,
        image,
        dark_image,
        flat_image
    )
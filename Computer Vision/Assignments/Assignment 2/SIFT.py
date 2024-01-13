def extractSIFTFeatures(gray, octaves, scales, sigma, sigmaN, k):
    """
    Extracts Scale-Invariant Feature Transform (SIFT) features from a grayscale image.

    Parameters:
    - gray_image (numpy.ndarray): The input grayscale image.
    - octaves (int): The number of octaves in the image pyramid.
    - scales (int): The number of scales per octave.
    - sigma (float): The initial scale of the Gaussian kernel.
    - sigmaN (float): The standard deviation for the DoG (Difference of Gaussians) kernel.
    - k (float): The factor between the scales in each octave.

    Returns:
    - list: List of SIFT features with coordinates, scale, and orientation.
    """

    
    r1, c1 = gray.shape

    gaussianOctaves = []
    gaussianOctaves.append(buildFirstGaussianOctave(gray, scales, sigma, sigmaN, k))

    for i in range(1, octaves):
        baseImage = gaussianOctaves[i - 1][2]
        row, col = baseImage.shape
        gaussianOctaves.append(buildGaussianOctave(baseImage[0:row:2, 0:col:2], scales, sigma, 0, k))

    dogOctaves = []
    for o in range(0, octaves):
        dogOctaves.append(buildDOGOctave(gaussianOctaves[o], scales))

    keypoints = []
    r = 10.0
    threshold = 0.03
    for o in range(0, octaves):
        keypoints.append(getKeyPoints(dogOctaves[o], threshold, r))

    O = []
    S = []
    for o in range(0, octaves):
        O.append(o - 1)
    for s in range(0, scales):
        S.append(np.power(k, s) * sigma)

    baseKeyPoints = []
    for o in range(0, octaves):
        kp = keypoints[o]
        numKP = len(kp)
        mag, ori = createGradientMagnitudeandOri(gaussianOctaves[o], S)
        Y, X = gaussianOctaves[o][0].shape
        p = np.power(2.0, O[o])
        for i in range(0, numKP):
            currKp = kp[i]
            x = np.multiply(currKp[0], p)
            y = np.multiply(currKp[1], p)
            s = currKp[2]
            if x < 0 or x > c1 - 1 or y < 0 or y > r1 - 1 or s < 0 or s > scales - 1:
                print("point coord out of range")
            sig = S[int(np.round(s))]
            newKps = generateOrientationHistogram(mag[int(np.round(s))], ori[int(np.round(s))], sig,
                                                  int(np.round(currKp[0])), int(np.round(currKp[1])))
            for pts in range(0, len(newKps)):
                newKp = []
                newKp.append(x)
                newKp.append(y)
                newKp.append(p)
                newKp.append(newKps[pts])
                baseKeyPoints.append(newKp)

    return baseKeyPoints


def buildFirstGaussianOctave(gray, scales, sigma, sigmaN, k):
    gaussianOctave = []
    dblGray = bilinearInterpolation(gray)
    

    for i in range(0, scales):
        desiredSigma = sigma * np.power(k, i)
        currSigma = np.sqrt(desiredSigma * desiredSigma - 2.0 * sigmaN * sigmaN)
        gaussianImage = cv2.GaussianBlur(dblGray, (0, 0), currSigma)
        gaussianOctave.append(gaussianImage)    

        # Plot the image after each Gaussian octave
        plt.imshow(gaussianImage, cmap='gray')
        plt.title(f'Gaussian Octave {i+1}')
        plt.show()

    return gaussianOctave


def buildGaussianOctave(baseImage, scales, sigma, sigmaN, k):
    """
    Builds a Gaussian octave for Scale-Invariant Feature Transform (SIFT).

    Parameters:
    - base_image (numpy.ndarray): The base image of the octave.
    - scales (int): The number of scales in the octave.
    - sigma (float): The initial scale of the Gaussian kernel.
    - sigma_n (float): The standard deviation for blurring.
    - k (float): The factor between the scales in the octave.

    Returns:
    - list: List of images representing the Gaussian octave.
    """

    gaussianOctave = []
    gaussianOctave.append(baseImage)

    for i in range(1, scales):
        desiredSigma = np.power(k, i) * sigma
        currSigma = np.sqrt(desiredSigma * desiredSigma - sigmaN * sigmaN)
        gaussianOctave.append(cv2.GaussianBlur(baseImage, (0, 0), currSigma))
    return gaussianOctave


def buildDOGOctave(gaussianOctave, scales):

    """
    Builds a Difference of Gaussians (DoG) octave for Scale-Invariant Feature Transform (SIFT).

    Parameters:
    - gaussian_octave (list): List of images representing the Gaussian octave.

    Returns:
    - list: List of images representing the DoG octave.
    """

    dogOctave = []

    for i in range(1, scales):
        dogOctave.append(np.subtract(gaussianOctave[i], gaussianOctave[i - 1]))

    return dogOctave


def createGradientMagnitudeandOri(gaussOctave, Scales):

    """
    Calculates gradient magnitude and orientation for an image in a Gaussian octave.

    Parameters:
    - gauss_octave (list): List of images representing the Gaussian octave.
    - scales (list): List of scales corresponding to the images in the octave.

    Returns:
    - tuple: A tuple containing lists of gradient magnitudes and orientations.
    """

    row, col = gaussOctave[0].shape
    magnitudes = []
    orientations = []
    eps = 1e-10
    for k in range(0, len(gaussOctave)):
        mag = np.zeros((row, col), gaussOctave[0].dtype)
        ori = np.zeros((row, col), gaussOctave[0].dtype)
        for j in range(1, row - 1):
            for i in range(1, col - 1):
                dx = gaussOctave[k][j, i + 1] - gaussOctave[k][j, i - 1]
                dy = gaussOctave[k][j + 1, i] - gaussOctave[k][j - 1, i]
                mag[j, i] = np.sqrt(dx * dx + dy * dy)
                ori[j, i] = np.arctan(dy / (dx + eps))
        sigma = Scales[k]
        mag = cv2.GaussianBlur(mag, (0, 0), 1.5 * sigma)
        magnitudes.append(mag)
        orientations.append(ori)
    return magnitudes, orientations


def generateOrientationHistogram(mag, ori, sig, x, y):
    """
    Generates orientation histogram for a given location in an image.

    Parameters:
    - magnitude (numpy.ndarray): Array representing the gradient magnitudes.
    - orientation (numpy.ndarray): Array representing the gradient orientations.
    - sigma (float): Standard deviation for the Gaussian blur.
    - x (int): x-coordinate of the location.
    - y (int): y-coordinate of the location.

    Returns:
    - list: List of dominant orientations.
    """

    wsize = int(2 * 1.5 * sig)
    nbins = 36
    hist = np.zeros((36,), dtype=mag.dtype)
    rows, cols = mag.shape

    for j in range(-wsize, wsize):
        for i in range(-wsize, wsize):
            r = y + j
            c = x + i
            if 0 <= r < rows and 0 <= c < cols:
                deg = ori[r, c] * 180.0 / np.pi
                hist[int(deg / 10)] += mag[r, c]

    peak_loc = np.argmax(hist)
    peak_val = hist[peak_loc]

    orientations = [peak_loc * 10 + 5]

    for k in range(nbins):
        if hist[k] >= 0.8 * peak_val and k != peak_loc:
            orientations.append(k * 10 + 5)

    # Plot the histogram
    plt.bar(np.arange(nbins) * 10 + 5, hist, width=10, align='edge')
    plt.title("Orientation Histogram")
    plt.xlabel("Degrees")
    plt.ylabel("Magnitude")
    plt.show()

    return orientations


def getKeyPoints(dogOctaves, threshold, r):
    """
    Detects keypoints in a Difference of Gaussians (DoG) octave.

    Parameters:
    - dog_octaves (list): List of images representing the DoG octave.
    - threshold (float): Threshold for keypoint detection.
    - r (float): Ratio for keypoint scoring.

    Returns:
    - list: List of keypoints as [x, y, z] coordinates.
    """

    keypoints = []
    max_iter = 5
    cnt1 = 0
    for DOG in range(1, len(dogOctaves) - 1):
        currDOG = dogOctaves[DOG]
        prevDOG = dogOctaves[DOG - 1]
        nextDOG = dogOctaves[DOG + 1]
        cnt = 0
        for j in range(1, currDOG.shape[0] - 1):
            for i in range(1, currDOG.shape[1] - 1):
                pix = currDOG[j, i]
                prevNeighborhood = prevDOG[j - 1:j + 2, i - 1:i + 2]
                currNeighborhood = currDOG[j - 1:j + 2, i - 1:i + 2]
                nextNeighborhood = nextDOG[j - 1:j + 2, i - 1:i + 2]

                fullNeighborhood = np.zeros((3, 3, 3), currNeighborhood.dtype)
                fullNeighborhood[:, :, 0] = prevNeighborhood[:, :]
                fullNeighborhood[:, :, 1] = currNeighborhood[:, :]
                fullNeighborhood[:, :, 2] = nextNeighborhood[:, :]

                minMax = localExtrema2(fullNeighborhood)
                if minMax == 0:
                    continue
                cnt += 1
                ptX = i
                ptY = j
                ptZ = DOG

                neighborHood = np.zeros((3, 3, 3), currNeighborhood.dtype)
                success = 0
                for iter in range(0, max_iter):
                    neighborHood[:, :, 0] = prevDOG[ptY - 1:ptY + 2, ptX - 1:ptX + 2]
                    neighborHood[:, :, 1] = currDOG[ptY - 1:ptY + 2, ptX - 1:ptX + 2]
                    neighborHood[:, :, 2] = nextDOG[ptY - 1:ptY + 2, ptX - 1:ptX + 2]
                    xHat, D_xHat, H, fail = getInterpolatedMaxima(fullNeighborhood)
                    if fail == 0:
                        break
                    if np.abs(xHat[0]) <= 0.5 and np.abs(xHat[1]) <= 0.5 and np.abs(xHat[2]) <= 0.5:
                        success = 1
                        break

                    if xHat[0] > 0.5:
                        ptX += 1
                    elif xHat[0] < -0.5:
                        ptX -= 1
                    if xHat[1] > 0.5:
                        ptY += 1
                    elif xHat[1] < -0.5:
                        ptY -= 1
                    if xHat[2] > 0.5:
                        ptZ += 1
                    elif xHat[2] < -0.5:
                        ptZ -= 1

                    if ptY < 1 or ptY > currDOG.shape[0] - 2:
                        break
                    if ptX < 1 or ptX > currDOG.shape[1] - 2:
                        break
                    if ptZ < 1 or ptZ > len(dogOctaves) - 2:
                        break

                if success == 1:
                    if np.abs(D_xHat) < threshold:
                        continue
                    score = np.square(H[0, 0] + H[1, 1]) / (H[0, 0] * H[1, 1] - np.square(H[0, 1]))
                    if score > (np.square(r + 1) / r):
                        continue
                    kp = []
                    kp.append(ptX + xHat[0])
                    kp.append(ptY + xHat[1])
                    kp.append(ptZ + xHat[2])
                    keypoints.append(kp)
                    cnt1 += 1
    return keypoints


def getInterpolatedMaxima(fullNeighborhood):
    """
    Calculates the interpolated maxima for a 3D neighborhood.

    Parameters:
    - full_neighborhood (numpy.ndarray): 3D array representing the neighborhood.

    Returns:
    - tuple: A tuple containing the interpolated maxima, the interpolated gradient, Hessian matrix, and success flag.
    """

    H, H1 = getHessianofDOG(fullNeighborhood)
    D = getDerivativeDOG(fullNeighborhood)
    minus_D = np.multiply(-1.0, D)
    xHat = np.zeros((3, 1), H.dtype)
    D_xhat = 0

    try:
        xHat = np.linalg.solve(H, minus_D)
        pix = fullNeighborhood[1, 1, 1]
        D_xhat = pix + 0.5 * (D[0] * xHat[0] + D[1] * xHat[1] + D[2] * xHat[2])
        return xHat, D_xhat, H1, 1
    except np.linalg.LinAlgError:
        return xHat, D_xhat, H1, 0


def getHessianofDOG(neighborhood):

    """
    Calculates the Hessian matrix and its 2x2 submatrix for a given 3D neighborhood.

    Parameters:
    - neighborhood (numpy.ndarray): 3D array representing the neighborhood.

    Returns:
    - tuple: A tuple containing the full Hessian matrix and its 2x2 submatrix.
    """

    i = 1
    j = 1
    sigma = 1
    D2_x2 = neighborhood[j, i + 1, sigma] - 2.0 * neighborhood[j, i, sigma] + neighborhood[j, i - 1, sigma]
    D2_y2 = neighborhood[j + 1, i, sigma] - 2.0 * neighborhood[j, i, sigma] + neighborhood[j - 1, i, sigma]
    D2_sigma2 = neighborhood[j, i, sigma + 1] - 2.0 * neighborhood[j, i, sigma] + neighborhood[j, i, sigma - 1]
    D2_x_y = neighborhood[j + 1, i + 1, sigma] - neighborhood[j - 1, i + 1, sigma] - neighborhood[j + 1, i - 1, sigma] + \
             neighborhood[j - 1, i - 1, sigma]
    D2_x_y /= 4.0
    D2_x_sigma = neighborhood[j, i + 1, sigma + 1] - neighborhood[j, i + 1, sigma - 1] - neighborhood[
        j, i - 1, sigma + 1] + neighborhood[j, i - 1, sigma - 1]
    D2_x_sigma /= 4.0
    D2_y_sigma = neighborhood[j + 1, i, sigma + 1] - neighborhood[j + 1, i, sigma - 1] - neighborhood[
        j - 1, i, sigma + 1] + neighborhood[j - 1, i, sigma - 1]
    D2_y_sigma /= 4.0

    Hessian = np.zeros((3, 3), neighborhood.dtype)
    Hessian[0, 0] = D2_x2
    Hessian[0, 1] = D2_x_y
    Hessian[0, 2] = D2_x_sigma

    Hessian[1, 0] = D2_x_y
    Hessian[1, 1] = D2_y2
    Hessian[1, 2] = D2_y_sigma

    Hessian[2, 0] = D2_x_sigma
    Hessian[2, 1] = D2_y_sigma
    Hessian[2, 2] = D2_sigma2

    Hessian_CurrentScale = np.zeros((2, 2), Hessian.dtype)
    Hessian_CurrentScale[0, 0] = D2_x2
    Hessian_CurrentScale[0, 1] = D2_x_y
    Hessian_CurrentScale[1, 0] = D2_x_y
    Hessian_CurrentScale[1, 1] = D2_y2

    return Hessian, Hessian_CurrentScale


def getDerivativeDOG(neighborhood):
    """
    Calculates the derivative of the Difference of Gaussians (DoG) for a given 3D neighborhood.

    Parameters:
    - neighborhood (numpy.ndarray): 3D array representing the neighborhood.

    Returns:
    - numpy.ndarray: Array containing the derivatives along x, y, and sigma dimensions.
    """

    i = 1
    j = 1
    sigma = 1
    Dx = (neighborhood[j, i + 1, sigma] - neighborhood[j, i - 1, sigma]) / 2.0
    Dy = (neighborhood[j + 1, i, sigma] - neighborhood[j - 1, i, sigma]) / 2.0
    Dsigma = (neighborhood[j, i, sigma + 1] - neighborhood[j, i, sigma - 1]) / 2.0
    D = np.zeros((3, 1), neighborhood.dtype)
    D[0] = Dx
    D[1] = Dy
    D[2] = Dsigma

    return D


def localExtrema1(n):
    isExtrema = 1
    pix = n[1, 1, 1]
    if pix >= 0:
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if i == 1 and j == 1 and k == 1:
                        continue
                    if pix < n[i, j, k]:
                        isExtrema = 0
    else:
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if i == 1 and j == 1 and k == 1:
                        continue
                    if pix > n[i, j, k]:
                        isExtrema = 0
    return isExtrema


def localExtrema(pix, neighborhood):

    """
    Checks if the center pixel of a 3x3x3 neighborhood is a local extrema.

    Parameters:
    - neighborhood (numpy.ndarray): 3D array representing the neighborhood.

    Returns:
    - bool: True if the center pixel is a local extrema, False otherwise.
    """

    lessThan = 0
    greaterThan = 0
    minMax = 1
    for k in range(3):
        for l in range(3):
            if greaterThan == 1 and lessThan == 1:
                minMax = 0
                break
            if pix >= neighborhood[k, l]:
                greaterThan = 1
            else:
                lessThan = 1

    return minMax


def localExtrema2(n):

    """
    Checks if the center pixel is a local extremum within a 3x3 neighborhood.

    A pixel is considered a local extremum if it is either greater than or less than all its neighbors.

    Parameters:
    - center_pixel (int): The value of the center pixel.
    - neighborhood (numpy.ndarray): A 2D array representing the 3x3 neighborhood.

    Returns:
    - bool: True if the center pixel is a local extremum, False otherwise.
    """

    pix = n[1, 1, 1]
    lessThan = 0
    greaterThan = 0
    isExtrema = 1
    numEq = 0

    for i in range(3):
        if isExtrema == 0:
            break
        for j in range(3):
            if isExtrema == 0:
                break
            for k in range(3):
                if lessThan == 1 and greaterThan == 1:
                    isExtrema = 0
                    break
                if i == 1 and j == 1 and k == 1:
                    continue
                if pix >= n[i, j, k]:
                    greaterThan = 1
                elif pix <= n[i, j, k]:
                    lessThan = 1
                else:
                    numEq += 1

    if numEq == 26:
        print("All same")
        isExtrema = 0

    return isExtrema


def bilinearInterpolation(gray):

    """
    Double the input image with bilinear interpolation in both dimensions.

    Parameters:
    - gray (numpy.ndarray): Input image assumed to be in the range [0, 1].

    Returns:
    - numpy.ndarray: Double-sized image obtained through bilinear interpolation.
    """

    r, c = gray.shape
    r1 = 2

def bilinearInterpolation(gray):
    # Double the input image with bilinear interpolation in both dimensions
    # Input image is assumed floating point b/w [0..1]
    r, c = gray.shape
    r1 = 2 * r
    c1 = 2 * c
    dest = np.zeros((r1, c1), gray.dtype)
    expanded = np.zeros((r + 2, c + 2), gray.dtype)
    expanded[1:r + 1, 1:c + 1] = gray[:, :]
    for j in range(1, r1 - 1):
        for i in range(1, c1 - 1):
            j1 = j / 2.0
            i1 = i / 2.0
            delY = j1 - int(j1)
            delX = i1 - int(i1)

            temp1 = (1.0 - delX) * expanded[int(j1), int(i1)] + delX * expanded[int(j1), int(i1) + 1]
            temp2 = (1.0 - delX) * expanded[int(j1) + 1, int(i1)] + delX * expanded[int(j1) + 1, int(i1) + 1]
            dest[j, i] = (1.0 - delY) * temp1 + delY * temp2

    return dest

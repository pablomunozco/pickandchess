import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import pickle


ext_cam = 0  # (= 1) portatil, (=0) raspi

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def capturar_frames(vc, num):

    if num < 1:  # Como minimo se debe capturar una imagen.
        print('ERROR: Como minimo se debe capturar un frame.')
        return

    if num == 1:  # Si solo se pide capturar un frame no devolveremos una lista.

        rval, frame = vc.read()

        if rval == False:
            print('ERROR: No se ha podido capturar la imagen.')
            return

        cv2.destroyWindow("Cam")
        return frame

    frames = []  # Aqui guardaremos las imagenes.

    for i in range(num):  # Capturaremos tantas imagenes como se indique en el parametro 'num'.

        rval, frame = vc.read()

        if rval == False:
            print('ERROR: No se ha podido capturar la imagen', str(i), '.')
            break

        else:

            cv2.imshow("Cam", frame)  # Muestra continuamente en la ventana lo que ve la webcam
            print('Pulsa cualquier tecla para capturar el frame', str(i), '.')
            key = cv2.waitKey()  # Espera a que se pulse una tecla
            # if key == 27:  # Si se pulsa la tecla 'ESC'.

            # plt.imshow(frame), plt.title(str(i)), plt.xticks([]), plt.yticks([]), plt.show()

            frames.append(frame)
            # cv2.imwrite('frames\i'+str(i)+'.jpg', frame)

    cv2.destroyWindow("Cam")
    return frames


def guardar_frames(frames):

    cont = 0
    for i in frames:
        cv2.imwrite('frames\i' + str(cont) + '.jpg', i)
        cont += 1


def cargar_frames(num, dir):

    imgs = glob.glob(str(dir)+'\*.jpg')

    frames = []
    
    for f in range(num):
        frames.append(cv2.imread(imgs[f]))

    return frames


def encender_webcam():

    cv2.namedWindow("Cam")
    vc = cv2.VideoCapture(ext_cam)

    if not vc.isOpened():
        print(" ERROR: no se ha podido conectar con la camara.")
        exit()

    # Aqui se define la resolucion de las imagenes originales tomadas.
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)

    vc.set(3, 1920)
    vc.set(4, 1080)

    return vc


def apagar_webcam(vc):

    cv2.destroyWindow("Cam")
    vc.release()


def transformar_gris(frames):

    for i in range(len(frames)):
        frames[i] = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

    return frames


def plot_frames(frames):

    cont = 0
    for i in frames:

        plt.imshow(i, 'gray'), plt.title(str(cont)), plt.xticks([]), plt.yticks([]), plt.show()

        cont += 1


def plot_frame(frame, title):

    plt.imshow(frame, 'gray'), plt.title(str(title)), plt.xticks([]), plt.yticks([]), plt.show()


def calibra():

    '''
    Antes que nada, es necesario obtener al menos 9 frames para la calibración.
    Se deben encontrar los corners de, almenos, 9 frames.
    '''

    vc = encender_webcam()

    frames = capturar_frames(vc, 10)

    if len(frames) < 9 or len(frames) > 25:
        print('OJO: Se intentarán capturar', str(len(frames)), 'frames.')
        print(' Se recomienda un numero entre 9 y 20 para calibrar.')

    frames = transformar_gris(frames)

    # print(len(frames)
    # print(frames[0].shape)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    cont = 0  # Declaramos el contador que determinara en cuantas imagenes se detectan los corners.

    for f in frames:

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(f, (7, 7), None)

        if ret == True:  # If found, add object points, image points (after refining them with SubPix).

            # print('Corners encontrados!')

            cont += 1

            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(f, corners, (11, 11), (-1, -1), criteria)

            imgpoints.append(corners2)

            # Draw and display the corners
            # aux = f.copy()
            f = cv2.drawChessboardCorners(f, (7, 7), corners2, ret)

            # plt.imshow(aux, 'gray'), plt.title(''), plt.xticks([]), plt.yticks([]), plt.show()

            # corners2 = corners2.reshape(7*7, 2)

            # print(corners2.shape)

        '''
        else:
            # print('ERROR: No se han encontrado los corners!')
        '''

    if cont < 9:  # Como minimo se necesitan 9 imagenes con los corners.

        print('ERROR: No se han podido encontrar 9 corners en los frames para calibrar.')
        exit()

    # Calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frames[0].shape[::-1], None, None)

    print('Se ha calibrado la camara correctamente.')
    print('ret del calibrate =', str(ret))

    return mtx, dist, vc


def guarda_calibra(mtx, dist):

    with open('calibra.p', 'wb') as f:
        pickle.dump((mtx, dist), f)

    # pickle.dump((mtx, dist), open("calibra.p", "wb"))

def carga_calibra():

    with open('calibra.p', 'rb') as f:
        (mtx, dist) = pickle.load(f)

    return mtx, dist

def undistort(img, mtx, dist):

    # Refine the camera matrix
    '''
    If the scaling parameter alpha=0, it returns undistorted image with minimum unwanted pixels.
    So it may even remove some pixels at image corners. If alpha=1, all pixels are retained with
    some extra black images. This function also returns an image ROI which can be used to crop
    the result.
    '''

    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))


    # Undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # Crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]

    cv2.destroyAllWindows()

    return dst


def calib_auto():
    '''
    Calibra la camara con imagenes ya existentes.
    Lee de la carpeta '\calibra' los frames que se usarán para calibrar.

    num: indica con cuantos frames se calibrará.
    '''

    num = 20

    frames = cargar_frames(num, 'calibra')

    print('num_frames =', len(frames))

    frames = transformar_gris(frames)

    # plot_frames(frames)  ##################################################

    if len(frames) < 9 or len(frames) > 25:
        print('OJO: Se cargarán', str(len(frames)), 'frames.')
        print(' Se recomienda un numero entre 9 y 20 para calibrar.')

    # frames = transformar_gris(frames)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    cont = 0  # Declaramos el contador que determinara en cuantas imagenes se detectan los corners.

    for f in frames:

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(f, (7, 7), None)

        if ret == True:  # If found, add object points, image points (after refining them with SubPix).

            # print('Corners encontrados!')

            cont += 1

            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(f, corners, (11, 11), (-1, -1), criteria)

            imgpoints.append(corners2)

            # Draw and display the corners
            # aux = f.copy()
            f = cv2.drawChessboardCorners(f, (7, 7), corners2, ret)

            # plt.imshow(aux, 'gray'), plt.title(''), plt.xticks([]), plt.yticks([]), plt.show()

            # corners2 = corners2.reshape(7*7, 2)

            # print(corners2.shape)


        else:
            print('ERROR: No se han encontrado los corners del siguiente frame.')
            plot_frame(f, 'NO')

    if cont < 9:  # Como minimo se necesitan 9 imagenes con los corners.

        print('ERROR: No se han podido encontrar 9 frames con los corners para calibrar.')
        exit()

    # Calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frames[0].shape[::-1], None, None)

    print('Se ha calibrado la camara correctamente.')
    print('ret del calibrate =', str(ret))

    return mtx, dist


def calib_manual():

    '''
    Calibra la camara capturando una serie de imagenes desde la misma.
    '''

    mtx, dist, vc = calibra()

    return mtx, dist, vc


def tablero_perfecto(casillas_costado, pixeles):

    '''
    casillas_costadp: Establece el numero de casillas del costado del tablero.
    pixeles: Establece el numero de pixeles de costado por casilla.
    '''

    negre = 255  # Estableix el valor que reresenta el negre (255 funciona).

    v = int(casillas_costado / 2)

    t = np.kron([[negre, 0] * v, [0, negre] * v] * v, np.ones((pixeles, pixeles)))

    # plt.imshow(t, 'gray'), plt.title(t.shape), plt.xticks([]), plt.yticks([]), plt.show()

    cv2.imwrite('perfecto_'+str(casillas_costado)+'.jpg', t)

    cv2.imwrite(t, t)  # Sirve para cambiar el formato. Permite que se pueda leer como una imagen cualquiera.

    return t


def order_points(pts):

    # print(pts.shape), print(pts[0])
    pts = np.reshape(pts, (-1, 2))  ###########################################
    # print(pts.shape), print(pts[0])


    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):

    # Obtiene los puntos ordenados
    rect = order_points(pts)
    (tl, tr, br, bl) = rect  # Top-left, top-right, bottom-right, bottom-left

    # Se calcula la anchura de la imagen, que sera la distancia maxima en el
    # eje x entre bottom-right y el bottom-left o el top-right y el top-left.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))  # Linea superior
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))  # Linea inferior
    maxWidth = max(int(widthA), int(widthB))

    # Se calcula la altura de la imagen, que sera la distancia maxima en el
    # eje y entre top-right y bottom-right o el top-left y el bottom-left.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Ahora ya tenemos las dimensiones de la nueva imagen.

    # Para que esta sea cuadrada, cojemos la dist. minima entre altura y anchura maximas.
    dt = max((maxHeight), (maxWidth))  # Distancia tablero
    dt = int(dt + (dt / 3))  # Le añadimos las dos casillas del borde.

    dc = int (dt / 8)  # Distancia casilla

    ''' Este array solo cojeria el tablero interno de 6x6.
    dst = np.array([
        [0, 0],  # top left
        [maxWidth - 1, 0],  # top right
        [maxWidth - 1, maxHeight - 1],  # bottom right
        [0, maxHeight - 1]], dtype="float32")  # bottom left
    '''

    dst = np.array([
        [dc - 1, dc - 1],  # top left
        [dt - dc - 1, dc - 1],  # top right
        [dt - dc - 1, dt - dc - 1],  # bottom right
        [dc - 1, dt - dc - 1]], dtype="float32")  # bottom left

    # print(dst)

    # Computa la matriz de transformación de perspectiva y la aplica.
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (dt, dt))
    # warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def obtener_casillas(tablero):

    altura, anchura = tablero.shape  # filas, columnas
    # print('altura =', altura, ' anchura =', anchura)

    casillas = []

    anc = int(anchura / 8)
    alt = int(altura / 8)

    object_detect()

    for i in range(8):
        for j in range(8):

            casillas.append(tablero[anc * i:anc * i + anc - 1, alt * j:alt * j + alt - 1])

    return casillas

def object_detect():
    tableroColor = cv2.imread('pieces.jpg')
    tablero = cv2.imread('pieces.jpg', 0)
    tablero.reshape(434, 434)
    hsv = cv2.cvtColor(tableroColor, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([10, 15, 100])
    upper_yellow = np.array([30, 255, 255])

    lower_black = np.array([0, 2, 22])
    upper_black = np.array([0, 0, 0])

    # maskBlack = cv2.inRange(hsv, lower_black, upper_black)

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    cv2.imshow('mask', mask)

    altura, anchura = tablero.shape  # filas, columnas
    # print('altura =', altura, ' anchura =', anchura)
    casillas = []

    anc = int(anchura / 8)
    alt = int(altura / 8)
    piece = np.arange(64).reshape(8, 8)

    positionPieces = []

    for i in range(8):
        for j in range(8):
            casillas.append(tablero[anc * i:anc * i + anc - 1, alt * j:alt * j + alt - 1])

            cv2.imwrite('casilla' + str(i) + str(j) + '.jpg',
                        tablero[anc * i:anc * i + anc - 1, alt * j:alt * j + alt - 1])
            imageCasilla = cv2.imread('casilla' + str(i) + str(j) + '.jpg')

            gray = cv2.cvtColor(imageCasilla, cv2.COLOR_BGR2GRAY)

            thresh = \
            cv2.threshold(mask[anc * i:anc * i + anc - 1, alt * j:alt * j + alt - 1], 220, 255, cv2.THRESH_BINARY)[1]

            cv2.imwrite('threshCasilla' + str(i) + str(j) + '.jpg', thresh)

            positionPieces = []
            nonZero = cv2.countNonZero(thresh)

            if (nonZero != 0):
                piece[i, j] = True


            else:
                piece[i, j] = False
            # print('nonZero' + str(nonZero))
            # cv2.imwrite('THRESH'+ str(i) + str(j) + '.jpg', thresh)

    bilateral_filtered_image = cv2.bilateralFilter(mask, 5, 175, 175)
    edge_detected_image = cv2.Canny(tablero, 75, 200)

    cv2.imshow("edges", edge_detected_image)
    cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_list = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        print('APPROX: ' + str(len(approx)))
        area = cv2.contourArea(contour)
        if ((len(approx) > 5) & (len(approx) < 20) & (area > 8)):
            contour_list.append(contour)

    cv2.drawContours(tableroColor, contour_list, -1, (77, 255, 0), 2)

    cv2.imshow('Objects Detected', tableroColor)
    cv2.waitKey(0)

    print('EMPTY:' + str(piece))
    print('len_empty' + str(len(piece)))
    print('casillas' + str(casillas))
    print('piece' + str(piece))
    return piece



def plot_casillas(casillas):

    for i in range(8):

        for j in range(8):

            plt.subplot(8, 8, i*8 + j + 1)

            plt.xticks([]), plt.yticks([])
            # plt.title('('+str(i)+','+str(j)+')')

            plt.imshow(casillas[i*8 + j], 'gray')

    plt.show()


# _____________________________________________________________________________________________________________


if __name__ == '__main__':

    print('opencv version =', cv2.__version__)  # Muestra la version de opencv

    # vc = encender_webcam()

    # mtx, dist = calib_auto()

    # guarda_calibra(mtx, dist)  # Guarda la información para la calibración.

    mtx, dist = carga_calibra()

    frame = cv2.imread('calibra\i5.jpg')

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    plot_frame(frame, 'original')

    ui = undistort(frame, mtx, dist)

    plot_frame(ui, 'undistorted')

    ret, corners = cv2.findChessboardCorners(ui, (7, 7), None)

    if ret == True:

        corners2 = cv2.cornerSubPix(ui, corners, (11, 11), (-1, -1), criteria)

        final = four_point_transform(ui, corners2)
        # cv2.imwrite('final.jpg', final)

        print('final.shape =', final.shape)

        plot_frame(final, 'final')

        casillas = obtener_casillas(final)

        plt.imshow(casillas[0], 'gray'), plt.title('casilla'), plt.xticks([]), plt.yticks([]), plt.show()

        plot_casillas(casillas)



    else:
        print('ERROR: No se han encontrado los corners!')


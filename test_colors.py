import cv2
import os
import numpy as np
import sys




def main(video):
    home = os.getenv('HOME')
    # Some Globals
    directorioDeVideos = home + '/trafficFlow/trialVideos'

    archivoParametrosACargar = video[0:video.rfind('.')] +'.npy'

    parametrosInstalacion = np.load(directorioDeVideos +'/' +archivoParametrosACargar)
    indicesSemaforo 	=  parametrosInstalacion[0]

    indicesSemaforo = np.array(indicesSemaforo)
    video_path = directorioDeVideos + '/' + video

    print('video path is', video_path)

    cap = cv2.VideoCapture(video_path)
    scale = 10
    while True:
        _, frameVideo = cap.read()


        pixeles = np.array([frameVideo[indicesSemaforo[0][1] ,indicesSemaforo[0][0]]])
        for indiceSemaforo in indicesSemaforo[1:]:
            pixeles = np.append(pixeles ,[frameVideo[indiceSemaforo[1] ,indiceSemaforo[0]]], axis=0)
        new_shape = ( int(8*scale), int(24*scale) )
        frame = cv2.resize(np.reshape(pixeles, (24, 8, 3)), new_shape)



        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red = np.array([140, 97, 51])
        upper_red = np.array([180, 255, 255])



        lower_yellow = np.array([0, 120, 120])
        upper_yellow = np.array([20, 255, 255])


        lower_green = np.array([40, 20, 0])
        upper_green = np.array([87, 255, 255])

        mask1 = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask2 = cv2.inRange(hsv, lower_red, upper_red)
        mask3 = cv2.inRange(hsv, lower_green, upper_green)

        mask = mask1 + mask2 + mask3

        res = cv2.bitwise_and(frame, frame, mask=mask)

        kernel = np.ones((15, 15), np.float32) / 225
        smoothed = cv2.filter2D(res, -1, kernel)
        cv2.imshow('Original', frame)
        cv2.imshow('Averaging', smoothed)

        _, puck = cv2.threshold(smoothed, 30, 255, cv2.THRESH_BINARY)
        cv2.imshow('Puck', puck)

        #cv2.imshow('Semaphoro', frame)

        ch = 0xFF & cv2.waitKey(5)
        if ch == ord('q'):
            break
    cv2.destroyAllWindows()



def track(video):

    home = os.getenv('HOME')
    # Some Globals
    directorioDeVideos = home + '/trafficFlow/trialVideos'

    archivoParametrosACargar = video[0:video.rfind('.')] + '.npy'

    parametrosInstalacion = np.load(directorioDeVideos + '/' + archivoParametrosACargar)
    indicesSemaforo = parametrosInstalacion[0]

    indicesSemaforo = np.array(indicesSemaforo)
    video_path = directorioDeVideos + '/' + video

    print('video path is', video_path)


    cap = cv2.VideoCapture(video_path)

    def nothing(x):
        pass

    # Creating a window for later use
    cv2.namedWindow('result')

    # Starting with 100's to prevent error while masking
    h, s, v = 100, 100, 100

    # Creating track bar
    cv2.createTrackbar('h', 'result', 0, 179, nothing)
    cv2.createTrackbar('s', 'result', 0, 255, nothing)
    cv2.createTrackbar('v', 'result', 0, 255, nothing)
    scale = 10
    while (1):


        _, frameVideo = cap.read()

        pixeles = np.array([frameVideo[indicesSemaforo[0][1] ,indicesSemaforo[0][0]]])
        for indiceSemaforo in indicesSemaforo[1:]:
            pixeles = np.append(pixeles ,[frameVideo[indiceSemaforo[1] ,indiceSemaforo[0]]], axis=0)
        new_shape = ( int(8*scale), int(24*scale) )
        frame = cv2.resize(np.reshape(pixeles, (24, 8, 3)), new_shape)


        # converting to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # get info from track bar and appy to result
        h = cv2.getTrackbarPos('h', 'result')
        s = cv2.getTrackbarPos('s', 'result')
        v = cv2.getTrackbarPos('v', 'result')

        # Normal masking algorithm
        lower_blue = np.array([h, s, v])
        upper_blue = np.array([180, 255, 255])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        result = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow('result', result)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cap.release()

    cv2.destroyAllWindows()
if __name__ == '__main__':
    print(sys.argv)
    archivoDeVideo = None
    for entrada in sys.argv:
        print('INPUTS ARE', entrada)
        if ('.mp4' in entrada ) |('.avi' in entrada):
            archivoDeVideo = entrada
    #main(archivoDeVideo)
    track(archivoDeVideo)

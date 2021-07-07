#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  Copyright 2020 Reso-nance Numérique <laurent@reso-nance.org>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
# usage: python3 playRandom_RAM.py monfichier.mp4

import os, sys, cv2, random, numpy
from datetime import datetime, timedelta

headlessMode = True # True pour le lancer dans la console (sans bureau) ou False pour le lancer sous le bureau
# 1 proba fixe + 1 liste
# temps min et temps max
screenSize = [600,600] # taille de l'écran d'affichage (colonne, ligne)
videoSize = [1920,1080] # dimension de la vidéo (colonne, ligne)
changePlayDirectionProbas = [0.02, 0.032, 0.1, 0.01] # liste de probabilités
fixedPlayDirectionChangeProba = .01 # probabilité de changement de direction supplémentaire
changeProbasEvery = [4.0, 7.0] # temps en secondes (min et max, peu importe l'ordre) au bout duquel une nouvelle proba est tirée au sort
changeConfigEvery = [3.0,3.1] # temps en seconde (min et max, peu importe l'ordre) au bout duquel une nouvelle config est choisie
maxFPS = 20 # nombre maximal d'images par secondes souhaitées

listPointsZoom = [[[[960,540],None],[1.2,None]],
                  [[[1670,250],None],[1.2,None]],
                  [[[1733,893],None],[1.6,None]],
                  [[[230,230],None],[1.3,None]],
                  [[[1763,157],None],[1.9,None]]] # liste des coordonnées/zooms (colonne, ligne) des différents points d'interets. La liste se présente comme suit: [[[[X,Y],None],[Zoom,None]],...] None en 2eme position pour les coordonnées si le point atteint doit être fixe (sinon on tirera aléatoirement un point entre les deux coordonnées), de même pour le zoom.
                                                  # on utilise listPointsZoom dans le code comme suit: listPointsZoom[n°config][0:coordonnées 1:zoom][0:1ere valeur 1:soit None soit une valeur qui donne un intervalle]
avgSpeed = 150 # vitesse moyenne de déplacement pendant les transitions en pixels par seconde
probaRandConfig = 1 # 0 que des configs provenant de la liste / 1 que de l'alea
randZoomInter = [1.2, 1.7] # intervalle dans lequel on ppeut piocher un zoom aléatoire
forkZoom = 0.01 # fourchette à partir de laquelle le zom est considéré comme atteint, par exemple pour une fourchette de 0.1, si l'on souhaite atteindre un zoom de 1.3 alors on considerera comme acceptable un zoom de 1.2 ou 1.4
portrait = False # portrait = False si on affiche en paysage, True si on affiche en portrait
mirror = False # inverser la video

angle = 0
if portrait:
    screenSize[0], screenSize[1] = screenSize[1], screenSize[0]
    videoSize[0], videoSize[1] = videoSize[1], videoSize[0]
    angle = 90
if mirror:
    angle += 180

# changement de coordonnée des points d'intérêts en fonction des rotations de l'image
for config in listPointsZoom:
    if angle == 90: 
       config[0][0] = [config[0][0][1],videoSize[1]-config[0][0][0]]
    elif angle == 180:
        config[0][0] = [videoSize[0] - config[0][0][0],videoSize[1] - config[0][0][1]]
    elif angle == 270:
        config[0][0] = [videoSize[0] - config[0][0][1],config[0][0][0]]
    if config[0][1] is not None:
        if angle == 90:
           config[0][1] = [config[0][1][1],videoSize[1]-config[0][1][0]]
        elif angle == 180:
            config[0][1] = [videoSize[0] - config[0][1][0],videoSize[1] - config[0][1][1]]
        elif angle == 270:
            config[0][1] = [videoSize[0] - config[0][1][1],config[0][1][0]]

if headlessMode :
    import pygame
    os.putenv('SDL_VIDEODRIVER', 'fbcon')
    os.environ["SDL_FBDEV"] = "/dev/fb0"
    os.environ['SDL_NOMOUSE'] = '1'
    print("running in headless mode using framebuffer",os.environ["SDL_FBDEV"])
    pygame.init()
    pygame.mouse.set_visible(False)
else : import time, imutils

class ImageSequence():

    def __init__(self, videoFilename):
        self.frameCount = 0
        self.frames=[] # will store cv2.imread() for each frame
        self.extractFrames(videoFilename)
        self.currentFrameIndex = 0 # play head position
        self.playingForward = True
        self.changePlayDirectionProba = random.choice(changePlayDirectionProbas)
        self.zoom = listPointsZoom[0][1][0]
        self.points = listPointsZoom[0][0][0]
        self.config = 0

    def extractFrames(self, videoPath):
        """ extracts frames from the video file to RAM, including cv2 -> pygame conversion in headless mode"""
        if os.path.isfile(videoPath):
            print(f"loading {videoPath} frames to RAM :")
            video = cv2.VideoCapture(os.path.abspath(videoPath)) # open the video file as cv2 capture
            frameReadSuccessfully, frame = video.read()
            while frameReadSuccessfully:
                if headlessMode :
                    # cv2 image -> pygame surface conversion
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = numpy.rot90(frame)
                    pygame.image.frombuffer(frame.tostring(), frame.shape[1::-1],"RGB")
                    frame = pygame.surfarray.make_surface(frame)
                    frame = pygame.transform.flip(frame, True, False)
                    frame = pygame.transform.rotate(frame, angle)
                else:
                    i = angle/90
                    while i != 0:
                        frame = numpy.rot90(frame)
                        i -= 1
                self.frames.append(frame)# stores the frame in RAM
                frameReadSuccessfully, frame = video.read()
                print('  processing frame #', len(self.frames), end="\r")
            video.release()

            self.frameCount = len(self.frames)
            if self.frameCount : print("\n done,",self.frameCount, "frames extracted")
            else : raise SystemError(f"error converting {videoPath} to frames")

        else : raise SystemError(f"file {videoPath} not found")


    def getCurrentFrame(self):
        return self.frames[self.currentFrameIndex]

    def getZoom(self):
        return self.zoom

    def setZoom(self, newZoom):
        self.zoom = newZoom

    def getPoints(self):
        return self.points

    def setPoints(self, newPoints):
        self.points = newPoints

    def getConfig(self):
        return self.config

    def changeConfig(self):
        old = self.getConfig()
        if len(listPointsZoom) == 1:
            self.config = old
        else:
            while old == self.getConfig():
                self.config = random.choice([i for i in range(len(listPointsZoom))])

    def step(self, loop=False):
        """ set the play head (self.currentFrameIndex) to it's new position depending of the change direction
        proba and the current position in the file (manages looping forward or backward) """
        if random.random() < self.changePlayDirectionProba : self.playingForward = not self.playingForward # random direction change
        if random.random() < fixedPlayDirectionChangeProba : self.playingForward = not self.playingForward
        if loop : # loops back to the end and forth to the beginning
            if self.playingForward :
                self.currentFrameIndex = self.currentFrameIndex +1 if self.currentFrameIndex+1<self.frameCount-1 else 0
            else :
                self.currentFrameIndex = self.currentFrameIndex -1 if self.currentFrameIndex-1>0 else self.frameCount-1
        else : # invert playing direction when an end is reached
            if self.playingForward :
                if self.currentFrameIndex+1 < self.frameCount-1 : self.currentFrameIndex+=1
                else :
                    self.currentFrameIndex -= 1
                    self.playingForward = False
                    return
            else :
                if self.currentFrameIndex-1>=0 : self.currentFrameIndex-=1
                else :
                    self.currentFrameIndex = 1
                    self.playingForward = True
                    return

def getProbaTimer():
    """ returns a datetime object of when the next proba change is due """
    delay = random.uniform(min(changeProbasEvery), max(changeProbasEvery))
    return datetime.now() + timedelta(seconds=delay)

def getConfigTimer():
    """ returns a datetime object of when the next config change is due """
    delay = random.uniform(min(changeConfigEvery), max(changeConfigEvery))
    return datetime.now() + timedelta(seconds=delay)

if __name__ == "__main__":
    if len(sys.argv) < 2 : raise SystemExit(f"usage : python3 {sys.argv[0]} monFichier.mp4")
    img = ImageSequence(sys.argv[1])

    timeStarted, framesShown, = datetime.now(), 0
    changeProbaTimer = getProbaTimer()
    changeConfigTimer = getConfigTimer()
    periodMillis = int(1000/maxFPS)
    deltaZoom = 0
    newZoom, oldZoom = listPointsZoom[img.getConfig()][1][0], img.getZoom()
    newPoints, oldPoints = listPointsZoom[img.getConfig()][0][0], img.getPoints()
    distance = 0
    timeTransiZoom = 0
    timeConfigStart = datetime.now()
    reachZoom, reachPoints = False, False
    
    # window or surface creation
    if headlessMode : surface = pygame.display.set_mode((screenSize[0], screenSize[1]))#,pygame.FULLSCREEN) # full screen
    else :
        cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN) # full screen, no titlebar
        cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    playing = True

    def Distance(P1, P2):
        return numpy.sqrt(numpy.square(P1[0] - P2[0]) + numpy.square(P1[1] - P2[1]))

    if headlessMode:
        
        def ZoomTranslatHeadlessMode(image, zoomSize, point):
            # Méthode Zoom et translation dans Pygame pour la Raspberry
            wnd_w, wnd_h = screenSize[0], screenSize[1]
            image_surface = pygame.Surface((image.get_width() / zoomSize, image.get_height() / zoomSize))
            image_surface.blit(image, (wnd_w/(2*zoomSize) - point[0],wnd_h/(2*zoomSize) - point[1]))
            image_surface = pygame.transform.scale(image_surface, (image.get_width(), image.get_height()))
            return image_surface


    else:

        def Zoom(cv2Object, zoomSize):
            # Resizes the image/video frame to the specified amount of "zoomSize".
            # A zoomSize of "2", for example, will double the canvas size
            #print(cv2Object.shape[1])
            cv2Object = imutils.resize(cv2Object, width=(int(zoomSize) * int(cv2Object.shape[1])))
            # center is simply half of the height & width (y/2,x/2)
            center = (int(cv2Object.shape[0] / 2), int(cv2Object.shape[1] / 2))
            # cropScale represents the top left corner of the cropped frame (y/x)
            cropScale = (int(center[0] / zoomSize), int(center[1] / zoomSize))
            # The image/video frame is cropped to the center with a size of the original picture
            # image[y1:y2,x1:x2] is used to iterate and grab a portion of an image
            # (y1,x1) is the top left corner and (y2,x1) is the bottom right corner of new cropped frame.
            cv2Object = cv2Object[(int(center[0]) - int(cropScale[0])):(int(center[0]) + int(cropScale[0])),(int(center[1]) - int(cropScale[1])):int((center[1]) + int(cropScale[1]))]
            return cv2Object

        def Translation(cv2Object, Points):
            rows, cols, color = cv2Object.shape
            M = numpy.float32([[1, 0, cols/2 - Points[0]], [0, 1, rows/2 - Points[1]]])
            dst = cv2.warpAffine(cv2Object, M, (cols, rows))
            return dst

    # play loop
    while playing :
        loopTime = datetime.now()

        # changing configuration
        if loopTime > changeConfigTimer and reachZoom and reachPoints:
            reachZoom = False
            reachPoints = False
            oldPoints = img.getPoints()
            oldZoom = img.getZoom()
            timeConfigStart = datetime.now()
            if random.random() < probaRandConfig:
                newZoom = numpy.random.uniform(randZoomInter[0],randZoomInter[1])
                if headlessMode:
                    newPoints = [numpy.random.randint(screenSize[0] / (2 * (newZoom + forkZoom)) - 1,img.getCurrentFrame().get_width() - screenSize[0] / (2 * (newZoom + forkZoom)) + 1), numpy.random.randint(screenSize[1] / (2 * (newZoom + forkZoom)) - 1,img.getCurrentFrame().get_height() - screenSize[1] / (2 * (newZoom + forkZoom)) + 1)]
                else:
                    newPoints = [numpy.random.randint(screenSize[0]/(2 * (newZoom + forkZoom)) - 1,img.getCurrentFrame().shape[1] - screenSize[0]/(2 * (newZoom + forkZoom)) + 1),numpy.random.randint(screenSize[1]/(2 * (newZoom + forkZoom)) - 1,img.getCurrentFrame().shape[0] - screenSize[1]/(2 * (newZoom + forkZoom)) + 1)]
                if angle == 90:
                   newPoints = [newPoints[1],videoSize[1]-newPoints[0]]
                elif angle == 180:
                    newPoints = [videoSize[0] - newPoints[0],videoSize[1] - newPoints[1]]
                elif angle == 270:
                    newPoints = [videoSize[0] - newPoints[1],newPoints[0]]
            else:
                img.changeConfig()
                if listPointsZoom[img.getConfig()][0][1] is None:
                    newPoints = listPointsZoom[img.getConfig()][0][0]
                else:
                    t = random.random()
                    X0, Y0, X1, Y1 = listPointsZoom[img.getConfig()][0][0][0], listPointsZoom[img.getConfig()][0][0][1], listPointsZoom[img.getConfig()][0][1][0], listPointsZoom[img.getConfig()][0][1][1]
                    newPoints = [int(X0 + (X1 - X0)*t), int(Y0 + (Y1 - Y0)*t)]
                if listPointsZoom[img.getConfig()][1][1] is None:
                    newZoom = listPointsZoom[img.getConfig()][1][0]
                else:
                    t = random.random()
                    newZoom = listPointsZoom[img.getConfig()][1][0] + (listPointsZoom[img.getConfig()][1][1] - listPointsZoom[img.getConfig()][1][0])*t
            distance = Distance(newPoints, oldPoints)
            timeTransiConfig = distance / avgSpeed

            # determination wether the zoom or the distance should be reached first
            if oldZoom > newZoom:
                timeTransiZoom = timeTransiConfig * 1.5
            else:
                timeTransiZoom = timeTransiConfig * 2/3

            changeConfigTimer = getConfigTimer()
        
        # seconds spent in this configuration
        configDuration = (datetime.now() - timeConfigStart).total_seconds()
        
        # display current frame
        if headlessMode :
            if not reachZoom:
    
                # update of deltaZoom each frame
                if timeTransiZoom > configDuration:
                    deltaZoom = (newZoom - img.getZoom())/(maxFPS*(timeTransiZoom - configDuration))
                else:
                    deltaZoom = 0
                
                img.setZoom(img.getZoom() + deltaZoom)
                if (img.getZoom() < newZoom + forkZoom) and (img.getZoom() >= newZoom - forkZoom):
                    reachZoom = True
                    changeConfigTimer = getConfigTimer()

            if not reachPoints:

                # update of the deltaCols, deltaRows each frame
                if Distance(img.getPoints(), newPoints) != 0:
                    deltaCols = numpy.sign(newPoints[0] - img.getPoints()[0]) * (avgSpeed / (maxFPS * img.getZoom())) * numpy.square((newPoints[0] - img.getPoints()[0]) / Distance(img.getPoints(), newPoints))
                    deltaRows = numpy.sign(newPoints[1] - img.getPoints()[1]) * (avgSpeed / (maxFPS * img.getZoom())) * numpy.square((newPoints[1] - img.getPoints()[1]) / Distance(img.getPoints(), newPoints))
                else:
                    deltaCols = 0
                    deltaRows = 0

                img.setPoints([img.getPoints()[0] + deltaCols, img.getPoints()[1] + deltaRows])

                if Distance(img.getPoints(), newPoints) < 5:
                    reachPoints = True
                    changeConfigTimer = getConfigTimer()


            surface.blit(ZoomTranslatHeadlessMode(img.getCurrentFrame(),img.getZoom(),img.getPoints()), (0,0))
            pygame.display.update()
            for event in pygame.event.get(): # exit on esc
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    playing = False
        else :
            if not reachZoom:
                img.setZoom(img.getZoom()+deltaZoom)
                if (img.getZoom() < newZoom + forkZoom) and (img.getZoom() >= newZoom - forkZoom):
                    reachZoom = True
                    changeConfigTimer = getConfigTimer()

            if not reachPoints:

                # update of the deltaCols, deltaRows each frame
                if Distance(img.getPoints(), newPoints) != 0:
                    deltaCols = numpy.sign(newPoints[0] - oldPoints[0]) * (avgSpeed / (maxFPS * img.getZoom())) * numpy.square((newPoints[0] - img.getPoints()[0]) / Distance(img.getPoints(), newPoints))
                    deltaRows = numpy.sign(newPoints[1] - oldPoints[1]) * (avgSpeed / (maxFPS * img.getZoom())) * numpy.square((newPoints[1] - img.getPoints()[1]) / Distance(img.getPoints(), newPoints))
                else:
                    deltaCols = 0
                    deltaRows = 0

                img.setPoints([img.getPoints()[0] + deltaCols, img.getPoints()[1] + deltaRows])
                if Distance(img.getPoints(), newPoints) < 5:
                    reachPoints = True
                    changeConfigTimer = getConfigTimer()
            cv2.imshow('window', Zoom(Translation(img.getCurrentFrame(),img.getPoints()),img.getZoom()))
        framesShown += 1 # used to calc the framerate

        # wait
        timeElapsed = (datetime.now() - loopTime).total_seconds()*1000 # in millis, used to calc the sleep delay between two frames
        if headlessMode :
            if timeElapsed < periodMillis : pygame.time.wait(int(periodMillis - timeElapsed))
        else :
            if cv2.waitKey(int(1000/maxFPS)) == 27: playing = False # exit on esc

        # set the playhead for the next frame
        if loopTime > changeProbaTimer :
            img.changePlayDirectionProba = random.choice(changePlayDirectionProbas)
            changeProbaTimer = getProbaTimer()
        img.step()

    # calculate FPS and exit graciously
    timeElapsed = (datetime.now()-timeStarted).total_seconds()
    if headlessMode : pygame.quit()
    else : cv2.destroyAllWindows()
    raise SystemExit(f"estimated FPS: {framesShown/timeElapsed:.02f}")
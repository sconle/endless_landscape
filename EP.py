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

headlessMode = False # True pour le lancer dans la console (sans bureau) ou False pour le lancer sous le bureau
# 1 proba fixe + 1 liste
# temps min et temps max
screenSize = [1920,1080] # taille de l'écran d'affichage (colonne, ligne)
changePlayDirectionProbas = [0.02, 0.032, 0.1, 0.01] # liste de probabilités
fixedPlayDirectionChangeProba = .01 # probabilité de changement de direction supplémentaire
changeProbasEvery = [4.0, 7.0] # temps en secondes (min et max, peu importe l'ordre) au bout duquel une nouvelle proba est tirée au sort
changeConfigEvery = [5.0, 5.1] # temps en seconde (min et max, peu importe l'ordre) au bout duquel une nouvelle config est choisie
maxFPS = 20 # nombre maximal d'images par secondes souhaitées
listZoom = [1, 2 ,2, 2, 2] # zoom moyen qu'atteint chaque config (1 correspond à aucun zoom)
listPoints = [(960,540),(0,0),(0,1080),(1920,0),(1920,1080)] # liste de coordonnées (colonne, ligne) des différents points d'interets
avgSpeed = 150 # vitesse moyenne de déplacement pendant les transitions en pixels par seconde
probaRandConfig = 0 # 0 que des configs provenant de la liste / 1 que de l'alea
randZoomInter = [1, 1.5] # intervalle dans lequel on ppeut piocher un zoom aléatoire


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
        self.zoom = listZoom[0]
        self.points = listPoints[0]
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
        old = self.config
        self.config = random.choice([i for i in range(len(listZoom))])

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
    if len(listZoom) != len(listPoints):
        raise SystemExit("La liste des zooms doit avoir la même longueur que celle des points")

    timeStarted, framesShown, = datetime.now(), 0
    changeProbaTimer = getProbaTimer()
    changeConfigTimer = getConfigTimer()
    periodMillis = int(1000/maxFPS)
    deltaZoom = 0
    newZoom, oldZoom = listZoom[img.getConfig()], img.getZoom()
    newPoints, oldPoints = listPoints[img.getConfig()], img.getPoints()
    distance = 0
    reachZoom, reachPoints = False, False

    # window or surface creation
    if headlessMode : surface = pygame.display.set_mode((0, 0), pygame.FULLSCREEN) # full screen
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
            image_surface = pygame.Surface((round(wnd_w / zoomSize), round(wnd_h / zoomSize)))
            image_surface.blit(image, (int(point[0] - (wnd_w)/2),int(point[1] - (wnd_h)/2)))
            image_surface = pygame.transform.scale(image_surface, (wnd_w, wnd_h))
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
            if random.random() < probaRandConfig:
                newZoom = numpy.random.uniform(randZoomInter[0],randZoomInter[1])
                if headlessMode:
                    newPoints = (numpy.random.randint(screenSize[0] / (2 * newZoom) - 1,img.getCurrentFrame().get_width() - screenSize[0] / (2 * newZoom) + 1), numpy.random.randint(screenSize[1] / (2 * newZoom) - 1,img.getCurrentFrame().get_height() - screenSize[1] / (2 * newZoom) + 1))
                else:
                    newPoints = (numpy.random.randint(screenSize[0]/(2*newZoom) - 1,img.getCurrentFrame().shape[1] - screenSize[0]/(2*newZoom) + 1),numpy.random.randint(screenSize[1]/(2*newZoom) - 1,img.getCurrentFrame().shape[0] - screenSize[1]/(2*newZoom) + 1))
            else:
                img.changeConfig()
                newPoints, newZoom = listPoints[img.getConfig()], listZoom[img.getConfig()]
            distance = Distance(newPoints, oldPoints)
            timeTransiConfig = distance / avgSpeed

            # changing Zoom
            if oldZoom > newZoom:
                timeTransiZoom = timeTransiConfig * 1.5
                if timeTransiZoom != 0:
                    deltaZoom = (newZoom - oldZoom)/(maxFPS*timeTransiZoom)
                else:
                    deltaZoom = 0
            else:
                timeTransiZoom = timeTransiConfig * 2/3
                if timeTransiZoom != 0:
                    deltaZoom = (newZoom - oldZoom)/(maxFPS*timeTransiZoom)
                else:
                    timeTransiZoom = 0

            changeConfigTimer = getConfigTimer() + timedelta(seconds=max(timeTransiZoom, timeTransiConfig))

        # display current frame
        if headlessMode :
            if not reachZoom:
                img.setZoom(img.getZoom() + deltaZoom)
                if (img.getZoom() < newZoom + 0.2) and (img.getZoom() > newZoom - 0.2):
                    reachZoom = True

            if not reachPoints:

                # update of the deltaCols, deltaRows each frame
                if Distance(img.getPoints(), newPoints) != 0:
                    deltaCols = numpy.sign(newPoints[0] - oldPoints[0]) * (avgSpeed / (maxFPS * img.getZoom())) * numpy.square((newPoints[0] - img.getPoints()[0]) / Distance(img.getPoints(), newPoints))
                    deltaRows = numpy.sign(newPoints[1] - oldPoints[1]) * (avgSpeed / (maxFPS * img.getZoom())) * numpy.square((newPoints[1] - img.getPoints()[1]) / Distance(img.getPoints(), newPoints))
                else:
                    deltaCols = 0
                    deltaRows = 0

                img.setPoints((img.getPoints()[0] + deltaCols, img.getPoints()[1] + deltaRows))

                if Distance(img.getPoints(), newPoints) < 5:
                    reachPoints = True


            surface.blit(ZoomTranslatHeadlessMode(img.getCurrentFrame(),img.getZoom(),img.getPoints()), (0,0))
            pygame.display.update()
            for event in pygame.event.get(): # exit on esc
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    playing = False
        else :
            if not reachZoom:
                img.setZoom(img.getZoom()+deltaZoom)
                if (img.getZoom() < newZoom + 0.2) and (img.getZoom() > newZoom - 0.2):
                    reachZoom = True

            if not reachPoints:

                # update of the deltaCols, deltaRows each frame
                if Distance(img.getPoints(), newPoints) != 0:
                    deltaCols = numpy.sign(newPoints[0] - oldPoints[0]) * (avgSpeed / (maxFPS * img.getZoom())) * numpy.square((newPoints[0] - img.getPoints()[0]) / Distance(img.getPoints(), newPoints))
                    deltaRows = numpy.sign(newPoints[1] - oldPoints[1]) * (avgSpeed / (maxFPS * img.getZoom())) * numpy.square((newPoints[1] - img.getPoints()[1]) / Distance(img.getPoints(), newPoints))
                else:
                    deltaCols = 0
                    deltaRows = 0

                img.setPoints((img.getPoints()[0] + deltaCols, img.getPoints()[1] + deltaRows))
                if Distance(img.getPoints(), newPoints) < 5:
                    reachPoints = True
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
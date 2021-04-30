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

import os, sys, cv2, random
from datetime import datetime, timedelta

headlessMode = True # True pour le lancer dans la console (sans bureau) ou False pour le lancer sous le bureau
# 1 proba fixe + 1 liste
# temps min et temps max
changePlayDirectionProbas = [0.02, 0.032, 0.1, 0.01] # liste de probabilités
fixedPlayDirectionChangeProba = .01 # probabilité de changement de direction supplémentaire
changeProbasEvery = [4.0, 7.0] # temps en secondes (min et max, peu importe l'ordre) au bout duquel une nouvelle proba est tirée au sort
maxFPS = 20 # nombre maximal d'images par secondes souhaitées

if headlessMode :
    import pygame, numpy
    os.putenv('SDL_VIDEODRIVER', 'fbcon')
    os.environ["SDL_FBDEV"] = "/dev/fb0"
    os.environ['SDL_NOMOUSE'] = '1'
    print("running in headless mode using framebuffer",os.environ["SDL_FBDEV"])
    pygame.init()
    pygame.mouse.set_visible(False)
else : import time

class ImageSequence():

    def __init__(self, videoFilename):
        self.frameCount = 0
        self.frames=[] # will store cv2.imread() for each frame
        self.extractFrames(videoFilename)
        self.currentFrameIndex = 0 # play head position
        self.playingForward = True
        self.changePlayDirectionProba = random.choice(changePlayDirectionProbas)

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

if __name__ == "__main__":
    if len(sys.argv) < 2 : raise SystemExit(f"usage : python3 {sys.argv[0]} monFichier.mp4")
    img = ImageSequence(sys.argv[1])

    timeStarted, framesShown, = datetime.now(), 0
    changeProbaTimer = getProbaTimer()
    periodMillis = int(1000/maxFPS)

    # window or surface creation
    if headlessMode : surface = pygame.display.set_mode((0, 0), pygame.FULLSCREEN) # full screen
    else :
        cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN) # full screen, no titlebar
        cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    playing = True

    # play loop
    while playing :
        loopTime = datetime.now()

        # display current frame
        if headlessMode :
            surface.blit(img.getCurrentFrame(), (0,0))
            pygame.display.update()
            for event in pygame.event.get(): # exit on esc
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    playing = False
        else : cv2.imshow('window', img.getCurrentFrame())
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
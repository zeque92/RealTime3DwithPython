# -*- coding: utf-8 -*-
import pygame
import numpy as np
from sys import exit
import TitleText
import TheStars
import SideEffectCube
import ShadowBobs
import TheGlobe
import BoxInABox
import MilkyWay
import TheWorld
import Raytracing
import Landscape
import EndCredits


class SoundVision:
    """
    Created on Sat Apr 10 18:10:33 2021

    Sound Vision Demo. Each demo part is a class and will be called from this main program.

    @author: Kalle
    """

    def __init__(self, screen, target_fps):
        self.screen = screen
        self.width = self.screen.get_size()[0]
        self.height = self.screen.get_size()[1]
        self.target_size = (self.width * 0.9, self.height * 0.9)  # target size for pictures
        self.screenCopy = screen.copy()
        self.music_file1 = "cats-eye.ogg"  # this mod by Jellybean is available at e.g. http://janeway.exotica.org.uk/release.php?id=45538
        self.music_file2 = "firepower.ogg"  # this mod by Jellybean is available at e.g. http://janeway.exotica.org.uk/release.php?id=45537
        self.music_file3 = "sinking2.ogg"  # this mod by Jellybean is available at e.g. http://janeway.exotica.org.uk/release.php?id=45536
        self.pic_file1 = "Alphaks.jpg"
        self.pic_file2 = "Guru.jpg"
        self.pic_file3 = "TheEnd.jpg"
        self.clock = pygame.time.Clock()
        self.target_fps = target_fps
        self.stop = False

    def run(self):

        # start music player
        pygame.mixer.music.load(self.music_file1)
        pygame.mixer.music.play(loops=-1)

        while not self.stop:

            # run the parts
            self.prepareScreen()
            self.showPicture(self.pic_file1, self.target_size, 1)
            self.wait(2500)
            if self.stop:
                break
            self.showPicture(self.pic_file1, self.target_size, 2)

            self.prepareScreen(False)
            self.stop = TitleText.TitleText(self.screen, self.target_fps).run()
            if self.stop:
                break

            # change tune
            pygame.mixer.music.fadeout(1500)
            self.prepareScreen(True)
            self.wait(1000)
            if self.stop:
                break
            pygame.mixer.music.load(self.music_file2)
            pygame.mixer.music.play(loops=-1)
            self.stop = TheStars.TheStars(self.screen, self.target_fps).run()
            if self.stop:
                break

            self.prepareScreen(False)
            self.stop = SideEffectCube.SideEffectCube(screen, self.target_fps, 25).run()
            if self.stop:
                break

            self.prepareScreen(True)
            self.stop = ShadowBobs.ShadowBobs(self.screen, 30).run()  # not target pfs = 30, optimised for max this
            if self.stop:
                break

            # re-start the tune, as looping does not work very well
            pygame.mixer.music.fadeout(1500)
            self.prepareScreen(True)
            self.wait(1000)
            if self.stop:
                break
            pygame.mixer.music.play(loops=-1)

            self.prepareScreen(True)
            self.stop = TheGlobe.TheGlobe(self.screen, self.target_fps, 25).run()
            if self.stop:
                break

            self.prepareScreen(True)
            self.stop = TheWorld.TheWorld(self.screen, self.target_fps).run()
            if self.stop:
                break

            self.prepareScreen(True)
            # change tune
            pygame.mixer.music.fadeout(1500)
            self.showPicture(self.pic_file2, self.target_size, 0)
            self.wait(1000)
            if self.stop:
                break
            pygame.mixer.music.load(self.music_file3)
            pygame.mixer.music.play(loops=-1)
            self.wait(2500)
            if self.stop:
                break

            self.prepareScreen(True)
            self.stop = Raytracing.Raytracing(self.screen, self.target_fps).run()
            if self.stop:
                break

            self.prepareScreen(True)
            self.stop = MilkyWay.MilkyWay(self.screen, self.target_fps, 25).run()
            if self.stop:
                break

            self.prepareScreen(True)
            self.stop = BoxInABox.BoxInABox(self.screen, self.target_fps, 25).run()
            if self.stop:
                break

            self.prepareScreen(True)
            self.stop = Landscape.Landscape(self.screen).run()
            if self.stop:
                break

            self.prepareScreen(True)
            self.stop = EndCredits.EndCredits(self.screen, self.target_fps).run()
            if self.stop:
                break

            self.prepareScreen(True)
            self.showPicture(self.pic_file3, (self.width, self.height), 0)
            self.wait(16000)

            self.stop = True  # do not loop from the start

        # exit; fade out, close display, stop music
        pygame.mixer.music.fadeout(2500)
        self.wait(2000)
        self.prepareScreen(True)

        pygame.quit()
        exit()

    def prepareScreen(self, fadeout=False):

        # prepare screen for next part.
        if fadeout:
            s = 30  # number of frames to use for fading
            orig_screen = self.screen.copy()  # get original image and store it here
            for i in range(s):
                # "multiply" current screen to fade it out with each step
                fadecol = int((s - i - 1) * 255.0 / s)
                self.screenCopy.fill((fadecol, fadecol, fadecol))
                self.screen.blit(orig_screen, (0, 0))
                self.screen.blit(self.screenCopy, (0, 0), None, pygame.BLEND_MULT)
                pygame.display.flip()
                self.clock.tick(self.target_fps)  # this keeps fadeout running at max target_fps
        self.screen.fill((0, 0, 0))

    def showPicture(self, picture_file, target_size, mode=0):

        # load and show image file.
        # target_size (width, height) is the basis for scaling the image, keeping its aspect ratio.
        # mode 0: fade in, mode 1: enter from left, mode 2: disappear to left

        surf = pygame.image.load(picture_file)
        surf.convert()
        image_size = surf.get_size()
        image_ratio = max(image_size[0] / target_size[0], image_size[1] / target_size[1])
        # scale image
        image_size = (int(image_size[0] / image_ratio), int(image_size[1] / image_ratio))
        surf2 = pygame.transform.scale(surf, image_size)
        image_position = (int((self.width - image_size[0]) / 2), int((self.height - image_size[1]) / 2))  # to center image on screen

        if mode == 0:
            surf3 = surf2.copy()
            for i in range(60):
                fadecol = int((i + 1) * 255.0 / 61)
                surf3.fill((fadecol, fadecol, fadecol))
                self.screen.blit(surf2, image_position)
                self.screen.blit(surf3, image_position, None, pygame.BLEND_MULT)
                pygame.display.flip()
                self.clock.tick(self.target_fps)  # this keeps fadeout running at max target_fps
        elif mode in (1, 2):
            s = 60.0
            for i in range(60):
                self.screen.fill((0, 0, 0))
                for j in range(image_size[1]):
                    # copy image line by line. If mode == 2 add 90 degrees --> change direction.
                    line_angle = (mode - 1) * 90.0 + max(0.0, min(90.0, (j * s / image_size[1]) - s + (i * 2 * (s/60)) * ((90.0 + s * 2) / (s * 2)) - s))
                    line_pos = np.sin(line_angle * np.pi / 180.0) * (image_position[0] + image_size[0]) - image_size[0]
                    if line_pos >= 0:
                        self.screen.blit(surf2, (line_pos, j + image_position[1]), (0, j, image_size[0], 1))
                    elif line_pos > -image_size[0]:
                        self.screen.blit(surf2, (0, j + image_position[1]), (-line_pos, j, image_size[0] + line_pos, 1))
                pygame.display.flip()
                self.clock.tick(self.target_fps)  # this keeps fadeout running at max target_fps
            self.screen.fill((0, 0, 0))
        # in the end just blit the pic
        if not mode == 2:
            self.screen.blit(surf2, image_position)
        pygame.display.flip()
        self.clock.tick(self.target_fps)  # this keeps fadeout running at max target_fps

    def wait(self, millisecs):

        # wait loop with key & mouse checks to enable skipping the wait
        start_time = pygame.time.get_ticks()
        running = True
        while running and pygame.time.get_ticks() - start_time <= millisecs:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    self.stop = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        self.stop = True
                    if event.key == pygame.K_s:
                        # save screen, at half the resolution, using class name as file name
                        pygame.image.save(pygame.transform.scale(self.screen, (int(self.screen.get_size()[0] / 2), int(self.screen.get_size()[1] / 2))),
                                          self.__class__.__name__ + '.jpg')
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        # left button
                        running = False
            pygame.time.wait(50)  # give up some CPU time


if __name__ == '__main__':
    """
    Prepare screen, etc.
    """

    # set screen size
    # first check available full screen modes
    pygame.display.init()
    # pick disp0lay mode from list or set a specific resolution
    # disp_modes = pygame.display.list_modes(0, pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE)
    # disp_size = disp_modes[9]  # selecting display size from available list. Assuming the 9th element is nice...
    disp_size = (1920, 1080)  # to force display size
    # disp_size = (1280, 720)
    target_fps = 60  # target frames per second - good to set to monitor refresh rate, but will affect the running speed of some parts.

    # set up screen
    pygame.display.set_caption('Sound Vision')
    screen = pygame.display.set_mode(disp_size)

    # inititalize fonts
    pygame.font.init()

    # initialize mixer
    pygame.mixer.init()

    SoundVision(screen, target_fps).run()

# -*- coding: utf-8 -*-
import pygame
from sys import exit


class EndCredits:
    """
    Displays end credits for parts with pictures.

    @author: kalle
    """

    def __init__(self, screen, target_fps):
        self.screen = screen
        self.screenCopy = self.screen.copy()
        self.width = self.screen.get_size()[0]
        self.height = self.screen.get_size()[1]
        self.screenSize = (self.width, self.height)
        self.midScreen = (int(self.width / 2), int(self.height / 2))
        self.textColor = (200, 200, 200)
        self.target_size = 0.48  # as % of screen size
        self.backgroundColor = (0, 0, 0)
        self.frameCount = 6 * 60    # nr of frames per "page"
        self.frameEntryCount = 60    # nr of frames to use for entry phase
        self.frameNr = 999999
        self.running = True
        self.stop = False
        self.target_fps = target_fps
        self.clock = pygame.time.Clock()

        self.imageList = [
            ['StripyVectors.jpg', ['Credits in Order of', 'Appearance:', '', '* Stripy *', '* Plane Vectors *', '', 'Coding by Zeque']],
            ['TitleText.jpg', ['* Metamorphing *', '* Title Text *', '', 'Coding by Zeque']],
            ['TheStars.jpg', ['* Movable 3D Stars *', '', 'Coding by Zeque']],
            ['SideEffectCube.jpg', ['* Side Effect Cube *', '', 'Amiga Coding by', 'Overlander', '', 'Python Coding by', 'Zeque']],
            ['ShadowBobs.jpg', ['* Shadowed Bobs *', '', 'Coding by Zeque']],
            ['TheGlobe.jpg', ['* The Globe *', '', 'Coding by Zeque']],
            ['TheWorld.jpg', ['* The World *', '', 'Coding and City Design', 'by Zeque']],
            ['MilkyWay.jpg', ['* The Milky Way *', '', 'Coding by Zeque']],
            ['BoxInABox.jpg', ['* Box in a Box *', '* in a Box... *', '', 'Coding by Zeque']],
            ['Landscape.jpg', ['* Fractal *', '* Landscapes *', '', 'Coding by Zeque']],
            ['Raytracing_01.jpg', ["* Raytracing *   by Overlander", "Logo by Nighthawk", "", "Music: 'Cats-eye', 'Firepower' and",
                                   "'Sinking' by Jellybean", '', "Pictures: 'Vengeance of Alphaks' and",
                                   "'Guru's Evening' by Red Baron", "'The End' by Nighthawk"]]
            ]
        self.imageList.reverse()  # reverse list for "popping"

    def run(self):

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    self.stop = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                        self.stop = True
                    if event.key == pygame.K_f:
                        self.toggleFullScreen()
                    if event.key == pygame.K_s:
                        # save screen, at half the resolution, using class name as file name
                        pygame.image.save(pygame.transform.scale(self.screen, (int(self.screen.get_size()[0] / 2), int(self.screen.get_size()[1] / 2))),
                                          self.__class__.__name__ + '.jpg')
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        # left button
                        self.running = False

            self.frameNr += 1
            if self.frameNr > self.frameCount:
                self.frameNr = 0
                # next image(s)
                if len(self.imageList) == 0:
                    self.running = False
                    break
                self.screenCopy.fill(self.backgroundColor)
                self.screenCopy.set_alpha(None)
                data = []
                image = []
                data.append(self.imageList.pop())
                image.append(self.loadImage(data[-1][0]))
                if len(self.imageList) > 0:
                    data.append(self.imageList.pop())
                    image.append(self.loadImage(data[-1][0]))
                    self.addTexts(data[0][1], (self.width * 0.75, self.height * 0.25))
                    self.addTexts(data[1][1], (self.width * 0.25, self.height * 0.75))
                else:
                    self.addTexts(data[0][1], (self.width * 0.50, self.height * 0.75))  # one image only

            if self.frameNr <= self.frameEntryCount:
                self.screen.fill(self.backgroundColor)
                if len(data) < 2:
                    # one image only - fade in
                    image[0].set_alpha(int(255 * self.frameNr / self.frameEntryCount))
                    self.screen.blit(image[0], (int(self.midScreen[0] - image[0].get_size()[0] / 2), int(0.25 * self.height - image[0].get_size()[1] / 2)))
                    self.screenCopy.set_alpha(int(255 * self.frameNr / self.frameEntryCount))
                    self.screen.blit(self.screenCopy, (0, self.midScreen[1]), (0, self.midScreen[1], self.width - 1, self.midScreen[1] - 1))
                else:
                    # two images - bring in from sides, show text top down
                    for i in range(2):
                        image_size = image[i].get_size()
                        x_pos = int(self.midScreen[0] * (i + 0.5) - image_size[0] / 2
                                    - (2 * i - 1) * (self.midScreen[0] + image_size[0]) * (1.0 - min(1.0, 2.0 * self.frameNr / self.frameEntryCount)))
                        self.screen.blit(image[i], (x_pos, int((i / 2 + 0.25) * self.height - image_size[1] / 2)))
                        s_lines = int(self.midScreen[1] * 2.0 * (self.frameNr / self.frameEntryCount - 0.5))
                        if s_lines > 0:
                            self.screen.blit(self.screenCopy, (-(i - 1) * self.midScreen[0], i * self.midScreen[1]),
                                             (-(i - 1) * self.midScreen[0], i * self.midScreen[1], self.midScreen[0] - 1, s_lines))
            elif self.frameNr >= self.frameCount - int(self.frameEntryCount / 2):
                # fade out
                if self.frameNr == self.frameCount - int(self.frameEntryCount / 2):
                    self.screenCopy = self.screen.copy()  # take a copy of the screen
                self.screenCopy.set_alpha(int(255 * (self.frameCount - self.frameNr) / int(self.frameEntryCount / 2)))
                self.screen.fill(self.backgroundColor)
                self.screen.blit(self.screenCopy, (0, 0))

            self.clock.tick(self.target_fps)  # this keeps code running at max target_fps
            # switch between currently showed and the next screen (prepared in "buffer")
            pygame.display.flip()

        return self.stop

    def loadImage(self, picture_file):

        # load image file, scale it, and return as a surface
        target_size = (self.target_size * self.width, self.target_size * self.height)
        surf = pygame.image.load(picture_file)
        surf.convert()
        image_size = surf.get_size()
        image_ratio = max(image_size[0] / target_size[0], image_size[1] / target_size[1])
        # scale image
        image_size = (int(image_size[0] / image_ratio), int(image_size[1] / image_ratio))
        return pygame.transform.scale(surf, image_size)

    def addTexts(self, texts, center):

        # write texts to screenCopy for later blitting to screen
        font_size = int(self.height / 25)
        self.font = pygame.font.SysFont('GillSansUltraCondensed,Arial', font_size)   # initialize font
        y = int(center[1] - len(texts) * int(font_size * 1.1) / 2)
        for text in texts:
            if text != '':
                f_screen = self.font.render(text, True, self.textColor, self.backgroundColor)
                self.screenCopy.blit(f_screen, (int(center[0] - f_screen.get_size()[0] / 2), y))
            y += int(font_size * 1.1)

    def toggleFullScreen(self):

        # toggle between full screen and windowed mode
        pygame.display.toggle_fullscreen()


if __name__ == '__main__':
    """
    Prepare screen, etc.
    """

    # set screen size
    # first check available full screen modes
    pygame.display.init()
    # pick disp0lay mode from list or set a specific resolution
    disp_modes = pygame.display.list_modes(0, pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE)
    # disp_size = disp_modes[0] # selecting display size from available list. Assuming the 9th element is nice...
    disp_size = (1920, 1080)  # to force display size
    # disp_size = (800, 600)
    pygame.font.init()

    screen = pygame.display.set_mode(disp_size)
    pygame.display.set_caption('End Credits')
    pygame.font.init()

    EndCredits(screen, 60).run()

    # exit; close display, stop music
    pygame.quit()
    exit()

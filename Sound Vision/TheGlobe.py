# -*- coding: utf-8 -*-
import pygame
import numpy as np
from os import chdir
from sys import exit


class TheGlobe:
    """
    Displays a rotating globe.

    @author: kalle
    """

    def __init__(self, screen, target_fps, run_seconds):
        self.run_seconds = run_seconds
        self.screen = screen
        self.screenGlobe = self.screen.copy()
        self.screenShade = self.screen.copy()
        self.screenSize = self.screen.get_size()
        self.width = self.screenSize[0]
        self.height = self.screenSize[1]
        self.midScreen = (int(self.width / 2), int(self.height / 2))
        self.font = pygame.font.SysFont('CourierNew', 14)   # initialize font and set font size
        self.backgroundColor = (0, 0, 0)
        self.landColor = (0, 255, 40)
        self.seaColor = (0, 40, 255)
        self.lightAngles = np.array([35, 0, -135])                 # light source angle (in degrees)
        self.lightRNodes = np.zeros((3, 3))                 # light source vectors rotated
        self.min_col = 20                                   # minimum shade color brightness
        self.max_col = 255                                  # maximum shade color brightness
        self.mid_col = (self.max_col + self.min_col) / 2 - self.min_col
        self.max_d = 0.97                                   # the percent of radius to use - no point going to 100 %
        self.angles = np.array([1.0, 0.5, 0.0])
        self.rotation = np.array([1.3, -0.7, 1.7]) * (np.pi / 180)  # rotation in degrees, converted to radius
        self.nodes = np.zeros((0, 3))
        self.rotatedNodes = np.zeros((0, 3))
        self.surfaces = []                                  # surfaces define nodes simply as (start node, nr of nodes)
        self.globeSize = 0
        self.globeQtr = 0.0
        self.globePosition = (0, 0)
        self.frameNr = 0
        self.landFadeFrameCount = 90
        self.landFadeFrameNr = 0
        self.clock = pygame.time.Clock()
        self.target_fps = target_fps                        # sets maximum refresh rate
        self.phase = 0
        self.running = True
        self.stop = False
        # the following for checking performance only
        self.infoDisplay = False
        self.paused = False
        self.millisecs = 0
        self.timer_avg_frames = 60
        self.timer_names = []
        self.timers = np.zeros((1, 1), dtype=int)
        self.timer_frame = 0
        self.start_timer = 0

        # set up timers
        self.timer_names.append("rotate")
        self.timer_names.append("draw globe")
        self.timer_names.append("draw land")
        self.timer_names.append("draw shade")
        self.timer_names.append("apply shade")
        self.timer_names.append("plot info")
        self.timer_names.append("display flip")
        self.timer_names.append("wait")
        self.timers = np.zeros((len(self.timer_names), self.timer_avg_frames), dtype=int)

        # initialize timers
        self.start_timer = pygame.time.get_ticks()
        self.millisecs = self.start_timer

        self.setupGlobeData()

    def run(self):

        self.setupGlobe()

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
                    if event.key == pygame.K_SPACE:
                        if self.paused:
                            self.paused = False
                        else:
                            self.paused = True
                    if event.key == pygame.K_s:
                        # save screen, at half the resolution, using class name as file name
                        pygame.image.save(pygame.transform.scale(self.screen, (int(self.screen.get_size()[0] / 2), int(self.screen.get_size()[1] / 2))),
                                          self.__class__.__name__ + '.jpg')
                    if event.key == pygame.K_i:
                        if self.infoDisplay:
                            self.infoDisplay = False
                        else:
                            self.infoDisplay = True
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        # left button: exit
                        self.running = False

            self.drawGlobe()

            if self.frameNr * 2 + self.min_col <= self.max_col:
                self.drawShade(self.frameNr * 2 + self.min_col)
                if self.frameNr * 2 + 1 + self.min_col <= self.max_col:
                    self.drawShade(self.frameNr * 2 + 1 + self.min_col)
                self.measureTime("draw shade")
            else:
                self.measureTime("draw shade")
                if not self.paused:
                    self.rotate()
                self.drawLand()

            self.addShade()

            if self.infoDisplay:
                self.plotInfo()

            self.frameNr += 1

            # switch between currently showed and the next screen (prepared in "buffer")
            pygame.display.flip()
            self.measureTime("display flip")
            self.clock.tick(self.target_fps)  # this keeps code running at max target_fps
            self.measureTime("wait")

            if pygame.time.get_ticks() > self.start_timer + 1000 * self.run_seconds:
                self.running = False

        return self.stop

    def setupGlobe(self):
        """
        Set up an empty globe (ocean) for drawing the continents on.
        Also, set up the shading globe.
        """
        self.screen.fill(self.backgroundColor)

        self.screenGlobe.fill(self.backgroundColor)
        pygame.draw.circle(self.screenGlobe, self.seaColor, self.midScreen, self.globeSize, 0)

        self.screenShade.fill(self.backgroundColor)

        # use orthogonal nodes at globe sides and rotate them to light angles
        lightNodes = np.array([
            [self.globeSize, 0, 0],
            [0, self.globeSize, 0],
            [0, 0, self.globeSize]
            ])
        self.lightRNodes = np.dot(lightNodes, self.rotateMatrix(self.lightAngles * np.pi / 180.0))

    def drawShade(self, layer):
        """
        Draw a filled ellipse with center on the Y axis of the globe to create a gray shaded ball. layer represents color as well.
        Works properly only when the light source is closer to viewer than the globe.
        """

        shade = int(layer ** 2 / 255)  # to get darker shading
        yd = ((layer - self.min_col - self.mid_col) * self.max_d / self.mid_col)  # percentage of y distance from origo
        yrad = np.sqrt(1.0 - yd ** 2)  # globe radius percentage at y latitude - from yd^2 + yrad^2 = 1 for a circle
        points = int(yrad * self.globeSize * 2 * np.pi / 6 + 3)  # draw ~6 pixel line segments - nr of points needed (approximate, depends on angles etc.)
        y = yd * self.lightRNodes[1]
        node_list = []
        node_list2 = []
        for j in range(points):
            x = yrad * np.sin(np.pi * 2 * j / points) * self.lightRNodes[0]
            z = yrad * np.cos(np.pi * 2 * j / points) * self.lightRNodes[2]
            xyz = x + y + z
            xyz2 = y + (x + z) * ((yrad - 0.15) / yrad)  # an inner circle
            node_list.append((int(xyz[0] + self.midScreen[0] + 0.5), int(xyz[1] + self.midScreen[1]) + 0.5))
            node_list2.append((int(xyz2[0] + self.midScreen[0] + 0.5), int(xyz2[1] + self.midScreen[1]) + 0.5))
        # draw it as polygon
        if yrad > 0.17 and yd < self.max_d:
            node_list.append(node_list[0])
            node_list2.append(node_list2[0])
            pygame.draw.polygon(self.screenShade, (shade, shade, shade), node_list + node_list2, 0)  # clear inner circle
        else:
            pygame.draw.polygon(self.screenShade, (shade, shade, shade), node_list, 0)

    def drawGlobe(self):
        """
        Copy the empty globe to screen; bounce it.
        """

        self.globePos = (self.midScreen[0], int(self.midScreen[1] * (1.2 - 0.4 * abs(np.sin(self.frameNr / 60)))))
        self.screen.blit(self.screenGlobe, (self.globePos[0] - self.globeSize, self.globePos[1] - self.globeSize - 10),
                         (self.midScreen[0] - self.globeSize, self.midScreen[1] - self.globeSize - 10, self.globeSize * 2 + 1, self.globeSize * 2 + 20))
        self.measureTime("draw globe")

    def rotate(self):
        """
        Rotate the nodes. First calculate rotation matrix.
        """

        # add rotation to angles
        self.angles += self.rotation

        # rotate all nodes
        self.rotatedNodes = np.dot(self.nodes, self.rotateMatrix(self.angles))
        self.measureTime("rotate")

    def rotateMatrix(self, angles):
        """
        Calculate and return rotation matrix.
        """
        # set matrix for rotation using angles.
        (sx, sy, sz) = np.sin(angles)
        (cx, cy, cz) = np.cos(angles)

        # build a matrix for X, Y, Z rotation (in that order, see https://en.wikipedia.org/wiki/Euler_angles).
        return np.array([[cy * cz               , -cy * sz              , sy      ],
                           [cx * sz + cz * sx * sy, cx * cz - sx * sy * sz, -cy * sx],
                           [sx * sz - cx * cz * sy, cz * sx + cx * sy * sz, cx * cy ]])

    def drawLand(self):
        """
        Draw the land mass / continents.
        """

        # set color applying fading from seaColor at the beginning
        if self.landFadeFrameNr < self.landFadeFrameCount:
            color = ([
                int(self.landFadeFrameNr / self.landFadeFrameCount * self.landColor[i]
                    + (1.0 - self.landFadeFrameNr / self.landFadeFrameCount) * self.seaColor[i])
                for i in range(3)
                ])
            halfColor = ([
                int(self.landFadeFrameNr / self.landFadeFrameCount * self.landColor[i] * 0.5
                    + (1.0 - self.landFadeFrameNr / self.landFadeFrameCount) * self.seaColor[i])
                for i in range(3)
                ])
            self.landFadeFrameNr += 1
        else:
            color = self.landColor
            halfColor = ([int(x * 0.5) for x in color])

        for surface in self.surfaces:
            beg_node = surface[0]
            end_node = beg_node + surface[1]
            # test if any part of surface is visible ie. z coordinate negative - only then draw
            if min(self.rotatedNodes[beg_node:end_node, 2]) < 0:
                if surface == self.surfaces[-1]:
                    # for last surface drawn, use half tone color
                    color = halfColor
                node_list = []
                edge_needed = False
                prev_node = self.rotatedNodes[end_node - 1, :]
                for node in self.rotatedNodes[beg_node: end_node, :]:
                    if prev_node[2] < 0:
                        # previous node is visible
                        if node[2] < 0:
                            # current node is visible
                            node_list.append((int(node[0] + self.globePos[0]), int(node[1] + self.globePos[1])))
                        else:
                            # current node is not visible. Approximate "edge" by calculating a node at z=0 and draw to that point
                            z0_node = prev_node + (prev_node[2] / (prev_node[2] - node[2])) * (node - prev_node)
                            node_list.append((int(z0_node[0] + self.globePos[0]), int(z0_node[1] + self.globePos[1])))
                    else:
                        # previous node is not visible
                        if node[2] < 0:
                            # current node is visible
                            z0_node = prev_node + (prev_node[2] / (prev_node[2] - node[2])) * (node - prev_node)
                            # if list already has nodes, add "globe square" edge nodes depending on quadrants.
                            if len(node_list) > 0:
                                pz0_node = (node_list[-1][0] - self.globePos[0], node_list[-1][1] - self.globePos[1])
                                self.edgeNode(node_list, pz0_node, z0_node)
                            else:
                                # make a note: need edge nodes at the end, as unable to add them now as previous node unknown.
                                edge_needed = True
                            node_list.append((int(z0_node[0] + self.globePos[0]), int(z0_node[1] + self.globePos[1])))
                            node_list.append((int(node[0] + self.globePos[0]), int(node[1] + self.globePos[1])))
                        # if both nodes not visible then add nothing
                    prev_node = node
                # check if started from behind and edge nodes needed.
                if edge_needed:
                    pz0_node = (node_list[-1][0] - self.globePos[0], node_list[-1][1] - self.globePos[1])
                    z0_node = (node_list[0][0] - self.globePos[0], node_list[0][1] - self.globePos[1])
                    self.edgeNode(node_list, pz0_node, z0_node)
                # draw
                if len(node_list) > 2:
                    pygame.draw.polygon(self.screen, color, node_list)

        self.measureTime("draw land")

    def edgeNode(self, node_list, pz0_node, z0_node):

        # adds nodes when they cross the edge. pz0 is the previous edge node and z0 the current edge node.
        if abs(pz0_node[1]) >= self.globeQtr:
            # prev node in up or down quarter - add an up or down node
            pz0_edge = (pz0_node[0], np.sign(pz0_node[1]) * self.globeSize)
        else:
            # prev node in left or right qtr
            pz0_edge = (np.sign(pz0_node[0]) * self.globeSize, pz0_node[1])
        node_list.append((pz0_edge[0] + self.globePos[0], pz0_edge[1] + self.globePos[1]))
        # add edge node using the current node - plus the required corner nodes.
        if abs(z0_node[1]) >= self.globeQtr:
            # node in up or down quarter - add an up or down node
            z0_edge = (z0_node[0], np.sign(z0_node[1]) * self.globeSize)
            if pz0_edge[1] != z0_edge[1]:
                # different quarter - needs corner node(s)
                if pz0_edge[1] == -z0_edge[1]:
                    # different halves - needs second corner node
                    node_list.append((int(np.sign(z0_edge[0] + pz0_edge[0]) * self.globeSize + self.globePos[0]), int(pz0_edge[1] + self.globePos[1])))
                node_list.append((int(np.sign(z0_edge[0] + pz0_edge[0]) * self.globeSize + self.globePos[0]), int(z0_edge[1] + self.globePos[1])))
        else:
            # prev node in left or right qtr
            z0_edge = (np.sign(z0_node[0]) * self.globeSize, z0_node[1])
            if pz0_edge[0] != z0_edge[0]:
                # different quarter - needs corner node(s)
                if pz0_edge[0] == -z0_edge[0]:
                    # different halves - needs second corner node
                    node_list.append((int(pz0_edge[0] + self.globePos[0]), int(np.sign(z0_edge[1] + pz0_edge[1]) * self.globeSize + self.globePos[1])))
                node_list.append((int(z0_edge[0] + self.globePos[0]), int(np.sign(z0_edge[1] + pz0_edge[1]) * self.globeSize + self.globePos[1])))
        node_list.append((z0_edge[0] + self.globePos[0], z0_edge[1] + self.globePos[1]))

    def addShade(self):
        """
        Add the shade by blending it on the two-color globe.
        """

        self.screen.blit(self.screenShade, (self.globePos[0] - self.globeSize, self.globePos[1] - self.globeSize),
                         (self.midScreen[0] - self.globeSize, self.midScreen[1] - self.globeSize, self.globeSize * 2 + 1, self.globeSize * 2 + 1),
                         pygame.BLEND_RGB_MULT)
        self.measureTime("apply shade")

    def plotInfo(self):
        """
        Add use instructions & data on screen.
        This can obviously be skipped, just for information.
        """

        # release any locks on screen
        while self.screen.get_locked():
            self.screen.unlock()

        pygame.draw.rect(self.screen, self.backgroundColor, (10, 10, 200, 180), 0)
        # add Frames Per Second
        fps = self.clock.get_fps()  # avg frame rate using the last ten frames
        self.plotInfoBlit(10,  10, ("fps" + ' '*16)[:16] + (' '*10 + str(round(fps, 1)))[-7:])
        tot_time = np.sum(self.timers)
        # add measured times as percentage of total
        if tot_time > 0:
            for i in range(len(self.timer_names)):
                self.plotInfoBlit(10, 25 + i * 15, (self.timer_names[i] + ' '*16)[:16]
                                  + (' '*10 + str(round(np.sum(self.timers[i, :]) * 100 / tot_time, 1)))[-7:])
        self.measureTime("plot info")

    def plotInfoBlit(self, x, y, msg):

        f_screen = self.font.render(msg, False, [255, 255, 255])
        self.screen.blit(f_screen, (x, y))

    def toggleFullScreen(self):

        # toggle between full screen and windowed mode
        pygame.display.toggle_fullscreen()

    def measureTime(self, timer_name):
        # add time elapsed from previous call to selected timer
        i = self.timer_names.index(timer_name)
        new_time = pygame.time.get_ticks()
        self.timers[i, self.timer_frame] += (new_time - self.millisecs)
        self.millisecs = new_time

    def nextTimeFrame(self):
        # move to next timer and clear data
        self.timer_frame += 1
        if self.timer_frame >= self.timer_avg_frames:
            self.timer_frame = 0
        self.timers[:, self.timer_frame] = 0

    def setupGlobeData(self):

        # prepare nodes and surfaces.
        # maps represents a list of continents and islands, with Mercator projection coordinates with respect to image size.
        image_size = (1320, 1320)
        maps = [
            [[ 493,  385],
             [ 472,  356],
             [ 473,  293],
             [ 441,  220],
             [ 406,  218],
             [ 393,  179],
             [ 448,  103],
             [ 537,   59],
             [ 597,  140],
             [ 573,  302],
             [ 509,  347],
             [ 501,  381]],
            [[ 632,  443],
             [ 619,  441],
             [ 625,  428],
             [ 635,  422]],
            [[ 642,  448],
             [ 643,  423],
             [ 636,  410],
             [ 642,  395],
             [ 654,  399],
             [ 655,  417],
             [ 664,  436]],
            [[ 648,  453],
             [ 669,  442],
             [ 692,  423],
             [ 693,  406],
             [ 700,  404],
             [ 700,  422],
             [ 733,  416],
             [ 745,  396],
             [ 771,  385],
             [ 771,  376],
             [ 750,  385],
             [ 740,  380],
             [ 740,  364],
             [ 754,  345],
             [ 749,  339],
             [ 739,  346],
             [ 727,  374],
             [ 729,  384],
             [ 719,  410],
             [ 706,  412],
             [ 700,  393],
             [ 684,  400],
             [ 679,  377],
             [ 699,  359],
             [ 712,  327],
             [ 748,  288],
             [ 771,  288],
             [ 809,  319],
             [ 807,  333],
             [ 784,  333],
             [ 790,  348],
             [ 812,  338],
             [ 856,  311],
             [ 902,  291],
             [ 919,  261],
             [ 938,  280],
             [ 987,  233],
             [1051,  198],
             [1072,  228],
             [1060,  257],
             [1116,  250],
             [1152,  282],
             [1184,  263],
             [1248,  295],
             [1319,  302],
             [1319,  357],
             [1263,  392],
             [1237,  447],
             [1230,  413],
             [1258,  376],
             [1259,  364],
             [1221,  388],
             [1183,  384],
             [1162,  418],
             [1180,  437],
             [1167,  475],
             [1129,  496],
             [1138,  515],
             [1126,  526],
             [1122,  500],
             [1099,  500],
             [1104,  548],
             [1089,  574],
             [1049,  583],
             [1063,  612],
             [1053,  623],
             [1038,  620],
             [1026,  609],
             [1043,  653],
             [1029,  652],
             [1012,  590],
             [ 991,  578],
             [ 974,  585],
             [ 957,  603],
             [ 945,  636],
             [ 926,  584],
             [ 904,  563],
             [ 870,  564],
             [ 846,  549],
             [ 837,  552],
             [ 855,  568],
             [ 869,  567],
             [ 878,  581],
             [ 866,  596],
             [ 821,  610],
             [ 816,  592],
             [ 787,  550],
             [ 782,  556],
             [ 816,  621],
             [ 856,  614],
             [ 838,  647],
             [ 807,  675],
             [ 811,  721],
             [ 792,  737],
             [ 788,  759],
             [ 761,  787],
             [ 728,  793],
             [ 716,  768],
             [ 713,  740],
             [ 700,  729],
             [ 711,  691],
             [ 688,  666],
             [ 694,  646],
             [ 667,  638],
             [ 649,  646],
             [ 625,  648],
             [ 596,  622],
             [ 597,  571],
             [ 642,  526],
             [ 696,  515],
             [ 729,  540],
             [ 745,  533],
             [ 783,  536],
             [ 791,  515],
             [ 766,  516],
             [ 760,  502],
             [ 784,  488],
             [ 798,  493],
             [ 816,  489],
             [ 795,  476],
             [ 771,  480],
             [ 746,  515],
             [ 732,  491],
             [ 708,  474],
             [ 705,  478],
             [ 729,  503],
             [ 715,  513],
             [ 714,  502],
             [ 694,  483],
             [ 669,  489],
             [ 656,  513],
             [ 623,  515],
             [ 628,  483],
             [ 655,  482],
             [ 659,  467]],
            [[1138,  526],
             [1152,  526],
             [1137,  538]],
            [[1158,  519],
             [1172,  505],
             [1181,  477],
             [1187,  479],
             [1180,  512],
             [1157,  530]],
            [[1070,  750],
             [1122,  712],
             [1153,  704],
             [1176,  728],
             [1188,  704],
             [1217,  756],
             [1220,  778],
             [1201,  814],
             [1175,  811],
             [1145,  783],
             [1112,  790],
             [1096,  799],
             [1075,  793],
             [1082,  776]],
            [[1269,  849],
             [1294,  821],
             [1298,  832],
             [1281,  852]],
            [[1302,  825],
             [1301,  815],
             [1313,  808]],
            [[1200,  698],
             [1192,  688],
             [1143,  671],
             [1146,  665],
             [1201,  681],
             [1208,  698]],
            [[1123,  693],
             [1098,  699],
             [1048,  687],
             [1011,  642],
             [1039,  664],
             [1050,  682],
             [1100,  695]],
            [[1061,  655],
             [1089,  636],
             [1095,  654],
             [1085,  673],
             [1063,  670]],
            [[1101,  662],
             [1097,  677],
             [1111,  679]],
            [[1114,  632],
             [1122,  640],
             [1121,  630]],
            [[1102,  593],
             [1101,  603],
             [1107,  598]],
            [[ 846,  715],
             [ 832,  758],
             [ 821,  759],
             [ 819,  739],
             [ 841,  707]],
            [[ 954,  637],
             [ 960,  638],
             [ 957,  632]],
            [[ 692,  493],
             [ 692,  504],
             [ 697,  500]],
            [[ 862,  285],
             [ 853,  274],
             [ 873,  227],
             [ 912,  203],
             [ 907,  215],
             [ 878,  241],
             [ 863,  274]],
            [[ 587,  358],
             [ 609,  342],
             [ 602,  337],
             [ 585,  337],
             [ 575,  348],
             [ 582,  351]],
            [[ 417,  906],
             [ 396,  903],
             [ 382,  869],
             [ 405,  731],
             [ 384,  719],
             [ 364,  677],
             [ 370,  657],
             [ 393,  617],
             [ 440,  625],
             [ 491,  660],
             [ 536,  684],
             [ 531,  696],
             [ 513,  716],
             [ 515,  740],
             [ 481,  756],
             [ 480,  774],
             [ 451,  796],
             [ 448,  817],
             [ 421,  825],
             [ 409,  868],
             [ 410,  889]],
            [[ 378,  640],
             [ 351,  632],
             [ 310,  601],
             [ 269,  590],
             [ 230,  543],
             [ 201,  492],
             [ 210,  454],
             [ 158,  395],
             [ 107,  380],
             [  89,  397],
             [  54,  371],
             [  71,  329],
             [  53,  310],
             [  89,  285],
             [ 115,  297],
             [ 165,  304],
             [ 203,  291],
             [ 244,  315],
             [ 305,  314],
             [ 325,  295],
             [ 343,  336],
             [ 319,  382],
             [ 330,  400],
             [ 369,  432],
             [ 379,  403],
             [ 375,  359],
             [ 401,  376],
             [ 408,  394],
             [ 423,  383],
             [ 453,  423],
             [ 464,  459],
             [ 439,  460],
             [ 434,  445],
             [ 418,  453],
             [ 425,  471],
             [ 391,  496],
             [ 385,  521],
             [ 367,  536],
             [ 370,  568],
             [ 340,  546],
             [ 307,  558],
             [ 308,  583],
             [ 318,  591],
             [ 341,  577],
             [ 342,  598],
             [ 354,  602],
             [ 358,  618],
             [ 382,  633]],
            [[ 386,  592],
             [ 405,  594],
             [ 401,  585],
             [ 386,  588]],
            [[   1, 1292],
             [  59, 1319],
             [ 108, 1319],
             [  64, 1274],
             [ 118, 1173],
             [  75, 1118],
             [ 121, 1125],
             [ 130, 1098],
             [ 249, 1082],
             [ 292, 1090],
             [ 296, 1069],
             [ 401, 1061],
             [ 412, 1003],
             [ 447,  966],
             [ 425, 1009],
             [ 438, 1073],
             [ 406, 1106],
             [ 356, 1137],
             [ 390, 1190],
             [ 437, 1258],
             [ 511, 1202],
             [ 549, 1157],
             [ 526, 1133],
             [ 639, 1038],
             [ 759, 1034],
             [ 862,  982],
             [ 918, 1006],
             [ 904, 1065],
             [ 962, 1000],
             [1070,  988],
             [1188,  994],
             [1284, 1043],
             [1256, 1097],
             [1274, 1138],
             [1247, 1163],
             [1264, 1230],
             [1319, 1285]],
            [[ 414,  367],
             [ 433,  330],
             [ 369,  253],
             [ 335,  253],
             [ 353,  335],
             [ 368,  299],
             [ 395,  324],
             [ 377,  344]],
            [[ 284,  299],
             [ 235,  300],
             [ 203,  271],
             [ 211,  241],
             [ 258,  265],
             [ 281,  254]],
            [[ 366,  235],
             [ 384,  166],
             [ 432,   92],
             [ 399,   72],
             [ 328,  111],
             [ 353,  170],
             [ 346,  228]],
            [[ 337,  178],
             [ 339,  158],
             [ 318,  125],
             [ 317,  165]],
            [[ 717,  204],
             [ 718,  175],
             [ 704,  159],
             [ 726,  146],
             [ 758,  151],
             [ 732,  168],
             [ 745,  193],
             [ 725,  180]],
            [[  90,  583],
             [  84,  591],
             [  93,  588]],
            [[ 349,  579],
             [ 367,  573],
             [ 383,  584],
             [ 367,  584]],
            [[ 771,  376],
             [ 750,  385],
             [ 740,  380],
             [ 740,  364],
             [ 755,  335],
             [ 751,  332],
             [ 751,  315],
             [ 745,  305],
             [ 749,  301],
             [ 754,  311],
             [ 763,  310],
             [ 767,  298],
             [ 775,  300],
             [ 771,  316],
             [ 774,  321],
             [ 770,  328],
             [ 773,  343],
             [ 774,  353],
             [ 778,  357]]
            ]

        # transform x, y coordinates to spherical angles (azimuth, inclination). x is straightforward; y see https://en.wikipedia.org/wiki/Mercator_projection
        R = 1.0 / np.log(np.tan(np.pi / 4 + (85.0 / 90.0) * np.pi / 4))  # with this R y is in [-1,1] between -85 and +85 degrees
        self.globeSize = int(min(self.width, self.height) / 3 + 0.5)
        self.globeQtr = np.sin(np.pi / 4) * self.globeSize
        for m in maps:
            ma = np.asarray(m, dtype=float)
            # transform to angles (radius): x = [0, pi) and y = (-pi/4, pi/4)
            ma[:, 0] = np.pi * 2.0 * ma[:, 0] / image_size[0]
            ma[:, 1] = 2.0 * np.arctan(np.exp(((ma[:, 1] / (image_size[1] / 2)) - 1) / R)) - np.pi / 2
            # transform from angles to 3D cartesian coordinates X,Y,Z (Z being distance)
            nodes = np.zeros((np.shape(ma)[0], 3))
            nodes[:, 0] = self.globeSize * np.cos(ma[:, 1]) * np.cos(ma[:, 0])
            nodes[:, 1] = self.globeSize * np.sin(ma[:, 1])
            nodes[:, 2] = self.globeSize * np.cos(ma[:, 1]) * np.sin(ma[:, 0])
            # add to nodes and surfaces
            node_nr = np.shape(self.nodes)[0]
            if node_nr == 0:
                self.nodes = nodes[:, :]
                manodes = ma[:, :]
            else:
                self.nodes = np.vstack((self.nodes, nodes))
                manodes = np.vstack((manodes, ma))
            self.surfaces.append((node_nr, np.shape(nodes)[0]))


if __name__ == '__main__':
    """
    Prepare screen, etc.
    """

    # set data directory
    chdir("C:/Users/Kalle/OneDrive/Asiakirjat/Python")

    # set screen size
    # first check available full screen modes
    pygame.display.init()
    # pick disp0lay mode from list or set a specific resolution
    disp_modes = pygame.display.list_modes(0, pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE)
    # disp_size = disp_modes[9] # selecting display size from available list. Assuming the 9th element is nice...
    disp_size = (1600, 1200)  # to force display size
    pygame.font.init()

    pygame.display.set_caption('The Globe')
    screen = pygame.display.set_mode(disp_size)

    # run the main program
    TheGlobe(screen, 60, 120).run()

    # exit
    pygame.quit()
    exit()

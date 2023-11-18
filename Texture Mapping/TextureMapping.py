# -*- coding: utf-8 -*-
import pygame
import numpy as np
from sys import exit


class TextureMapping:
    """
    Mapping an image on a 3D rotating vector object.

    @author: kalle
    """

    def __init__(self, screen, target_fps):

        self.tutorial_mode = False                           # if True then show picture and sections for a cube, if not then don't and switch between shapes
        self.screen = screen
        self.screen_copy = self.screen.copy()
        self.background_color = (0, 0, 0)
        self.width = self.screen.get_size()[0]
        self.height = self.screen.get_size()[1]
        self.full_screen = False
        self.mode = 1                                       # mode 1: cube; mode 2: dodecahedron; mode 3: icosahedron
        self.mode_timer = pygame.time.get_ticks()
        self.mode_time = 10000                              # time to show each mode in milliseconds
        self.fade_time = 1000                               # time to fade in/out milliseconds
        self.phase = 1
        if self.tutorial_mode:
            self.mid_screen = np.array([int(2 * self.width / 3), int(self.height / 2)], dtype=float)
        else:
            self.mid_screen = np.array([int(self.width / 2), int(self.height / 2)], dtype=float)
        self.z_scale = self.height * 2.2                    # scaling for z coordinates
        self.z_pos = 1000.0
        self.target_fps = target_fps
        self.running = True
        self.paused = False
        self.clock = pygame.time.Clock()
        self.start_timer = pygame.time.get_ticks()
        self.rotate_timer = pygame.time.get_ticks()         # timer for rotation - used to keep rotation constant irrespective of frame rate
        self.rotate_speed = np.array([  0.12,  -0.1,  0.04])
        self.angles = np.array([ 0.5, 0.5, 0.5])
        self.nodes = np.zeros((0, 3))                       # nodes will have unrotated X,Y,Z coordinates
        self.surfaces = np.zeros((0, 4))
        self.surface_arrays = []                            # a list containing references to surfarrays or surfarray lists (for animation) used for each surface
        self.surface_arrays_item = []                       # a list containing the item used in surface_arrays (for surfarray lists)
        self.surface_array_nodes = np.zeros((0, 4, 2), dtype=np.int16)  # surfarray corners
        self.surface_angle_viewer = np.zeros((0), dtype=float)
        self.rotated_nodes = np.zeros((16, 3))              # rotated_nodes will have X,Y,Z coordinates after rotation ("final 3D coordinates")
        self.trans_nodes = np.zeros((0, 2))                 # trans_nodes will have X,Y screen coordinates
        self.screen_rect = pygame.Rect((self.mid_screen), (1, 1))

        self.image_A = pygame.image.load('Alphaks.jpg').convert()
        self.image_array_A = pygame.surfarray.pixels2d(self.image_A)
        image_size = np.asarray(self.image_A.get_size(), np.int16)
        # image_nodes must define the same number of nodes as the surface that uses them
        self.image_nodes_A3 = self.setup_image_nodes(image_size, np.array([0, 0]), 3)
        self.image_nodes_A4 = self.setup_image_nodes(image_size, np.array([0, 0]), 4)
        self.image_nodes_A5 = self.setup_image_nodes(image_size, np.array([0, 0]), 5)

        self.image_M = pygame.image.load('Milky_Way2.jpg').convert()
        self.image_array_M = pygame.surfarray.pixels2d(self.image_M)
        image_size = np.asarray(self.image_M.get_size(), np.int16)
        self.image_nodes_M3 = self.setup_image_nodes(image_size, np.array([0, 0]), 3)
        self.image_nodes_M4 = self.setup_image_nodes(image_size, np.array([0, 0]), 4)
        self.image_nodes_M5 = self.setup_image_nodes(image_size, np.array([0, 0]), 5)

        self.image_S = pygame.image.load('ShadowBobs.jpg').convert()
        self.image_array_S = pygame.surfarray.pixels2d(self.image_S)
        image_size = np.asarray(self.image_S.get_size(), np.int16)
        self.image_nodes_S3 = self.setup_image_nodes(image_size, np.array([0, 0]), 3)
        self.image_nodes_S4 = self.setup_image_nodes(image_size, np.array([0, 0]), 4)
        self.image_nodes_S5 = self.setup_image_nodes(image_size, np.array([0, 0]), 5)

        self.image_W = pygame.image.load('TheWorld.jpg').convert()
        self.image_array_W = pygame.surfarray.pixels2d(self.image_W)
        image_size = np.asarray(self.image_W.get_size(), np.int16)
        self.image_nodes_W3 = self.setup_image_nodes(image_size, np.array([0, 0]), 3)
        self.image_nodes_W4 = self.setup_image_nodes(image_size, np.array([0, 0]), 4)
        self.image_nodes_W5 = self.setup_image_nodes(image_size, np.array([0, 0]), 5)

        self.image_G = pygame.image.load('Guru.jpg').convert()
        self.image_array_G = pygame.surfarray.pixels2d(self.image_G)
        image_size = (np.asarray(self.image_G.get_size()) * 0.95).astype(np.int16)
        offset = np.array([10, image_size[1] / 8], np.int16)
        self.image_nodes_G3 = self.setup_image_nodes(image_size - offset, offset, 3)
        offset = np.array([10, image_size[1] / 4], np.int16)
        self.image_nodes_G4 = self.setup_image_nodes(image_size - offset, offset, 4)
        offset = np.array([10, image_size[1] / 5], np.int16)
        self.image_nodes_G5 = self.setup_image_nodes(image_size - offset, offset, 5)

        self.image_R_list = []                              # an animated list of images. All must have same size.
        for i in range(1, 11):
            self.image_R = pygame.image.load('Raytracing_' + ('0' + str(i))[-2:] + '.jpg').convert()
            self.image_array_R = pygame.surfarray.pixels2d(self.image_R)
            self.image_R_list.append(self.image_array_R.copy())
        image_size = np.asarray(self.image_R.get_size(), np.int16)
        self.image_nodes_R3 = self.setup_image_nodes(image_size, np.array([0, 0]), 3)
        self.image_nodes_R4 = self.setup_image_nodes(image_size, np.array([0, 0]), 4)
        self.image_nodes_R5 = self.setup_image_nodes(image_size, np.array([0, 0]), 5)

        # scr data will be populated with screen x, y coordinates and the color, then to be set in a single step
        self.scr_x = np.zeros((self.width * self.height), dtype=np.int16)      # reserve memory for x coordinates
        self.scr_y = np.zeros((self.width * self.height), dtype=np.int16)      # reserve memory for y coordinates
        self.img_xy = np.zeros((self.width * self.height, 2), dtype=np.int16)  # reserve memory for image x, y coordinates
        self.scr_col = np.zeros((self.width * self.height), dtype=np.int32)    # reserve memory for colors
        self.scr_section_data = []                                             # list to store section data
        self.scr_cnt = 0
        self.scr_lines = 0
        self.scr_sections = 0
        self.tutorial_img_data = []

        self.setup_object()

        # the following for checking performance only
        if self.tutorial_mode:
            self.info_display = False
        else:
            self.info_display = True
        self.millisecs = 0
        self.timer_avg_frames = 60
        self.timer_names = []
        self.timer_frame = 0
        self.font = pygame.font.SysFont('CourierNew', 15)

        # set up timers
        self.timer_name = []
        self.timer_names.append("rotate")
        self.timer_names.append("calculate angles")
        self.timer_names.append("clear")
        self.timer_names.append("calculate")
        self.timer_names.append("draw")
        self.timer_names.append("show pic")
        self.timer_names.append("display flip")
        self.timer_names.append("plot info")
        self.timer_names.append("wait")
        self.timers = np.zeros((len(self.timer_names), self.timer_avg_frames), dtype=int)
        self.timer_kpixels = np.zeros((self.timer_avg_frames), dtype=int)
        self.timer_time = np.zeros((self.timer_avg_frames), dtype=int)

    def run(self):
        """
        Main loop.
        """
        # initialize timers
        self.start_timer = pygame.time.get_ticks()
        self.millisecs = self.start_timer
        self.timer = pygame.time.get_ticks()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    if event.key == pygame.K_f:
                        self.toggle_fullscreen()
                    if event.key == pygame.K_SPACE:
                        self.pause()
                    if event.key == pygame.K_s:
                        # save screen, at half the resolution, using class name as file name
                        pygame.image.save(self.screen, self.__class__.__name__ + '.jpg')
                        # pygame.image.save(pygame.transform.scale(self.screen, (int(self.screen.get_size()[0] / 2), int(self.screen.get_size()[1] / 2))),
                        #                   self.__class__.__name__ + '.jpg')
                    if event.key == pygame.K_i:
                        self.toggle_info_display()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        # left button: exit
                        self.running = False
                   # # user rotation control
                   #  if event.key == pygame.K_z:
                   #      self.rotate_speed = np.zeros((3))
                   #      self.angles[2] -= 0.04
                   #  if event.key == pygame.K_a:
                   #      self.rotate_speed = np.zeros((3))
                   #      self.angles[2] += 0.04
                   #  if event.key == pygame.K_LEFT:
                   #      self.rotate_speed = np.zeros((3))
                   #      self.angles[1] -= 0.04
                   #  if event.key == pygame.K_RIGHT:
                   #      self.rotate_speed = np.zeros((3))
                   #      self.angles[1] += 0.04
                   #  if event.key == pygame.K_DOWN:
                   #      self.rotate_speed = np.zeros((3))
                   #      self.angles[0] -= 0.04
                   #  if event.key == pygame.K_UP:
                   #      self.rotate_speed = np.zeros((3))
                   #      self.angles[0] += 0.04
            # user rotation control
            keys = pygame.key.get_pressed()
            if keys[pygame.K_z]:
                self.rotate_speed = np.zeros((3))
                self.angles[2] -= 0.04
            if keys[pygame.K_a]:
                self.rotate_speed = np.zeros((3))
                self.angles[2] += 0.04
            if keys[pygame.K_LEFT]:
                self.rotate_speed = np.zeros((3))
                self.angles[1] -= 0.04
            if keys[pygame.K_RIGHT]:
                self.rotate_speed = np.zeros((3))
                self.angles[1] += 0.04
            if keys[pygame.K_DOWN]:
                self.rotate_speed = np.zeros((3))
                self.angles[0] -= 0.04
            if keys[pygame.K_UP]:
                self.rotate_speed = np.zeros((3))
                self.angles[0] += 0.04

            if self.paused:
                pygame.time.wait(100)
                self.rotate_timer = pygame.time.get_ticks()
                self.millisecs = pygame.time.get_ticks()
                self.mode_timer += 100

            else:
                # main components executed here
                if self.phase == 1:
                    # fade in
                    if pygame.time.get_ticks() > self.mode_timer + self.fade_time:
                        self.phase += 1
                        self.z_pos = 1000.0
                        self.mode_timer = pygame.time.get_ticks()
                    else:
                        # adjust Z position
                        self.z_pos = 1000.0 + ((1.0 - (pygame.time.get_ticks() - self.mode_timer) / self.fade_time) ** 2) * 30000.0
                elif self.phase == 2:
                    # just show
                    if pygame.time.get_ticks() > self.mode_timer + self.mode_time:
                        if not self.tutorial_mode:
                            # in tutorial mode, never increase phase - will stick to cube
                            self.phase += 1
                        self.mode_timer = pygame.time.get_ticks()
                elif self.phase == 3:
                    # fade out
                    if pygame.time.get_ticks() > self.mode_timer + self.fade_time:
                        self.phase += 1
                        self.mode_timer = pygame.time.get_ticks()
                    else:
                        # adjust Z position
                        self.z_pos = 1000.0 + (((pygame.time.get_ticks() - self.mode_timer) / self.fade_time) ** 2) * 30000.0
                elif self.phase == 4:
                    # switch mode
                    self.mode += 1
                    if self.mode > 3:
                        self.mode = 1
                    self.phase = 1
                    self.setup_object()
                    self.mode_timer = pygame.time.get_ticks()

                self.clear()
                self.add_angles()
                self.rotate()
                self.calculate_angle_viewer()
                self.draw()
                if self.tutorial_mode:
                    self.show_tutorial()
                if self.info_display:
                    self.plot_info()
                    self.measure_time("plot info")

            # release any locks on screen
            while self.screen.get_locked():
                self.screen.unlock()

            # switch between currently showed and the next screen (prepared in "buffer")
            pygame.display.flip()
            self.measure_time("display flip")
            self.clock.tick(self.target_fps)  # this keeps code running at max target_fps
            self.measure_time("wait")
            self.next_time_frame()

    def add_angles(self):

        self.angles += self.rotate_speed * (pygame.time.get_ticks() - self.rotate_timer) * np.pi / 500.0  # rotation defined as rounds per second
        self.angles[self.angles > np.pi * 2] -= np.pi * 2
        self.angles[self.angles < 0] += np.pi * 2
        self.rotate_timer = pygame.time.get_ticks()

    def rotate(self):

        matrix = self.rotate_matrix_XYZ(self.angles)
        self.rotated_nodes = np.matmul(self.nodes, matrix)
        self.trans_nodes = ((self.rotated_nodes[:, 0:2] * self.z_scale) / (self.rotated_nodes[:, 2:3] + self.z_pos) + self.mid_screen).astype(np.int16)
        self.screen_rect = pygame.Rect(np.min(self.trans_nodes, axis=0) - 3, np.max(self.trans_nodes, axis=0) - np.min(self.trans_nodes, axis=0) + 6)
        self.measure_time("rotate")

    def rotate_matrix_XYZ(self, angles):

        # define rotation matrix for given angles
        (sx, sy, sz) = np.sin(angles)
        (cx, cy, cz) = np.cos(angles)

        # build a matrix for X, Y, Z rotation (in that order, see Wikipedia: Euler angles).
        return np.array([[cy * cz               , -cy * sz              , sy      ],
                         [cx * sz + cz * sx * sy, cx * cz - sx * sy * sz, -cy * sx],
                         [sx * sz - cx * cz * sy, cz * sx + cx * sy * sz, cx * cy ]])

    def clear(self):

        # clear screen.
        self.screen.fill(self.background_color, self.screen_rect)
        self.measure_time("clear")

    def draw(self):

        # draw the cube.

        self.scr_cnt = 0
        self.scr_lines = 0
        self.scr_sections = 0
        self.scr_section_data = []

        # update picture mapping data for each visible surface
        for i in range(np.shape(self.surface_angle_viewer)[0]):
            if self.surface_angle_viewer[i] > 0:
                self.map_texture(i)
        self.measure_time("calculate")

        # update screen in one go by setting the colors for the mapped pixels.
        if self.scr_cnt > 0:
            screen_array = pygame.surfarray.pixels2d(self.screen)
            screen_array[self.scr_x[0:self.scr_cnt], self.scr_y[0:self.scr_cnt]] = self.scr_col[0:self.scr_cnt]

        # draw the surface borders
        for i in range(np.shape(self.surface_angle_viewer)[0]):
            if self.surface_angle_viewer[i] > 0:
                color = (180, 180, 180)
                nodes = list(self.trans_nodes[self.surfaces[i, :]])
                pygame.draw.aalines(self.screen, color, True, nodes)

        self.measure_time("draw")

    def show_tutorial(self):

        # in tutorial mode, add picture (for surfaces 0 or 1) and section lines
        if self.surface_angle_viewer[0] > 0:
            surface = 0
        elif self.surface_angle_viewer[1] > 0:
            surface = 1
        else:
            surface = -1

        if surface == -1:
            # clear picture area
            size = (np.amax(np.vstack((self.tutorial_img_data[0][1] * self.tutorial_img_data[0][2],
                                      self.tutorial_img_data[1][1] * self.tutorial_img_data[1][2])), axis=0)
                    + 2).astype(np.int16)
            pos = np.amin(np.vstack((self.tutorial_img_data[0][3], self.tutorial_img_data[1][3])), axis=0)
            self.screen.fill(self.background_color, (pos, size))
        else:
            # copy picture to screen
            self.screen.blit(self.tutorial_img_data[surface][0], self.tutorial_img_data[surface][3])
            # show sections
            color = np.array([40, 40, 220], dtype=np.uint8)
            for sd in self.scr_section_data:
                if sd[0] == surface:
                    # draw section on rotating cube
                    pygame.draw.polygon(self.screen, color, sd[1], 2)
                    # draw section on still image. Get coordinates, apply scale and add position
                    nodes = (np.asarray(sd[2]) * self.tutorial_img_data[surface][1] + self.tutorial_img_data[surface][3]).astype(np.int16)
                    pygame.draw.polygon(self.screen, color, nodes, 2)
                    # change color for next section
                    color = np.array([color[2], color[0], color[1]])

        self.measure_time("show pic")

    def map_texture(self, surface):

        """
        Map an image on a 3D rotated surface. See https://en.wikipedia.org/wiki/Texture_mapping#Affine_texture_mapping
        Start from the topmost node (minimum y coordinate) and proceed through each section, line by line, between this and the next y.
        Store the screen coordinates and the mapped color from the picture.
        """

        image_nodes = self.surface_array_nodes[surface]
        image_node_cnt = np.shape(image_nodes)[0]
        if isinstance(self.surface_arrays[surface], list):
            # for lists of image arrays (animation), pick from list and add to the counter
            image_array = self.surface_arrays[surface][self.surface_arrays_item[surface]]
            if self.surface_arrays_item[surface] < len(self.surface_arrays[surface]) - 1:
                self.surface_arrays_item[surface] += 1
            else:
                self.surface_arrays_item[surface] = 0
        else:
            # not a list, pick image array directly
            image_array = self.surface_arrays[surface]

        # build a node array where the nodes appear twice - enabling going "over" the right edge
        nodes_x2 = np.hstack((self.surfaces[surface, :], self.surfaces[surface, :]))
        # find the topmost node (minimum y cooridnate) and maximum y coordinate
        min_y_node = np.argmin(self.trans_nodes[self.surfaces[surface, :], 1])
        max_y = np.max(self.trans_nodes[self.surfaces[surface, :], 1])
        y_beg = self.trans_nodes[nodes_x2[min_y_node], 1]
        # when going "left" and "right" through the nodes, start with the top node in both
        (left, right) = (min_y_node, min_y_node)

        # loop through each section from this y coordinate to the next y coordinate until all sections processed
        while y_beg < max_y:
            # image node depends on the node order
            img_node_beg = image_nodes[np.array([left % image_node_cnt, right % image_node_cnt]), :]
            img_node_end = image_nodes[np.array([(left - 1) % image_node_cnt, (right + 1) % image_node_cnt]), :]
            img_node_diff = img_node_end - img_node_beg
            # cube node comes from surface node list
            node_beg = self.trans_nodes[nodes_x2[np.array([left, right])], :]
            node_end = self.trans_nodes[nodes_x2[np.array([left - 1, right + 1])], :]
            node_diff = node_end - node_beg

            # find section end = y_end
            if node_end[1, 1] < node_end[0, 1]:
                # right node comes first (i.e. its Y coordinate is before left's)
                right += 1
                y_end = node_end[1, 1]
            else:
                # left node comes first (i.e. its Y coordinate is before or equal to right's)
                left -= 1
                y_end = node_end[0, 1]

            if y_end > y_beg:
                y = np.arange(y_beg, y_end, dtype=np.int16)
                # node multipliers for each y for left and right side. Since y_end is the first node down, node_diff[:, 1] is here always > 0
                m = (y[:, None] - node_beg[:, 1]) / node_diff[:, 1]
                # respective screen x coordinates for left and right side
                x = (np.round(node_beg[:, 0] + m * node_diff[:, 0])).astype(np.int16)
                x_cnt = np.abs(x[:, 1] - x[:, 0])  # + 1   - use +1 when using linspace method below
                # count cumulative pixel count to use as the offset when storing data
                x_cnt_cum = np.hstack((np.array([0]), np.cumsum(x_cnt))) + self.scr_cnt
                # respective image coordinates, interpolating between image nodes (usually corners)
                img_l = img_node_beg[0, :] + m[:, 0:1] * img_node_diff[0, :]
                img_r = img_node_beg[1, :] + m[:, 1:2] * img_node_diff[1, :]

# =============================================================================
#                 # linspace method; slower
#                 for y_line in range(np.shape(y)[0]):
#                     # linspace returns the x,y screen coordinates and the respective x,y image coordinates
#                     # by interpolating all the screen x coordinates on the static y=y_line and, for the image, between "left" and "right" x,y
#                     # store the screen x,y coordinates here
#                     (self.scr_x[x_cnt_cum[y_line]:x_cnt_cum[y_line + 1]], self.scr_y[x_cnt_cum[y_line]:x_cnt_cum[y_line + 1]], img_x, img_y) \
#                         = np.linspace(
#                         (x[y_line, 0], y_beg + y_line, img_l[y_line, 0], img_l[y_line, 1]),  # starting points screen x, y and image x, y
#                         (x[y_line, 1], y_beg + y_line, img_r[y_line, 0], img_r[y_line, 1]),  # ending points screen x, y and image x, y
#                         num=x_cnt[y_line],
#                         endpoint=True,
#                         dtype=np.int16,
#                         axis=1
#                         )
#                     # store the color found in each interpolated image pixel in self.scr_col
#                     self.scr_col[x_cnt_cum[y_line]:x_cnt_cum[y_line + 1]] = image_array[img_x, img_y]
#                 # add the count of pixels etc.
#                 self.scr_cnt += x_cnt_cum[-1] - x_cnt_cum[0]
#                 self.scr_sections += 1
#                 self.scr_lines += y_end - y_beg
#                 # continue from next line onwwards
#                 y_beg = y_end.copy()
# =============================================================================

                for y_line in range(y_end - y_beg):
                    # process each horizontal line, these are the x coordinates from x left to x right
                    if x_cnt[y_line] > 1:
                        # if "left" not on the left side, use negative step.
                        scr_x = np.arange(x[y_line, 0], x[y_line, 1], np.sign(x[y_line, 1] - x[y_line, 0]), dtype=np.int16)
                        # add x coordinates to self.scr_x array
                        self.scr_x[x_cnt_cum[y_line]:x_cnt_cum[y_line + 1]] = scr_x
                        # add y coordinates similarly - y is constant
                        self.scr_y[x_cnt_cum[y_line]:x_cnt_cum[y_line + 1]] = y_line + y_beg
                        # interpolate between line begin and end coordinates in image
                        self.img_xy[x_cnt_cum[y_line]:x_cnt_cum[y_line + 1], :] = (img_l[y_line, :] + ((scr_x - scr_x[0]) / (scr_x[-1] - scr_x[0]))[:, None] * (img_r[y_line, :] - img_l[y_line, :])).astype(np.int16)
                        # store the color found in each interpolated pixel in self.scr_col
                        self.scr_col[x_cnt_cum[y_line]:x_cnt_cum[y_line + 1]] = image_array[self.img_xy[x_cnt_cum[y_line]:x_cnt_cum[y_line + 1], 0], self.img_xy[x_cnt_cum[y_line]:x_cnt_cum[y_line + 1], 1]]
                # add to the counts of pixels, sections, and lines processed
                self.scr_cnt += x_cnt_cum[-1] - x_cnt_cum[0]
                self.scr_sections += 1
                self.scr_lines += y_end - y_beg
                if np.shape(x)[0] > 2:
                    # store surface number, surface section coordinates (4), picture section coordinates (4), and pixel data pointers (2)
                    self.scr_section_data.append((surface,
                                                  ((x[0, 0], y_beg), (x[0, 1], y_beg), (x[-2, 1], y_end - 1), (x[-2, 0], y_end - 1)),
                                                  ((img_l[0, 0], img_l[0, 1]), (img_r[0, 0], img_r[0, 1]), (img_r[-2, 0], img_r[-2, 1]), (img_l[-2, 0], img_l[-2, 1])),
                                                  x_cnt_cum[0], x_cnt_cum[-1]
                                                  ))
                # continue from next line onwards
                y_beg = y_end.copy()

    def calculate_angle_viewer(self):

        # this function returns the angle to viewer (unscaled!).
        vec_a = self.rotated_nodes[self.surfaces[:, 2], :] - self.rotated_nodes[self.surfaces[:, 1], :]
        vec_b = self.rotated_nodes[self.surfaces[:, 0], :] - self.rotated_nodes[self.surfaces[:, 1], :]
        cp_vector = np.hstack((
            (vec_b[:, 1] * vec_a[:, 2] - vec_b[:, 2] * vec_a[:, 1])[:, None],
            (vec_b[:, 2] * vec_a[:, 0] - vec_b[:, 0] * vec_a[:, 2])[:, None],
            (vec_b[:, 0] * vec_a[:, 1] - vec_b[:, 1] * vec_a[:, 0])[:, None]
            ))
        # cp_length = np.linalg.norm(cp_vector, axis=1)

        vec_viewer = self.rotated_nodes[self.surfaces[:, 1], :] + np.array([0.0, 0.0, self.z_pos])
        self.surface_angle_viewer = np.sum(cp_vector * vec_viewer, axis=1)
        self.measure_time("calculate angles")

    def toggle_fullscreen(self):

        # toggle between fullscreen and windowed mode
        pygame.display.toggle_fullscreen()

    def pause(self):

        if self.paused:
            self.paused = False
            self.timer += pygame.time.get_ticks() - self.pause_timer  # adjust timer for pause time
        else:
            self.paused = True
            self.pause_timer = pygame.time.get_ticks()

    def toggle_info_display(self):

        # switch between a windowed display and full screen
        if self.info_display:
            self.screen.fill(self.background_color, (10, 10, 250, 105 + len(self.timer_names) * 15))
            self.info_display = False
        else:
            self.info_display = True

    def plot_info(self):

        # show info on object and performance
        while self.screen.get_locked():
            self.screen.unlock()

        if self.timer_time[self.timer_frame - self.timer_avg_frames + 1] > 0:
            kpixels = np.sum(self.timer_kpixels) / (self.timer_time[self.timer_frame] - self.timer_time[self.timer_frame - self.timer_avg_frames + 1])
        else:
            kpixels = 0

        self.screen.fill(self.background_color, (10, 10, 250, 105 + len(self.timer_names) * 15))
        # print object info
        self.plot_info_msg(self.screen, 10, 10, 'frames per sec: ' + (' '*10 + str(int(self.clock.get_fps())))[-7:])
        self.plot_info_msg(self.screen, 10, 25, 'kpixels per sec:' + (' '*10 + str(int(kpixels)))[-7:])
        self.plot_info_msg(self.screen, 10, 40, 'kpixels mapped: ' + (' '*10 + str(int(self.scr_cnt / 1000)))[-7:])
        self.plot_info_msg(self.screen, 10, 55, 'lines mapped:   ' + (' '*10 + str(int(self.scr_lines)))[-7:])
        self.plot_info_msg(self.screen, 10, 70, 'sections mapped:' + (' '*10 + str(int(self.scr_sections)))[-7:])

        # add measured times as percentage of total
        tot_time = np.sum(self.timers)
        if tot_time > 0:
            for i in range(len(self.timer_names)):
                info_msg = (self.timer_names[i] + ' '*16)[:16] + (' '*10 + str(round(np.sum(self.timers[i, :]) * 100 / tot_time, 1)))[-7:]
                self.plot_info_msg(self.screen, 10, 90 + i * 15, info_msg)

    def plot_info_msg(self, screen, x, y, msg):
        f_screen = self.font.render(msg, False, (255, 255, 255))
        screen.blit(f_screen, (x, y))

    def measure_time(self, timer_name):

        # add time elapsed from previous call to selected timer
        i = self.timer_names.index(timer_name)
        new_time = pygame.time.get_ticks()
        self.timers[i, self.timer_frame] += (new_time - self.millisecs)
        self.millisecs = new_time

    def next_time_frame(self):

        # move to next timer and clear data
        self.timer_frame += 1
        if self.timer_frame >= self.timer_avg_frames:
            self.timer_frame = 0
        self.timers[:, self.timer_frame] = 0
        self.timer_kpixels[self.timer_frame] = self.scr_cnt
        self.timer_time[self.timer_frame] = pygame.time.get_ticks()

    def setup_object(self):

        gr = (1.0 + np.sqrt(5)) / 2.0  # golden ratio
        rg = 1.0 / gr  # inverse of golden ratio

        if self.mode == 2:
            # define a regular dodecahedron. See https://en.wikipedia.org/wiki/Regular_dodecahedron

            self.nodes = (100.0 * np.array([
                [  1,  1, -1],
                [  1,  1,  1],
                [ -1,  1,  1],
                [ -1,  1, -1],
                [  1, -1, -1],
                [  1, -1,  1],
                [ -1, -1,  1],
                [ -1, -1, -1],
                [  0, gr,-rg],
                [  0, gr, rg],
                [  0,-gr, rg],
                [  0,-gr,-rg],
                [ rg,  0,-gr],
                [-rg,  0,-gr],
                [-rg,  0, gr],
                [ rg,  0, gr],
                [ gr, rg,  0],
                [ gr,-rg,  0],
                [-gr,-rg,  0],
                [-gr, rg,  0]
                ])).astype(float)
            self.surfaces = np.array([
                [ 4, 11,  7, 13, 12],
                [ 4, 17,  5, 10, 11],
                [11, 10,  6, 18,  7],
                [ 7, 18, 19,  3, 13],
                [13,  3,  8,  0, 12],
                [12,  0, 16, 17,  4],
                [ 5, 15, 14,  6, 10],
                [ 6, 14,  2, 19, 18],
                [19,  2,  9,  8,  3],
                [ 8,  9,  1, 16,  0],
                [16,  1, 15,  5, 17],
                [ 1,  9,  2, 14, 15]
                ])
            self.surface_arrays = [
                self.image_R_list,
                self.image_array_A,
                self.image_array_W,
                self.image_array_G,
                self.image_array_M,
                self.image_array_S,
                self.image_array_G,
                self.image_array_M,
                self.image_array_S,
                self.image_array_A,
                self.image_array_W,
                self.image_R_list
                ]
            self.surface_arrays_item = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.surface_array_nodes = np.array([
                self.image_nodes_R5,
                self.image_nodes_A5,
                self.image_nodes_W5,
                self.image_nodes_G5,
                self.image_nodes_M5,
                self.image_nodes_S5,
                self.image_nodes_G5,
                self.image_nodes_M5,
                self.image_nodes_S5,
                self.image_nodes_A5,
                self.image_nodes_W5,
                self.image_nodes_R5
                ], dtype=np.int16)

        elif self.mode == 3:
            # define a regular icosahedron. See https://en.wikipedia.org/wiki/Regular_icosahedron

            self.nodes = (100.0 * gr * np.array([
                [ 0.0,  1.0, -rg],
                [ 0.0,  1.0,  rg],
                [ 0.0, -1.0,  rg],
                [ 0.0, -1.0, -rg],
                [ 1.0,  rg,  0.0],
                [ 1.0, -rg,  0.0],
                [-1.0, -rg,  0.0],
                [-1.0,  rg,  0.0],
                [ rg,  0.0, -1.0],
                [-rg,  0.0, -1.0],
                [-rg,  0.0,  1.0],
                [ rg,  0.0,  1.0]
                ])).astype(float)
            self.surfaces = np.array([
                [ 1,  0,  7],
                [ 7,  0,  9],
                [ 9,  0,  8],
                [ 8,  0,  4],
                [ 4,  0,  1],
                [ 1,  7, 10],
                [ 7,  9,  6],
                [ 9,  8,  3],
                [ 8,  4,  5],
                [ 4,  1, 11],
                [ 6, 10,  7],
                [ 3,  6,  9],
                [ 5,  3,  8],
                [11,  5,  4],
                [10, 11,  1],
                [ 2, 10,  6],
                [ 2,  6,  3],
                [ 2,  3,  5],
                [ 2,  5, 11],
                [ 2, 11, 10]
                ])
            self.surface_arrays = [
                self.image_R_list,
                self.image_array_A,
                self.image_array_W,
                self.image_array_G,
                self.image_array_M,
                self.image_array_S,
                self.image_R_list,
                self.image_array_A,
                self.image_array_W,
                self.image_array_G,
                self.image_array_M,
                self.image_array_S,
                self.image_R_list,
                self.image_array_A,
                self.image_array_W,
                self.image_array_G,
                self.image_array_M,
                self.image_array_S,
                self.image_R_list,
                self.image_array_A
                ]
            self.surface_arrays_item = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.surface_array_nodes = np.array([
                self.image_nodes_R3,
                self.image_nodes_A3,
                self.image_nodes_W3,
                self.image_nodes_G3,
                self.image_nodes_M3,
                self.image_nodes_S3,
                self.image_nodes_R3,
                self.image_nodes_A3,
                self.image_nodes_W3,
                self.image_nodes_G3,
                self.image_nodes_M3,
                self.image_nodes_S3,
                self.image_nodes_R3,
                self.image_nodes_A3,
                self.image_nodes_W3,
                self.image_nodes_G3,
                self.image_nodes_M3,
                self.image_nodes_S3,
                self.image_nodes_R3,
                self.image_nodes_A3
                ], dtype=np.int16)

        else:
            # define a cube.

            self.nodes = (100.0 * np.array([
                [ 1,  1,  1],
                [ 1,  1, -1],
                [-1,  1, -1],
                [-1,  1,  1],
                [ 1, -1,  1],
                [ 1, -1, -1],
                [-1, -1, -1],
                [-1, -1,  1]
                ])).astype(float)
            self.surfaces = np.array([
                [0, 1, 2, 3],
                [4, 7, 6, 5],
                [0, 4, 5, 1],
                [2, 6, 7, 3],
                [1, 5, 6, 2],
                [3, 7, 4, 0]
                ])
            self.surface_arrays = [
                self.image_array_G,
                self.image_array_M,
                self.image_array_S,
                self.image_array_W,
                self.image_array_A,
                self.image_R_list
                ]
            self.surface_arrays_item = [0, 0, 0, 0, 0, 0]
            self.surface_array_nodes = np.array([
                self.image_nodes_G4,
                self.image_nodes_M4,
                self.image_nodes_S4,
                self.image_nodes_W4,
                self.image_nodes_A4,
                self.image_nodes_R4
                ], dtype=np.int16)

            if self.tutorial_mode:
                self.tutorial_img_data = []
                # for tutorial mode, prepare images of first two surfaces of the cube
                img = self.image_G
                img_size0 = np.asarray(img.get_size())
                img_scale0 = min((self.width / 3) / img_size0[0], (self.height * 0.9) / img_size0[1])
                img_surf0 = pygame.transform.scale(img, (img_scale0 * img_size0).astype(np.int16))
                img_pos0 = np.array([10, (self.height - img_scale0 * img_size0[1]) // 2])
                self.tutorial_img_data.append((img_surf0, img_scale0, img_size0, img_pos0))
                img = self.image_M
                img_size1 = np.asarray(img.get_size())
                img_scale1 = min((self.width / 3) / img_size1[0], (self.height * 0.9) / img_size1[1])
                img_surf1 = pygame.transform.scale(img, (img_scale1 * img_size1).astype(np.int16))
                img_pos1 = np.array([10, (self.height - img_scale1 * img_size1[1]) // 2])
                self.tutorial_img_data.append((img_surf1, img_scale1, img_size1, img_pos1))

    def setup_image_nodes(self, image_size, offset, nr_nodes):

        img_x = image_size[0] - 1
        img_y = image_size[1] - 1
        if nr_nodes == 3:
            # return a triangle, "pointing up" and starting from top
            return (np.array([
                [img_x / 2, 0],
                [img_x, img_y],
                [0, img_y]
                ]) + offset).astype(np.int16)
        elif nr_nodes == 5:
            # return a pentagon, "pointing up" and starting from top
            return (np.array([
                [img_x / 2, 0],
                [img_x, 0.382 * img_y],
                [0.809 * img_x, img_y],
                [0.191 * img_x, img_y],
                [0, 0.382 * img_y]
                ]) + offset).astype(np.int16)
        else:
            # return a rectangle
            return (np.array([
                [img_x, 0],
                [0, 0],
                [0, img_y],
                [img_x, img_y]
                ]) + offset).astype(np.int16)


if __name__ == '__main__':
    """
    Prepare screen, objects etc.
    """

    # set screen size
    # first check available full screen modes
    pygame.display.init()
    # disp_modes = pygame.display.list_modes(0, pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE)
    # disp_size = disp_modes[4] # selecting display size from available list. Assuming the 5th element is nice...
    # disp_size = (1920, 1080)
    disp_size = (1280, 720)

    pygame.font.init()
    pygame.mixer.init()
    music_file = "simplicity.ogg"  # this mod by Jellybean is available at e.g. https://demozoo.org/sceners/7713/
    pygame.mixer.music.load(music_file)
    pygame.mixer.music.play(loops=-1)

    screen = pygame.display.set_mode(disp_size)
    pygame.display.set_caption('Texture Mapping')
    TextureMapping(screen, 60).run()

    # exit; close display, stop music
    pygame.quit()
    exit()

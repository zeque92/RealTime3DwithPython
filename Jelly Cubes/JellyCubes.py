# -*- coding: utf-8 -*-
import pygame
import numpy as np
from sys import exit


class JellyCubes:
    """
    Two jelly cubes (ie. transparent cubes) change size so that they take turns in one being inside the other.
    While moving from inside to outside, the cut surfaces are calculated and drawn.

    @author: kalle
    """

    def __init__(self, screen, target_fps):

        self.screen = screen
        self.screen_copy = self.screen.copy()
        self.background_color = (0, 0, 0)
        self.alpha_value = 128
        self.screen_copy.set_alpha(self.alpha_value)
        self.screen_copy.set_colorkey(self.background_color)
        self.width = self.screen.get_size()[0]
        self.height = self.screen.get_size()[1]
        self.full_screen = False
        self.mid_screen = np.array([int(self.width / 2), int(self.height / 2)], dtype=float)
        self.z_scale = self.height * 2                      # scaling for z coordinates
        self.z_pos = 1000.0
        self.target_fps = target_fps
        self.running = True
        self.paused = False
        self.clock = pygame.time.Clock()
        self.start_timer = pygame.time.get_ticks()
        self.rotate_timer = pygame.time.get_ticks()         # timer for rotation - used to keep rotation constant irrespective of frame rate
        self.angles = np.zeros((2, 3), dtype=float)
        self.rotate_speed = np.zeros((2, 3), dtype=float)
        self.nodes = np.zeros((0, 3))                       # nodes will have unrotated X,Y,Z coordinates
        self.edges = np.zeros((0, 2))
        self.edges_both = np.zeros((0, 2))                  # edges both ways; e.g. from node 0 to 1 and node 1 to 0
        self.surfaces = np.zeros((0, 4))
        self.surfaces_joined = np.zeros((0, 5))
        self.surface_egdes = np.zeros((0, 4))
        self.surface_axes = np.zeros((7))
        self.surfaces_both = np.zeros((0, 4))
        self.surface_angle_viewer = np.zeros((0), dtype=float)
        self.rotated_nodes_small = np.zeros((8, 3))         # rotated_nodes_small will have X,Y,Z coordinates for the smaller cube as if big cube still
        self.rotated_nodes = np.zeros((16, 3))              # rotated_nodes will have X,Y,Z coordinates after rotation ("final 3D coordinates")
        self.trans_nodes = np.zeros((0, 2))                 # trans_nodes will have X,Y screen coordinates
        self.small_surf = []                                # small_surf stores data on cut surfaces
        self.size_angle = 0.0
        self.size_speed = 0.08
        self.cube_sizes = np.zeros((2), dtype=float)
        self.cube_small = 0
        self.cube_big = 1
        self.screen_rect = pygame.Rect((self.mid_screen), (1, 1))

        self.setup_cubes()

        # the following for checking performance only
        self.info_display = True
        self.millisecs = 0
        self.timer_avg_frames = 180
        self.timer_names = []
        self.timers = np.zeros((1, 1), dtype=int)
        self.timer_frame = 0
        self.start_timer = 0
        self.font = pygame.font.SysFont('CourierNew', 15)

        # set up timers
        self.timer_name = []
        self.timer_names.append("rotate")
        self.timer_names.append("calculate cuts")
        self.timer_names.append("draw: clear")
        self.timer_names.append("draw: big back")
        self.timer_names.append("draw: small back")
        self.timer_names.append("draw: small front")
        self.timer_names.append("draw: big front")
        self.timer_names.append("draw: cuts")
        self.timer_names.append("plot info")
        self.timer_names.append("display flip")
        self.timer_names.append("wait")
        self.timers = np.zeros((len(self.timer_names), self.timer_avg_frames), dtype=int)

        # initialize timers
        self.start_timer = pygame.time.get_ticks()
        self.millisecs = self.start_timer

    def run(self):
        """
        Main loop.
        """
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
                        pygame.image.save(pygame.transform.scale(self.screen, (int(self.screen.get_size()[0] / 2), int(self.screen.get_size()[1] / 2))),
                                          self.__class__.__name__ + '.jpg')
                    if event.key == pygame.K_i:
                        self.toggle_info_display()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        # left button: exit
                        self.running = False

            if self.paused:
                pygame.time.wait(100)
                self.rotate_timer = pygame.time.get_ticks()
                self.millisecs = pygame.time.get_ticks()

            else:
                # main components executed here
                self.change_sizes()
                self.add_angles()
                self.rotate()
                self.measure_time("rotate")
                self.calculate_angle_viewer()
                self.calc_cuts()
                self.measure_time("calculate cuts")
                self.draw()
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

    def change_sizes(self):

        self.size_angle += self.size_speed * (pygame.time.get_ticks() - self.rotate_timer) * np.pi / 500.0  # size speed defined as rounds per second
        self.cube_sizes = np.array([
            np.sin(self.size_angle) * 0.3 + 1.0,
            np.sin(self.size_angle + np.pi) * 0.3 + 1.0
            ])

    def add_angles(self):

        self.angles += self.rotate_speed * (pygame.time.get_ticks() - self.rotate_timer) * np.pi / 500.0  # rotation defined as rounds per second
        self.angles[self.angles > np.pi * 2] -= np.pi * 2
        self.angles[self.angles < 0] += np.pi * 2
        self.rotate_timer = pygame.time.get_ticks()

    def rotate(self):

        if self.cube_sizes[0] < self.cube_sizes[1] and self.cube_small == 1:
            self.cube_small = 0
            self.cube_big = 1
        elif self.cube_sizes[0] > self.cube_sizes[1] and self.cube_big == 1:
            self.cube_small = 1
            self.cube_big = 0

        matrix_small = self.rotate_matrix_XYZ(self.angles[self.cube_small, :])
        matrix_big = self.rotate_matrix_XYZ(self.angles[self.cube_big, :])
        matrix_big_inv = np.linalg.inv(matrix_big)

        # rotate small cube
        self.rotated_nodes[0:8, :] = np.matmul(self.nodes * self.cube_sizes[self.cube_small], matrix_small)
        # rotate big cube
        self.rotated_nodes[8:16, :] = np.matmul(self.nodes * self.cube_sizes[self.cube_big], matrix_big)
        # make a special rotated copy of the small cube by rotating it with the inverse of big cube matrix.
        # the result is small cube rotated as if big cube was stationary (or not rotated at all).
        self.rotated_nodes_small = np.matmul(self.rotated_nodes[0:8, :] * self.cube_sizes[self.cube_small], matrix_big_inv)

    def rotate_matrix_XYZ(self, angles):

        # define rotation matrix for given angles
        (sx, sy, sz) = np.sin(angles)
        (cx, cy, cz) = np.cos(angles)

        # build a matrix for X, Y, Z rotation (in that order, see Wikipedia: Euler angles).
        return np.array([[cy * cz               , -cy * sz              , sy      ],
                         [cx * sz + cz * sx * sy, cx * cz - sx * sy * sz, -cy * sx],
                         [sx * sz - cx * cz * sy, cz * sx + cx * sy * sz, cx * cy ]])

    def draw(self):

        # clear screen.
        self.screen.fill(self.background_color, self.screen_rect)

        self.trans_nodes = ((self.rotated_nodes[:, 0:2] * self.z_scale) / (self.rotated_nodes[:, 2:3] + self.z_pos) + self.mid_screen).astype(np.int16)
        rect_small = pygame.Rect(np.min(self.trans_nodes[0:8, :], axis=0),
                                 np.max(self.trans_nodes[0:8, :], axis=0) - np.min(self.trans_nodes[0:8, :], axis=0) + 1)
        rect_big = pygame.Rect(np.min(self.trans_nodes[8:16, :], axis=0),
                               np.max(self.trans_nodes[8:16, :], axis=0) - np.min(self.trans_nodes[8:16, :], axis=0) + 1)
        self.screen_rect = pygame.Rect(np.min(self.trans_nodes, axis=0), np.max(self.trans_nodes, axis=0) - np.min(self.trans_nodes, axis=0) + 1)
        self.measure_time("draw: clear")

        # draw the back side of the big cube. This can be drawn directly to main surface using the alpha value to adjust color.
        self.screen.lock()
        self.screen_copy.lock()
        for i in range(6):
            if self.surface_angle_viewer[i + 6] < 0:
                color = (self.surface_colors[self.cube_big * 6 + i, :] * (self.alpha_value / 255.0)).astype(np.uint8)
                nodes = list(self.trans_nodes[self.surfaces_both[i + 6, :]])
                pygame.draw.polygon(self.screen, color, nodes)
        self.measure_time("draw: big back")
        while self.screen.get_locked():
            self.screen.unlock()

        # add the small cube.
        for i in range(6):
            if self.surface_angle_viewer[i] < 0:
                color = self.surface_colors[self.cube_small * 6 + i, :]
                nodes = list(self.trans_nodes[self.surfaces_both[i, :]])
                pygame.draw.polygon(self.screen_copy, color, nodes)
        self.screen_copy.unlock()
        self.screen.blit(self.screen_copy, rect_small, rect_small)
        # self.screen_copy.fill(self.background_color, rect_small)  # clearing here not necessary - fron will always cover back
        self.measure_time("draw: small back")
        self.screen_copy.lock()
        for i in range(6):
            if self.surface_angle_viewer[i] > 0:
                color = self.surface_colors[self.cube_small * 6 + i, :]
                nodes = list(self.trans_nodes[self.surfaces_both[i, :]])
                pygame.draw.polygon(self.screen_copy, color, nodes)
        self.screen_copy.unlock()
        self.screen.blit(self.screen_copy, rect_small, rect_small)
        self.screen_copy.fill(self.background_color, rect_small)
        self.measure_time("draw: small front")

        # add the front side of the big cube.
        self.screen_copy.lock()
        for i in range(6):
            if self.surface_angle_viewer[i + 6] > 0:
                color = self.surface_colors[self.cube_big * 6 + i, :]
                nodes = list(self.trans_nodes[self.surfaces_both[i + 6, :]])
                pygame.draw.polygon(self.screen_copy, color, nodes)
        self.screen_copy.unlock()
        self.screen.blit(self.screen_copy, rect_big, rect_big)
        self.screen_copy.fill(self.background_color, rect_big)
        self.measure_time("draw: big front")

        # add the small cube sections which pierce big cube surfaces which are front
        self.screen_copy.lock()
        nodes = []
        for s_surf in self.small_surf:
            if self.surface_angle_viewer[s_surf[1] + 6] >= 0:
                color = self.surface_colors[self.cube_small * 6 + s_surf[0], :]
                nodes = list(self.trans_nodes[node, :] for node in s_surf[2])
                pygame.draw.polygon(self.screen_copy, color, nodes)
        self.screen_copy.unlock()
        if len(nodes) > 0:
            # do only if something was drawn
            self.screen.blit(self.screen_copy, rect_small, rect_small)
            self.screen_copy.fill(self.background_color, rect_small)
        self.measure_time("draw: cuts")

    def calculate_angle_viewer(self):

        # this function returns the angle to viewer (unscaled!).
        vec_a = self.rotated_nodes[self.surfaces_both[:, 2], :] - self.rotated_nodes[self.surfaces_both[:, 1], :]
        vec_b = self.rotated_nodes[self.surfaces_both[:, 0], :] - self.rotated_nodes[self.surfaces_both[:, 1], :]
        cp_vector = np.hstack((
            (vec_b[:, 1] * vec_a[:, 2] - vec_b[:, 2] * vec_a[:, 1])[:, None],
            (vec_b[:, 2] * vec_a[:, 0] - vec_b[:, 0] * vec_a[:, 2])[:, None],
            (vec_b[:, 0] * vec_a[:, 1] - vec_b[:, 1] * vec_a[:, 0])[:, None]
            ))
        # cp_length = np.linalg.norm(cp_vector, axis=1)

        vec_viewer = self.rotated_nodes[self.surfaces_both[:, 1], :] + np.array([0.0, 0.0, self.z_pos])
        self.surface_angle_viewer = np.sum(cp_vector * vec_viewer, axis=1)

    def calc_cuts(self):

        cut_size = abs(self.nodes[0, 0] * self.cube_sizes[self.cube_big])
        rotated_nodes_max = np.max(np.abs(self.rotated_nodes_small), axis=1)[:, None]  # if > cut_size, this node is "out" and needs drawing
        # figure out which axis the "out node" is on - to find out which surface of the big cube it comes out of
        big_surf_axis = np.argmax(np.abs(self.rotated_nodes_small), axis=1)[:, None]
        big_surf = (3 + (1 + big_surf_axis) * np.sign(self.rotated_nodes_small[np.arange(8)[:, None], big_surf_axis[:]])).astype(np.int16)
        # multip is the multiplier from edge start node to end node, representing the point when hitting the bigger cube (cut size)
        multip = (np.sign(self.rotated_nodes_small[self.edges_both[:, 0]]) * cut_size - self.rotated_nodes_small[self.edges_both[:, 0]]) \
            / (self.rotated_nodes_small[self.edges_both[:, 1]] - self.rotated_nodes_small[self.edges_both[:, 0]] + 0.000001)
        multip_min = np.minimum(1.0, np.min(abs(multip) + (1 - np.sign(multip)) * 99999, axis=1))[:, None]
        # calculate the cut points for edges by applying the multiplier, both directions
        rotated_nodes_cut = self.rotated_nodes[self.edges_both[:, 0]] \
            + multip_min * (self.rotated_nodes[self.edges_both[:, 1]] - self.rotated_nodes[self.edges_both[:, 0]])
        rn_nr = np.shape(self.rotated_nodes)[0]  # number of rotated nodes (always = 16)
        eb_nr = np.shape(self.edges_both)[0]  # number of edges both ways (always = 24)
        self.rotated_nodes = np.vstack((self.rotated_nodes, rotated_nodes_cut))  # add the cut nodes to rotated nodes
        # edge data combines start and end node + multiplier + cut node nr for easier search and use
        edge_data = np.hstack((self.edges_both[:, 0:1] * 100 + self.edges_both[:, 1:2], multip_min, np.arange(rn_nr, rn_nr + eb_nr)[:, None]))

        self.small_surf = []
        # loop surfaces and (for the visible) add new rotated nodes and build polygons
        for s in range(np.shape(self.surfaces)[0]):
            if self.surface_angle_viewer[s] > 0:
                # process all 4 corners
                for q in range(4):
                    node_1 = self.surfaces[s, q]  # corner node
                    if rotated_nodes_max[node_1] > cut_size:
                        # this corner node is "out" and needs drawing
                        node_2 = self.surfaces_joined[s, q + 1]  # next node - use "joined" surfaces
                        data_2 = edge_data[edge_data[:, 0] == node_1 * 100 + node_2]  # find the data based on node_1-->node_2
                        if data_2[0, 1] != 2.0:
                            # multiplier != 1: this node not fully connected to next node, continue - otherwise do nothing
                            node_3 = self.surfaces[s, q - 1]  # preceding node
                            data_3 = edge_data[edge_data[:, 0] == node_1 * 100 + node_3]  # find the data based on node_1-->node_3
                            if data_3[0, 1] == 1.0:
                                # preceding node and corner node fully connected, need another node - node_3 is another corner node
                                node_4 = self.surfaces[s, q - 2]  # pre-preceding node
                                data_4 = edge_data[edge_data[:, 0] == node_3 * 100 + node_4]  # find the data based on node_3-->node_4
                                # add as a polygon - data_2[2] and data_4[2] refer to the cut nodes
                                self.small_surf.append((s, self.surface_axes[big_surf[node_1]], (node_1, int(data_2[0, 2]), int(data_4[0, 2]), node_3)))
                            else:
                                # add as a polygon - data_2[2] and data_3[2] refer to the cut nodes
                                self.small_surf.append((s, self.surface_axes[big_surf[node_1]], (node_1, int(data_2[0, 2]), int(data_3[0, 2]))))

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
            self.info_display = False
            self.screen.fill(self.background_color, (10, 10, 250, 45 + len(self.timer_names) * 15))
        else:
            self.info_display = True

    def plot_info(self):

        # show info on object and performance
        while self.screen.get_locked():
            self.screen.unlock()

        self.screen.fill(self.background_color, (10, 10, 250, 45 + len(self.timer_names) * 15))
        # print object info
        self.plot_info_msg(self.screen, 10, 10, 'frames per sec: ' + str(int(self.clock.get_fps())))
        self.plot_info_msg(self.screen, 10, 25, 'small / big:    ' + str(self.cube_small) + ' / ' + str(self.cube_big))
        # self.plot_info_msg(self.screen, 10, 40, 'angles:         ' + (' '*10 + str(int(self.angles[0, 0] * 180 / np.pi)))[-7:]
        #                    + (' '*10 + str(int(self.angles[0, 1] * 180 / np.pi)))[-7:] + (' '*10 + str(int(self.angles[0, 2] * 180 / np.pi)))[-7:])
        # self.plot_info_msg(self.screen, 10, 55, '                ' + (' '*10 + str(int(self.angles[1, 0] * 180 / np.pi)))[-7:]
        #                    + (' '*10 + str(int(self.angles[1, 1] * 180 / np.pi)))[-7:] + (' '*10 + str(int(self.angles[1, 2] * 180 / np.pi)))[-7:])

        # add measured times as percentage of total
        tot_time = np.sum(self.timers)
        if tot_time > 0:
            for i in range(len(self.timer_names)):
                info_msg = (self.timer_names[i] + ' '*16)[:17] + (' '*10 + str(round(np.sum(self.timers[i, :]) * 100 / tot_time, 1)))[-7:]
                self.plot_info_msg(self.screen, 10, 45 + i * 15, info_msg)

    def plot_info_msg(self, screen, x, y, msg):
        f_screen = self.font.render(msg, False, (255, 255, 255))
        screen.blit(f_screen, (x, y))

    def fade_out_screen(self):

        # fade screen to background color.
        s = 60  # number of frames to use for fading
        orig_screen = self.screen.copy()  # get original image and store it here

        for i in range(s):
            # "multiply" current screen to fade it out with each step
            fadecol = int((s - i - 1) * 255.0 / s)
            orig_screen.set_alpha(fadecol)
            self.screen.fill(self.background_color)
            self.screen.blit(orig_screen, (0, 0))
            self.plot_info()
            pygame.display.flip()
            self.clock.tick(self.target_fps)  # this keeps fadeout running at max target_fps
        self.screen.fill(self.background_color)

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

    def setup_cubes(self):

        # define two cubes. As they are both cubes just one set of nodes will suffice.

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
        self.edges = np.array([
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7]
            ])
        self.surfaces = np.array([
            [0, 1, 2, 3],
            [4, 7, 6, 5],
            [0, 4, 5, 1],
            [2, 6, 7, 3],
            [1, 5, 6, 2],
            [3, 7, 4, 0]
            ])
        self.surface_edges = np.array([
            [0, 1, 2, 3],
            [7, 6, 5, 3],
            [8, 4, 9, 0],
            [9, 5, 10, 1],
            [10, 6, 11, 2],
            [11, 7, 8, 3]
            ])

        # just one axis is constant for each surface - for others min and max cancel each other
        a = np.max(self.nodes[self.surfaces], axis=1) + np.min(self.nodes[self.surfaces], axis=1)
        # find the axis
        b = np.hstack((np.nonzero(a)[0][:, None], np.nonzero(a)[1][:, None]))
        # add 1 and use sign to label axis 1 to 3, positive and negative, then add 3 and [3, 99] to get indexed from 0 to 6
        c = np.vstack((np.array([3, 99]), np.hstack((((3 + (1 + b[:, 1]) * np.sign(a[b[:, 0], b[:, 1]])))[:, None], b[:, 0:1]))))
        # sort and store the value
        self.surface_axes = (c[c[:, 0].argsort()][:, 1]).astype(np.int16)

        self.surface_colors = np.array([
            [ 40,  0, 250],
            [ 40,  0, 250],
            [ 40,  0, 185],
            [ 40,  0, 185],
            [ 40,  0, 120],
            [ 40,  0, 120],
            [250,  0,  40],
            [250,  0,  40],
            [185,  0,  40],
            [185,  0,  40],
            [120,  0,  40],
            [120,  0,  40]
            ], dtype=np.uint8)

        self.edges_both = np.vstack((self.edges, np.hstack((self.edges[:, 1][:, None], self.edges[:, 0][:, None]))))  # edges "both ways"
        self.surfaces_joined = np.hstack((self.surfaces, self.surfaces[:, 0:1]))  # adds first node again as last node for each surface
        self.surfaces_both = np.vstack((self.surfaces, self.surfaces + 8))
        self.cube_sizes = np.array([1.1, 1.0])
        self.rotate_speed = np.array([
            [  0.12,  -0.1,  0.04],
            [ -0.09,  0.08, 0.06]
            ])
        self.angles = np.array([
            [ 0.0, 0.0, 0.0],
            [ 0.0, 0.0, 0.0]
            ])


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
    # pygame.mixer.init()
    # music_file = "river road.ogg"  # this mod by Jellybean is available at e.g. http://janeway.exotica.org.uk/release.php?id=39506
    # pygame.mixer.music.load(music_file)
    # pygame.mixer.music.play(loops=-1)

    screen = pygame.display.set_mode(disp_size)
    pygame.display.set_caption('Jelly Cubes')
    JellyCubes(screen, 60).run()

    # exit; close display, stop music
    pygame.quit()
    exit()

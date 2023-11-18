# -*- coding: utf-8 -*-
import pygame
import numpy as np
import math
import bisect
from sys import exit


class RGBSphere:
    """
    Builds a rotating spherical object from a regular tetrahedron or a regular icosahedron.
    Can be shown either in wireframe or solid with or without image coloring, and shaded with one or three moving light sources.

    @author: kalle
    """

    def __init__(self, screen, image_filename, image_type, image_brightness, image_contrast, target_fps):

        self.screen = screen
        self.screen_info = self.screen.copy()
        self.width = self.screen.get_size()[0]
        self.height = self.screen.get_size()[1]
        self.full_screen = False
        self.background_color = (0, 0, 0)
        self.mid_screen = np.array([int(self.width / 2), int(self.height / 2)], dtype=float)
        self.z_scale = self.height * 2                      # scaling for z coordinates
        self.target_fps = target_fps                        # affects movement speeds
        self.running = True
        self.paused = False
        self.status_msg = 'Preparing object'
        self.fill_surfaces = False
        self.fill_colors = False
        self.fill_colors_fade = 0.0
        self.pause_timer = 0
        self.clock = pygame.time.Clock()
        self.start_timer = pygame.time.get_ticks()
        self.rotate_timer = pygame.time.get_ticks()         # timer for rotation - used to keep rotation constant irrespective of frame rate

        self.edge_color_set = [(128, 128, 128),             # back and front edge colors, surface color = front color
                               (255, 255, 255)]
        self.edge_colors = self.edge_color_set.copy()
        self.angles = np.zeros((4, 3), dtype=float)
        self.rotate_speed = np.zeros((4, 3), dtype=float)
        self.z_pos = 1200.0
        self.nodes = np.zeros((0, 3))                       # nodes will have unrotated X,Y,Z coordinates
        self.rotated_nodes = np.zeros((0, 3))               # rotatedNodes will have X,Y,Z coordinates after rotation ("final 3D coordinates")
        self.trans_nodes = np.zeros((0, 2))                 # transNodes will have X,Y coordinates
        self.push_nodes = np.zeros((0, 3))                  # nodes stored for "pushing"
        self.push_length = np.zeros((0, 3))                 # the distance of push nodes from center
        self.sphere_radius = 0.0                            # the distance of sphere from center - the target for pushing
        self.edges = []                                     # edge is a connection between 2 nodes
        self.surfaces = []                                  # surface is a plane within (in this case 3) edges
        self.depth = 0                                      # depth from tetra to sphere
        self.nodes_at_depth = [0]
        self.edges_at_depth = [0]
        self.surfaces_at_depth = [0]
        self.light_nodes = np.array([[-250.0, -80.0, -250.0],
                                     [-250.0, -80.0, -250.0],
                                     [-250.0, -80.0, -250.0]])
        self.light_rotated_nodes = self.light_nodes.copy()
        self.light_trans_nodes = np.zeros((0, 2))
        self.light_nr = 1
        self.light_speed = 0
        self.light_pics = []
        self.light_pic_zcoords = []
        self.light_pic_min_size = 0
        self.light_pic_max_size = 0
        self.light_size = 20                                # light size at z_pos
        self.timer = 0
        self.rotate_time = 4000                             # time (ms) to just rotate the current object
        self.fade_time = 2000                               # time (ms) to fade in the next phase
        self.wait_time = 1000                               # time (ms) between fade and push
        self.push_time = 2000                               # time (ms) to push the next phase from flat to sphere level
        self.change_time = 1000                             # time (ms) to change between filled and wireframe modes (fades out and in)
        self.light_time = 2000                              # time (ms) to move all 3 light sources together when going from 3 to 1 lights
        self.phase = 0
        self.fade = 1.0

        # the following for checking performance only
        self.info_display = True
        self.millisecs = 0
        self.timer_avg_frames = 180
        self.timer_names = []
        self.timers = np.zeros((1, 1), dtype=int)
        self.timer_frame = 0
        self.start_timer = 0
        self.cross_product_count = 0
        self.draw_count = 0
        self.font = pygame.font.SysFont('CourierNew', 15)
        # set up timers
        self.timer_name = []
        self.timer_names.append("rotate")
        self.timer_names.append("push nodes")
        self.timer_names.append("clear screen")
        self.timer_names.append("prepare edges")
        self.timer_names.append("prepare surfaces")
        self.timer_names.append("draw")
        self.timer_names.append("plot info")
        self.timer_names.append("display flip")
        self.timer_names.append("wait")
        self.timers = np.zeros((len(self.timer_names), self.timer_avg_frames), dtype=int)

        # load image for surface coloring
        self.image = pygame.image.load(image_filename).convert()
        self.image_type = image_type
        self.image_contrast = image_contrast
        self.image_brightness = image_brightness
        self.image_size = np.asarray(self.image.get_size())
        self.image_ext = 1.2  # multiplier for horizontal size, as use may overflow from right to left edge.
        self.image_array = np.zeros((0, 0, 0))
        self.image_screen = self.image  # will be resized
        self.image_screen_size = self.image_size  # will be resized
        self.image_ratio = 1.0
        self.image_mercator_R = 1.0 / np.log(np.tan(np.pi / 4 + (85.0 / 90.0) * np.pi / 4))  # with this R y is in [-1,1] between -85 and +85 degrees
        self.image_setup()
        self.image_copy()
        self.color_surfaces = 0  # 0 = do not draw; 1 = draw edges; 2 = draw filled polygons

        self.setup_info_screen()

        # two setups; choose one
        # self.setup_tetra()
        self.setup_ico()

        # add some depth to it
        self.image_copy()
        self.add_depth()
        self.add_depth()
        self.add_depth()
        self.add_depth()
        self.color_surfaces = 2  # show surfaces by coloring them
        self.add_depth()
        self.copy_colors_down()
        self.fade_out_screen()
        # pygame.time.wait(1000)

        # return to starting depth and "unpushed" nodes
        self.depth = 0
        self.nodes = self.push_nodes.copy()

        # form surface_nodes
        self.surface_nodes = np.asarray(([surf.nodes for surf in self.surfaces]), dtype=np.int16)
        self.surface_is_mid = np.asarray(([surf.mid_surface for surf in self.surfaces]), dtype=np.int8)
        self.surface_edges = np.asarray(([surf.edge_ixs for surf in self.surfaces]), dtype=np.int16)
        self.surface_colors = np.asarray(([surf.color for surf in self.surfaces]), dtype=np.uint8)
        self.surface_parent_colors = np.vstack((
            np.asarray(([surf.color for surf in self.surfaces if surf.depth == 0]), dtype=np.uint8),  # use color as parent color for depth = 0 (not needed)
            np.asarray(([surf.parent.color for surf in self.surfaces if surf.depth > 0]), dtype=np.uint8)
            ))
        self.surface_used_colors = self.surface_colors.copy()
        self.edge_nodes = np.asarray(([edge.nodes for edge in self.edges]), dtype=np.int16)

        self.status_msg = 'Preparing lights'
        self.setup_lights()
        self.status_msg = 'Ready'

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
                    if event.key == pygame.K_DOWN:
                        # decrease depth, only allowed when phase = 0 and there is less depth available
                        if self.phase == 0 and self.depth > 0:
                            self.timer = pygame.time.get_ticks()
                            self.status_msg = 'Pushing in'
                            self.phase = 5
                    if event.key == pygame.K_UP:
                        # increase depth, only allowed when phase = 0 and there is more depth available
                        if self.phase == 0 and self.depth < len(self.nodes_at_depth) - 2:
                            self.phase = 1
                    if event.key == pygame.K_LEFT:
                        if self.phase == 0 and self.fill_surfaces and not self.fill_colors:
                            # change to wireframe mode. Only allowed when phase = 0
                            self.timer = pygame.time.get_ticks()
                            self.phase = 9
                            self.status_msg = 'Changing mode'
                        elif self.phase == 0 and self.fill_surfaces and self.fill_colors:
                            # change to filled surface mode. Only allowed when phase = 0
                            self.timer = pygame.time.get_ticks()
                            self.phase = 13
                            self.status_msg = 'Changing mode'
                    if event.key == pygame.K_RIGHT:
                        if self.phase == 0 and not self.fill_surfaces:
                            # change to filled surface mode. Only allowed when phase = 0
                            self.timer = pygame.time.get_ticks()
                            self.phase = 9
                            self.status_msg = 'Changing mode'
                        elif self.phase == 0 and self.fill_surfaces and not self.fill_colors:
                            # change to filled surface with image colors mode. Only allowed when phase = 0
                            self.timer = pygame.time.get_ticks()
                            self.fill_colors = True
                            self.phase = 12
                            self.status_msg = 'Changing mode'
                    if event.key == pygame.K_l:
                        # change light source mode, if in filled surface mode. Only allowed when phase = 0
                        if self.phase == 0 and self.fill_surfaces:
                            self.timer = pygame.time.get_ticks()
                            if self.light_nr == 1:
                                self.light_nr = 3
                                self.angles[2:4, :] = self.angles[1, :]  # all light angles start from the same
                                self.light_rotated_nodes[1:3, :] = self.light_rotated_nodes[0, :]
                            elif self.light_nr == 3 and self.light_speed > 0:  # grouping only allowed when lights moving
                                target_angles = (self.angles[1] + self.rotate_speed[1] * self.light_speed * self.light_time * np.pi / (1000.0 / 2)) % (2 * np.pi)
                                angle_add = (target_angles - (self.angles[2:4] % (2 * np.pi))) / (self.light_time * np.pi / (1000.0 / 2))
                                self.phase = 11
                                self.status_msg = 'Grouping lights'
                    if event.key == pygame.K_m:
                        # add light source speed
                        if self.fill_surfaces and self.light_speed < 16 and self.phase != 11:
                            if self.light_speed == 0:
                                self.light_speed = 1
                            else:
                                self.light_speed = int(self.light_speed * 2)
                    if event.key == pygame.K_n:
                        # reduce light source speed
                        if self.fill_surfaces and self.light_speed > 0 and self.phase != 11:
                            if self.light_speed == 1:
                                self.light_speed = 0
                            else:
                                self.light_speed = int(self.light_speed / 2)
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
                if self.phase == 1:
                    # add depth
                    self.depth += 1
                    self.phase += 1
                    self.fade = 0.0
                    self.timer = pygame.time.get_ticks()
                    self.status_msg = 'Fading in'
                    if self.fill_surfaces and self.fill_colors:
                        self.surface_used_colors = self.surface_parent_colors.copy()
                elif self.phase == 2:
                    # fade in the next depth, if not filled
                    if (self.fill_surfaces and not self.fill_colors) or pygame.time.get_ticks() - self.timer >= self.fade_time:
                        self.phase += 1
                        self.timer = pygame.time.get_ticks()
                        self.fade = 1.0
                        self.status_msg = 'Waiting'
                        self.surface_used_colors = self.surface_colors.copy()
                    elif self.fill_surfaces and self.fill_colors:
                        fade = (pygame.time.get_ticks() - self.timer) / self.fade_time
                        self.surface_used_colors = (fade * self.surface_colors + (1.0 - fade) * self.surface_parent_colors).astype(np.uint8)
                    else:
                        self.fade = (pygame.time.get_ticks() - self.timer) / self.fade_time
                elif self.phase == 3:
                    # wait between fade in and push
                    if (self.fill_surfaces and not self.fill_colors) or pygame.time.get_ticks() - self.timer >= self.wait_time:
                        self.phase += 1
                        self.timer = pygame.time.get_ticks()
                        self.status_msg = 'Pushing out'
                elif self.phase == 4:
                    # push the new nodes to the surface of the sphere
                    if pygame.time.get_ticks() - self.timer >= self.push_time:
                        self.push_to_radius(1.0)
                        self.phase = 0  # return to "no action" state
                        self.status_msg = 'Ready'
                        # surfaces ready; calculate all their cross product vector lengths
                        # for surface in (surf for surf in self.surfaces if surf.depth == self.depth):
                        #     self.update_surface_cross_product_vector(surface)
                        #     self.update_surface_cross_product_length(surface)
                    else:
                        self.push_to_radius((pygame.time.get_ticks() - self.timer) / self.push_time)
                    self.measure_time("push nodes")
                elif self.phase == 5:
                    # push the newest nodes back to level with lower level nodes
                    if pygame.time.get_ticks() - self.timer >= self.push_time:
                        self.push_to_radius(0.0)
                        self.phase += 1
                        self.status_msg = 'Waiting'
                    else:
                        self.push_to_radius(1.0 - (pygame.time.get_ticks() - self.timer) / self.push_time)
                    self.measure_time("push nodes")
                elif self.phase == 6:
                    # wait between push and fade
                    if (self.fill_surfaces and not self.fill_colors) or pygame.time.get_ticks() - self.timer >= self.wait_time:
                        self.phase += 1
                        self.timer = pygame.time.get_ticks()
                elif self.phase == 7:
                    # subtract depth
                    self.phase += 1
                    self.fade = 1.0
                    self.timer = pygame.time.get_ticks()
                    self.status_msg = 'Fading out'
                elif self.phase == 8:
                    # fade out current depth, if not filled
                    if (self.fill_surfaces and not self.fill_colors) or pygame.time.get_ticks() - self.timer >= self.fade_time:
                        self.depth -= 1
                        self.phase = 0
                        self.timer = pygame.time.get_ticks()
                        self.fade = 0.0
                        self.status_msg = 'Ready'
                        self.surface_used_colors = self.surface_colors.copy()
                    elif self.fill_surfaces and self.fill_colors:
                        fade = 1.0 - (pygame.time.get_ticks() - self.timer) / self.fade_time
                        self.surface_used_colors = (fade * self.surface_colors + (1.0 - fade) * self.surface_parent_colors).astype(np.uint8)
                    else:
                        self.fade = 1.0 - (pygame.time.get_ticks() - self.timer) / self.fade_time
                elif self.phase == 9:
                    # fade out, change mode
                    if pygame.time.get_ticks() - self.timer >= self.change_time / 2:
                        self.phase += 1
                        if self.fill_surfaces:
                            self.fill_surfaces = False
                        else:
                            self.fill_surfaces = True
                    else:
                        for i in range(len(self.edge_color_set)):
                            self.edge_colors[i] = ([int((1.0 - (pygame.time.get_ticks() - self.timer)
                                                         / (self.change_time / 2)) * x) for x in self.edge_color_set[i]])
                elif self.phase == 10:
                    # fade in
                    if pygame.time.get_ticks() - self.timer >= self.change_time:
                        self.phase = 0
                        self.edge_colors = self.edge_color_set.copy()
                        self.status_msg = 'Ready'
                    else:
                        for i in range(len(self.edge_color_set)):
                            self.edge_colors[i] = ([int((((pygame.time.get_ticks() - self.timer) - (self.change_time / 2))
                                                         / (self.change_time / 2)) * x) for x in self.edge_color_set[i]])
                elif self.phase == 11:
                    # move all three light sources together
                    if pygame.time.get_ticks() - self.timer >= self.light_time:
                        self.light_nr = 1
                        self.angles[2:4] = self.angles[1:2]
                        self.phase = 0
                        self.status_msg = 'Ready'
                    else:
                        self.angles[2:4] += (angle_add - self.rotate_speed[2:4] * self.light_speed) * (pygame.time.get_ticks() - self.rotate_timer) * np.pi / (1000.0 / 2)
                elif self.phase == 12:
                    # fade image in
                    if pygame.time.get_ticks() - self.timer >= self.change_time:
                        self.fill_colors_fade = 1.0
                        self.phase = 0
                        self.status_msg = 'Ready'
                    else:
                        self.fill_colors_fade = (pygame.time.get_ticks() - self.timer) / self.change_time
                elif self.phase == 13:
                    # fade image out
                    if pygame.time.get_ticks() - self.timer >= self.change_time:
                        self.fill_colors_fade = 0.0
                        self.fill_colors = False
                        self.phase = 0
                        self.status_msg = 'Ready'
                    else:
                        self.fill_colors_fade = 1.0 - (pygame.time.get_ticks() - self.timer) / self.change_time

                self.rotate()
                self.measure_time("rotate")
                self.clear_screen()
                self.measure_time("clear screen")
                if self.fill_surfaces:
                    self.draw_lights(1)  # lights behind the object
                    self.draw_surfaces()
                    self.draw_lights(-1)  # lights in front of the object
                else:
                    self.draw_edges()
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

    def rotate(self):

        # set matrix for rotation using angles, calculate rotated nodes (3D) and trans nodes (2D).

        self.angles[0, :] += self.rotate_speed[0, :] * (pygame.time.get_ticks() - self.rotate_timer) * np.pi / 500.0  # rotation defined as rounds per second
        self.angles[1:4, :] += self.rotate_speed[1:4, :] * self.light_speed * (pygame.time.get_ticks() - self.rotate_timer) * np.pi / 500.0
        self.rotate_timer = pygame.time.get_ticks()

        # rotate object
        matrix = self.rotate_matrix(self.angles[0])
        # rotate onject nodes but only to required depth
        self.rotated_nodes = np.dot(self.nodes[:self.nodes_at_depth[self.depth + 1]], matrix)
        # add zPos and transform from 3D to 2D; add midScreen to center on screen.
        self.trans_nodes = (self.rotated_nodes[:, 0:2] * self.z_scale) / (self.rotated_nodes[:, 2:3] + self.z_pos) + self.mid_screen

        # rotate lights
        if self.fill_surfaces:
            if self.light_speed > 0:
                matrix = self.rotate_matrix(self.angles[1])
                self.light_rotated_nodes[0, :] = np.dot(self.light_nodes[0, :], matrix)
                if self.light_nr == 3:
                    matrix = self.rotate_matrix(self.angles[2])
                    self.light_rotated_nodes[1, :] = np.dot(self.light_nodes[1, :], matrix)
                    matrix = self.rotate_matrix(self.angles[3])
                    self.light_rotated_nodes[2, :] = np.dot(self.light_nodes[2, :], matrix)
            self.light_trans_nodes = (self.light_rotated_nodes[:, 0:2] * self.z_scale) / (self.light_rotated_nodes[:, 2:3] + self.z_pos) + self.mid_screen

    def rotate_matrix(self, angles):

        # define rotation matrix for given angles
        (sx, sy, sz) = np.sin(angles)
        (cx, cy, cz) = np.cos(angles)

        # build a matrix for X, Y, Z rotation (in that order, see Wikipedia: Euler angles).
        return np.array([[cy * cz               , -cy * sz              , sy      ],
                         [cx * sz + cz * sx * sy, cx * cz - sx * sy * sz, -cy * sx],
                         [sx * sz - cx * cz * sy, cz * sx + cx * sy * sz, cx * cy ]])

    def clear_screen(self):

        # clear screen.
        self.screen.fill(self.background_color)
        self.draw_count = 0

    def draw_edges(self):

        # draw all edges.

        if self.phase in (2, 8):
            # fade in/out phase - draw middle surfaces with faded color, and still previous depth with full color
            start_depth = self.depth - 1
        else:
            start_depth = self.depth
        # set up "normal" edges - when fading in/out these will be one depth less.
        surf_beg = self.surfaces_at_depth[start_depth]
        surf_end = self.surfaces_at_depth[start_depth + 1]
        edge_beg = self.edges_at_depth[start_depth]
        edge_end = self.edges_at_depth[start_depth + 1]
        colors = self.edge_colors[:]

        surfs = self.surface_nodes[surf_beg:surf_end, :]
        angle_viewer, shade = self.calculate_surf(surfs, False)  # shade will be None
        # surf_vis = ((-np.sign(angle_viewer - 0.0000000001) + 1) / 2).astype(np.int8)  # 0 for back, 1 for front
        surf_vis = np.sign(angle_viewer)  # 1 for back, -1 for front
        surf_edges = self.surface_edges[surf_beg:surf_end, :][surf_vis == -1]  # visible (front) surface edges

        self.measure_time("prepare surfaces")

        edge_cols = np.zeros((edge_end))
        edge_cols[surf_edges[:, 0]] += 1
        edge_cols[surf_edges[:, 1]] += 1
        edge_cols[surf_edges[:, 2]] += 1
        edges_front = self.edge_nodes[edge_beg:edge_end, :][edge_cols[edge_beg:edge_end] > 0]
        if self.edges_at_depth[self.depth] < 1000:
            edges_back = self.edge_nodes[edge_beg:edge_end, :][edge_cols[edge_beg:edge_end] == 0]

        self.measure_time("prepare edges")

        if self.phase in (2, 8):
            # fade in/out phase - draw middle surfaces with faded color ("current" depth)
            start_depth = self.depth
            surf_beg = self.surfaces_at_depth[start_depth]
            surf_end = self.surfaces_at_depth[start_depth + 1]
            edge_beg = self.edges_at_depth[start_depth]
            edge_end = self.edges_at_depth[start_depth + 1]
            # add faded colors
            colors.append(([int(self.fade * x) for x in self.edge_colors[0]]))
            colors.append(([int(self.fade * x) for x in self.edge_colors[1]]))

            surfs = self.surface_nodes[surf_beg:surf_end, :]
            surf_edges = self.surface_edges[surf_beg:surf_end, :]
            angle_viewer, shade = self.calculate_surf(surfs, False)  # shade will be None
            # no overlap, so surf_vis can be -1 (back) or 1 (front); 0 if not drawn. surface_is_mid is 0 or 1.
            surf_vis = ((-np.sign(angle_viewer - 0.0000000001)) * self.surface_is_mid[surf_beg:surf_end]).astype(np.int8)

            self.measure_time("prepare surfaces")

            edge_cols = np.zeros((edge_end))
            # different approach to above; surf_vis can be -1, 0 or 1, and there are no overlapping edges.
            edge_cols[surf_edges[:, 0]] += surf_vis
            edge_cols[surf_edges[:, 1]] += surf_vis
            edge_cols[surf_edges[:, 2]] += surf_vis
            mid_edges_front = self.edge_nodes[edge_beg:edge_end, :][edge_cols[edge_beg:edge_end] > 0]
            if self.edges_at_depth[self.depth] < 1000:
                mid_edges_back = self.edge_nodes[edge_beg:edge_end, :][edge_cols[edge_beg:edge_end] < 0]

            self.measure_time("prepare edges")

        # draw, back edges first and fade in/out in the middle
        # if edge count exceeds 1000 at highest depth, skip drawing back side edges
        self.draw_count = 0
        if self.edges_at_depth[self.depth] < 1000:
            self.draw_count += np.shape(edges_back)[0]
            for i in range(np.shape(edges_back)[0]):
                pygame.draw.aaline(self.screen, colors[0], self.trans_nodes[edges_back[i, 0], :], self.trans_nodes[edges_back[i, 1], :])
        if self.phase in (2, 8):
            if self.edges_at_depth[self.depth] < 1000:
                self.draw_count += np.shape(mid_edges_back)[0]
                for i in range(np.shape(mid_edges_back)[0]):
                    pygame.draw.aaline(self.screen, colors[2], self.trans_nodes[mid_edges_back[i, 0], :], self.trans_nodes[mid_edges_back[i, 1], :])
            self.draw_count += np.shape(mid_edges_front)[0]
            for i in range(np.shape(mid_edges_front)[0]):
                pygame.draw.aaline(self.screen, colors[3], self.trans_nodes[mid_edges_front[i, 0], :], self.trans_nodes[mid_edges_front[i, 1], :])
        self.draw_count += np.shape(edges_front)[0]
        for i in range(np.shape(edges_front)[0]):
            pygame.draw.aaline(self.screen, colors[1], self.trans_nodes[edges_front[i, 0], :], self.trans_nodes[edges_front[i, 1], :])

        self.measure_time("draw")

        # # add little spheres to show original nodes
        # for i in range(self.nodes_at_depth[1]):
        #     if self.rotated_nodes[i, 2] > 0:
        #         color = self.edge_colors[0]
        #     else:
        #         color = self.edge_colors[1]
        #     pygame.draw.circle(self.screen, color, self.trans_nodes[i], 4)
        # for i in range(np.shape(self.trans_nodes)[0]):
        #     self.plot_info_msg(self.trans_nodes[i, 0], self.trans_nodes[i, 1], str(i))

    def draw_surfaces(self):

        # draw filled and light source shaded surfaces.

        s_start = self.surfaces_at_depth[self.depth]
        s_end = self.surfaces_at_depth[self.depth + 1]
        surfs = self.surface_nodes[s_start:s_end, :]
        angle_viewer, shade = self.calculate_surf(surfs, True)

        draw_surfs = surfs[angle_viewer < 0]
        col_shade = 0.05 + np.maximum(0.0, np.minimum(0.95, 0.3 + 0.7 * shade[angle_viewer < 0]))
        if self.fill_colors:
            # use image colors for surfaces
            if self.light_nr == 3:
                color = ((self.fill_colors_fade * self.surface_used_colors[s_start:s_end, :][angle_viewer < 0]
                         + (1.0 - self.fill_colors_fade) * np.asarray(self.edge_colors[1]))
                         * col_shade).astype(np.uint8)
            else:
                color = ((self.fill_colors_fade * self.surface_used_colors[s_start:s_end, :][angle_viewer < 0]
                         + (1.0 - self.fill_colors_fade) * np.asarray(self.edge_colors[1]))
                         * col_shade[:, 0:1]).astype(np.uint8)
        else:
            if self.light_nr == 3:
                color = (np.asarray(self.edge_colors[1]) * col_shade).astype(np.uint8)
            else:
                color = (np.asarray(self.edge_colors[1]) * col_shade[:, 0:1]).astype(np.uint8)

        self.measure_time("prepare surfaces")

        self.draw_count = np.shape(draw_surfs)[0]
        for i in range(np.shape(draw_surfs)[0]):
            pygame.draw.polygon(self.screen, color[i, :], self.trans_nodes[draw_surfs[i, :], :])

        self.measure_time("draw")

        # # much slower version with more loops and using classes
        # self.cross_product_count = 0
        # for surface in self.surfaces[self.surfaces_at_depth[self.depth]:self.surfaces_at_depth[self.depth + 1]]:
        #     # update surface visibility (front or back)
        #     # it is slow to calculate the cross products and angles - first check if it is obvious surface is not visible
        #     min_z = min(self.rotated_nodes[surface.nodes, 2])
        #     if min_z >= 0:
        #         surface.visible = 0
        #     else:
        #         self.update_surface_cross_product_vector(surface)
        #         self.cross_product_count += 1
        #         vec_viewer = self.rotated_nodes[surface.nodes[1], :] + np.array([0.0, 0.0, self.z_pos])
        #         surface.update_angle_to_viewer(vec_viewer)
        #         surface.update_visible()
        #         if surface.visible == 1:
        #             # for visible surfaces, calculate shading based on light source
        #             if self.phase in (4, 5):
        #                 # nodes being pushed - need to update
        #                 self.update_surface_cross_product_length(surface)
        #             vec_light = self.rotated_nodes[surface.nodes[1], :] - self.light_nodes[0]
        #             surface.update_angle_to_lightsource(vec_light)
        #             surface.update_shade()

        # self.measure_time("prepare surfaces")

        # for surface in (surf for surf in self.surfaces[self.surfaces_at_depth[self.depth]:self.surfaces_at_depth[self.depth + 1]] if surf.visible == 1):
        #     color = ([int(surface.shade * x) for x in self.edge_colors[1]])
        #     # node_list = [self.trans_nodes[surface.nodes[0]], self.trans_nodes[surface.nodes[1]], self.trans_nodes[surface.nodes[2]]]
        #     node_list = self.trans_nodes[surface.nodes, :]
        #     pygame.draw.polygon(self.screen, color, node_list)
        #     self.draw_count += 1

        # self.measure_time("draw")

    def calculate_surf(self, surfs, calc_shade):

        # this function returns the angle to viewer (unscaled!) and, if calc_shade is True, shade numpy arrays.

        vec_a = self.rotated_nodes[surfs[:, 2], :] - self.rotated_nodes[surfs[:, 1], :]
        vec_b = self.rotated_nodes[surfs[:, 0], :] - self.rotated_nodes[surfs[:, 1], :]
        cp_vector = np.hstack((
            (vec_b[:, 1] * vec_a[:, 2] - vec_b[:, 2] * vec_a[:, 1])[:, None],
            (vec_b[:, 2] * vec_a[:, 0] - vec_b[:, 0] * vec_a[:, 2])[:, None],
            (vec_b[:, 0] * vec_a[:, 1] - vec_b[:, 1] * vec_a[:, 0])[:, None]
            ))
        cp_length = np.linalg.norm(cp_vector, axis=1)

        # get surface mid points for more accurate and consistent calculations
        surf_mids = (self.rotated_nodes[surfs[:, 0], :] + self.rotated_nodes[surfs[:, 1], :] + self.rotated_nodes[surfs[:, 2], :]) / 3.0
        # vec_viewer = self.rotated_nodes[surfs[:, 1], :] + np.array([0.0, 0.0, self.z_pos])
        vec_viewer = surf_mids + np.array([0.0, 0.0, self.z_pos])
        angle_viewer = np.sum(cp_vector * vec_viewer, axis=1)
        if calc_shade:
            shade = np.zeros((np.shape(vec_viewer)[0], 3), dtype=float)
            for i in range(self.light_nr):
                # vec_light = self.rotated_nodes[surfs[:, 1], :] - self.light_rotated_nodes[i]
                vec_light = surf_mids - self.light_rotated_nodes[i]
                light_length = np.linalg.norm(vec_light, axis=1)
                shade[:, i] = -np.arcsin(np.sum(cp_vector * vec_light, axis=1) / (cp_length * light_length)) / (np.pi / 2)
            return angle_viewer, shade
        else:
            return angle_viewer, None

    def draw_lights(self, direction):

        # draw lights by copying a predrawn image to screen. If direction = 1, Z > 0 i.e. lights behind the object, -1: Z < 0: in front of the object

        for light in range(self.light_nr):
            if self.light_rotated_nodes[light, 2] * direction >= 0:
                pic = bisect.bisect_right(self.light_pic_zcoords, self.light_rotated_nodes[light, 2])  # find pic number using light Z coordinate
                c = int(self.light_pic_max_size / 2 - pic)  # center of max size pic, adjust byt pic index (pics grow smaller)
                if self.light_nr == 3:
                    self.screen.blit(self.light_pics[light + 1][pic], self.light_trans_nodes[light] - c)
                else:
                    # only plot the first light as white light
                    self.screen.blit(self.light_pics[0][pic], self.light_trans_nodes[light] - c)

    def add_depth(self):

        # add a level of depth to the object.
        # for each surface at the current depth, add the required nodes at the middle of its edges and build four new surfaces

        self.depth += 1
        used_edges = []
        used_edge_mid_nodes = []
        node_nr = np.shape(self.nodes)[0]
        # add nr of nodes to nodes_at_depth etc - then update these as progresses
        self.nodes_at_depth.append(node_nr)
        self.edges_at_depth.append(len(self.edges))
        self.surfaces_at_depth.append(len(self.surfaces))

        plot_update_freq = 30
        plot_update_nr = 0

        for surface in (surf for surf in self.surfaces if surf.depth == self.depth - 1):
            # get or if needed add mid nodes for each (three) edges
            mid_nodes = []
            mid_edges = []
            for i in range(3):
                edge = surface.edges[i]
                if edge in used_edges:
                    mid_nodes.append(used_edge_mid_nodes[used_edges.index(edge)])
                else:
                    # add a new node at the middle
                    new_node = (self.nodes[edge.nodes[0]] + self.nodes[edge.nodes[1]]) / 2
                    self.nodes = np.vstack((self.nodes, new_node))
                    mid_nodes.append(node_nr)
                    used_edges.append(edge)
                    used_edge_mid_nodes.append(node_nr)
                    node_nr += 1
            # then, build the corner surfaces and edges
            for i in range(3):
                edge = surface.edges[i]
                if edge.nodes[0] in surface.edges[i - 1].nodes:
                    # common node with previous edge - this is the corner
                    corner_node = edge.nodes[0]
                else:
                    corner_node = edge.nodes[1]
                # add new edges - if necessary
                edge_1 = self.add_edge(corner_node, mid_nodes[i])
                edge_2 = self.add_edge(mid_nodes[i],  mid_nodes[i - 1])
                edge_3 = self.add_edge(mid_nodes[i - 1], corner_node)
                # add the surface
                self.add_surface(surface, edge_1, edge_2, edge_3, 0)
                # store the mid edge
                mid_edges.append(edge_2)
            # finally, use the middle edges to build the middle surface
            self.add_surface(surface, mid_edges[0], mid_edges[1], mid_edges[2], 1)

            plot_update_nr += 1
            if plot_update_nr == plot_update_freq:
                plot_update_nr = 0
                # update these data to show progress
                self.nodes_at_depth[-1] = node_nr
                self.edges_at_depth[-1] = len(self.edges)
                self.surfaces_at_depth[-1] = len(self.surfaces)
                self.screen.fill(self.background_color, (10, 10, 600, 120))  # clear screen
                self.plot_info()
                # pygame.display.flip()

        # update these data to final levels and show progress
        self.nodes_at_depth[-1] = node_nr
        self.edges_at_depth[-1] = len(self.edges)
        self.surfaces_at_depth[-1] = len(self.surfaces)
        self.screen.fill(self.background_color, (10, 10, 600, 120))  # clear screen
        self.plot_info()
        pygame.display.flip()

        # pre-calculate surface cross product vector lengths
        self.rotated_nodes = self.nodes
        for surface in self.surfaces[self.surfaces_at_depth[-2]:]:
            self.update_surface_cross_product_vector(surface)
            self.update_surface_cross_product_length(surface)

        # store the new nodes for "pushing"
        node_nr = self.nodes_at_depth[-2]
        self.push_nodes = np.vstack((self.push_nodes[:node_nr, :], self.nodes[node_nr:, :]))
        self.push_length = np.sqrt(self.push_nodes[:, 0] ** 2 + self.push_nodes[:, 1] ** 2 + self.push_nodes[:, 2] ** 2)
        ratio = self.sphere_radius / self.push_length[node_nr:]
        # apply to nodes for next iterative addition of nodes
        self.nodes[node_nr:, :] = self.push_nodes[node_nr:, :] * ratio[:, None]

    def copy_colors_down(self):

        # copy colors from highest depth all the way down by averaging parent surface color from its children.

        for depth in range(self.depth, 0, -1):
            prev_parent = None
            child_count = 0
            color = np.zeros((3), dtype=np.int32)
            for surface in (surf for surf in self.surfaces if surf.depth == depth):
                # children are always next to each other
                parent = surface.parent
                if prev_parent is None:
                    prev_parent = parent
                if parent == prev_parent:
                    color += surface.color
                    child_count += 1
                else:
                    if child_count > 0:
                        prev_parent.color = (color / child_count).astype(np.uint8)
                    prev_parent = parent
                    child_count = 0
                    color = np.zeros((3), dtype=np.int32)
            if child_count > 0:
                prev_parent.color = (color / child_count).astype(np.uint8)

    def push_to_radius(self, push_ratio):

        # push new nodes to sphere radius. push_ratio goes from 0 to 1 --> ratios go from 1 to required x
        node_beg = self.nodes_at_depth[self.depth]
        node_end = self.nodes_at_depth[self.depth + 1]
        ratio = (self.push_length[node_beg:node_end] + push_ratio * (self.sphere_radius - self.push_length[node_beg:node_end])) \
            / self.push_length[node_beg:node_end]
        self.nodes[node_beg:node_end, :] = self.push_nodes[node_beg:node_end] * ratio[:, None]

    def map_single_surface_to_image(self, surface):

        # using the image, get average color for a surface

        nodes = self.nodes[surface.nodes, :]
        # transform Cartesian coordinates (X. Y. Z) to Polar (longitude, latitude) between 0 and 1
        polar = np.hstack((np.arctan2(nodes[:, 0], nodes[:, 2])[:, None] / (2.0 * np.pi) + 0.5,
                           np.arccos(nodes[:, 1] / self.sphere_radius)[:, None] / np.pi))

        if self.image_type == 'Mercator':
            # Mercator mode; y in image assumed to be between -85 and 85 degrees. (https://en.wikipedia.org/wiki/Web_Mercator_projection)
            R = 1.0 / np.log(np.tan(np.pi / 4 + (85.0 / 90.0) * np.pi / 4))  # with this R y is in [-1,1] between -85 and +85 degrees
            surf_coord = np.hstack((polar[:, 0:1] * self.image_size[0],  # x as in "normal"
                                    ((1.0 + R * np.log(np.tan(np.pi / 4 + 0.4999 * (polar[:, 1:2] * np.pi - np.pi / 2)))) / 2.0) * self.image_size[1]))
        #     surf_coord_y = (np.minimum(0.9999, np.maximum(0.0, ((1.0 + R * np.log(np.tan(np.pi / 4 + 0.4999 * (np.arccos(self.rotated_nodes_flat[:, 1] / self.radius)
        #                    - np.pi / 2)))) / 2.0))) * self.image_size[1]).astype(np.int16)
            # Mercator results in poles being outsied of picture - clip y coordinates
            surf_coord[:, 1][surf_coord[:, 1] < 0] = 0
            surf_coord[:, 1][surf_coord[:, 1] > self.image_size[1] - 1] = self.image_size[1] - 1
        else:
            # Normal mode: picture is simply wrapped
            surf_coord = polar * self.image_size  # as original image size

        min_coord = np.min(surf_coord, axis=0)
        max_coord = np.max(surf_coord, axis=0)
        if min_coord[0] < self.image_size[0] / 4 and max_coord[0] > self.image_size[0] * 3 / 4:
            # move leftmost to the right, if others close to right edge
            surf_coord[surf_coord[:, 0] < self.image_size[0] / 4] += np.array([self.image_size[0], 0])
        if min_coord[1] < self.image_size[1] / 4 and max_coord[1] > self.image_size[1] * 3 / 4:
            # move uppermost down, if others close to bottom edge
            surf_coord[surf_coord[:, 1] < self.image_size[1] / 4] += np.array([0, self.image_size[1]])
        if np.shape(surf_coord)[0] == 3 and np.shape(surf_coord[surf_coord[:, 1] == 0])[0] == 1:
            # triangle has one node at north pole; make it a quadrilateral (i.e. four node plane)
            surf_coord = np.vstack((
               surf_coord[surf_coord[:, 1] != 0],                       # the nodes not at the pole
               np.array([surf_coord[surf_coord[:, 1] != 0][1, 0], 0]),  # the second node X at Y = 0
               np.array([surf_coord[surf_coord[:, 1] != 0][0, 0], 0]),  # the first node X at Y = 0
               ))
        elif np.shape(surf_coord)[0] == 3 and np.shape(surf_coord[surf_coord[:, 1] == self.image_size[1] - 1])[0] == 1:
            # triangle has one node at south pole; make it a quadrilateral (i.e. four node plane)
            surf_coord = np.vstack((
               surf_coord[surf_coord[:, 1] != self.image_size[1] - 1],                                   # the nodes not at the pole
               np.array([surf_coord[surf_coord[:, 1] != self.image_size[1]][1, 0], self.image_size[1] - 1]),  # the second node X at Y = self.image_size Y
               np.array([surf_coord[surf_coord[:, 1] != self.image_size[1]][0, 0], self.image_size[1] - 1]),  # the first node X at Y = self.image_size Y
               ))
        line_coord = (surf_coord / self.image_ratio + self.image_pos).astype(np.int16)  # for drawing change to screen image size and add position
        if self.color_surfaces == 2:
            # calculate the average color of each surface using the original image
            surface.color = self.get_surface_color(self.image_array, surf_coord.astype(np.int16))
            pygame.draw.polygon(self.screen, surface.color, line_coord)
        elif self.color_surfaces == 1:
            # if not colored, just draw outlines
            for j in range(np.shape(line_coord)[0]):
                pygame.draw.aaline(self.screen, (200, 200, 200), line_coord[j - 1, :], line_coord[j, :])

        if pygame.time.get_ticks() > self.start_timer + 4 * 1000 / self.target_fps:
            # show progress on screen
            pygame.display.flip()
            self.start_timer = pygame.time.get_ticks()

    def get_surface_color(self, image_array, coords):

        # takes an image array and coordinates of a convex polygon (coord) and calculates the average color of the image within the polygon.
        # polygon left edge is included and right edge is excluded, top edge is included and bottom edge is excluded.
        # coords must be integers.

        color = np.zeros((3), dtype=np.int32)
        pixel_count = 0
        min_y = np.min(coords[:, 1])
        max_y = np.max(coords[:, 1])
        if max_y > np.shape(image_array)[1]:
            max_y = np.shape(image_array)[1]

        coords_nr = np.shape(coords)[0]
        # find topmost nodes
        co_0 = -1
        for i in range(coords_nr):
            if coords[i, 1] == min_y:
                if co_0 == -1:
                    # first top node found
                    co_0 = i
                    co_1 = i  # if only one found, use it for both
                else:
                    # found another at the top, use it instead
                    if i > co_0 + 1:
                        co_1 = i  # the nodes are at the first and last coord
                    else:
                        co_0 = i
                    break

        co_2 = co_0 + 1   # next node "clockwise"
        if co_2 == coords_nr:
            co_2 = 0
        co_3 = co_1 - 1   # next node "counter clockwise"  (actually the directions are not known but all that matters is that we go "two ways")
        if co_3 < 0:
            co_3 += coords_nr

        (x_0, y_0) = coords[co_0, :]
        (x_0_add, y_0_add) = coords[co_2, :] - coords[co_0, :]
        (x_1, y_1) = coords[co_1, :]
        (x_1_add, y_1_add) = coords[co_3, :] - coords[co_1, :]
        if y_0_add > 0 and y_1_add > 0:
            # go through each horizontal line and calculate both color sum and increment pixel count
            for y in range(min_y, max_y):
                # check for need to change coords at the edges
                if y > coords[co_2, 1]:
                    co_0 += 1
                    if co_0 == coords_nr:
                        co_0 = 0
                    co_2 += 1
                    if co_2 == coords_nr:
                        co_2 = 0
                    (x_0, y_0) = coords[co_0, :]
                    (x_0_add, y_0_add) = coords[co_2, :] - coords[co_0, :]
                    if y_0_add == 0:
                        break
                if y > coords[co_3, 1]:
                    co_1 -= 1
                    if co_1 < 0:
                        co_1 += coords_nr
                    co_3 -= 1
                    if co_3 < 0:
                        co_3 += coords_nr
                    (x_1, y_1) = coords[co_1, :]
                    (x_1_add, y_1_add) = coords[co_3, :] - coords[co_1, :]
                    if y_1_add == 0:
                        break
                # interpolate X coordinates
                x_left = int(x_0 + x_0_add * (y - y_0) / y_0_add)
                x_right = int(x_1 + x_1_add * (y - y_1) / y_1_add)
                if x_right < x_left:
                    # "wrong" order
                    color += np.sum(image_array[x_right:x_left, y, :], axis=0)
                    pixel_count += x_left - x_right
                elif x_right > x_left:
                    color += np.sum(image_array[x_left:x_right, y, :], axis=0)
                    pixel_count += x_right - x_left

        if pixel_count == 0:
            return np.zeros((3), dtype=np.uint8)
        else:
            col = color / pixel_count
            # apply brightness first
            col += (255.0 - col) * self.image_brightness
            # apply contrast
            col = (1.0 - self.image_contrast) * col + self.image_contrast * (1 + np.sin((np.pi / 2) * (col - 127.5) / 127.5)) * 127.5
            return col.astype(np.uint8)

    def image_setup(self):

        # set up image for screen use and for array use.

        screen_size = np.array([self.width, self.height]).astype(np.int16)
        self.image_ratio = max(self.image_size[0] * self.image_ext / self.width, self.image_size[1] / self.height) / 0.9
        if self.image_ratio > 1.0:
            # scale image if larger than desired
            self.image_screen_size = (self.image_size / self.image_ratio).astype(np.int16)
            self.image_screen = pygame.transform.scale(self.image, self.image_screen_size)
        else:
            self.image_ratio = 1.0
            self.image_screen_size = self.image_size
            self.image_screen = self.image
        self.image_pos = ((screen_size - self.image_screen_size * np.array([self.image_ext, 1.0])) / 2).astype(np.int16)

        # make a bigger version of the original image as well as a surfarray. Using vstack as first axis is the X coordinate
        self.image_array = pygame.surfarray.pixels3d(self.image)
        self.image_array = np.vstack((self.image_array, self.image_array[0:int(self.image_size[0] * (self.image_ext - 1.0)), :]))

    def image_copy(self):

        # copy image to screen

        self.screen.blit(self.image_screen, self.image_pos)
        # additional section
        self.screen.blit(self.image_screen, self.image_pos + np.array([self.image_screen_size[0], 0]),
                         (0, 0, int((self.image_ext - 1.0) * self.image_screen_size[0]), self.image_screen_size[1]))

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

    def toggle_fill_surfaces(self):

        if self.fill_surfaces:
            self.fill_surfaces = False
        else:
            self.fill_surfaces = True

    def toggle_info_display(self):

        # switch between a windowed display and full screen
        if self.info_display:
            self.info_display = False
        else:
            self.info_display = True

    def plot_info(self):

        # show info on object and performance
        while self.screen.get_locked():
            self.screen.unlock()

        if not self.fill_surfaces:
            light_msg = 'Wireframe -> None'
        elif self.light_nr == 1 and self.fill_colors:
            light_msg = 'One light, image, speed ' + str(self.light_speed)
        elif self.light_nr == 1 and not self.fill_colors:
            light_msg = 'One light, speed ' + str(self.light_speed)
        elif self.light_nr == 3 and self.fill_colors:
            light_msg = 'Three lights, image, speed ' + str(self.light_speed)
        elif self.light_nr == 3 and not self.fill_colors:
            light_msg = 'Three lights, speed ' + str(self.light_speed)

        self.screen_info.fill(self.background_color, (10, 10, 550, 160 + len(self.timer_names) * 15))
        # print object info
        self.plot_info_msg(self.screen_info, 10, 10, 'status:         ' + self.status_msg)
        self.plot_info_msg(self.screen_info, 10, 25, 'lighting:       ' + light_msg)
        self.plot_info_msg(self.screen_info, 10, 40, 'depth:          ' + str(self.depth + 1))
        self.plot_info_msg(self.screen_info, 10, 55, 'nr of nodes:    ' + ' + '.join([str(x - y) for x, y in zip(self.nodes_at_depth[1:self.depth + 2],
                                                                                                 self.nodes_at_depth[:self.depth + 1])])
                                                                        + ' = ' + str(self.nodes_at_depth[self.depth + 1]))
        self.plot_info_msg(self.screen_info, 10, 70, 'nr of edges:    ' + str(self.edges_at_depth[self.depth + 1] - self.edges_at_depth[self.depth]))
        self.plot_info_msg(self.screen_info, 10, 85, 'nr of surfaces: ' + str(self.surfaces_at_depth[self.depth + 1] - self.surfaces_at_depth[self.depth]))
        self.plot_info_msg(self.screen_info, 10, 100, 'draw count:     ' + str(self.draw_count))
        self.plot_info_msg(self.screen_info, 10, 115, 'frames per sec: ' + str(int(self.clock.get_fps())))

        # add measured times as percentage of total
        tot_time = np.sum(self.timers)
        if tot_time > 0:
            for i in range(len(self.timer_names)):
                info_msg = (self.timer_names[i] + ' '*16)[:16] + (' '*10 + str(round(np.sum(self.timers[i, :]) * 100 / tot_time, 1)))[-7:]
                self.plot_info_msg(self.screen_info, 10, 140 + i * 15, info_msg)

        # add keyboard info:
        self.screen.blit(self.screen_info, (10, 10), (10, 10, 550, 15 + 15 + 90 + 160 + len(self.timer_names) * 15))

    def setup_info_screen(self):

        # show keys
        while self.screen_info.get_locked():
            self.screen_info.unlock()
        i = 160 + 15 + len(self.timer_names) * 15
        self.plot_info_msg(self.screen_info, 10,  0 + i, 'change mode:    left / right')
        self.plot_info_msg(self.screen_info, 10, 15 + i, 'change depth:   up / down')
        self.plot_info_msg(self.screen_info, 10, 30 + i, 'lighting mode:  l')
        self.plot_info_msg(self.screen_info, 10, 45 + i, 'light speed:    n / m')
        self.plot_info_msg(self.screen_info, 10, 60 + i, 'fullscreen o/o: f')
        self.plot_info_msg(self.screen_info, 10, 75 + i, 'info o/o:       i')
        self.plot_info_msg(self.screen_info, 10, 90 + i, 'pause:          space')

        self.screen_info.set_colorkey(self.background_color)

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

    def update_surface_cross_product_vector(self, surface):

        # calculate cross product vector for surface using rotated_nodes
        # always use vectors (1, 0) and (1, 2) (numbers representing nodes)
        vec_a = self.rotated_nodes[surface.nodes[2], :] - self.rotated_nodes[surface.nodes[1], :]
        vec_b = self.rotated_nodes[surface.nodes[0], :] - self.rotated_nodes[surface.nodes[1], :]
        surface.cross_product_vector = ([
            vec_b[1] * vec_a[2] - vec_b[2] * vec_a[1],
            vec_b[2] * vec_a[0] - vec_b[0] * vec_a[2],
            vec_b[0] * vec_a[1] - vec_b[1] * vec_a[0]
            ])

    def update_surface_cross_product_length(self, surface):

        surface.cross_product_length = self.vector_length(surface.cross_product_vector)

    def vector_length(self, vector):

        return math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])

    def setup_tetra(self):

        # set up a regular tetrahedron inside a sphere with center (0,0,0).
        # nodes see https://en.wikipedia.org/wiki/Tetrahedron
        self.nodes = 200 * np.array([
            [0.0, 0.0, 1.0],
            [np.sqrt(8.0/9.0), 0, -1.0 / 3.0],
            [-np.sqrt(2.0/9.0), np.sqrt(2.0 / 3.0), -1.0 / 3.0],
            [-np.sqrt(2.0/9.0), -np.sqrt(2.0 / 3.0), -1.0 / 3.0]
            ])
        self.rotated_nodes = self.nodes.copy()  # init with nodes
        self.push_nodes = self.nodes.copy()  # init with nodes
        self.nodes_at_depth.append(np.shape(self.nodes)[0])

        # set sphere radius for later "pushing" and surface mapping
        self.sphere_radius = self.vector_length(self.nodes[0])

        # define edges
        self.add_edge(0, 1)  # 0
        self.add_edge(0, 2)  # 1
        self.add_edge(0, 3)  # 2
        self.add_edge(1, 2)  # 3
        self.add_edge(2, 3)  # 4
        self.add_edge(3, 1)  # 5
        self.edges_at_depth.append(len(self.edges))

        # define surfaces: Three edges in clockwise order, and a 0/1 for mid_surface
        self.add_surface(None, self.edges[0], self.edges[1], self.edges[3], 0)
        self.add_surface(None, self.edges[1], self.edges[2], self.edges[4], 0)
        self.add_surface(None, self.edges[2], self.edges[0], self.edges[5], 0)
        self.add_surface(None, self.edges[3], self.edges[4], self.edges[5], 0)
        self.surfaces_at_depth.append(len(self.surfaces))

        # pre-calculate surface cross product vector lengths
        for surface in self.surfaces:
            self.update_surface_cross_product_vector(surface)
            self.update_surface_cross_product_length(surface)

        # rotation speed defined as rounds per second
        self.rotate_speed = np.array([[0.06, 0.043, -0.07],
                                      [0.033, -0.027, 0.05],
                                      [-0.024, 0.057, 0.038],
                                      [0.037, -0.052, -0.03]])

    def setup_ico(self):

        # set up a regular icosahedron inside a sphere with center (0,0,0).
        # nodes see https://en.wikipedia.org/wiki/Regular_icosahedron
        grt = 1 / ((1 + np.sqrt(5)) / 2)  # golden ratio (inverted)
        scale = np.sqrt(1 + grt ** 2)
        self.nodes = (200.0 / scale) * np.array([
            [ 0.0,  1.0,  grt],
            [ 0.0,  1.0, -grt],
            [ 0.0, -1.0, -grt],
            [ 0.0, -1.0,  grt],
            [ 1.0,  grt,  0.0],
            [ 1.0, -grt,  0.0],
            [-1.0, -grt,  0.0],
            [-1.0,  grt,  0.0],
            [ grt,  0.0,  1.0],
            [-grt,  0.0,  1.0],
            [-grt,  0.0, -1.0],
            [ grt,  0.0, -1.0]
            ])
        self.rotated_nodes = self.nodes.copy()  # init with nodes
        self.push_nodes = self.nodes.copy()  # init with nodes
        self.nodes_at_depth.append(np.shape(self.nodes)[0])

        # set sphere radius for later "pushing" and surface mapping
        self.sphere_radius = self.vector_length(self.nodes[0])

        # define edges
        self.add_edge(0, 1)  # 0
        self.add_edge(0, 7)  # 1
        self.add_edge(0, 9)  # 2
        self.add_edge(0, 8)  # 3
        self.add_edge(0, 4)  # 4
        self.add_edge(1, 7)  # 5
        self.add_edge(7, 9)  # 6
        self.add_edge(9, 8)  # 7
        self.add_edge(8, 4)  # 8
        self.add_edge(4, 1)  # 9
        self.add_edge(1, 10)  # 10
        self.add_edge(10, 7)  # 11
        self.add_edge(7, 6)  # 12
        self.add_edge(6, 9)  # 13
        self.add_edge(9, 3)  # 14
        self.add_edge(3, 8)  # 15
        self.add_edge(8, 5)  # 16
        self.add_edge(5, 4)  # 17
        self.add_edge(4, 11)  # 18
        self.add_edge(11, 1)  # 19
        self.add_edge(10, 6)  # 20
        self.add_edge(6, 3)  # 21
        self.add_edge(3, 5)  # 22
        self.add_edge(5, 11)  # 23
        self.add_edge(11, 10)  # 24
        self.add_edge(2, 10)  # 25
        self.add_edge(2, 6)  # 26
        self.add_edge(2, 3)  # 27
        self.add_edge(2, 5)  # 28
        self.add_edge(2, 11)  # 29
        self.edges_at_depth.append(len(self.edges))

        # define surfaces: Three edges in clockwise order, and a 0/1 for mid_surface
        self.add_surface(None, self.edges[0], self.edges[1], self.edges[5], 0)
        self.add_surface(None, self.edges[1], self.edges[2], self.edges[6], 0)
        self.add_surface(None, self.edges[2], self.edges[3], self.edges[7], 0)
        self.add_surface(None, self.edges[3], self.edges[4], self.edges[8], 0)
        self.add_surface(None, self.edges[4], self.edges[0], self.edges[9], 0)
        self.add_surface(None, self.edges[5], self.edges[11], self.edges[10], 0)
        self.add_surface(None, self.edges[6], self.edges[13], self.edges[12], 0)
        self.add_surface(None, self.edges[7], self.edges[15], self.edges[14], 0)
        self.add_surface(None, self.edges[8], self.edges[17], self.edges[16], 0)
        self.add_surface(None, self.edges[9], self.edges[19], self.edges[18], 0)
        self.add_surface(None, self.edges[20], self.edges[11], self.edges[12], 0)
        self.add_surface(None, self.edges[21], self.edges[13], self.edges[14], 0)
        self.add_surface(None, self.edges[22], self.edges[15], self.edges[16], 0)
        self.add_surface(None, self.edges[23], self.edges[17], self.edges[18], 0)
        self.add_surface(None, self.edges[24], self.edges[19], self.edges[10], 0)
        self.add_surface(None, self.edges[25], self.edges[20], self.edges[26], 0)
        self.add_surface(None, self.edges[26], self.edges[21], self.edges[27], 0)
        self.add_surface(None, self.edges[27], self.edges[22], self.edges[28], 0)
        self.add_surface(None, self.edges[28], self.edges[23], self.edges[29], 0)
        self.add_surface(None, self.edges[29], self.edges[24], self.edges[25], 0)
        self.surfaces_at_depth.append(len(self.surfaces))

        # pre-calculate surface cross product vector lengths
        for surface in self.surfaces:
            self.update_surface_cross_product_vector(surface)
            self.update_surface_cross_product_length(surface)

        # rotation speed defined as rounds per second. First is the object, then three light sources.
        self.rotate_speed = np.array([[0.06, 0.083, -0.07],
                                      [0.033, -0.027, 0.05],
                                      [-0.024, 0.057, 0.038],
                                      [0.037, -0.052, -0.03]])

    def add_edge(self, node_1, node_2):

        # adds an edge connecting two nodes
        e = Edge()
        e.nodes.append(node_1)
        e.nodes.append(node_2)
        e.depth = self.depth
        # first check if edge exists already. Only testing against same depth edges
        if len(self.edges_at_depth) > 1:
            for edge in self.edges[self.edges_at_depth[-1]:]:
                if e == edge:
                    return edge
        # not found - add to edges
        e.edge_ix = len(self.edges)
        self.edges.append(e)
        return e

    def add_surface(self, parent, edge_1, edge_2, edge_3, mid_surface):

        # adds a surface
        s = Surface(parent)
        s.edges.append(edge_1)
        s.edges.append(edge_2)
        s.edges.append(edge_3)
        s.edge_ixs = np.array([edge_1.edge_ix, edge_2.edge_ix, edge_3.edge_ix], dtype=np.int16)
        s.mid_surface = mid_surface
        s.depth = self.depth
        s.update_nodes()
        self.surfaces.append(s)
        self.map_single_surface_to_image(s)

    def setup_lights(self):

        light_dist = np.linalg.norm(self.light_nodes[0])  # light distance from origo
        min_size = self.light_size * self.z_scale / (self.z_pos + light_dist)  # light size at far out
        max_size = self.light_size * self.z_scale / (self.z_pos - light_dist)  # light size closest to viewer
        self.light_pic_min_size = int(min_size / 2) * 2
        self.light_pic_max_size = (int(max_size / 2) + 1) * 2
        for light in range(4):
            color = np.asarray(self.edge_color_set[1])
            if light == 1:
                color[1:3] = 0  # Red only
            elif light == 2:
                color[0] = 0  # Green only
                color[2] = 0  # Green only
            elif light == 3:
                color[0:2] = 0  # Blue only

            light_pics = []
            for c in range(int(max_size / 2) + 1, int(min_size / 2) - 1, -1):
                surf = pygame.Surface((c * 2, c * 2))
                surf.set_colorkey(self.background_color)
                # set color shades
                rads = np.arange(1, c + 1)
                col_shades = (np.sin(0.3 + 0.7 * (c - rads + 1) / c * np.pi / 2)[:, None] * color).astype(np.uint8)
                # draw cocentric circles
                for r in range(c, 0, -1):
                    pygame.draw.circle(surf, col_shades[r - 1, :], (c, c), r)
                # add to list
                light_pics.append(surf)
                # store min z position for this light pic
                if light == 0:
                    self.light_pic_zcoords.append(int(self.light_size * self.z_scale / (c * 2) - self.z_pos))
            self.light_pics.append(light_pics)


class Edge:

    """
    An edge connects two nodes.

    @author: kalle
    """

    def __init__(self):
        self.depth = 0
        self.edge_ix = -1   # stores edge index in self.edges
        self.nodes = []

    def __eq__(self, other):
        # same depth edges are equal if they have the same nodes in either direction
        return (self.depth == other.depth and ((self.nodes[0] == other.nodes[0] and self.nodes[1] == other.nodes[1])
                                               or (self.nodes[1] == other.nodes[0] and self.nodes[0] == other.nodes[1])))


class Surface:

    """
    A surface is a plane defined be three edges.

    @author: kalle
    """

    def __init__(self, parent):

        self.parent = parent    # refers to parent surface, as levels are based on splitting one surface to four. "None" for first level depth.
        self.depth = 0          # the "depth" of the surface - original tetra = 0, then increasing
        self.mid_surface = 0    # this is "1" if, when adding depth, this surface is the "middle" of the bigger triangle
        self.edges = []
        self.edge_ixs = np.zeros((1, 3), dtype=np.int16)      # edge indexes in self.edges
        self.nodes = []
        self.cross_product_vector = np.zeros((0, 3))
        self.cross_product_length = 0.0   # precalculated length of the cross product vector - this is constant
        self.angle_to_viewer = 0.0
        self.angle_to_lightsource = 0.0
        self.shade = 1.0
        self.color = np.array([0, 0, 0], dtype=np.uint8)
        self.visible = 1

    def update_nodes(self):

        # update list of nodes based on edges. Take both nodes from first edge and the remaining third from the second edge.
        # assuming the edges are listed in clockwise order, list also nodes clockwise by having the common node in the middle.
        if self.edges[0].nodes[0] in self.edges[1].nodes:
            # common node = [0][0]
            self.nodes.append(self.edges[0].nodes[1])
            self.nodes.append(self.edges[0].nodes[0])
            if self.edges[1].nodes[0] == self.edges[0].nodes[0]:
                self.nodes.append(self.edges[1].nodes[1])
            else:
                self.nodes.append(self.edges[1].nodes[0])
        else:
            # common node = [0][1]
            self.nodes.append(self.edges[0].nodes[0])
            self.nodes.append(self.edges[0].nodes[1])
            if self.edges[1].nodes[0] == self.edges[0].nodes[1]:
                self.nodes.append(self.edges[1].nodes[1])
            else:
                self.nodes.append(self.edges[1].nodes[0])

    def update_angle_to_viewer(self, vec_viewer):

        # instead of true angle calculation using asin and vector lengths, a simple np.dot is sufficient to find the sign (which defines if surface is visible)
        # self.angle_to_viewer = math.asin(np.dot(self.cross_product_vector, vec_viewer) / (self.cross_product_length * np.linalg.norm(vec_viewer)))
        self.angle_to_viewer = np.dot(self.cross_product_vector, vec_viewer)

    def update_angle_to_lightsource(self, vec_light):

        self.angle_to_lightsource = math.asin(min(1.0, max(-1.0, np.dot(self.cross_product_vector, vec_light)
                                                           / (self.cross_product_length * self.vector_length(vec_light)))))

    def update_shade(self):

        self.shade = 0.1 + min(0.9, max(0.0, 0.1 - 0.85 * self.angle_to_lightsource / (np.pi / 2)))

    def update_visible(self):

        if self.angle_to_viewer < 0:
            self.visible = 1
        else:
            self.visible = 0

    def vector_length(self, vector):

        # equivalent to numpy.linalg.norm for a 3D real vector, but much faster. math.sqrt is faster than numpy.sqrt.
        return math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])


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
    music_file = "alacrity.ogg"  # this mod by Jellybean is available at e.g. http://janeway.exotica.org.uk/release.php?id=39506
    pygame.mixer.music.load(music_file)
    pygame.mixer.music.play(loops=-1)

    screen = pygame.display.set_mode(disp_size)
    image_filename = 'Normal_Mercator_map_85deg.jpg'
    image_type = 'Mercator'  # supported: Normal, and Mercator (assumed image between -85 and 85 degrees)
    image_brightness = 0.3  # add some brightness. This will brighten dark colors.
    image_contrast = 1.0  # add some contrast. This is how much weight is given to sine curve based contrast addition.
    # image_filename = 'image.png'
    # image_type = 'Normal'  # supported: Normal, and Mercator (assumed image between -85 and 85 degrees)
    # image_brightness = 0.0  # add some brightness. This will brighten dark colors.
    # image_contrast = 0.0  # add some contrast. This is how much weight is given to sine curve based contrast addition.
    pygame.display.set_caption('RGB Sphere')
    RGBSphere(screen, image_filename, image_type, image_brightness, image_contrast, 60).run()

    # exit; close display, stop music
    pygame.quit()
    exit()

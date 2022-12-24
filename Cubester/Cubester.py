# -*- coding: utf-8 -*-
import pygame
import numpy as np
from sys import exit


class Cubester:
    """
    @author: kalle
    """

    def __init__(self, screen, target_fps):

        self.screen = screen
        self.target_fps = target_fps
        self.width = self.screen.get_size()[0]
        self.height = self.screen.get_size()[1]
        self.mid_screen = np.array([int(self.width / 2), int(self.height / 2)], dtype=np.float32)
        self.background_color = (0, 0, 0)
        self.screen_info = pygame.Surface((300, 300), 0, self.screen)
        self.screen_info.set_colorkey(self.background_color)
        self.screen_buttons = pygame.Surface((self.height // 2.8, self.height), 0, self.screen)
        self.screen_buttons.set_colorkey(self.background_color)
        self.screen_button_mouseover = pygame.Surface((1, 1), 0, self.screen)  # will be set in setup_buttons
        self.screen_button_selected = pygame.Surface((1, 1), 0, self.screen)  # will be set in setup_buttons

        self.running = True
        self.paused = False
        self.clock = pygame.time.Clock()

        self.size = 3  # starting cube size

        self.disc_keys_x = 'qwertyuio'
        self.disc_keys_y = 'asdfghjkl'
        self.disc_keys_z = 'zxcvbnm,.'
        self.disc_keys_len = len(self.disc_keys_x)  # x, y, z must have equal length
        self.disc_keys_all = [self.disc_keys_x, self.disc_keys_y, self.disc_keys_z]

        self.disc = np.array([-1, -1, -1])
        self.disc_direction = 1
        self.disc_anim_ms = np.sqrt(self.size) * 200  # make anims slower with bigger cubes
        self.disc_anim_start = 0
        self.disc_anim_phase = 0
        self.disc_anim_cubies = np.zeros((0), dtype=np.int32)
        self.disc_anim_neighbors = np.zeros((0), dtype=np.int32)

        self.shuffle_count = -1
        self.shuffle_anim_ms = 0
        self.shuffle_undo = False

        self.font = pygame.font.SysFont('Arial', self.height // 50)
        self.button_coords = np.zeros((0, 4), dtype=np.int16)
        self.button_types = []
        self.button_pressed = -1
        self.button_y_space = self.height // 100
        self.mouse_rotate = ''
        self.show_buttons = True
        self.show_keys = True
        self.setup_buttons()

        self.cube = Cube(self.size, 0.3, self.height, self.disc_keys_all)
        self.add_cube_data(self.cube)

        # the following for checking performance only
        self.info_display = False
        self.millisecs = 0
        self.timer_avg_frames = 30
        self.timer_names = []
        self.timers = np.zeros((1, 1), dtype=int)
        self.timer_frame = 0
        # self.font = pygame.font.SysFont('CourierNew', 15)

        # set up timers
        self.timer_name = []
        # self.timer_names.append("clear")
        self.timer_names.append("clear")
        self.timer_names.append("buttons")
        self.timer_names.append("rotate disc")
        self.timer_names.append("rotate")
        self.timer_names.append("draw")
        self.timer_names.append("add labels")
        self.timer_names.append("calc visible")
        self.timer_names.append("plot info")
        self.timer_names.append("display flip")
        self.timer_names.append("wait")
        self.timers = np.zeros((len(self.timer_names), self.timer_avg_frames), dtype=int)

        # initialize timers
        self.millisecs = pygame.time.get_ticks()

    def run(self):
        """
        Main loop.
        """
        prev_time = pygame.time.get_ticks()

        while self.running:

            time = pygame.time.get_ticks()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    if event.key == pygame.K_SPACE:
                        self.paused = not(self.paused)
                    if event.key == pygame.K_F1:
                        self.shuffle_count = 50
                        if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                            self.shuffle_undo = True
                    if event.key == pygame.K_F2:
                        angles = self.cube.angles + 0
                        if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                            if self.size > 2:
                                self.size -= 1
                        else:
                            self.size += 1
                        self.cube = Cube(self.size, 0.3, self.height, self.disc_keys_all)
                        self.add_cube_data(self.cube)
                        self.cube.angles = angles + 0
                    if event.key == pygame.K_F3:
                        self.show_keys = not(self.show_keys)
                    if event.key == pygame.K_F4:
                        self.show_buttons = not(self.show_buttons)
                    if event.key == pygame.K_F5:
                        self.toggle_fullscreen()
                    if event.key == pygame.K_F11:
                        self.info_display = not(self.info_display)
                    if event.key == pygame.K_F12:
                        # save screen, at half the resolution, using class name as file name
                        pygame.image.save(pygame.transform.scale(self.screen, (int(self.screen.get_size()[0] / 2), int(self.screen.get_size()[1] / 2))),
                                          self.__class__.__name__ + '.jpg')
                    if self.disc_anim_phase == 0 and self.shuffle_count < 0:
                        key = pygame.key.name(event.key)
                        x = self.disc_keys_x.find(key)  # find will return -1 if not found
                        y = self.disc_keys_y.find(key)
                        z = self.disc_keys_z.find(key)
                        (key_mult, direction) = self.get_key_mods(pygame.key.get_mods())
                        x += key_mult * self.disc_keys_len * (x >= 0)
                        y += key_mult * self.disc_keys_len * (y >= 0)
                        z += key_mult * self.disc_keys_len * (z >= 0)
                        # check that there is a rotation and it is valid (within cube size).
                        if max(x, y, z) >= 0 and max(x, y, z) < self.size:
                            self.disc = np.array([x, y, z])
                            self.disc_anim_neighbors = self.get_disc_neighbors(self.cube, self.disc)
                            self.disc_anim_start = time
                            self.disc_anim_phase = 1
                            self.disc_direction = direction
                # elif event.type == pygame.MOUSEBUTTONDOWN:
                #     if event.button == 1:
                #         # left button: exit
                #         self.running = False

            # main components executed here
            self.clear(self.screen)
            if self.show_buttons and not self.info_display:
                self.button_display(self.screen)
            if self.shuffle_count >= 0:
                self.shuffle_or_undo(self.cube, time, self.shuffle_undo)
            if self.disc_anim_phase > 0:
                self.rotate_disc(self.cube, time)
            self.angle_add(self.cube, time, prev_time)
            self.rotate(self.cube)
            self.draw(self.screen, self.cube)
            if self.show_keys:
                self.add_labels(self.screen, self.cube)
            if self.info_display:
                self.plot_info()

            # release any locks on screen
            while self.screen.get_locked():
                self.screen.unlock()

            # switch between currently showed and the next screen (prepared in "buffer")
            pygame.display.flip()
            self.measure_time("display flip")
            self.clock.tick(self.target_fps)  # this keeps code running at max target_fps
            self.measure_time("wait")
            prev_time = time + 0

        # print(self.cube.cubie_surf_color_nrs)

    def clear(self, screen):

        screen.fill(self.background_color)

        self.measure_time("clear")

    def button_display(self, screen):

        # copy button info to screen
        self.screen.blit(self.screen_buttons, (0, 0))

        # check for mouse: is it on top of any of the buttons?
        mouse_pos = np.asarray(pygame.mouse.get_pos())
        buttons_over = np.nonzero(
            (mouse_pos[0] >= self.button_coords[:, 0]) &
            (mouse_pos[0] <= self.button_coords[:, 2]) &
            (mouse_pos[1] >= self.button_coords[:, 1]) &
            (mouse_pos[1] <= self.button_coords[:, 3])
            )

        self.mouse_rotate = ''  # initialize as no "mouse rotate" as rotate buttons work differently than the others
        if np.shape(buttons_over)[1] == 1:
            # mouse over one button
            button_nr = buttons_over[0][0]
            if pygame.mouse.get_pressed()[0]:
                # left button pressed down; show as selected and mark as pressed
                self.button_pressed = button_nr
                screen.blit(self.screen_button_selected, self.button_coords[button_nr, 0:2] + np.array([-10, -(self.button_y_space - 3)]))
                if self.button_types[button_nr][0].find('Rotate') >= 0:
                    self.mouse_rotate = self.button_types[button_nr][2]  # use the short cut key to signal direction
            else:
                # left button not pressed; show as "mouse over"
                screen.blit(self.screen_button_mouseover, self.button_coords[button_nr, 0:2] + np.array([-10, -(self.button_y_space - 3)]))
                # if this was pressed and now released: act accordingly
                if self.button_pressed == button_nr:
                    self.button_press(button_nr)
                self.button_pressed = -1  # mark as no longer pressed
        else:
            self.button_pressed = -1

        self.measure_time("buttons")

    def button_press(self, button_nr):

        # act on mouse left button release, if screen button pressed

        (btext, bnum, bkeys) = self.button_types[button_nr]
        if btext == 'Auto Rotate':
            self.paused = not(self.paused)
        elif btext == 'Full Screen':
            self.toggle_fullscreen()
        elif btext == 'Size Up':
            angles = self.cube.angles + 0
            self.size += 1
            self.cube = Cube(self.size, 0.3, self.height, self.disc_keys_all)
            self.add_cube_data(self.cube)
            self.cube.angles = angles + 0
        elif btext == 'Size Down' and self.size > 2:
            angles = self.cube.angles + 0
            self.size -= 1
            self.cube = Cube(self.size, 0.3, self.height, self.disc_keys_all)
            self.add_cube_data(self.cube)
            self.cube.angles = angles + 0
        if btext == 'Show Keys':
            self.show_keys = not(self.show_keys)
        if btext == 'Show Buttons':
            self.show_buttons = not(self.show_buttons)
        elif btext == 'Shuffle':
            self.shuffle_count = bnum
        elif btext == 'Undo':
            self.shuffle_count = bnum
            self.shuffle_undo = True
        elif btext == 'Undo All':
            self.shuffle_count = len(self.cube.moves)
            self.shuffle_undo = True
        elif btext == 'Exit':
            self.running = False

    def rotate_disc(self, cube, time):

        # rotate a single disc 90 degrees - animated.

        (x, y, z) = self.disc
        if x >= 0:
            cubies = cube.cube[x, :, :]
        elif y >= 0:
            cubies = cube.cube[:, y, :]
        elif z >= 0:
            cubies = cube.cube[:, :, z]

        # select the cubies on the outer edge of the cube (inner cubies have reference to -1).
        self.disc_anim_cubies = cubies[cubies >= 0]
        cube.cubie_rotated_disc_nodes = cube.cubie_nodes.copy()

        # animation: calculate angles based on time to keep rotation speed constant.
        if time - self.disc_anim_start < self.disc_anim_ms:
            angles = (self.disc >= 0) * (np.pi / 180) * (90 * self.disc_direction * (time - self.disc_anim_start) / self.disc_anim_ms)
        else:
            # animation finished, rotate full 90 degrees.
            angles = (self.disc >= 0) * (np.pi / 180) * 90 * self.disc_direction

        # rotate the nodes of the animated cubies. Note a simpler rotation would be sufficient (this is 3D but the rotation is around a single axis).
        disc_nodes = np.repeat(self.disc_anim_cubies, 8) * 8 + np.resize(np.arange(8), np.size(self.disc_anim_cubies) * 8)
        matrix = self.rotate_matrix(angles)
        cube.cubie_rotated_disc_nodes[disc_nodes, :] = np.matmul(cube.cubie_rotated_disc_nodes[disc_nodes, :], matrix)

        if time - self.disc_anim_start >= self.disc_anim_ms:
            # animation finished. Copy the rotated cubie nodes to nodes.
            # as the nodes have been rotated 90 degrees, also the surfaces will have correct direction.
            cube.cubie_nodes = cube.cubie_rotated_disc_nodes.copy()
            # then rotate the disc in the cube itself to keep track which cubie is where.
            # note due to rotation matrix and cube design axes selection differences, the y direction is of opposite sign to x & z.
            if x >= 0:
                cube.cube[x, :, :] = np.rot90(cube.cube[x, :, :], -self.disc_direction)
            elif y >= 0:
                cube.cube[:, y, :] = np.rot90(cube.cube[:, y, :], self.disc_direction)
            elif z >= 0:
                cube.cube[:, :, z] = np.rot90(cube.cube[:, :, z], -self.disc_direction)
            # rotate colors for these cubies
            surf_rot = cube.surface_rotation[np.sign(y) * 1 + np.sign(z) * 2, :]  # x = 0, y = 1, z = 2
            for cubie in self.disc_anim_cubies:
                cube.cubie_surf_color_nrs[cubie, surf_rot] = np.roll(cube.cubie_surf_color_nrs[cubie, surf_rot], self.disc_direction)

            self.disc_anim_phase = 0
            if not self.shuffle_undo:
                cube.moves.append([self.disc, self.disc_direction])

        self.measure_time("rotate disc")

    def get_disc_neighbors(self, cube, disc):

        # store cubies on rotated disc and its neighbors (edge cubies only).
        (x, y, z) = disc
        neighbors_1a = np.zeros((0), dtype=np.int32)
        neighbors_1b = np.zeros((0), dtype=np.int32)
        neighbors_2a = np.zeros((0), dtype=np.int32)
        neighbors_2b = np.zeros((0), dtype=np.int32)
        if x >= 0:
            cubies = cube.cube[x, :, :]
            if x > 0:
                neighbors_1a = cube.cube[x - 1, 0:cube.size:cube.size - 1, :]
                neighbors_1b = cube.cube[x - 1, 1:-1, 0:cube.size:cube.size - 1]
            if x < cube.size - 1:
                neighbors_2a = cube.cube[x + 1, 0:cube.size:cube.size - 1, :]
                neighbors_2b = cube.cube[x + 1, 1:-1, 0:cube.size:cube.size - 1]
        elif y >= 0:
            cubies = cube.cube[:, y, :]
            if y > 0:
                neighbors_1a = cube.cube[0:cube.size:cube.size - 1, y - 1, :]
                neighbors_1b = cube.cube[1:-1, y - 1, 0:cube.size:cube.size - 1]
            if y < cube.size - 1:
                neighbors_1a = cube.cube[0:cube.size:cube.size - 1, y + 1, :]
                neighbors_1b = cube.cube[1:-1, y + 1, 0:cube.size:cube.size - 1]
        elif z >= 0:
            cubies = cube.cube[:, :, z]
            if z > 0:
                neighbors_1a = cube.cube[0:cube.size:cube.size - 1, :, z - 1]
                neighbors_1b = cube.cube[1:-1, 0:cube.size:cube.size - 1, z - 1]
            if z < cube.size - 1:
                neighbors_2a = cube.cube[0:cube.size:cube.size - 1, :, z + 1]
                neighbors_2b = cube.cube[1:-1, 0:cube.size:cube.size - 1, z + 1]

        return np.hstack((cubies.ravel(), neighbors_1a.ravel(), neighbors_1b.ravel(), neighbors_2a.ravel(), neighbors_2b.ravel()))

    def get_key_mods(self, keys):

        key_mult = 0
        direction = 1
        if keys & pygame.KMOD_LCTRL:
            # Left control key pressed. Add another set of discs (if disc_keys_n has 10 characters, this gives another 10 discs).
            key_mult += 1
        if keys & pygame.KMOD_RCTRL:
            # Right control key pressed. Add another set of discs (if disc_keys_n has 10 characters, this gives another 20 discs after disc 20).
            key_mult += 2
        if keys & pygame.KMOD_LALT:
            # Left ALT key pressed. Add another set of discs (if disc_keys_n has 10 characters, this gives another 40 discs after disc 40).
            key_mult += 4
        if keys & pygame.KMOD_SHIFT:
            # SHIFT key pressed. Add another set of discs if a huge cube; otherwise, change direction.
            if self.size > 8 * self.disc_keys_len:
                key_mult += 8
            else:
                direction = -1

        return (key_mult, direction)

    def angle_add(self, cube, time, prev_time):

        unit_vector = np.array([0, 0, 0], dtype=np.float32)
        time_adjustment = (time - prev_time) / (20 * 17)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT] or self.mouse_rotate == 'RIGHT':
            unit_vector[0] += 1
        if keys[pygame.K_LEFT] or self.mouse_rotate == 'LEFT':
            unit_vector[0] -= 1
        if keys[pygame.K_UP] or self.mouse_rotate == 'UP':
            unit_vector[1] += 1
        if keys[pygame.K_DOWN] or self.mouse_rotate == 'DOWN':
            unit_vector[1] -= 1
        if keys[pygame.K_PAGEUP] or self.mouse_rotate == 'PAGE UP':
            unit_vector[2] += 1
        if keys[pygame.K_PAGEDOWN] or self.mouse_rotate == 'PAGE DOWN':
            unit_vector[2] -= 1
        if not np.all(np.equal(unit_vector, np.array([0, 0, 0]))):

            # rotate along one of the cube's axes, based on the unit vector chosen. See https://en.wikipedia.org/wiki/Rotation_matrix
            # if more than one axes chosen (e.g. UP and LEFT), that will work as well.
            # first rotate the unit vector to point according to current cube angles, then rotate around it.
            unit_vector /= np.sqrt(np.sum(unit_vector ** 2))  # scale to unit vector if multiple axes used
            matrix = self.rotate_matrix(cube.angles)
            (x, y, z) = np.matmul(unit_vector, matrix)
            angle = 1.0 * time_adjustment
            sa = np.sin(angle)
            ca = np.cos(angle)
            matrix_uv = np.array([[ca + x * y * (1 - ca)    , x * y * (1 - ca) - z * sa, x * z * (1 - ca) + y * sa],
                                  [y * x * (1 - ca) + z * sa, ca + y * y * (1 - ca)    , y * z * (1 - ca) - x * sa],
                                  [z * x * (1 - ca) - y * sa, z * y * (1 - ca) + x * sa, ca + z * z * (1 - ca)    ]])

            # final rotation matrix is current matrix rotated around unit vector.
            matrix_final = np.matmul(matrix, matrix_uv)
            # and the resulting angles can be calculated from that matrix.
            angle_x = np.arctan2(-matrix_final[1, 2], matrix_final[2, 2])
            angle_y = np.arctan2(matrix_final[0, 2], np.sqrt(1 - matrix_final[0, 2] ** 2))
            angle_z = np.arctan2(-matrix_final[0, 1], matrix_final[0, 0])
            cube.angles = np.array([angle_x, angle_y, angle_z])
            self.paused = True

        if not self.paused:
            # automatic rotation
            cube.angles += np.array([0.32, 0.37, 0.24]) * time_adjustment

    def rotate(self, cube):

        matrix = self.rotate_matrix(cube.angles)
        cube.cubie_rotated_nodes = np.matmul(cube.cubie_rotated_disc_nodes, matrix)
        cube.cube_corner_rotated_nodes = np.matmul(cube.cube_corner_nodes, matrix)

        size_adj = self.height * cube.cube_size_adj * cube.z_pos / ((2 + cube.distance) * cube.size)
        cube.cubie_trans_nodes = (cube.cubie_rotated_nodes[:, 0:2] / (cube.z_pos + cube.cubie_rotated_nodes[:, 2:3]) * size_adj + self.mid_screen).astype(np.int16)

        self.measure_time("rotate")

        cube.cubie_surface_angle_viewer = self.calculate_angle_viewer(cube, cube.cubie_rotated_nodes, cube.surf_nodes, np.array([0.0, 0.0, cube.z_pos]))
        cube.cube_surface_angle_viewer = self.calculate_angle_viewer(cube, cube.cube_corner_rotated_nodes, cube.surfaces, np.array([0.0, 0.0, cube.z_pos]))

        self.measure_time("calc visible")

    def rotate_matrix(self, angles):

        (sx, sy, sz) = np.sin(angles)
        (cx, cy, cz) = np.cos(angles)

        # build a matrix for X, Y, Z rotation (in that order, see Wikipedia: Euler angles).
        return np.array([[cy * cz               , -cy * sz              , sy      ],
                         [cx * sz + cz * sx * sy, cx * cz - sx * sy * sz, -cy * sx],
                         [sx * sz - cx * cz * sy, cz * sx + cx * sy * sz, cx * cy ]])

    def add_labels(self, screen, cube):

        # calculate cube corner screen coordinates and add key pics as labels for each separate disc rotation.

        (key_mult, direction) = self.get_key_mods(pygame.key.get_mods())

        size_adj = self.height * cube.cube_size_adj * cube.z_pos / ((2 + cube.distance) * cube.size)
        trans_nodes = (cube.cube_corner_rotated_nodes[:, 0:2] / (cube.z_pos + cube.cube_corner_rotated_nodes[:, 2:3]) * size_adj + self.mid_screen).astype(np.int16)

        # find the number of times each cube node is used in visible surfaces
        visible_surf = cube.surfaces[cube.cube_surface_angle_viewer < 0]
        visible_nodes_count = np.bincount(visible_surf.ravel(), minlength=8)
        for i in range(3):
            node_use = np.sum(visible_nodes_count[cube.cube_corner_edges[i, :, :]], axis=1)
            node_use_select = np.max(node_use[node_use < 4])
            if node_use_select > 1:
                avg_coord = np.sum(trans_nodes[cube.cube_corner_edges[i, :, :], :], axis=(1, 2))
                edge_sort = np.argsort(avg_coord)[::-1]
                edge = edge_sort[node_use[edge_sort] == node_use_select]
                opp_edge = edge[0] - 2
                for cubie in range(cube.size):
                    (key_pic, key_pic_size) = cube.disc_key_pics[i * int(cube.disc_key_pics_nr / 3) + cubie % self.disc_keys_len + (int(cubie / self.disc_keys_len) != key_mult) * self.disc_keys_len]
                    cubie_pos = (cubie * (2 + cube.distance) + 1) / ((cube.size - 1) * (2 + cube.distance) + 2)  # cubie mid point relative position on edge
                    edge_node = (1 - cubie_pos) * cube.cube_corner_rotated_nodes[cube.cube_corner_edges[i, edge[0]][0], :] \
                        + cubie_pos * cube.cube_corner_rotated_nodes[cube.cube_corner_edges[i, edge[0]][1], :]
                    opp_edge_node = (1 - cubie_pos) * cube.cube_corner_rotated_nodes[cube.cube_corner_edges[i, opp_edge][0], :] \
                        + cubie_pos * cube.cube_corner_rotated_nodes[cube.cube_corner_edges[i, opp_edge][1], :]
                    key_pic_node = opp_edge_node + 1.06 * (edge_node - opp_edge_node)
                    key_pic_coord = (key_pic_node[0:2] / (cube.z_pos + key_pic_node[2]) * size_adj + self.mid_screen - key_pic_size / 2).astype(np.int16)
                    screen.blit(key_pic, key_pic_coord)

        self.measure_time("add labels")

    def draw(self, screen, cube):

        # draw the cubies.

        # find maximum color for all cubies of all surfaces towards the viewer. This will be greater than color_0 if any surface shown is colored.
        max_colors = np.max((cube.surf_color_nrs * (cube.cubie_surface_angle_viewer < 0)).reshape(cube.cubies_nr, 6), axis=1)
        # make sure all currently rotated cubies are included, even if they otherwise would not.
        if self.disc_anim_phase > 0 and np.size(self.disc_anim_cubies) > 0:
            max_colors[self.disc_anim_cubies] = 1

        # select cubies to draw, including only those with a colored visible surface or being rotated.
        cubies = np.arange(cube.cubies_nr)[max_colors > 0]
        # select surfaces to draw
        surfs = (np.repeat(cubies, 6) * 6 + np.resize(np.arange(6), (np.size(cubies) * 6)))

        if cube.size <= 20:
            surfs_visible = surfs[cube.cubie_surface_angle_viewer[surfs] < 0]
        else:
            # for large cubes, do not draw the noncolored surfaces (i.e. only draw the outermost surface(s)) unless they are being rotated.
            if self.disc_anim_phase > 0 and np.size(self.disc_anim_cubies) > 0 and cube.size > 20:
                rotated_surfs = np.zeros(np.size(cube.surf_color_nrs), dtype=np.uint8)
                rotated_surfs[(np.repeat(self.disc_anim_neighbors, 6) * 6 + np.resize(np.arange(6), (np.size(self.disc_anim_neighbors) * 6)))] = 1
                surfs_visible = surfs[(cube.cubie_surface_angle_viewer[surfs] < 0) & ((cube.surf_color_nrs[surfs] > 0) | (rotated_surfs[surfs] == 1))]
            else:
                surfs_visible = surfs[(cube.cubie_surface_angle_viewer[surfs] < 0) & (cube.surf_color_nrs[surfs] > 0)]

        # calculate surface average z coordinate
        surf_z = np.average(cube.cubie_rotated_nodes[cube.surf_nodes[surfs_visible, :], 2], axis=1)
        # sort surfaces so that the most distant will be drawn first
        sorted_z = surf_z.argsort()[::-1]
        for surf_nr in sorted_z:
            surf = surfs_visible[surf_nr]
            # get surface four screen coordinates by going through its surf_nodes in trans_nodes
            coords = cube.cubie_trans_nodes[cube.surf_nodes[surf], :]
            color = cube.surf_colors[surf]
            pygame.draw.polygon(screen, color, coords)
            if cube.size <= 20:
                pygame.draw.lines(screen, (20, 20, 20), True, coords, cube.cube_surface_line_width)

        self.measure_time("draw")

    def calculate_angle_viewer(self, cube, rotated_nodes, surf_nodes, viewer_pos):

        # calculate angle to viewer (unscaled!). If negative, surface is visible to viewer.

        vec_a = rotated_nodes[surf_nodes[:, 2], :] - rotated_nodes[surf_nodes[:, 1], :]
        vec_b = rotated_nodes[surf_nodes[:, 0], :] - rotated_nodes[surf_nodes[:, 1], :]
        cp_vector = np.hstack((
            (vec_b[:, 1] * vec_a[:, 2] - vec_b[:, 2] * vec_a[:, 1])[:, None],
            (vec_b[:, 2] * vec_a[:, 0] - vec_b[:, 0] * vec_a[:, 2])[:, None],
            (vec_b[:, 0] * vec_a[:, 1] - vec_b[:, 1] * vec_a[:, 0])[:, None]
            ))
        # cp_length = np.linalg.norm(cp_vector, axis=1)

        vec_viewer = rotated_nodes[surf_nodes[:, 1], :] + viewer_pos
        return np.sum(cp_vector * vec_viewer, axis=1)

    def shuffle_or_undo(self, cube, time, undo=False):

        # quickly shuffle the cube, still animated. If undo=True, undo the latest shuffles.
        if self.disc_anim_phase == 0:
            direction = 0
            disc = np.array([0, 0, 0])
            self.shuffle_count -= 1
            if self.shuffle_count < 0:
                # ending shuffle
                self.disc_anim_ms = self.shuffle_anim_ms + 0
                self.shuffle_anim_ms = 0
                self.shuffle_undo = False
            else:
                if self.shuffle_anim_ms == 0:
                    # starting shuffle
                    self.shuffle_anim_ms = self.disc_anim_ms + 0
                    self.disc_anim_ms = int(1000 / 60 * 8)  # assuming 60Hz, animate in 4 frames

                if undo:
                    if len(cube.moves) > 0:
                        undo_move = cube.moves.pop()
                        disc = undo_move[0]
                        direction = -undo_move[1]  # reverse original direction
                    else:
                        self.shuffle_count = 0
                else:
                    # random shuffle. Do not replicate the latest move backward, though.
                    while direction == 0:
                        disc_axis = np.random.randint(0, 3)
                        disc_nr = np.random.randint(0, cube.size)
                        disc = ((np.arange(3) == disc_axis) * (disc_nr + 1) - 1).astype(np.int16)
                        direction = int(np.random.randint(0, 2) * 2 - 1)
                        if len(cube.moves) > 0:
                            if np.all(np.equal(disc, cube.moves[-1][0])) and direction == -cube.moves[-1][1]:
                                direction = 0

                if direction != 0:
                    self.disc = disc
                    self.disc_direction = direction
                    self.disc_anim_start = time
                    self.disc_anim_phase = 1

    def setup_buttons(self):

        # set up button data and prepare screen_buttons

        button_list = [
            ['Auto Rotate', None, 'SPACE'],
            ['Size Up', None, 'F2'],
            ['Size Down', None, 'SHIFT + F2'],
            ['Show Keys', None, 'F3'],
            ['Show Buttons', None, 'F4'],
            ['Full Screen', None, 'F5'],
            ['Shuffle', 10, None],
            ['Shuffle', 50, 'F1'],
            ['Shuffle', 250, None],
            ['Undo', 1, None],
            ['Undo', 10, None],
            ['Undo', 50, 'SHIFT + F1'],
            ['Undo All', None, None],
            ['Rotate X', None, 'RIGHT'],
            ['Rotate Y', None, 'UP'],
            ['Rotate Z', None, 'PAGE UP'],
            ['Counter Rotate X', None, 'LEFT'],
            ['Counter Rotate Y', None, 'DOWN'],
            ['Counter Rotate Z', None, 'PAGE DOWN'],
            ['Exit', None, 'ESC']
            ]

        bcol = (255, 255, 200)
        f_screen = self.font.render(button_list[0][0], True, (0, 0, 0), bcol)
        f_size = f_screen.get_size()

        y_space = self.button_y_space
        x_size = int(f_size[0] * 1.5)
        y_add = f_size[1] + y_space
        y = int(y_space - y_add * 1.5)
        prev_btext = 'aaaa'

        # create button frame for selection
        self.screen_button_mouseover = pygame.Surface((x_size + 20, y_add + 8), 0, self.screen)
        self.screen_button_mouseover.set_colorkey(self.background_color)
        pygame.draw.rect(self.screen_button_mouseover, (40, 40, 200), (10, y_space - 3, x_size, y_add - 4), 4, 4)
        self.screen_button_selected = pygame.Surface((x_size + 20, y_add + 8), 0, self.screen)
        self.screen_button_selected.set_colorkey(self.background_color)
        pygame.draw.rect(self.screen_button_selected, (40, 200, 40), (10, y_space - 3, x_size, y_add - 4), 4, 4)

        for button in button_list:
            (btext, bnum, bkeys) = button

            # new "line": add additional space when button title changes
            if btext[:4] == prev_btext[:4]:
                y += y_add
            else:
                y += int(y_add * 1.5)
            if bnum is not None:
                btext_use = btext + ' ' + str(bnum)
            else:
                btext_use = btext
            prev_btext = btext + ''

            # add button
            f_screen = self.font.render(btext_use, True, (0, 0, 0), bcol)
            f_size = f_screen.get_size()
            rect = pygame.draw.rect(self.screen_buttons, bcol, (10, y - 3, x_size, y_add - 4), 0, 4)
            self.screen_buttons.blit(f_screen, (10 + (x_size - f_size[0]) // 2, rect[1] + (rect[3] - f_size[1]) // 2))
            coords = np.asarray(rect) + np.array([0, 0, rect[0], rect[1]])  # convert rect to coordinates
            self.button_coords = np.vstack((self.button_coords, coords[None, :]))  # add button coords to button coordinate array
            self.button_types.append([btext, bnum, bkeys])  # add button data to list

            # add key short cut text
            if bkeys is not None:
                f_screen = self.font.render(bkeys, True, (255, 255, 255), self.background_color)
                self.screen_buttons.blit(f_screen, (x_size + 30, y))

    def add_cube_data(self, cube):

        # add some info on the cube to screen_buttons
        self.shuffle_count = -1

        if self.button_coords[-1, 3] < self.height - 50:
            y = self.button_coords[-1, 3] + 10
            self.screen_buttons.fill(self.background_color, (10, y, self.screen_buttons.get_size()[0] - 10, self.screen_buttons.get_size()[1] - y))
            x_size = self.button_coords[-1, 2] - self.button_coords[-1, 0]
            y_add = self.button_coords[-1, 3] - self.button_coords[-1, 1]
            self.add_cube_data_txt('Rotate Disc:', 10, y)
            self.add_cube_data_txt('[Disc key]', x_size + 30, y)
            y += y_add
            if cube.size > self.disc_keys_len:
                self.add_cube_data_txt('Higher discs:', 10, y)
                if cube.size > self.disc_keys_len * 8:
                    txt = 'LCTRL/RCTRL/ALT/SHIFT'
                elif cube.size > self.disc_keys_len * 4:
                    txt = 'LCTRL/RCTRL/ALT'
                elif cube.size > self.disc_keys_len * 2:
                    txt = 'LCTRL/RCTRL'
                else:
                    txt = 'LCTRL'
                self.add_cube_data_txt(txt, x_size + 30, y)
                y += y_add
            if cube.size <= self.disc_keys_len * 8:
                self.add_cube_data_txt('Change dir:', 10, y)
                self.add_cube_data_txt('SHIFT', x_size + 30, y)
                y += y_add
            self.add_cube_data_txt('Cube size:', 10, y)
            self.add_cube_data_txt(str(cube.size) + ' x ' + str(cube.size) + ' x ' + str(cube.size), x_size + 30, y)
            y += y_add
            self.add_cube_data_txt('Cubies:', 10, y)
            self.add_cube_data_txt(str(cube.cubies_nr), x_size + 30, y)

    def add_cube_data_txt(self, txt, x, y):

        f_screen = self.font.render(txt, True, (255, 255, 255), self.background_color)
        self.screen_buttons.blit(f_screen, (x, y))

    def toggle_fullscreen(self):

        # toggle between fullscreen and windowed mode
        pygame.display.toggle_fullscreen()

    def plot_info(self):

        # show info
        while self.screen_info.get_locked():
            self.screen_info.unlock()
        self.screen_info.fill(self.background_color)

        y = 0
        y_add = self.height // 50 + 2
        x_size = self.screen_info.get_size()[0]
        info_msg = 'frames per sec:'
        info_val = f'{int(self.clock.get_fps()):5.0f}'
        self.plot_info_msg(self.screen_info, 0, y, x_size, info_msg, info_val)
        y += y_add
        info_msg = 'size:'
        info_val = f'{self.size} x {self.size} x {self.size}'
        self.plot_info_msg(self.screen_info, 0, y, x_size, info_msg, info_val)
        y += y_add
        angles = np.mod(self.cube.angles / np.pi * 180, 360)
        info_msg = 'angles:'
        info_val = f'{angles[0]:4.0f} {angles[1]:4.0f} {angles[2]:4.0f}'
        self.plot_info_msg(self.screen_info, 0, y, x_size, info_msg, info_val)
        y += y_add
        # add measured times as percentage of total
        tot_time = np.sum(self.timers)
        if tot_time > 0:
            for i in range(len(self.timer_names)):
                info_msg = self.timer_names[i]
                info_val = f'{np.sum(self.timers[i, :]) * 100 / tot_time:5.1f}'
                self.plot_info_msg(self.screen_info, 0, y + i * y_add, x_size, info_msg, info_val)

        # copy to screen
        self.screen.blit(self.screen_info, (10, 10))

        self.measure_time("plot info")

    def plot_info_msg(self, screen, x, y, x_size, msg, val):
        f_screen = self.font.render(msg, True, (255, 255, 255))
        f_screen.set_colorkey(self.background_color)
        screen.blit(f_screen, (x, y))
        f_screen = self.font.render(val, True, (255, 255, 255))
        f_screen.set_colorkey(self.background_color)
        f_size = f_screen.get_size()[0]
        screen.blit(f_screen, (x_size - f_size, y))

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


class Cube:
    """
    holds cube data
    """

    def __init__(self, size, distance, screen_height, disc_keys_all):

        self.size = size
        self.distance = distance  # distance between small cubes (cubies)
        self.z_pos = size * 10
        self.cube_size_adj = 0.8 / np.sqrt(3)  # cube maximum size (percentage of screen y) adjusted for rotation
        self.cube_surface_line_width = int(max(1, screen_height / (self.size * 35)))
        self.cube = -np.ones((size, size, size), dtype=np.int32)
        self.cubies_nr = size ** 3 - max(0, size - 2) ** 3  # cubies_nr = outer edge small cubes; subtract inner cubes from all cubes.
        self.nodes = np.array([
            [-1, -1, -1],
            [-1, -1, +1],
            [+1, -1, +1],
            [+1, -1, -1],
            [-1, +1, -1],
            [-1, +1, +1],
            [+1, +1, +1],
            [+1, +1, -1]
            ])
        self.surfaces = np.array([
            [0, 1, 2, 3],
            [7, 6, 5, 4],
            [0, 3, 7, 4],
            [1, 0, 4, 5],
            [2, 1, 5, 6],
            [3, 2, 6, 7]
            ])
        # colors of cube sides (1-6) and "non-colored" side (color 0)
        self.color_0 = 60
        self.colors = np.array([
            [self.color_0, self.color_0, self.color_0],
            [255, 255, 255],
            [255, 255,   0],
            [  0, 200,   0],
            [  0,   0, 200],
            [235, 150,   0],
            [200,   0,   0]
            ], dtype=np.uint8)
        self.surface_directions = np.average(self.nodes[self.surfaces], axis=1).astype(np.int32)
        # surface rotation defines the surfaces rotated for each (x, y, z) rotation, direction = 1
        self.surface_rotation = np.array([
            [0, 2, 1, 4],
            [3, 4, 5, 2],
            [3, 0, 5, 1]
            ], dtype=np.int32)
        self.cubie_nodes = np.zeros((8 * self.cubies_nr, 3), dtype=np.float32)
        # surf_nodes has the rotated node numbers for all surfaces
        self.surf_nodes = np.resize(self.surfaces, (self.cubies_nr * 6, 4)) + np.repeat(np.arange(self.cubies_nr) * 8, 6)[:, None]
        self.surf_colors = np.zeros((6 * self.cubies_nr, 3), np.uint8)
        self.surf_color_nrs = np.zeros((6 * self.cubies_nr), np.uint8)
        self.cubie_surf_color_nrs = np.zeros((self.cubies_nr, 6), np.uint8)

        self.moves = []
        self.angles = np.zeros((3), dtype=np.float32)

        cubie_nr = 0
        mid = (size - 1) / 2
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    # find all colors and directions this cubie (small cube) needs - zero to three each
                    colors = []
                    directions = []
                    if x == 0:
                        colors.append(1)
                        directions.append(np.array([-1, 0, 0]))
                    elif x == size - 1:
                        colors.append(2)
                        directions.append(np.array([1, 0, 0]))
                    if y == 0:
                        colors.append(3)
                        directions.append(np.array([0, -1, 0]))
                    elif y == size - 1:
                        colors.append(4)
                        directions.append(np.array([0, 1, 0]))
                    if z == 0:
                        colors.append(5)
                        directions.append(np.array([0, 0, -1]))
                    elif z == size - 1:
                        colors.append(6)
                        directions.append(np.array([0, 0, 1]))
                    if len(colors) > 0:
                        self.cube[x, y, z] = cubie_nr
                        cubie_colors = np.zeros((6), dtype=np.uint8)
                        for i in range(len(directions)):
                            surf = (self.surface_directions == directions[i]).all(axis=1).nonzero()[0]
                            cubie_colors[surf] = colors[i]

                        self.cubie_nodes[8 * cubie_nr:8 * (cubie_nr + 1), :] = self.nodes + np.array([x - mid, y - mid, z - mid]) * (2 + distance)
                        self.surf_colors[6 * cubie_nr:6 * (cubie_nr + 1), :] = self.colors[cubie_colors]
                        self.surf_color_nrs[6 * cubie_nr:6 * (cubie_nr + 1)] = cubie_colors
                        self.cubie_surf_color_nrs[cubie_nr, :] = cubie_colors
                        cubie_nr += 1

        self.cubie_rotated_disc_nodes = self.cubie_nodes.copy()
        self.cubie_rotated_nodes = self.cubie_nodes.copy()
        self.cubie_trans_nodes = np.zeros((8 * self.cubies_nr, 2), dtype=np.int16)
        self.cubie_surface_angle_viewer = np.zeros((6 * self.cubies_nr), dtype=np.float32)

        self.cube_corner_nodes = self.nodes * (1 + mid * (2 + distance))
        self.cube_corner_rotated_nodes = self.cube_corner_nodes.copy()
        self.cube_surface_angle_viewer = np.zeros((6), dtype=np.float32)
        self.cube_corner_edges = np.array([
            [[0, 3], [4, 7], [5, 6], [1, 2]],  # edges where Y and Z constant and X from - to +
            [[0, 4], [1, 5], [2, 6], [3, 7]],  # edges where X and Z constant and Y from - to +
            [[0, 1], [3, 2], [7, 6], [4, 5]]   # edges where X and Y constant and Z from - to +
            ])

        self.font = pygame.font.SysFont('Arial', int(max(14, min(30, screen_height / (size * 2)))))
        self.disc_key_pics = []
        self.disc_key_pics_nr = 0
        self.setup_disc_key_pics(self.size, disc_keys_all)

        # print(self.surf_nodes)
        # print(self.surf_color_nrs)
        # print(self.cubie_surf_color_nrs)
        # print(self.surface_directions)

    def setup_disc_key_pics(self, size, disc_keys_all):

        # create pictures to use for disc keys. Use two colors.
        bkg_color = (0, 0, 0)
        for keys in disc_keys_all:
            mult = 2
            for m in range(mult):
                for key in keys:
                    if m % 2 == 0:
                        f_color = (255, 255, 255)
                    else:
                        f_color = (255, 255, 0)
                    f_screen = self.font.render(key, True, f_color, bkg_color)
                    f_screen.set_colorkey((0, 0, 0))
                    f_size = np.asarray(f_screen.get_size())
                    self.disc_key_pics.append((f_screen, f_size))
        self.disc_key_pics_nr = len(self.disc_key_pics)


if __name__ == '__main__':
    """
    Prepare screen, objects etc.
    """

    # set screen size
    # first check available full screen modes
    pygame.display.init()
    # disp_modes = pygame.display.list_modes(0, pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE)
    # disp_size = disp_modes[4] # selecting display size from available list. Assuming the 5th element is nice...
    # disp_size = (2560, 1440)
    disp_size = (1920, 1080)
    # disp_size = (1280, 720)
    # disp_size = (800, 600)

    pygame.font.init()
    screen = pygame.display.set_mode(disp_size)  # , pygame.FULLSCREEN | pygame.DOUBLEBUF)
    pygame.display.set_caption('Cubester')
    Cubester(screen, 60).run()

    # exit; close display, stop music
    pygame.quit()
    exit()

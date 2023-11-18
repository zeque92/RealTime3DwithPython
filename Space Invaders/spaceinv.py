# -*- coding: utf-8 -*-

import pygame
import numpy as np
import random
import os.path
import json
import base64
import hashlib
from cryptography.fernet import Fernet
from io import BytesIO
from sys import exit


class SpaceInvaders:

    """
    Space Invaders Game

    (c) Kalle & Eelis 2022
    """

    def __init__(self, screen):

        pygame.font.init()
        pygame.mixer.init()

        # set up display surfaces
        self.screen = screen
        self.screen_size = np.array(self.screen.get_size())
        self.mid_screen = self.screen_size // 2
        self.background_color = (0, 0, 0)
        self.screen_titles = screen.copy()
        self.screen_titles.set_colorkey(self.background_color)
        self.screen_credits = self.screen.copy()
        self.screen_credits.set_colorkey(self.background_color)
        self.screen_instructions = self.screen.copy()
        self.screen_instructions.set_colorkey(self.background_color)
        self.screen_info = screen.copy()
        self.screen_info.set_colorkey(self.background_color)
        self.title_color = (220, 220, 160)
        self.score_color = (200, 200, 0)
        self.angle = 0
        self.angle_add = 0
        self.info_screen = 'titles'
        self.info_screen_next = 'credits'
        self.f = Fernet(base64.urlsafe_b64encode(hashlib.md5('<password here>'.encode()).hexdigest().encode("utf-8")))
        self.namehints = {'p':'png','j':'jpg','o':'ogg','m':'mp3','w':'wav','t':'txt'}

        # load data files - images
        (fileobj, namehint) = self.load_dat('ship p')
        self.ship_pic = pygame.image.load(fileobj, namehint).convert()
        self.ship_pic.set_colorkey((255, 255, 255))
        shield_size = int(np.max(np.asarray(self.ship_pic.get_size()) * 1.6))
        (fileobj, namehint) = self.load_dat('Shield_small p')
        self.ship_shield_pic = pygame.transform.scale(pygame.image.load(fileobj, namehint).convert(), (shield_size, shield_size))
        self.ship_shield_pic.set_colorkey((0, 0, 0))
        self.ship_shield_pic.set_alpha(96)
        (fileobj, namehint) = self.load_dat('alien 1 p')
        self.alien1_pic = pygame.image.load(fileobj, namehint).convert()
        self.alien1_pic.set_colorkey((255, 255, 255))
        (fileobj, namehint) = self.load_dat('alien 2 p')
        self.alien2_pic = pygame.image.load(fileobj, namehint).convert()
        self.alien2_pic.set_colorkey((0, 0, 0))
        (fileobj, namehint) = self.load_dat('alien 3 p')
        self.alien3_pic = pygame.image.load(fileobj, namehint).convert()
        self.alien3_pic.set_colorkey((0, 0, 0))
        (fileobj, namehint) = self.load_dat('alien 4 p')
        self.alien4_pic = pygame.image.load(fileobj, namehint).convert()
        self.alien4_pic.set_colorkey((0, 0, 0))
        (fileobj, namehint) = self.load_dat('alien 5 p')
        self.alien5_pic = pygame.image.load(fileobj, namehint).convert()
        self.alien5_pic.set_colorkey((0, 0, 0))
        (fileobj, namehint) = self.load_dat('alien 6 p')
        self.alien6_pic = pygame.image.load(fileobj, namehint).convert()
        self.alien6_pic.set_colorkey((0, 0, 0))
        (fileobj, namehint) = self.load_dat('alien 7 p')
        self.alien7_pic = pygame.image.load(fileobj, namehint).convert()
        self.alien7_pic.set_colorkey((0, 0, 0))
        (fileobj, namehint) = self.load_dat('alien 8 p')
        self.alien8_pic = pygame.image.load(fileobj, namehint).convert()
        self.alien8_pic.set_colorkey((0, 0, 0))
        (fileobj, namehint) = self.load_dat('alien 9 p')
        self.alien9_pic = pygame.image.load(fileobj, namehint).convert()
        self.alien9_pic.set_colorkey((0, 0, 0))
        (fileobj, namehint) = self.load_dat('alien 10 p')
        self.alien10_pic = pygame.image.load(fileobj, namehint).convert()
        self.alien10_pic.set_colorkey((0, 0, 0))
        (fileobj, namehint) = self.load_dat('alien 11 p')
        self.alien11_pic = pygame.image.load(fileobj, namehint).convert()
        self.alien11_pic.set_colorkey((0, 0, 0))
        (fileobj, namehint) = self.load_dat('alien 12 p')
        self.alien12_pic = pygame.image.load(fileobj, namehint).convert()
        self.alien12_pic.set_colorkey((0, 0, 0))
        (fileobj, namehint) = self.load_dat('explosion alien w')
        self.alien1_sound_explosion = pygame.mixer.Sound(fileobj)
        self.alien1_sound_explosion.set_volume(0.2)
        (fileobj, namehint) = self.load_dat('alien boss 1 p')
        self.alien_boss1_pic = pygame.image.load(fileobj, namehint).convert()
        self.alien_boss1_hit_area = pygame.Rect(206 * 2 // 3, 358 * 2 // 3, 100 * 2 // 3, 57 * 2 // 3)
        self.alien_boss1_pic = pygame.transform.scale(self.alien_boss1_pic, np.array(self.alien_boss1_pic.get_size()) * 2 // 3)
        self.pic_colorkey(self.alien_boss1_pic, (36, 36, 36))
        self.alien_boss1_cannon_pos = np.array([[self.alien_boss1_pic.get_size()[0] * 0.2 - 10, self.alien_boss1_pic.get_size()[1] - 5],
                                                [self.alien_boss1_pic.get_size()[0] * 0.8 - 10, self.alien_boss1_pic.get_size()[1] - 5]], dtype=float)
        (fileobj, namehint) = self.load_dat('alien boss 2 p')
        self.alien_boss2_pic = pygame.image.load(fileobj, namehint).convert()
        self.alien_boss2_pic.set_colorkey((0, 0, 0))
        self.alien_boss2_hit_area = pygame.Rect(87, 300, 106, 65)
        self.alien_boss2_cannon_pos = np.array([[self.alien_boss2_pic.get_size()[0] * 0.43 - 10, self.alien_boss2_pic.get_size()[1] - 25],
                                                [self.alien_boss2_pic.get_size()[0] * 0.57 - 10, self.alien_boss2_pic.get_size()[1] - 25]], dtype=float)
        (fileobj, namehint) = self.load_dat('alien boss 3 p')
        self.alien_boss3_pic = pygame.image.load(fileobj, namehint).convert()
        self.alien_boss3_pic.set_colorkey((0, 0, 0))
        self.alien_boss3_hit_area = pygame.Rect(135, 210, 52, 45)
        self.alien_boss3_cannon_pos = np.array([[-10, 225], [110, 210], [192, 210], [312, 225]], dtype=float)
        (fileobj, namehint) = self.load_dat('alien boss 4 p')
        self.alien_boss4_pic = pygame.image.load(fileobj, namehint).convert()
        self.alien_boss4_pic.set_colorkey((0, 0, 0))
        self.alien_boss4_hit_area = pygame.Rect(153, 340, 72, 35)
        self.alien_boss4_cannon_pos = np.array([[27, 368], [146, 350], [212, 350], [321, 368]], dtype=float)
        (fileobj, namehint) = self.load_dat('alien_ufo p')
        self.alien_ufo_pic = pygame.image.load(fileobj, namehint).convert()
        self.alien_ufo_pic.set_colorkey((0, 0, 0))
        (fileobj, namehint) = self.load_dat('bullet_small p')
        self.bullet_alien1_pic = pygame.image.load(fileobj, namehint).convert()
        self.bullet_alien1_pic = pygame.transform.flip(pygame.transform.scale(self.bullet_alien1_pic, (np.array(self.bullet_alien1_pic.get_size()) / 2.6).astype(np.int16)), 0, 1)
        self.bullet_alien1_pic = self.recolor(self.bullet_alien1_pic, 'B')
        self.bullet_alien1_pic.set_colorkey((0, 0, 0))
        (fileobj, namehint) = self.load_dat('bullet_double p')
        self.bullet_alien2_pic = pygame.image.load(fileobj, namehint).convert()
        self.bullet_alien2_pic = pygame.transform.flip(pygame.transform.scale(self.bullet_alien2_pic, np.array(self.bullet_alien2_pic.get_size()) // 2), 0, 1)
        self.bullet_alien2_pic.set_colorkey((0, 0, 0))
        (fileobj, namehint) = self.load_dat('bullet_medium p')
        self.bullet_ship1_pic = pygame.image.load(fileobj, namehint).convert()
        self.bullet_ship1_pic = pygame.transform.scale(self.bullet_ship1_pic, np.array(self.bullet_ship1_pic.get_size()) // 3)
        self.bullet_ship1_pic.set_colorkey((0, 0, 0))
        (fileobj, namehint) = self.load_dat('bullet_ufo p')
        self.bullet_ufo1_pic = pygame.image.load(fileobj, namehint).convert()
        self.bullet_ufo1_pic.set_colorkey((0, 0, 0))
        (fileobj, namehint) = self.load_dat('power_up p')
        self.powerup_template = pygame.image.load(fileobj, namehint).convert()

        # load data files - sounds
        (fileobj, namehint) = self.load_dat('explo 12 p')
        self.explosion1_pic = self.adjust_pic(pygame.image.load(fileobj, namehint).convert(), np.array([-2, 7]), np.array([554, 554]))  # adjust picture position and size
        self.pic_colorkey(self.explosion1_pic, (16, 16, 16))
        self.explosion1_pic_grid = np.array([5, 5])
        self.explosion1_freq = 2
        (fileobj, namehint) = self.load_dat('explosion p')
        self.explosion_ship_pic = pygame.image.load(fileobj, namehint).convert()
        self.pic_colorkey(self.explosion_ship_pic, (16, 16, 16))
        self.explosion_ship_pic_grid = np.array([4, 4])
        self.explosion_ship_freq = 2
        (fileobj, namehint) = self.load_dat('ship gun w')
        self.ship_sound_gun = pygame.mixer.Sound(fileobj)
        self.ship_sound_gun.set_volume(0.2)
        (fileobj, namehint) = self.load_dat('explosion ship boss w')
        self.ship_sound_explosion = pygame.mixer.Sound(fileobj)
        self.ship_sound_explosion.set_volume(1.0)
        (fileobj, namehint) = self.load_dat('explosion ship boss w')
        self.alien_boss_sound_explosion = pygame.mixer.Sound(fileobj)
        self.alien_boss_sound_explosion.set_volume(0.5)
        (fileobj, namehint) = self.load_dat('explosion ufo m')
        self.alien_ufo_sound_explosion = pygame.mixer.Sound(fileobj)
        self.alien_ufo_sound_explosion.set_volume(0.5)
        (fileobj, namehint) = self.load_dat('miss w')
        self.sound_miss = pygame.mixer.Sound(fileobj)
        self.sound_miss.set_volume(0.4)
        (fileobj, namehint) = self.load_dat('pling w')
        self.powerup_sound = pygame.mixer.Sound(fileobj)
        self.powerup_sound.set_volume(0.6)
        (fileobj, namehint) = self.load_dat('victory w')
        self.sound_victory = pygame.mixer.Sound(fileobj)
        self.sound_victory.set_volume(0.4)
        (fileobj, namehint) = self.load_dat('defeat w')
        self.sound_defeat = pygame.mixer.Sound(fileobj)
        self.sound_defeat.set_volume(0.4)

        # set up powerups
        self.font_powerup = pygame.font.SysFont('Arial Black', 10)
        self.font_powerup_desc = pygame.font.SysFont('stencil', 16)
        self.powerup_data = self.setup_powerups(self.powerup_template, self.font_powerup, self.font_powerup_desc)
        self.powerup_desc_width = 0  # filled by the next loop
        for pdata in self.powerup_data:
            if pdata[3].get_size()[0] > self.powerup_desc_width:
                self.powerup_desc_width = pdata[3].get_size()[0]

        # set up high scores
        self.highscore_file = 'spaceinv_scores t'
        self.highscore_nr = 10
        self.highscore_name_length = 8
        self.latest_hs_nr = 0
        self.latest_hs_keys = []
        self.highscores = []
        self.read_scores()

        # level and game data
        self.level_data = []
        self.setup_level_data()
        self.level = 0
        self.level_time = 0
        self.level_completed_time = 0
        self.level_ufo_probability = 0.05
        self.level_power_up_probability = 0.8  # probability of getting a power up by shooting an ufo
        self.level_new_life = False

        self.running = True
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.game_mode = 'start page'
        self.game_over = False
        self.game_over_time = 0
        self.freeze_aliens = False
        self.cheat_mode = 0
        self.font_title = pygame.font.SysFont('stencil', self.screen_size[0] // 10)
        self.font_highscorelist = pygame.font.SysFont('stencil', self.screen_size[1] // 30)
        self.font_score = pygame.font.SysFont('stencil', 25)
        self.font_game_score = pygame.font.SysFont('stencil', 50)
        self.game_score = 0
        self.show_score = -1
        self.game_over_pic = self.font_game_score.render('GAME OVER', True, self.title_color)
        self.game_over_pic.set_colorkey(self.background_color)
        self.score_pic = self.font_game_score.render('0', True, self.score_color, self.background_color)
        self.score_pic.set_colorkey(self.background_color)
        self.score_pic_position = (10, 10)
        self.level_pic = self.font_game_score.render('Level 0', True, self.score_color, self.background_color)
        self.level_pic.set_colorkey(self.background_color)
        self.level_pic_position = (self.screen_size[0] - self.level_pic.get_size()[0] - 10, 10)
        self.switch_time = 0
        self.switch_time_ms = 2000  # milliseconds between game and start page modes

        # game object lists
        self.alien_bullets = []
        self.aliens = []
        self.ufos = []
        self.scores = []
        self.explosions = []
        self.powerups = []

        # setup title screen
        self.title_hs_pos = np.array([0, 0])
        self.title_hs_pos_end = np.array([0, 0])
        self.title_hs_rect = np.zeros((4))
        self.title_cr_rect = np.zeros((4))
        self.title_ins_rect = np.zeros((4))
        self.title_rect = np.zeros((4))
        self.title_z_pos = 1000
        self.setup_titles(0)
        self.setup_credits()
        self.setup_instructions()
        self.screen_info = self.screen_titles.copy()  # copy to info screen
        self.credits_time = pygame.time.get_ticks()
        self.title_change = 0
        self.title_change_ctr = 0

        # set up a random batch of stars for the background
        self.nr_stars = 4000
        self.z_range = (50, 2000)                           # range for Z coordinates of stars
        self.stars = np.random.rand(self.nr_stars, 3) * np.array([self.screen_size[0] - 2, self.screen_size[1] - 2, 1.0]) \
            + np.array([-self.mid_screen[0], -self.mid_screen[1], 0.0])
        # adjust Z coordinates as more stars needed at distance for a balanced view
        self.stars[:, 2] = (self.stars[:, 2] ** 0.5) * (self.z_range[1] - self.z_range[0]) + self.z_range[0]
        self.star_move = np.array([0.0, 0.0, -0.5])

        # create the ship
        self.ship_shoot_freq = 280
        self.ship = Ship(self.ship_pic, self.bullet_ship1_pic, self.ship_shield_pic, self.ship_sound_gun, self.ship_sound_explosion,
                         self.screen_size, 7.0, self.ship_shoot_freq, 3)

        # load music data and start looping it
        (fileobj, namehint) = self.load_dat('music theme o')
        pygame.mixer.music.load(fileobj)
        pygame.mixer.music.set_volume(0.2)
        pygame.mixer.music.play(loops=-1)
        self.nr_channels = pygame.mixer.get_num_channels()
        self.channel = 1  # reserving channel 0 for prioritized sounds

    # ----------------   main loop   ----------------
    def run(self):

        prev_time = pygame.time.get_ticks()
        while self.running:

            time = pygame.time.get_ticks()

            # keyboard activity
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if self.game_mode == 'new high score':
                    if event.type == pygame.TEXTINPUT:
                        unicode = event.text
                        key = ord(unicode)
                        self.latest_hs_keys.append((key, unicode))
                    if event.type == pygame.KEYDOWN:
                        # get special keys
                        key = event.key
                        if key in (pygame.K_RETURN, pygame.K_BACKSPACE):
                            self.latest_hs_keys.append((key, ''))

            if not self.game_mode == 'new high score':
                keys = pygame.key.get_pressed()
                if keys[pygame.K_ESCAPE]:
                    self.running = False
                if keys[pygame.K_SPACE]:
                    if self.game_mode == 'start page':
                        self.game_mode = 'go to game'
                        self.switch_time = time
                    elif self.game_mode == 'game':
                        if self.ship.shoot(time):
                            # shot fired, play sound
                            self.play_sound(self.ship.sound_gun)
                else:
                    self.ship.space_down = False
                if keys[pygame.K_i] and self.angle_add == 0:
                    self.angle_add = 2
                    self.info_screen_next = 'instructions'
                    self.credits_time = time
                if keys[pygame.K_c]:
                    if self.cheat_mode == 0:
                        self.cheat_mode = 1
                    else:
                        self.cheat_mode = 3
                if keys[pygame.K_h] and self.cheat_mode in (1, 3):
                    self.cheat_mode = 2
                    pid = random.randint(0, len(self.powerup_data) - 1)  # randomly pick power up
                    self.powerups.append(PowerUp(pid, self.powerup_data, self.powerup_sound, (0, 0)))
                    powerup = self.powerups[-1]
                    self.ship.add_powerup(powerup, self.ship_shoot_freq)
                    self.powerups.remove(powerup)
                if keys[pygame.K_LEFT]:
                    self.ship.move(-1, self.screen_size)
                if keys[pygame.K_RIGHT]:
                    self.ship.move(1, self.screen_size)

            # clear screen for a new frame
            self.screen.fill(self.background_color)

            # main loop depending on game mode
            if self.game_mode == 'start page':
                self.move_stars(time, prev_time)
                self.plot_stars()

                self.ufo_ops_back(time, prev_time)
                self.copy_titles()
                self.rotate_titles(time)
                self.bullet_ops(time)
                self.ufo_ops_front(time)

            elif self.game_mode == 'go to game':
                star_move_frac = max(1.0 - (time - self.switch_time) / self.switch_time_ms, 0.0)
                self.star_move = np.array([0.0,
                                           125.0 * np.cos(star_move_frac * np.pi / 2.0),
                                           -0.5 * np.sin(star_move_frac * np.pi / 2.0)])
                self.move_stars(time, prev_time)
                self.plot_stars()
                self.ufo_ops_back(time, prev_time)
                self.screen_titles.set_alpha(int(255 * star_move_frac))
                self.copy_titles()
                self.ufo_ops_front(time)
                if time - self.switch_time > self.switch_time_ms + 1000:
                    self.star_move = np.array([0.0, 125.0, 0.0])
                    self.screen_titles.set_alpha(255)
                    self.new_game(time)

            elif self.game_mode == 'go to start page':
                self.move_stars(time, prev_time)
                self.plot_stars()
                if time > self.switch_time + 1000:
                    self.info_screen = 'titles'
                    self.angle = 0
                    self.angle_add = 0
                    if self.cheat_mode in (0, 1) and self.game_score > self.highscores[self.highscore_nr - 1][1]:
                        self.latest_hs_nr = self.highscore_add()
                        self.game_mode = 'new high score'
                        pygame.key.start_text_input()
                    else:
                        self.game_mode = 'start page'
                        self.credits_time = time

            elif self.game_mode == 'new high score':
                self.highscore_get_name(time)
                self.move_stars(time, prev_time)
                self.plot_stars()
                self.setup_titles(time)
                self.copy_titles()

            else:  # actual game mode ('game')
                self.move_stars(time, prev_time)
                self.plot_stars()

                self.level_ops(time)
                self.ship_ops(time, prev_time)
                self.ufo_ops_back(time, prev_time)
                self.alien_ops(time, prev_time)
                self.bullet_ops(time)
                self.explosion_ops()
                self.score_ops(time)
                self.powerup_ops(time)
                self.ufo_ops_front(time)
                self.info_ops()

                if self.game_over:
                    self.game_over_ops(time)

            self.clock.tick(self.fps)  # keeps code running at maximum self.fps frames per second
            pygame.display.flip()
            prev_time = time + 0

    # ----------------   initialize a new game   ----------------
    def new_game(self, time):

        # set up a new game
        self.game_mode = 'game'
        self.ship = Ship(self.ship_pic, self.bullet_ship1_pic, self.ship_shield_pic, self.ship_sound_gun, self.ship_sound_explosion,
                         self.screen_size, 7.0, self.ship_shoot_freq, 3)
        self.level = 0
        self.aliens.clear()
        self.alien_bullets.clear()
        self.scores.clear()
        self.explosions.clear()
        self.powerups.clear()
        self.game_over = False
        self.game_over_time = 0
        self.game_score = 0
        self.show_score = -1
        self.cheat_mode = 0

    # ----------------   main loop "operations" for different entities   ----------------
    def level_ops(self, time):

        # test if level completed
        if len(self.aliens) == 0 and len(self.ufos) == 0 and not self.game_over:
            if (self.level_completed_time > 120 and len(self.explosions) == 0 and len(self.scores) == 0 and self.show_score == self.game_score) or self.level == 0:
                self.level_completed_time = 0
                self.level += 1
                self.setup_level(time)
            else:
                if self.level_completed_time == 0:
                    # add score, play fanfare
                    self.scores.append(Score(self, 200 + self.level * 50, str(200 + self.level * 50),
                                             self.font_game_score, self.screen_size // 2 + np.array([-20, -50]),
                                             self.score_color, self.background_color))
                    self.play_sound(self.sound_victory, 0)
                    if self.level_new_life and self.ship.lives < 5:
                        self.ship.lives += 1  # add extra life
                self.level_completed_time += 1
                pic = self.font_game_score.render('LEVEL ' + str(self.level) + ' COMPLETED', True, self.title_color, self.background_color)
                pic.set_colorkey((self.background_color))
                position = ((self.screen_size - pic.get_size()) / 2).astype(int)
                self.screen.blit(pic, position)

        # test if Freeze Alien powerup is active
        for powerup in self.ship.powerups:
            if self.powerup_data[powerup[0]][2] == 'Freeze Aliens':
                self.freeze_aliens = True
                break
        else:
            self.freeze_aliens = False

    def bullet_ops(self, time):

        # check alien bullets hitting ship or its shield
        for bullet in self.alien_bullets:
            bullet.move()
            if self.ship.shield:
                # ship protected by a shield
                if bullet.rect.colliderect(self.ship.shield_rect):
                    mask_offset = (bullet.rect[0] - self.ship.shield_rect[0], bullet.rect[1] - self.ship.shield_rect[1])
                    if self.ship.shield_mask.overlap(bullet.mask, mask_offset):
                        self.play_sound(self.sound_miss)
                        bullet.position[1] = -9999
            elif bullet.rect.colliderect(self.ship.rect) and self.ship.status == 0 and self.game_mode == 'game':
                mask_offset = (bullet.rect[0] - self.ship.rect[0], bullet.rect[1] - self.ship.rect[1])
                if self.ship.mask.overlap(bullet.mask, mask_offset):
                    self.ship.is_hit(time, self.cheat_mode)
                    self.play_sound(self.ship.sound_explosion)
                    bullet.position[1] = -9999
                    self.explosions.append(Explosion(self.explosion_ship_pic, (int(self.ship.position[0] + self.ship.size[0] / 2),
                                                                               int(self.ship.position[1] + self.ship.size[1] / 2)),
                                                     self.explosion_ship_pic_grid, self.explosion_ship_freq))

            if bullet.position[0] < -bullet.size[0] or bullet.position[0] > self.screen_size[0] \
                    or bullet.position[1] < -bullet.size[1] or bullet.position[1] > self.screen_size[1]:
                self.alien_bullets.remove(bullet)
            else:
                bullet.draw(self.screen)

        # check ship bullets hitting ufos, aliens, or powerups
        for bullet in self.ship.bullets:
            bullet.move()

            for ufo in self.ufos:
                if ufo.may_be_shot and bullet.rect.colliderect(ufo.rect):
                    mask_offset = (bullet.rect[0] - ufo.rect[0], bullet.rect[1] - ufo.rect[1])
                    if ufo.mask.overlap(bullet.mask, mask_offset):

                        # ufo killed
                        self.scores.append(Score(self, ufo.score, str(ufo.score), self.font_score,
                                                 (int(bullet.position[0] + bullet.size[0]), int(bullet.position[1] - 20)),
                                                 self.score_color, self.background_color))
                        self.play_sound(ufo.sound_explosion)
                        self.explosions.append(Explosion(self.explosion_ship_pic, (int(ufo.position[0] + ufo.size[0] / 2),
                                                                                   int(ufo.position[1] + ufo.size[1] / 2)),
                                                         self.explosion_ship_pic_grid, self.explosion_ship_freq))
                        self.ufos.remove(ufo)
                        bullet.position[1] = -9999

                        # if player is lucky, create a power up
                        if random.random() <= self.level_power_up_probability:
                            pos = (int(ufo.position[0] + ufo.size[0] / 2), int(ufo.position[1] + ufo.size[1] / 2))
                            self.create_random_powerup(pos)
                        break
            else:

                for alien in self.aliens:
                    if bullet.rect.colliderect(alien.rect):
                        mask_offset = (bullet.rect[0] - alien.rect[0], bullet.rect[1] - alien.rect[1])
                        if alien.mask.overlap(bullet.mask, mask_offset):
                            # check if bullet hit the "damage area", if it is defined
                            if not alien.hit_area or bullet.rect.colliderect(alien.hit_area.move(alien.position)):
                                # hit caused damage
                                alien.hit_nr -= 1
                                self.play_sound(alien.sound_explosion)

                                if alien.hit_nr == 0:
                                    # alien killed
                                    self.scores.append(Score(self, alien.score, str(alien.score), self.font_score,
                                                             (int(bullet.position[0] + bullet.size[0]), int(bullet.position[1] - 20)),
                                                             self.score_color, self.background_color))
                                    if alien.hit_total > 5:
                                        # Boss alien - big explosion
                                        self.explosions.append(Explosion(self.explosion_ship_pic, (int(alien.position[0] + alien.size[0] / 2),
                                                                                                   int(alien.position[1] + alien.size[1] / 2)),
                                                                         self.explosion_ship_pic_grid, self.explosion_ship_freq))
                                        # Boss killed - create TWO powerups
                                        pos = (int(alien.position[0] + alien.size[0] / 2 + random.randint(-50, 50)),
                                               int(alien.position[1] + alien.size[1] / 2) + random.randint(-50, 50))
                                        self.create_random_powerup(pos)
                                        pos = (int(alien.position[0] + alien.size[0] / 2 + random.randint(-50, 50)),
                                               int(alien.position[1] + alien.size[1] / 2) + random.randint(-50, 50))
                                        self.create_random_powerup(pos)
                                    else:
                                        # normal alien
                                        self.explosions.append(Explosion(self.explosion1_pic, (int(alien.position[0] + alien.size[0] / 2),
                                                                                               int(alien.position[1] + alien.size[1] / 2)),
                                                                         self.explosion1_pic_grid, self.explosion1_freq))
                                    self.aliens.remove(alien)
                                else:
                                    # hit but not killed
                                    self.scores.append(Score(self, int(alien.score / alien.hit_total), str(int(alien.score / alien.hit_total)), self.font_score,
                                                             (int(bullet.position[0] + bullet.size[0]), int(bullet.position[1] - 20)),
                                                             self.score_color, self.background_color))
                                    self.explosions.append(Explosion(self.explosion1_pic, (int(bullet.position[0] + bullet.size[0] / 2),
                                                                                           int(bullet.position[1])),
                                                                     self.explosion1_pic_grid, self.explosion1_freq))
                            else:
                                # hit did not cause damage
                                self.play_sound(self.sound_miss)
                            # mark bullet for removal
                            bullet.position[1] = -9999
                            break
                else:

                    for powerup in self.powerups:
                        if bullet.rect.colliderect(powerup.rect):
                            mask_offset = (bullet.rect[0] - powerup.rect[0], bullet.rect[1] - powerup.rect[1])
                            if powerup.mask.overlap(bullet.mask, mask_offset):
                                self.play_sound(powerup.sound_award, 0)
                                # show powerup description as score (0 points)
                                self.scores.append(Score(self, 0, powerup.desc, self.font_score,
                                                         (int(bullet.position[0] - 60), int(bullet.position[1] - 40)),
                                                         self.score_color, self.background_color))
                                self.ship.add_powerup(powerup, self.ship_shoot_freq)
                                self.powerups.remove(powerup)
                                # mark bullet for removal
                                bullet.position[1] = -9999
                                break

            if bullet.position[0] < -bullet.size[0] or bullet.position[0] > self.screen_size[0] \
                    or bullet.position[1] < -bullet.size[1] or bullet.position[1] > self.screen_size[1]:
                self.ship.bullets.remove(bullet)
            else:
                bullet.draw(self.screen)

    def ship_ops(self, time, prev_time):

        if self.ship.status == 0 or (self.ship.status == 1 and (time - self.ship.start_time) % 100 < 50):
            # if status = 0 (normal), draw ship. If status = 1 ("new" ship), draw a blinking ship.
            self.ship.draw(self.screen)

        if self.ship.status != 0 and not self.game_over:
            if self.ship.lives == 0:
                self.game_over = True
            elif time - self.ship.start_time > 2000:
                # after a pause, bring next ship (status: "new") and, after a pause, make it "normal".
                self.ship.status -= 1
                self.ship.start_time = time

        # check ship's powerups' life times
        if len(self.ship.powerups) > 0:
            for i in range(len(self.ship.powerups) - 1, -1, -1):
                powerup = self.ship.powerups[i]
                # if powerup initially has -1 as life_time (item [1]), it will never end (so not decreased)
                if powerup[1] >= 0 and powerup[1] <= time - prev_time:
                    self.ship.end_powerup(powerup, self.powerup_data, self.ship_shoot_freq)
                else:
                    if powerup[1] > time - prev_time:
                        # decrease life time, but preserve Double Fire life if using Triple Fire
                        if not (self.ship.bullet_type == 3 and self.powerup_data[powerup[0]][2] == 'Double Fire'):
                            powerup[1] -= time - prev_time
                if powerup[0] < 0:
                    del self.ship.powerups[i]

    def alien_ops(self, time, prev_time):

        if self.freeze_aliens:
            self.level_time += time - prev_time  # this keeps bosses frozen

        for alien in self.aliens:
            if self.freeze_aliens:
                alien.last_move += time - prev_time  # this keeps 'normal' aliens frozen
            else:
                alien.move(time, self.level_time, self.screen_size)
            alien.draw(self.screen)

            # test if alien hits the ship or has passed lower than the ship
            if (alien.rect.colliderect(self.ship.rect) or alien.position[1] >= self.ship.position[1]) and not self.game_over:
                mask_offset = (alien.rect[0] - self.ship.rect[0], alien.rect[1] - self.ship.rect[1])
                if self.ship.mask.overlap(alien.mask, mask_offset):
                    self.ship.is_hit(time)
                    self.play_sound(self.ship.sound_explosion)
                    self.ship.lives = 0  # all lives lost if aliens make it!
                    self.explosions.append(Explosion(self.explosion_ship_pic, (int(self.ship.position[0] + self.ship.size[0] / 2),
                                                                               int(self.ship.position[1] + self.ship.size[1] / 2)),
                                                     self.explosion_ship_pic_grid, self.explosion_ship_freq))

            # alien shoots
            if not self.game_over and not self.freeze_aliens and random.random() < self.fps / alien.shoot_freq:
                for i in range(np.shape(alien.cannon_pos)[0]):
                    self.alien_bullets.append(Bullet(alien.bullet_pic, alien.position + alien.cannon_pos[i, :], np.array([0.0, 6.5]), np.array([0.0, 0.0])))

    def ufo_ops_back(self, time, prev_time):

        # add random ufos, but only if aliens left (or not in game mode)
        if random.random() < self.level_ufo_probability / 50 and (self.game_mode != 'game' or (len(self.aliens) > 0 and not self.freeze_aliens)):
            # add a ufo
            speed = 0.35 + (self.level ** 0.5) * random.random() * 0.1
            from_side = random.randint(0, 1) * 2 - 1
            self.ufos.append(Ufo(self.alien_ufo_pic, self.bullet_ufo1_pic, self.alien_ufo_sound_explosion, 150, speed, from_side, self.screen_size))

        # move and draw ufo
        for ufo in self.ufos:
            last_phase = ufo.phase + 0
            if self.freeze_aliens:
                # slow down ufos, no total freeze
                freeze_speed = 0.2
            else:
                freeze_speed = 1.0

            ufo.move(time, freeze_speed, self.screen_size)  # moving updates ufo.phase

            if ufo.phase >= ufo.turning_point + 75 / 2 and last_phase < ufo.turning_point + 75 / 2:
                # ufo crossed aliens (in z coordinate) in mid turn, shoot
                for i in range(-2, 3):
                    self.alien_bullets.append(Bullet(ufo.bullet_pic,
                                                     (ufo.position[0] + ufo.size[0] / 2 - ufo.bullet_size[0] / 2, ufo.position[1] + ufo.size[1]),
                                                     np.array([i, (2 - abs(i)) * 0.3]), np.array([0.0, 0.2])))

            if ufo.phase > ufo.turning_point + 75 and (np.min(ufo.position + ufo.size) < 0 or np.max(ufo.position - self.screen_size)) > 0:
                # ufo outside of screen: remove
                self.ufos.remove(ufo)
            else:
                if ufo.phase < ufo.turning_point + 75 / 2:
                    # ufo behind the aliens
                    ufo.draw(self.screen)

    def ufo_ops_front(self, time):

        # draw ufo when in front of the aliens
        for ufo in self.ufos:
            if ufo.phase >= ufo.turning_point + 75 / 2:
                ufo.draw(self.screen)

    def score_ops(self, time):

        # show scores
        for score in self.scores:
            if time - score.start_time > score.show_time:
                self.scores.remove(score)
            else:
                score.move()
                score.draw(self.screen, time)

    def explosion_ops(self):

        for explosion in self.explosions:
            explosion.draw(self.screen)
            explosion.freq_cnt += 1
            if explosion.freq_cnt == explosion.freq:
                explosion.freq_cnt = 0
                explosion.phase[0] += 1
                if explosion.phase[0] == explosion.grid[0]:
                    explosion.phase[0] = 0
                    explosion.phase[1] += 1
                    if explosion.phase[1] == explosion.grid[1]:
                        self.explosions.remove(explosion)

    def powerup_ops(self, time):

        # show powerups. These are powerups on screen (not ship's powerups).
        for powerup in self.powerups:
            if time - powerup.start_time > powerup.show_time:
                self.powerups.remove(powerup)
            else:
                powerup.draw(self.screen, time)

    def info_ops(self):

        # show powerups:
        y = self.score_pic_position[1]
        x = 220
        max_w = 260
        for powerup in self.ship.powerups:  # powerup is an array of [pid, life_time], powerup_data a list of (pic, life_time, desc, text_pic)
            pid = powerup[0]
            life_time = powerup[1]
            self.screen.blit(self.powerup_data[pid][0], (x, y))  # draw powerup pic
            self.screen.blit(self.powerup_data[pid][3], (x + self.powerup_data[pid][0].get_size()[0] + 6, y + 4))  # add powerup description
            # add bar showing life time
            if life_time < 0:
                p_width = max_w
            else:
                p_width = min(max_w, max_w * life_time // self.powerup_data[pid][1])
            color = (int((max_w - p_width) * 200 / max_w), 0, int(p_width * 200 / max_w))
            pygame.draw.rect(self.screen, color, (x + self.powerup_data[pid][0].get_size()[0] + self.powerup_desc_width + 12, y + 4, p_width, 12))
            y += 20

        # show game score
        if self.show_score != self.game_score:
            if self.show_score < self.game_score - 500000:
                self.show_score += 10000
            elif self.show_score < self.game_score - 50000:
                self.show_score += 1000
            elif self.show_score < self.game_score - 5000:
                self.show_score += 100
            elif self.show_score < self.game_score - 1000:
                self.show_score += 20
            elif self.show_score < self.game_score - 100:
                self.show_score += 5
            else:
                self.show_score += 1
            # create new score pic only when needed
            self.score_pic = self.font_game_score.render(str(self.show_score), True, self.score_color, self.background_color)
            self.score_pic.set_colorkey(self.background_color)
        self.screen.blit(self.score_pic, self.score_pic_position)

        # show level nr
        self.screen.blit(self.level_pic, self.level_pic_position)

        # show lives
        if self.ship.status == 2:
            lives = self.ship.lives + 1  # ship is dead but not yet using the backup ships
        else:
            lives = self.ship.lives
        for i in range(1, lives):
            self.screen.blit(self.ship.pic_small, (self.level_pic_position[0] - 20 - int(1.2 * i * self.ship.pic_small_size[0]), self.level_pic_position[1]))

    def game_over_ops(self, time):

        if self.game_over_time == 0:
            self.game_over_time = time
            self.play_sound(self.sound_defeat, 0)
        elif time > self.game_over_time + 2 * self.switch_time_ms and len(self.ship.bullets) == 0 and self.show_score == self.game_score:
            self.game_mode = 'go to start page'
            self.switch_time = time
            self.star_move = np.array([0.0, 0.0, -0.5])
            self.level_ufo_probability = 0.05
            self.freeze_aliens = False
            return

        star_move_frac = max(min((time - (self.game_over_time + self.switch_time_ms)) / self.switch_time_ms, 1.0), 0.0)
        self.star_move = np.array([0.0,
                                   125.0 * np.cos(star_move_frac * np.pi / 2.0),
                                   -0.5 * np.sin(star_move_frac * np.pi / 2.0)])

        position = ((self.screen_size - self.game_over_pic.get_size()) / 2).astype(int)
        self.screen.blit(self.game_over_pic, position)

    # ----------------   new level setup   ----------------
    def setup_level(self, time):

        self.level_pic = self.font_game_score.render('Level ' + str(self.level), True, self.score_color, self.background_color)
        self.level_pic.set_colorkey(self.background_color)
        self.level_pic_position = (self.screen_size[0] - self.level_pic.get_size()[0] - 10, 10)
        self.level_time = time

        # create aliens
        self.aliens.clear()

        # find mamimum "non-boss" level
        for i in range(len(self.level_data) - 1, 0, -1):
            if self.level_data[i][12] > 0:
                lvl_max = i
                break

        # set level
        lvl = (self.level - 1) % len(self.level_data)  # re-use all levels
        lvl_data = self.level_data[lvl]
        self.level_ufo_probability = 1.0 - lvl_data[15] ** (self.level / 2)
        self.level_new_life = lvl_data[16]
        move_delay = lvl_data[13] / (self.level ** 0.5)
        if lvl_data[6] == 1:
            hit_nr = 1
        else:
            hit_nr = int(lvl_data[6] * (self.level / 3) ** 0.5)  # nr of hits required to kill

        if lvl_data[12] == 0:
            # y_move = 0: boss level
            shoot_freq = lvl_data[14] // ((self.level / 3) ** 0.3)  # the bigger the less shots
            pos = ((self.screen_size - np.array(lvl_data[0].get_size())) / 2).astype(np.int16)
            self.aliens.append(Alien(
                lvl_data[0], lvl_data[1], lvl_data[2], lvl_data[3], lvl_data[4],
                lvl_data[5] * hit_nr, hit_nr, pos, lvl_data[11], lvl_data[12], 0, move_delay, 0, 1, shoot_freq
                ))
        else:
            # group of aliens
            # for "re-used" levels, use aliens from two different levels. If first round, lvl2 = lvl
            lvl2 = (lvl + int((self.level - 1) / len(self.level_data))) % len(self.level_data)
            if self.level_data[lvl2][12] == 0:
                lvl2 = (lvl2 + 1) % len(self.level_data)  # boss level --> use the enxt level
            lvl2_data = self.level_data[lvl2]
            # pick number of aliens and size multipliers from maximum level, if re-using levels
            if lvl2 == lvl:
                lvl_max_data = self.level_data[lvl]
            else:
                lvl_max_data = self.level_data[lvl_max]
            shoot_freq = lvl_data[14] // (self.level ** 0.3)  # the bigger the less shots
            alien_size = np.maximum(np.array(lvl_data[0].get_size()), np.array(lvl2_data[0].get_size()))
            x_size = int(alien_size[0] * lvl_max_data[9])
            x_times = int(((self.screen_size[0] - alien_size[0]) - (lvl_max_data[7] - 1) * x_size - 10) / lvl_data[11])
            for y in range(lvl_max_data[8]):
                # make sure there's some room between the ship and the aliens
                if 90 + (y + 4) * alien_size[1] * lvl_max_data[10] < self.ship.position[1]:
                    for x in range(lvl_max_data[7]):
                        pos = np.array([10 + x * x_size, 90 + y * alien_size[1] * lvl_max_data[10]]).astype(np.int16)
                        if (y + x) % 2 == 0:
                            self.aliens.append(Alien(
                                lvl_data[0], lvl_data[1], lvl_data[2], lvl_data[3], lvl_data[4],
                                lvl_data[5] * hit_nr, hit_nr, pos, lvl_data[11], lvl_data[12], x_times, move_delay, (y + x) * 30, 1, shoot_freq
                                ))
                        else:
                            self.aliens.append(Alien(
                                lvl2_data[0], lvl2_data[1], lvl2_data[2], lvl2_data[3], lvl2_data[4],
                                lvl2_data[5] * hit_nr, hit_nr, pos, lvl_data[11], lvl_data[12], x_times, move_delay, (y + x) * 30, 1, shoot_freq
                                ))

    def setup_level_data(self):

        # setup all levels
        # data structure for a level:
        #   [0, 1]   alien picture, alien_boss1_hit_area (for bosses)
        #   [2, 3]   alien bullet picture, cannon positions (array)
        #   [4]      alien explosion sound
        #   [5, 6]   alien score per hit, hit nr (hits required before killed - will be adjusted by level nr if > 1)
        #   [7 - 12] alien x_nr, y_nr (number of aliens in a matrix), x_size, y_size (multipliers defining distances), x_move, y_move (y_move = 0 for boss)
        #   [13, 14] alien move delay (bigger = slower), shooting frequency (bigger = less shots). Will be adjusted by level nr
        #   [15, 16] 1 - ufo probability (ill be adjusted by level nr), add life if level completed (True/False)

        # level 1
        self.level_data.append((
            self.alien1_pic, None,
            self.bullet_alien1_pic, np.array([[self.alien1_pic.get_size()[0] // 2 - 1, self.alien1_pic.get_size()[1]]], dtype=float),
            self.alien1_sound_explosion,
            25, 1,
            6, 5, 1.8, 1.5, 20, 40,
            500, 32000,
            0.93, False
            ))
        # level 2
        self.level_data.append((
            self.alien2_pic, None,
            self.bullet_alien1_pic, np.array([[self.alien2_pic.get_size()[0] // 2 - 1, self.alien2_pic.get_size()[1]]], dtype=float),
            self.alien1_sound_explosion,
            28, 1,
            8, 5, 1.6, 1.4, 20, 45,
            500, 32000,
            0.93, False
            ))
        # level 3
        self.level_data.append((
            self.alien3_pic, None,
            self.bullet_alien1_pic, np.array([[self.alien3_pic.get_size()[0] // 2 - 1, self.alien3_pic.get_size()[1]]], dtype=float),
            self.alien1_sound_explosion,
            30, 1,
            7, 6, 1.6, 1.4, 20, 45,
            500, 32000,
            0.93, False
            ))
        # level 4 (boss)
        self.level_data.append((
            self.alien_boss2_pic, self.alien_boss2_hit_area,
            self.bullet_alien2_pic, self.alien_boss2_cannon_pos,
            self.alien_boss_sound_explosion,
            20, 12,
            0, 0, 0, 0, 0, 0,
            1000, 5000,
            0.96, False
            ))
        # level 5
        self.level_data.append((
            self.alien4_pic, None,
            self.bullet_alien1_pic, np.array([[self.alien4_pic.get_size()[0] // 2 - 1, self.alien4_pic.get_size()[1]]], dtype=float),
            self.alien1_sound_explosion,
            32, 1,
            7, 7, 1.6, 1.3, 20, 50,
            500, 32000,
            0.93, False
            ))
        # level 6
        self.level_data.append((
            self.alien5_pic, None,
            self.bullet_alien1_pic, np.array([[self.alien5_pic.get_size()[0] // 2 - 1, self.alien5_pic.get_size()[1]]], dtype=float),
            self.alien1_sound_explosion,
            35, 1,
            8, 7, 1.5, 1.4, 22, 45,
            500, 32000,
            0.93, False
            ))
        # level 7
        self.level_data.append((
            self.alien6_pic, None,
            self.bullet_alien1_pic, np.array([[self.alien6_pic.get_size()[0] // 2 - 1, self.alien6_pic.get_size()[1]]], dtype=float),
            self.alien1_sound_explosion,
            38, 1,
            8, 8, 1.5, 1.3, 24, 50,
            500, 32000,
            0.93, False
            ))
        # level 8 (boss)
        self.level_data.append((
            self.alien_boss1_pic, self.alien_boss1_hit_area,
            self.bullet_alien2_pic, self.alien_boss1_cannon_pos,
            self.alien_boss_sound_explosion,
            25, 12,
            0, 0, 0, 0, 0, 0,
            1000, 5000,
            0.96, True
            ))
        # level 9
        self.level_data.append((
            self.alien7_pic, None,
            self.bullet_alien1_pic, np.array([[self.alien7_pic.get_size()[0] // 2 - 1, self.alien7_pic.get_size()[1]]], dtype=float),
            self.alien1_sound_explosion,
            40, 1,
            8, 8, 1.5, 1.4, 25, 50,
            500, 32000,
            0.93, False
            ))
        # level 10
        self.level_data.append((
            self.alien8_pic, None,
            self.bullet_alien1_pic, np.array([[self.alien8_pic.get_size()[0] // 2 - 1, self.alien8_pic.get_size()[1]]], dtype=float),
            self.alien1_sound_explosion,
            42, 1,
            8, 8, 1.5, 1.3, 25, 50,
            500, 32000,
            0.93, False
            ))
        # level 11
        self.level_data.append((
            self.alien9_pic, None,
            self.bullet_alien1_pic, np.array([[self.alien9_pic.get_size()[0] // 2 - 1, self.alien9_pic.get_size()[1]]], dtype=float),
            self.alien1_sound_explosion,
            44, 1,
            8, 8, 1.5, 1.3, 25, 50,
            500, 32000,
            0.93, False
            ))
        # level 12 (boss)
        self.level_data.append((
            self.alien_boss3_pic, self.alien_boss3_hit_area,
            self.bullet_alien2_pic, self.alien_boss3_cannon_pos,
            self.alien_boss_sound_explosion,
            32, 12,
            0, 0, 0, 0, 0, 0,
            1000, 5000,
            0.96, False
            ))
        # level 13
        self.level_data.append((
            self.alien10_pic, None,
            self.bullet_alien1_pic, np.array([[self.alien10_pic.get_size()[0] // 2 - 1, self.alien10_pic.get_size()[1]]], dtype=float),
            self.alien1_sound_explosion,
            46, 1,
            9, 8, 1.5, 1.4, 25, 50,
            500, 32000,
            0.93, False
            ))
        # level 14
        self.level_data.append((
            self.alien11_pic, None,
            self.bullet_alien1_pic, np.array([[self.alien11_pic.get_size()[0] // 2 - 1, self.alien11_pic.get_size()[1]]], dtype=float),
            self.alien1_sound_explosion,
            48, 1,
            9, 9, 1.5, 1.4, 25, 50,
            500, 32000,
            0.93, False
            ))
        # level 15
        self.level_data.append((
            self.alien12_pic, None,
            self.bullet_alien1_pic, np.array([[self.alien12_pic.get_size()[0] // 2 - 1, self.alien12_pic.get_size()[1]]], dtype=float),
            self.alien1_sound_explosion,
            50, 1,
            9, 9, 1.5, 1.5, 25, 50,
            500, 32000,
            0.93, False
            ))
        # level 16 (boss)
        self.level_data.append((
            self.alien_boss4_pic, self.alien_boss4_hit_area,
            self.bullet_alien2_pic, self.alien_boss4_cannon_pos,
            self.alien_boss_sound_explosion,
            35, 12,
            0, 0, 0, 0, 0, 0,
            1000, 5000,
            0.96, True
            ))

    # ----------------   powerup setup   ----------------
    def setup_powerups(self, pic, font, font_desc):

        # set up a list of power ups.
        # if pic has green color, it will be changed to a power up specific color
        old_color = (0, 255, 0)
        powerup_data = []

        desc = 'Double Fire'
        powerup_data.append(self.setup_powerups_pic(pic.copy(), old_color, (120, 20, 220), desc, desc[:1], 24000, font, font_desc))
        desc = 'Triple Fire'
        powerup_data.append(self.setup_powerups_pic(pic.copy(), old_color, (120, 20, 20), desc, desc[:1], 15000, font, font_desc))
        desc = 'Rapid Fire'
        powerup_data.append(self.setup_powerups_pic(pic.copy(), old_color, (20, 20, 120), desc, desc[:1], 40000, font, font_desc))
        desc = 'Auto Fire'
        powerup_data.append(self.setup_powerups_pic(pic.copy(), old_color, (220, 20, 120), desc, desc[:1], -1, font, font_desc))
        desc = 'Freeze Aliens'
        powerup_data.append(self.setup_powerups_pic(pic.copy(), old_color, (160, 220, 160), desc, desc[:1], 5000, font, font_desc))
        desc = 'Shield'
        powerup_data.append(self.setup_powerups_pic(pic.copy(), old_color, (160, 160, 220), desc, desc[:1], 9000, font, font_desc))

        return powerup_data

    def setup_powerups_pic(self, new_pic, old_color, new_color, desc, letter, life_time, font, font_desc):

        rgb_array = pygame.surfarray.pixels2d(new_pic)
        self.recolor2(rgb_array, old_color, new_color)
        rgb_array = None
        new_pic.set_colorkey((255, 255, 255))
        if new_color[0] + new_color[1] + new_color[2] > 3 * 128:
            letter_color = (1, 1, 1)  # use black if letter background very light colored
        else:
            letter_color = (220, 220, 220)
        pic_letter = font.render(letter, False, letter_color, (0, 0, 0))
        pic_letter.set_colorkey((0, 0, 0))
        pic_desc = font_desc.render(desc, False, (220, 220, 220))
        pic_desc.set_colorkey((0, 0, 0))
        new_pic.blit(pic_letter, ((new_pic.get_size()[0] - pic_letter.get_size()[0]) // 2, (new_pic.get_size()[1] - pic_letter.get_size()[1]) // 2))

        return (new_pic, life_time, desc, pic_desc)

    # ----------------   setup titles, credits, and instructions screens   ----------------
    def setup_titles(self, time):

        # setup title screen
        self.screen_titles.fill(self.background_color)
        title_pos = np.array([0, 80])
        title_pos[1] = self.add_text_row(self.screen_titles, self.font_title, 'SPACE', title_pos[1], None, 1.1)
        title_pos[1] = self.add_text_row(self.screen_titles, self.font_title, 'INVADERS', title_pos[1], None, 2.0)

        # high score list titles
        self.title_hs_pos = title_pos.copy()
        title_pos[1] = self.add_text_row(self.screen_titles, self.font_highscorelist, '- TOP PLAYERS -', title_pos[1], None, 1.5)
        title_height = (title_pos[1] - self.title_hs_pos[1]) // 1.5

        # store the left, right and right position of player, score and level
        hs_pos_x = np.array([(self.screen_size[0] - 600) // 2,
                             -((self.screen_size[0] - 600) // 2 + 480),
                             -((self.screen_size[0] - 600) // 2 + 600)])
        title_pos[1] = self.add_text_row(self.screen_titles, self.font_highscorelist, 'PLAYER', title_pos[1], hs_pos_x[0], 0.0)
        title_pos[1] = self.add_text_row(self.screen_titles, self.font_highscorelist, 'SCORE', title_pos[1], hs_pos_x[1], 0.0)
        title_pos[1] = self.add_text_row(self.screen_titles, self.font_highscorelist, 'LVL', title_pos[1], hs_pos_x[2], 1.5)

        # texts at the bottom
        self.title_hs_pos_end = np.array([0, self.screen_size[1] - title_height * 3])
        title_pos[1] = self.title_hs_pos_end[1]
        if self.game_mode == 'new high score':
            title_pos[1] = self.add_text_row(self.screen_titles, self.font_highscorelist, 'CONGRATULATIONS! NEW HIGH SCORE', title_pos[1], None, 1.2)
            title_pos[1] = self.add_text_row(self.screen_titles, self.font_highscorelist, 'ENTER YOUR NAME AND PRESS RETURN', title_pos[1], None, 1.2)
        else:
            title_pos[1] = self.add_text_row(self.screen_titles, self.font_highscorelist, 'PRESS SPACE BAR FOR A NEW GAME', title_pos[1], None, 1.2)
            title_pos[1] = self.add_text_row(self.screen_titles, self.font_highscorelist, 'ESC TO EXIT     I FOR INSTRUCTIONS', title_pos[1], None, 1.2)

        # add high scores at the middle
        title_pos[1] = self.title_hs_pos[1] + 1.5 * title_height * 2.0
        for i in range(0, self.highscore_nr):
            hs = self.highscores[i]
            if hs[1] > 0:
                # if getting high score name, add a blinking underscore
                text = hs[0]
                if self.game_mode == 'new high score' and i == self.latest_hs_nr and len(hs[0]) < self.highscore_name_length and time % 1000 > 500:
                    text = text + '_'
                title_pos[1] = self.add_text_row(self.screen_titles, self.font_highscorelist, text, title_pos[1], hs_pos_x[0], 0.0)
                title_pos[1] = self.add_text_row(self.screen_titles, self.font_highscorelist, str(hs[1]), title_pos[1], hs_pos_x[1], 0.0)
                title_pos[1] = self.add_text_row(self.screen_titles, self.font_highscorelist, str(hs[2]), title_pos[1], hs_pos_x[2], 1.1)
                # test for overlap with end texts
                if title_pos[1] >= self.title_hs_pos_end[1]:
                    break

        subsurf = self.screen_titles.subsurface((0, self.title_hs_pos[1], self.screen_size[0] - 1, self.title_hs_pos_end[1] - self.title_hs_pos[1] - 1))
        self.title_hs_rect = np.array([
            subsurf.get_bounding_rect()[0] + subsurf.get_offset()[0],
            subsurf.get_bounding_rect()[1] + subsurf.get_offset()[1],
            subsurf.get_bounding_rect()[2],
            subsurf.get_bounding_rect()[3]
            ])

    def setup_credits(self):

        # setup credits screen. Take a copy of titles screen and clear the middle for credits.
        self.screen_credits = self.screen_titles.copy()
        self.screen_credits.fill(self.background_color,
                                      (0, self.title_hs_pos[1], self.screen_size[0] - 1, self.title_hs_pos_end[1] - self.title_hs_pos[1] - 1))

        title_pos = self.title_hs_pos.copy()
        title_pos[1] = self.add_text_row(self.screen_credits, self.font_highscorelist, '- CREDITS -', title_pos[1], None, 3.0)
        title_pos[1] = self.add_text_row(self.screen_credits, self.font_highscorelist, 'LEAD PROGRAMMER', title_pos[1], None, 1.5)
        title_pos[1] = self.add_text_row(self.screen_credits, self.font_highscorelist, 'KALLE SAARIAHO', title_pos[1], None, 3.0)
        title_pos[1] = self.add_text_row(self.screen_credits, self.font_highscorelist, 'ASSISTANT PROGRAMMER', title_pos[1], None, 1.5)
        title_pos[1] = self.add_text_row(self.screen_credits, self.font_highscorelist, 'EELIS SAARIAHO', title_pos[1], None, 3.0)
        title_pos[1] = self.add_text_row(self.screen_credits, self.font_highscorelist, '(c) 2022', title_pos[1], None, 1.1)

        subsurf = self.screen_credits.subsurface((0, self.title_hs_pos[1], self.screen_size[0] - 1, self.title_hs_pos_end[1] - self.title_hs_pos[1] - 1))
        self.title_cr_rect = np.array([
            subsurf.get_bounding_rect()[0] + subsurf.get_offset()[0],
            subsurf.get_bounding_rect()[1] + subsurf.get_offset()[1],
            subsurf.get_bounding_rect()[2],
            subsurf.get_bounding_rect()[3]
            ])

    def setup_instructions(self):

        # setup instructions screen. Take a copy of titles screen and clear the middle for instructions.
        self.screen_instructions = self.screen_titles.copy()
        self.screen_instructions.fill(self.background_color,
                                      (0, self.title_hs_pos[1], self.screen_size[0] - 1, self.title_hs_pos_end[1] - self.title_hs_pos[1] - 1))

        title_pos = self.title_hs_pos.copy()
        title_pos[1] = self.add_text_row(self.screen_instructions, self.font_highscorelist, '- INSTRUCTIONS -', title_pos[1], None, 2.0)
        title_pos[1] = self.add_text_row(self.screen_instructions, self.font_highscorelist, 'ALIEN INVASION IS IMMINENT.', title_pos[1], None, 2.0)
        title_pos[1] = self.add_text_row(self.screen_instructions, self.font_highscorelist, 'YOU AND YOUR THREE SPACE FIGHTERS', title_pos[1], None, 1.1)
        title_pos[1] = self.add_text_row(self.screen_instructions, self.font_highscorelist, 'ARE ALL THAT\'S LEFT TO STOP THEM.', title_pos[1], None, 1.5)
        title_pos[1] = self.add_text_row(self.screen_instructions, self.font_highscorelist, 'WATCH OUT FOR UFOS AND THEIR BOMBS!', title_pos[1], None, 1.1)
        title_pos[1] = self.add_text_row(self.screen_instructions, self.font_highscorelist, 'UFOS CAN ONLY BE SHOT IN THE MIDDLE', title_pos[1], None, 1.1)
        title_pos[1] = self.add_text_row(self.screen_instructions, self.font_highscorelist, 'OF THEIR ATTACK PATTERN. THEY', title_pos[1], None, 1.1)
        title_pos[1] = self.add_text_row(self.screen_instructions, self.font_highscorelist, 'MAY LEAVE POWER UPS IF SHOT DOWN.', title_pos[1], None, 1.5)
        title_pos[1] = self.add_text_row(self.screen_instructions, self.font_highscorelist, 'USE LEFT AND RIGHT CURSOR KEYS TO', title_pos[1], None, 1.1)
        title_pos[1] = self.add_text_row(self.screen_instructions, self.font_highscorelist, 'MOVE AND SPACE BAR TO FIRE.', title_pos[1], None, 1.1)

        subsurf = self.screen_instructions.subsurface((0, self.title_hs_pos[1], self.screen_size[0] - 1, self.title_hs_pos_end[1] - self.title_hs_pos[1] - 1))
        self.title_ins_rect = np.array([
            subsurf.get_bounding_rect()[0] + subsurf.get_offset()[0],
            subsurf.get_bounding_rect()[1] + subsurf.get_offset()[1],
            subsurf.get_bounding_rect()[2],
            subsurf.get_bounding_rect()[3]
            ])

        # get the minimum Rect which covers all high scores, credits, and instructions.
        self.title_rect = np.array([
            min(self.title_hs_rect[0], self.title_ins_rect[0], self.title_cr_rect[0]),
            min(self.title_hs_rect[1], self.title_ins_rect[1], self.title_cr_rect[1]),
            max(self.title_hs_rect[0] + self.title_hs_rect[2], self.title_ins_rect[0] + self.title_ins_rect[2], self.title_cr_rect[0] + self.title_cr_rect[2]) \
                - min(self.title_hs_rect[0], self.title_ins_rect[0], self.title_cr_rect[0]),
            max(self.title_hs_rect[1] + self.title_hs_rect[3], self.title_ins_rect[1] + self.title_ins_rect[3], self.title_cr_rect[1] + self.title_cr_rect[3]) \
                - min(self.title_hs_rect[1], self.title_ins_rect[1], self.title_cr_rect[1])
            ])

        # set z_pos of titles to control the rotation perspective
        self.title_z_pos = max(self.title_rect[3] * 3, 1.5 * (self.title_rect[3] / 2) / (self.screen_size[0] / self.title_rect[2] - 1))

    def add_text_row(self, screen, font, text, y, x=None, spacing=1.5):

        # add text to titles/instructions. If x is None then center text, if x<0 then right align at abs(x). Returns next y.
        f_screen = font.render(text, True, self.title_color, self.background_color)
        f_size = f_screen.get_size()
        if x is None:
            pos = ((self.screen_size[0] - f_size[0]) // 2, y)  # center
        elif x >= 0:
            pos = (x, y)  # left align
        else:
            pos = (-x - f_size[0], y)  # right align at abs(x)
        screen.blit(f_screen, pos)
        return int(y + f_size[1] * spacing)

    def copy_titles(self):

        if self.angle != 0:
            self.screen.blit(self.screen_info, (0, 0))
        elif self.info_screen == 'titles':
            self.screen.blit(self.screen_titles, (0, 0))
        elif self.info_screen == 'credits':
            self.screen.blit(self.screen_credits, (0, 0))
        else:
            self.screen.blit(self.screen_instructions, (0, 0))

    def rotate_titles(self, time):

        # rotates between high scores and instructions in 3D but only around the horizontal axis.

        # first check if rotation required.
        if time > self.credits_time + 20000 and self.angle_add == 0 and self.info_screen == 'credits':
            self.angle_add = 2
            self.info_screen_next = 'titles'
            self.credits_time = time

        if time > self.credits_time + 15000 and self.angle_add == 0 and self.info_screen == 'titles':
            self.angle_add = 2
            self.info_screen_next = 'credits'

        if self.angle_add != 0:
            self.angle += self.angle_add
            if self.angle == 0:
                self.angle_add = 0
            elif self.angle == 90:
                # rotated 90 degrees - switch source image and continue from -90 degrees
                self.angle = -90
                if self.info_screen == 'titles':
                    self.info_screen = self.info_screen_next
                else:
                    self.info_screen = 'titles'
        else:
            return  # no rotation

        self.screen_info.fill(self.background_color, (0, self.title_rect[1], self.screen_size[0], self.title_rect[3]))

        if self.info_screen == 'titles':
            while self.screen_titles.get_locked():
                self.screen_titles.unlock()
            rgb_array_src = pygame.surfarray.pixels2d(self.screen_titles.subsurface(self.title_rect))
        elif self.info_screen == 'credits':
            while self.screen_credits.get_locked():
                self.screen_credits.unlock()
            rgb_array_src = pygame.surfarray.pixels2d(self.screen_credits.subsurface(self.title_rect))
        else:
            while self.screen_instructions.get_locked():
                self.screen_instructions.unlock()
            rgb_array_src = pygame.surfarray.pixels2d(self.screen_instructions.subsurface(self.title_rect))

        rect_mid = self.title_rect[2:4] // 2
        sa, ca = np.sin(self.angle * np.pi / 180), np.cos(self.angle * np.pi / 180)  # sin and cos of angle
        y_top = int(-rect_mid[1] * ca)
        z_top = -rect_mid[1] * sa

        # yr = range of y coordinates covered by destination
        # y_mapr = range of y coordinates mapped to source data
        # zr = range of z coordinates covered by destination (one z for each y in yr)
        # z_multip = multiplier used for getting x coordinate mapping for each z in zr
        if y_top < 0:
            yr = np.arange(y_top, -y_top, dtype=np.int16)
        elif y_top > 0:
            yr = np.arange(y_top - 1, -y_top - 1, -1, dtype=np.int16)
        else:
            return  # exit if no rows to draw
        y_num = np.shape(yr)[0]
        y_mapr = np.linspace(0, self.title_rect[3], y_num, False).astype(np.int16)
        zr = np.linspace(z_top, -z_top, y_num)
        z_multip = (zr + self.title_z_pos) / self.title_z_pos

        # destination must be wider than source due to perspective - when zr is negative (closer to viewer). Pick m = maximum multiplier for x.
        m = 1.0 / min(z_multip[0], z_multip[-1])

        # define destination subsurface, wider but less high than source data (self.title_rect)
        dst_rect = np.array([
            int(max(self.title_rect[0] - (m - 1.0) * rect_mid[0], 0)),
            self.title_rect[1] + self.title_rect[3] // 2 - abs(y_top),
            int(min(self.title_rect[2] + 2.0 * (m - 1.0) * rect_mid[0], self.screen_size[0])),
            y_num
            ])

        while self.screen_info.get_locked():
            self.screen_info.unlock()
        rgb_array_dst = pygame.surfarray.pixels2d(self.screen_info.subsurface(dst_rect))

        # map x coordinates and y coordinates: for each destination pixel, one pixel in source data
        # x_map
        #   - get range of all x coordinates in destination np.arange(dst_rect[2])
        #   - subtract the middle dst_rect[2] / 2, and convert to "vertical" array [:, None]
        #   - multiply with z multipliers z_multip to apply perspective, and add source data middle x rect_mid[0]
        #   - result: when the line is close to the viewer ie. z_multip << 0, the wider destination is mapped 1:1 to source ("source widens)
        #           e.g. leftmost pixel in destination is mapped to the leftmost pixel in source, some source pixels are used multiple times.
        #         when the line is far from the viewer ie. z_multip >> 0, the wider destination is mapped "over" the source ("source narrows")
        #           e.g. leftmost pixel in destination is mapped left to the leftmost pixel in source: x coordinate is negative, and at the right edge >> source width
        # y_map_flat
        #   - make a matrix (the same size as x_map) by multiplying the range of y coordinates (y_mapr) with ones for all x coordinates (dst_rect[2])
        #   - filter out the destination coordinates outside of destination surface: [(x_map >= 0) & (x_map < self.title_rect[2])]
        #   - flatten the range
        # x_map_flat
        #   - filter as y_map_flat
        #   - flatten the range
        x_map = ((np.arange(dst_rect[2]) - (dst_rect[2] / 2))[:, None] * z_multip + rect_mid[0]).astype(np.int16)
        y_map_flat = ((y_mapr * np.ones((dst_rect[2]), dtype=np.int16)[:, None])[(x_map >= 0) & (x_map < self.title_rect[2])]).flatten()
        x_map_flat = (x_map[(x_map >= 0) & (x_map < self.title_rect[2])]).flatten()

        # draw all pixels of destination range but filtered as x_map_flat and y_map_flat, by picking the mapped image coordinates from source data
        rgb_array_dst[(x_map >= 0) & (x_map < self.title_rect[2])] = rgb_array_src[x_map_flat, y_map_flat]

    # ----------------   handle high scores   ----------------
    def read_scores(self):

        # read high scores from file
        if os.path.isfile(self.highscore_file + '.dat'):
            (fileobj, namehint) = self.load_dat(self.highscore_file)
            # with open(self.highscore_file, newline='') as f:
            #     self.highscores = json.load(f)[:self.highscore_nr]
            self.highscores = json.load(fileobj)[:self.highscore_nr]
        # make sure list is full length for easier handling
        for i in range(len(self.highscores), self.highscore_nr + 1):
            self.highscores.append(['', 0, 0])

    def write_scores(self):

        # write high scores to file
        enc_data = self.f.encrypt(json.dumps(self.highscores[:self.highscore_nr]).encode())
        open(self.highscore_file + '.dat', 'wb').write(enc_data)

    def highscore_add(self):

        # new high score to be added.
        i = 0
        # find the spot.
        while self.highscores[i][1] >= self.game_score:
            i += 1

        # make room and update.
        if i < self.highscore_nr - 1:
            self.highscores[i + 1:] = self.highscores[i:-1]
        self.highscores[i] = ['', self.game_score, self.level]

        # return the index with blank name
        return i

    def highscore_get_name(self, time):

        # process key presses and build a name
        while len(self.latest_hs_keys) > 0:
            key, unicode = self.latest_hs_keys.pop(0)
            if key == pygame.K_RETURN:
                # typing finished. Store high scores and move on.
                self.write_scores()
                self.latest_hs_keys.clear()
                self.game_mode = 'start page'
                self.credits_time = time
            elif key == pygame.K_BACKSPACE:
                if len(self.highscores[self.latest_hs_nr][0]) > 0:
                    # remove last character
                    self.highscores[self.latest_hs_nr][0] = self.highscores[self.latest_hs_nr][0][:-1]
            elif (((unicode >= 'A' and unicode <= 'Z')
                    or (unicode >= 'a' and unicode <= 'z')
                    or (unicode >= '0' and unicode <= '9')
                    or unicode in 'åäöÅÄÖ-_§!#$%&/()=+?\*.:<>|')
                  and len(self.highscores[self.latest_hs_nr][0]) < self.highscore_name_length):
                # add character, if on allowed characters list and name not too long
                self.highscores[self.latest_hs_nr][0] = self.highscores[self.latest_hs_nr][0] + unicode

    # ----------------   3D stars background   ----------------
    def move_stars(self, time, prev_time):

        # move stars in X,Y depending on their Z coordinate - the closer the faster / bigger move. Hence divide star_move X & Y by star Z
        self.stars += (time - prev_time) * self.star_move / np.hstack((self.stars[:, 2:3], self.stars[:, 2:3], np.ones((self.nr_stars, 1))))

        # return stars outside of X, Y range to the other edge. Here only out of Y down is needed.
        # self.stars[:, 0][self.stars[:, 0] < -self.mid_screen[0]] += self.screen_size[0] - 2
        # self.stars[:, 0][self.stars[:, 0] > self.mid_screen[0] - 2] -= self.screen_size[0] - 2
        # self.stars[:, 1][self.stars[:, 1] < -self.mid_screen[1]] += self.screen_size[1] - 2
        self.stars[:, 1][self.stars[:, 1] > self.mid_screen[1] - 2] -= self.screen_size[1] - 2

        # move stars using Z coordinate and Z move
        if self.star_move[2] != 0.0:
            self.stars[:, 0:2] *= self.stars[:, 2:3] / (self.stars[:, 2:3] + (time - prev_time) * self.star_move[2])

        # if outside of screen, normally replace with a new random star at a random X, Y edge and random Z
        nr_half = self.nr_stars // 2
        # first half: vertical edge
        self.stars[0:nr_half, :][(self.stars[0:nr_half, 2] > self.z_range[1])] = np.hstack((
            np.random.randint(0, 2, (np.shape(self.stars[0:nr_half, :][(self.stars[0:nr_half, 2] > self.z_range[1])])[0], 1)) * (self.screen_size[0] - 2) - self.mid_screen[0],
            np.random.rand(np.shape(self.stars[0:nr_half, :][(self.stars[0:nr_half, 2] > self.z_range[1])])[0], 1) * (self.screen_size[1] - 2) - self.mid_screen[1],
            np.random.rand(np.shape(self.stars[0:nr_half, :][(self.stars[0:nr_half, 2] > self.z_range[1])])[0], 1) * (self.z_range[1] - self.z_range[0]) + self.z_range[0]
            ))
        # second half: horizontal edge
        self.stars[nr_half:, :][(self.stars[nr_half:, 2] > self.z_range[1])] = np.hstack((
            np.random.rand(np.shape(self.stars[nr_half:, :][(self.stars[nr_half:, 2] > self.z_range[1])])[0], 1) * (self.screen_size[0] - 2) - self.mid_screen[0],
            np.random.randint(0, 2, (np.shape(self.stars[nr_half:, :][(self.stars[nr_half:, 2] > self.z_range[1])])[0], 1)) * (self.screen_size[1] - 2) - self.mid_screen[1],
            np.random.rand(np.shape(self.stars[nr_half:, :][(self.stars[nr_half:, 2] > self.z_range[1])])[0], 1) * (self.z_range[1] - self.z_range[0]) + self.z_range[0]
            ))
        # if Z too close OR X, Y out of bounds due to Z move, replace with a new random star at maximum Z
        self.stars[(self.stars[:, 2] < self.z_range[0]) | (abs(self.stars[:, 0] + 1) > self.mid_screen[0] - 1) | (abs(self.stars[:, 1] + 1) > self.mid_screen[1] - 1)] \
            = np.random.rand(np.shape(self.stars[(self.stars[:, 2] < self.z_range[0]) | (abs(self.stars[:, 0] + 1) > self.mid_screen[0] - 1) | (abs(self.stars[:, 1] + 1) > self.mid_screen[1] - 1)])[0], 3) \
            * np.array([self.screen_size[0] - 2, self.screen_size[1] - 2, 0]) + np.array([-self.mid_screen[0], -self.mid_screen[1], self.z_range[1]])

    def plot_stars(self):

        while self.screen.get_locked():
            self.screen.unlock()
        rgb_array = pygame.surfarray.pixels3d(self.screen)

        # define color as a function of distance
        c_shades = np.array([0.8, 0.8, 1.0])  # percentage of maximum R, G, B color used to tilt to Blue
        colors = (c_shades * ((1.0 - self.stars[:, 2:3] / (self.z_range[1] - self.z_range[0])) * 200 + 55)).astype(np.uint8)
        stars_int = (self.stars[:, 0:2]).astype(np.int16)
        rgb_array[(stars_int[:, 0] + self.mid_screen[0]), (stars_int[:, 1] + self.mid_screen[1]), 0:3] = colors
        # add additional pixels to those which are closest (color is above a threshold)
        rgb_array[(stars_int[:, 0][colors[:, 2] > 130] + self.mid_screen[0] + 1),
                  (stars_int[:, 1][colors[:, 2] > 130] + self.mid_screen[1]), 0:3] = colors[colors[:, 2] > 130]
        rgb_array[(stars_int[:, 0][colors[:, 2] > 180] + self.mid_screen[0]),
                  (stars_int[:, 1][colors[:, 2] > 180] + self.mid_screen[1] + 1), 0:3] = colors[colors[:, 2] > 180]
        rgb_array[(stars_int[:, 0][colors[:, 2] > 220] + self.mid_screen[0] + 1),
                  (stars_int[:, 1][colors[:, 2] > 220] + self.mid_screen[1] + 1), 0:3] = colors[colors[:, 2] > 220]

    # ----------------   small help functions   ----------------
    def play_sound(self, sound, channel=None):

        # plays a game sound on the next channel (all channels used in order).
        # if channel not specified, sounds will be missed as sometimes all channels are busy - rotates through channels.
        if channel is None:
            ch = pygame.mixer.Channel(self.channel)
            self.channel += 1  # move to next channel
            if self.channel == self.nr_channels:
                self.channel = 1
        else:
            ch = pygame.mixer.Channel(channel)
        ch.play(sound)

    def create_random_powerup(self, position):

        # adds a random powerup at position
        pid = -1
        while pid == -1:
            pid = random.randint(0, len(self.powerup_data) - 1)  # randomly pick power up
            # test if this powerup is already active with life time < 0 (i.e. eternal) - if so, pick a new one
            for powerup in self.ship.powerups:
                if powerup[0] == pid and powerup[1] < 0:
                    pid = -1
        self.powerups.append(PowerUp(pid, self.powerup_data, self.powerup_sound, position))

    def pic_colorkey(self, pic, color):

        # gives pic a colorkey and makes lighter (if light color) or darker (if dark color) shades equal to colorkey (important for some jpgs)
        pic.set_colorkey(color)
        if color != (0, 0, 0) and color != (255, 255, 255):
            # process the image only if color not totally black or white
            pic_array = pygame.surfarray.pixels3d(pic)
            if color[0] + color[1] + color[2] <= 48 * 3:
                pic_array[:, :, 0:3][(pic_array[:, :, 0] <= color[0]) & (pic_array[:, :, 1] <= color[1]) & (pic_array[:, :, 2] <= color[2])] = \
                    np.array([color[0], color[1], color[2]], dtype=np.uint8)
            elif color[0] + color[1] + color[2] >= 208 * 3:
                pic_array[:, :, 0:3][(pic_array[:, :, 0] >= color[0]) & (pic_array[:, :, 1] >= color[1]) & (pic_array[:, :, 2] >= color[2])] = \
                    np.array([color[0], color[1], color[2]], dtype=np.uint8)

    def adjust_pic(self, image, offset, size):

        # adjust a picture by moving it (offset) and fitting it in a new size.
        # first copy the original
        image_copy = image.copy()
        image_size = np.asarray(image.get_size(), dtype=int)
        # create a new one, with desired final size
        image = pygame.Surface(size)
        # apply offset
        image_offset = np.maximum(offset, np.zeros((2)))
        blit_offset = -np.minimum(offset, np.zeros((2)))
        blit_size = np.minimum(image_size - image_offset, size - blit_offset)
        # copy original image data back to it
        image.blit(image_copy, blit_offset, (image_offset, blit_size))

        return image

    def recolor(self, image, mode):

        # recolor image by exchanging two color components with each other, then return the resulting image.
        rgb_array = pygame.surfarray.pixels3d(image)
        if mode == 'R':
            # exchange green and blue
            rgb_array = np.stack((rgb_array[:, :, 0], rgb_array[:, :, 2], rgb_array[:, :, 1]), axis=-1)
        elif mode == 'G':
            # exchange red and blue
            rgb_array = np.stack((rgb_array[:, :, 2], rgb_array[:, :, 1], rgb_array[:, :, 0]), axis=-1)
        elif mode == 'B':
            # exchange red and green
            rgb_array = np.stack((rgb_array[:, :, 1], rgb_array[:, :, 0], rgb_array[:, :, 2]), axis=-1)

        return pygame.surfarray.make_surface(rgb_array)

    def recolor2(self, rgb_array, old_color, new_color):

        # recolor image by overwriting old color with new color.
        # colors given as (R, G, B).
        old_col_2d = 256 * 256 * old_color[0] + 256 * old_color[1] + old_color[2]
        new_col_2d = 256 * 256 * new_color[0] + 256 * new_color[1] + new_color[2]
        rgb_array[rgb_array == old_col_2d] = new_col_2d
        return rgb_array

    def load_dat(self, filename):

        # load and return a data file and its name hint
        enc_data = open(filename + '.dat', 'rb').read()
        dec_data = BytesIO(self.f.decrypt(enc_data))
        return (dec_data, self.namehints[filename[-1:]])


class Ship:
    """
    Player Space Ship
    """

    def __init__(self, pic, bullet_pic, shield_pic, sound_gun, sound_explosion, screen_size, speed, shoot_freq, lives):

        self.pic = pic
        self.mask = pygame.mask.from_surface(pic)
        self.size = np.asarray(self.pic.get_size())
        self.position = np.array(((screen_size[0] - self.size[0]) / 2, (screen_size[1] - self.size[1] - 70)))
        self.rect = pygame.Rect(self.position, self.size)
        self.bullet_pic = bullet_pic
        self.double_fire_angle = 10.0  # degree angle for double/triple fire
        self.bullet_pic_left = pygame.transform.rotate(self.bullet_pic, self.double_fire_angle)
        self.bullet_pic_right = pygame.transform.rotate(self.bullet_pic, -self.double_fire_angle)
        # self.bullet_mask = pygame.mask.from_surface(bullet_pic)
        self.bullet_size = np.asarray(self.bullet_pic.get_size())
        self.bullet_left_size = np.asarray(self.bullet_pic_left.get_size())
        self.bullet_right_size = np.asarray(self.bullet_pic_right.get_size())
        self.shield_pic = shield_pic
        self.shield_mask = pygame.mask.from_surface(shield_pic)
        self.shield_size = np.asarray(self.shield_pic.get_size())
        self.shield_rect = pygame.Rect(self.position - (self.shield_size - self.size) // 2, self.shield_size)
        self.sound_gun = sound_gun
        self.sound_explosion = sound_explosion
        self.speed = speed
        self.shoot_freq = shoot_freq
        self.last_shot_time = 0
        self.start_time = 0
        self.status = 0  # 0 = normal status; 1 = new ship (momentarily protected), 2 = dead
        self.lives = lives
        self.auto_fire = False
        self.shield = False
        self.space_down = False
        self.bullet_type = 1
        self.bullets = []
        self.powerups = []  # a list of arrays (pid, life_time) of active powerups
        self.pic_small = self.pic  # pygame.transform.scale(self.pic, (self.size[0] // 2, self.size[1] // 2))
        self.pic_small_size = self.pic_small.get_size()

    def move(self, direction, screen_size):

        if self.status in (0, 1):
            self.position[0] += direction * self.speed
            if self.position[0] < 0:
                self.position[0] = 0
            if self.position[0] > screen_size[0] - self.size[0]:
                self.position[0] = screen_size[0] - self.size[0]
            self.rect = pygame.Rect(self.position, self.size)
            self.shield_rect = pygame.Rect(self.position - (self.shield_size - self.size) // 2, self.shield_size)

    def draw(self, screen):

        screen.blit(self.pic, self.position.astype(np.int16))
        if self.shield:
            screen.blit(self.shield_pic, self.shield_rect)

    def shoot(self, time):

        if time > self.last_shot_time + self.shoot_freq and (not self.space_down or self.auto_fire) and self.status in (0, 1):
            self.last_shot_time = time
            self.space_down = True
            # triple fire (3) combines single (1) and double (2)
            if self.bullet_type in (1, 3):
                self.bullets.append(Bullet(self.bullet_pic, self.position + np.array([(self.size[0] - self.bullet_size[0]) / 2, -self.bullet_size[1]]),
                                           np.array([0.0, -6.5]), np.array([0.0, 0.0])))
            if self.bullet_type in (2, 3):
                # Double Fire powerup
                self.bullets.append(Bullet(self.bullet_pic_left, self.position + np.array([self.size[0] / 2 - self.bullet_left_size[0], -self.bullet_left_size[1]]),
                                           np.array([np.cos((90.0 + self.double_fire_angle) * np.pi / 180) * 6.5,
                                                     np.sin((90.0 + self.double_fire_angle) * np.pi / 180) * -6.5]), np.array([0.0, 0.0])))
                self.bullets.append(Bullet(self.bullet_pic_right, self.position + np.array([self.size[0] / 2, -self.bullet_right_size[1]]),
                                           np.array([np.cos((90.0 - self.double_fire_angle) * np.pi / 180) * 6.5,
                                                     np.sin((90.0 - self.double_fire_angle) * np.pi / 180) * -6.5]), np.array([0.0, 0.0])))
            return True  # shot fired
        else:
            return False  # shot not fired

    def is_hit(self, time, cheat_mode=0):

        # ship is hit
        if cheat_mode in (0, 1):
            self.status = 2  # ship marked dead
            self.start_time = time
            self.lives -= 1

    def add_powerup(self, new_powerup, ship_shoot_freq):

        # add a new powerup to the ship, or if already has it, extend its life
        for powerup in self.powerups:
            if powerup[0] == new_powerup.pid:
                # existing powerup - add to life_time only
                powerup[1] += new_powerup.life_time
                break
        else:
            # new powerup
            self.powerups.append(np.array([new_powerup.pid, new_powerup.life_time], dtype=np.int32))
            if new_powerup.desc == 'Rapid Fire':
                self.shoot_freq = ship_shoot_freq // 2
            elif new_powerup.desc == 'Auto Fire':
                self.auto_fire = True
            elif new_powerup.desc == 'Shield':
                self.shield = True
            elif new_powerup.desc[-4:] == 'Fire':
                self.fire_powerup(new_powerup.desc)

    def fire_powerup(self, desc):

        if desc == 'Double Fire' and self.bullet_type < 2:
            self.bullet_type = 2
        elif desc == 'Triple Fire'and self.bullet_type < 3:
            self.bullet_type = 3

    def end_powerup(self, powerup, powerup_data, ship_shoot_freq):

        # remove powerup and its effect
        desc = powerup_data[powerup[0]][2]
        if desc == 'Rapid Fire':
            self.shoot_freq = ship_shoot_freq
        elif desc == 'Auto Fire':
            self.auto_fire = False
        elif desc == 'Shield':
            self.shield = False
        elif desc[-4:] == 'Fire':
            # extra bullets end - check if any other such powerup still active
            self.bullet_type = 1
            for pup in self.powerups:
                if powerup_data[pup[0]][2] != desc and powerup_data[pup[0]][2][-4:] == 'Fire':
                    self.fire_powerup(powerup_data[pup[0]][2])

        # mark for removal
        powerup[0] = -1


class Bullet:
    """
    Bullet
    """

    def __init__(self, pic, start_pos, move_vec, accelerate_vec):

        self.pic = pic
        self.mask = pygame.mask.from_surface(pic)
        self.position = start_pos
        self.move_vec = move_vec
        self.accelerate_vec = accelerate_vec
        self.size = self.pic.get_size()
        # self.size = np.abs(np.array([2, move_vec[1]])).astype(np.int16)
        self.rect = pygame.Rect(self.position, self.size)
        # self.mask = pygame.mask.Mask(self.size, True)

    def move(self):

        self.position += self.move_vec
        self.move_vec += self.accelerate_vec
        self.rect = pygame.Rect(self.position, self.size)

    def draw(self, screen):

        screen.blit(self.pic, self.position.astype(np.int16))


class Alien:
    """
    Alien
    """

    def __init__(self, pic, hit_area, bullet_pic, cannon_pos, sound_explosion, score, hit_nr,
                 start_pos, x_move, y_move, x_times, move_delay, start_delay, start_dir, shoot_freq):

        self.pic = pic
        self.mask = pygame.mask.from_surface(pic)
        self.hit_area = hit_area  # hit area is a Rect (within mask) where hit causes damage. If None then not used (all hits cause damage)
        self.bullet_pic = bullet_pic
        self.cannon_pos = cannon_pos
        self.sound_explosion = sound_explosion
        self.score = score
        self.hit_nr = hit_nr
        self.hit_total = hit_nr
        self.size = self.pic.get_size()
        self.life_bar_size = np.array([0.8 * self.size[0], 0.04 * self.size[0] + 1]).astype(np.int16)
        self.position = start_pos
        self.x_move = x_move
        self.y_move = y_move
        self.boss_move = np.random.randint(move_delay * 1.5, move_delay * 4, 4)
        self.x_times = x_times
        self.move_delay = move_delay  # set move_delay == 0 for "boss movement"
        self.direction = start_dir
        self.last_move = pygame.time.get_ticks() + start_delay
        self.x_move_cnt = 0
        self.shoot_freq = shoot_freq
        self.rect = pygame.Rect(self.position, self.size)

    def move(self, time, level_time, screen_size):

        if self.x_times == 0:
            # boss - move using sine/cosine
            s = np.sin((time - level_time) / self.boss_move)
            self.position = np.array([screen_size[0] * (0.5 + (s[0] + s[1]) * (screen_size[0] - self.size[0] * 1.1)
                                                        / screen_size[0] / 4.0) - (self.size[0] / 2),
                                     screen_size[1] * (0.3 + (s[2] + s[3]) * (screen_size[1] / (0.5 / 0.3) - self.size[1] * 1.1)
                                                       / screen_size[1] / 4.0) - (self.size[1] / 2)], dtype=float)
            self.rect = pygame.Rect(self.position, self.size)

        elif time >= self.last_move + self.move_delay:
            if self.x_move_cnt == self.x_times:
                self.position[1] += self.y_move
                self.x_move_cnt = 0
                self.direction = -self.direction
            else:
                self.position[0] += self.x_move * self.direction
                self.x_move_cnt += 1
            self.last_move += self.move_delay
            self.rect = pygame.Rect(self.position, self.size)

    def draw(self, screen):

        screen.blit(self.pic, self.position.astype(np.int16))
        if self.hit_total > 1:
            # life bar for boss aliens
            pos = self.position + np.array([(self.size[0] - self.life_bar_size[0]) // 2, -self.life_bar_size[1] - 5])
            pygame.draw.rect(screen, (255, 255, 255), (pos - 1, self.life_bar_size + 2), 1)
            pygame.draw.rect(screen, (200, 0, 0), (pos, (self.life_bar_size[0] * self.hit_nr // self.hit_total, self.life_bar_size[1])), 0)


class Ufo:
    """
    Alien ufo
    Contrary to a regular Alien, the ufo has a distance (z coordinate) and can only be shot when at a suitable distance.
    """

    def __init__(self, pic, bullet_pic, sound_explosion, score, speed, from_side, screen_size):

        self.orig_size = np.array(pic.get_size())
        self.z_pos_start = self.orig_size[0] / 50  # starting z position
        self.size = (self.orig_size / self.z_pos_start).astype(np.int16)  # define start size in pixels (80) based on ufo pic width
        # position 3D is relative to screen center
        self.position_3D = np.array([
            from_side * (screen_size[0] + self.size[0]) / 2,  # x start is just outside of screen (either side)
            -screen_size[1] / 2 * (random.random() * 0.2 + 0.05),  # y start is between 0.25 and 0.45 times screen height
            1.0]) * self.z_pos_start
        self.position = (self.position_3D[0:2] / self.position_3D[2] + screen_size / 2)
        self.orig_pic = pic
        self.pic = pygame.transform.scale(self.orig_pic, self.size)
        self.mask = pygame.mask.from_surface(self.pic)
        self.bullet_pic = bullet_pic
        self.bullet_size = np.array(bullet_pic.get_size())
        self.sound_explosion = sound_explosion
        self.score = score
        self.speed = speed
        self.from_side = from_side  # -1 is left, +1 is right
        self.last_move = pygame.time.get_ticks()
        self.phase = 0  # phase 0 (start) to turning_point to turning_point + 75 to out of screen
        self.turning_point = random.random() * 50 + 1  # turning point between 10 and 60 % of screen width
        self.rect = pygame.Rect(self.position, self.size)
        self.may_be_shot = False

    def move(self, time, freeze_speed, screen_size):

        speed_adj = self.speed * freeze_speed * (time - self.last_move) / 20  # adjust speed with time
        self.phase += speed_adj
        self.last_move = time
        if self.phase < self.turning_point:
            # ufo entering in the distance, move sideways
            self.position_3D[0] -= self.from_side * speed_adj * screen_size[0] * self.z_pos_start / 100.0
        elif self.phase < self.turning_point + 75:
            # ufo making a turn and coming closer
            cx = np.cos((self.phase - self.turning_point) * np.pi / 75)  # cx ranges from 1.0 to 0.0 to -1.0
            sz = np.sin((self.phase - self.turning_point) * np.pi / 75)  # sz ranges from 0.0 to 1.0 to 0.0
            self.position_3D[0] -= self.from_side * cx * speed_adj * screen_size[0] * self.z_pos_start / 100.0
            self.position_3D[2] -= sz * 1.3 * self.z_pos_start * speed_adj / 75
            # adjust size and mask as ufo now closer = bigger
            self.size = (self.orig_size / self.position_3D[2]).astype(np.int16)
            if np.min(self.size) > 0:
                self.pic = pygame.transform.scale(self.orig_pic, self.size)
            self.mask = pygame.mask.from_surface(self.pic)
            if self.phase > self.turning_point + 75 / 4 and self.phase < self.turning_point + 75 * 3 / 4:
                # mark ufo as vulnerable when close to turn mid point - may be shot down
                self.may_be_shot = True
        else:
            # after turning, ufo going back to where it came from, Cannot be shot any more.
            self.may_be_shot = False
            self.position_3D[0] += self.from_side * speed_adj * screen_size[0] * self.z_pos_start / 100.0

        # convert 3D position to 2D
        self.position = (self.position_3D[0:2] / self.position_3D[2] + screen_size / 2)
        self.rect = pygame.Rect(self.position, self.size)

    def draw(self, screen):

        screen.blit(self.pic, self.position.astype(np.int16))


class Score:
    """
    score
    """

    def __init__(self, space_inv, score_amount, score_text, font, start_pos, color, background_color):

        space_inv.game_score += score_amount
        self.position = np.array(start_pos).astype(float)
        self.pic = font.render(score_text, True, color)
        self.pic.set_colorkey(background_color)
        self.show_time = 2000
        self.start_time = pygame.time.get_ticks()

    def draw(self, screen, time):

        self.pic.set_alpha(35 + 220 * (1.0 - (time - self.start_time) / self.show_time))
        screen.blit(self.pic, self.position.astype(np.int16))

    def move(self):

        self.position += np.array([0.2, -0.1])


class Explosion:
    """
    explosion
    """

    def __init__(self, pic, mid_position, grid, freq):

        self.pic = pic
        self.size = self.pic.get_size()
        self.grid = grid
        self.grid_size = (self.size[0] / self.grid[0], self.size[1] / self.grid[1])
        self.position = (int(mid_position[0] - self.grid_size[0] / 2), int(mid_position[1] - self.grid_size[1] / 2))
        self.freq = freq
        self.freq_cnt = 0
        self.phase = np.array([0, 0])

    def draw(self, screen):

        rect = (int(self.phase[0] * self.grid_size[0]), int(self.phase[1] * self.grid_size[1]), int(self.grid_size[0]), int(self.grid_size[1]))
        screen.blit(self.pic, self.position, rect)


class PowerUp:
    """
    power up
    """

    def __init__(self, pid, powerup_data, sound_award, mid_position):

        self.pid = pid
        self.pic = powerup_data[pid][0]
        self.life_time = powerup_data[pid][1]
        self.desc = powerup_data[pid][2]
        self.size = self.pic.get_size()
        self.mask = pygame.mask.from_surface(self.pic)
        self.position = (int(mid_position[0] - self.size[0] / 2), int(mid_position[1] - self.size[1] / 2))
        self.sound_award = sound_award
        self.show_time = 7000
        self.start_time = pygame.time.get_ticks()
        self.rect = pygame.Rect(self.position, self.size)

    def draw(self, screen, time):

        screen.blit(self.pic, self.position)


if __name__ == '__main__':
    """
    Prepare screen, etc.
    """

    pygame.display.init()
    desktops = pygame.display.get_desktop_sizes()
    # define display/window height based on (the first) desktop size
    if desktops[0][1] > 1080:
        disp_size = (960, 1140)
    else:
        disp_size = (960, 960)
    pygame.display.set_caption('Space Invaders')
    screen = pygame.display.set_mode(disp_size)
    SpaceInvaders(screen).run()

    # exit; close everything
    pygame.quit()
    exit()

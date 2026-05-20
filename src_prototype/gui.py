import pygame
import pygame.freetype
import numpy as np
import matplotlib.cm as cm

viridis = cm.get_cmap("viridis")

class Window:
    def __init__(self):
        self.GRID_SIZE = 28
        self.PIXEL_SIZE = 20
        self.SIDEBAR_WIDTH = 500
        self.WINDOW_SIZE = (self.GRID_SIZE * self.PIXEL_SIZE + self.SIDEBAR_WIDTH, self.GRID_SIZE * self.PIXEL_SIZE)

        self.brush_size = 3
        self.brush_strength = 0.001

        self.mouse_prev = (0, 0)

        pygame.init()
        pygame.freetype.init()
        self.font = pygame.freetype.Font("CascadiaMono.ttf", 20)
        self.screen = pygame.display.set_mode(self.WINDOW_SIZE)
        pygame.display.set_caption("Neural Network")

        self.running = True
        self.drawing = False

        self.prediction = np.zeros(10)

        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE))

    def get_grid(self):
        return self.grid

    def draw_text(self, x, y, text):
        self.font.render_to(self.screen, (x, y), text, (255, 255, 255))

    def draw_grid(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                value = self.grid[y, x]  # float 0–1
                r, g, b, _ = viridis(value)  # returns RGBA (0–1)
                color = (int(r * 255), int(g * 255), int(b * 255))
                rect = pygame.Rect(x * self.PIXEL_SIZE, y * self.PIXEL_SIZE, self.PIXEL_SIZE, self.PIXEL_SIZE)
                pygame.draw.rect(self.screen, color, rect)

    def draw_predictions(self):
        gap = 10
        width = self.SIDEBAR_WIDTH / 10 - gap * 1.1
        height = 150
        left = self.GRID_SIZE * self.PIXEL_SIZE + gap
        for i in range(10):
            color = (100, 150, 200)
            value = float(self.prediction[i]) * height
            rect = pygame.Rect((left + i * (width + gap), 300 - height), (width, height))
            pygame.draw.rect(self.screen, (70, 70, 70), rect)
            rect = pygame.Rect((left + i * (width + gap), 300 - value), (width, value+1))
            pygame.draw.rect(self.screen, color, rect)
            self.draw_text(left + i * (width + gap) + 10, 310, str(i))

        self.draw_text(self.WINDOW_SIZE[0] / 3 * 2 - 40, self.WINDOW_SIZE[1] * 3/4, "[Click to clear screen]")

    def draw(self, x, y):
        gx = x // self.PIXEL_SIZE
        gy = y // self.PIXEL_SIZE
        for dy in range(-self.brush_size, self.brush_size + 1):
            for dx in range(-self.brush_size, self.brush_size + 1):
                if 0 <= gx + dx < self.GRID_SIZE and 0 <= gy + dy < self.GRID_SIZE:
                    strength = (dx**2 + dy**2)**0.5
                    max_strength = (2*self.brush_size**2)**0.5
                    self.grid[gy+dy, gx+dx] = min(1.0, self.grid[gy+dy, gx+dx] + (max_strength - strength) ** 4 * self.brush_strength)

    def update(self):
        mx, my = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if mx > self.GRID_SIZE * self.PIXEL_SIZE:
                    self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE))
                self.drawing = True
            if event.type == pygame.MOUSEBUTTONUP:
                self.drawing = False

        if self.drawing:
            self.draw((self.mouse_prev[0] + mx) // 2, (self.mouse_prev[1] + my) // 2)
            self.draw(mx, my)

        self.mouse_prev = (mx, my)

        self.draw_grid()
        self.draw_predictions()

        pygame.display.flip()
        self.screen.fill((0, 0, 0))

    @staticmethod
    def quit():
        pygame.quit()
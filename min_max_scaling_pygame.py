import pygame
import random

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1600, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Min Max Scaling Demo")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Fonts
title_font = pygame.font.Font(None, 64)
text_font = pygame.font.Font(None, 32)

# Data points
num_points = 10
original_data = [random.randint(0, 100) for _ in range(num_points)]
scaled_data = []

# Min-Max scaling function
def min_max_scale(data, new_min=0, new_max=1):
    old_min, old_max = min(data), max(data)
    return [(x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min for x in data]

# Button class
class Button:
    def __init__(self, x, y, width, height, text, color, text_color, action):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.text_color = text_color
        self.action = action

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)
        text_surface = text_font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.action()

# Create buttons
regenerate_button = Button(50, 700, 200, 50, "Regenerate Data", GREEN, BLACK, lambda: regenerate_data())
scale_button = Button(300, 700, 200, 50, "Apply Min-Max Scaling", BLUE, WHITE, lambda: apply_scaling())

def regenerate_data():
    global original_data, scaled_data
    original_data = [random.randint(0, 100) for _ in range(num_points)]
    scaled_data = []

def apply_scaling():
    global scaled_data
    scaled_data = min_max_scale(original_data, 0, 100)

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        regenerate_button.handle_event(event)
        scale_button.handle_event(event)

    screen.fill(WHITE)

    # Draw title
    title_surface = title_font.render("Min Max Scaling Demo", True, BLACK)
    screen.blit(title_surface, (WIDTH // 2 - title_surface.get_width() // 2, 20))

    # Draw developer name
    dev_surface = text_font.render("Developed by: Venugopal Adep", True, BLACK)
    screen.blit(dev_surface, (WIDTH // 2 - dev_surface.get_width() // 2, 80))

    # Draw original data
    for i, value in enumerate(original_data):
        x = 100 + i * 150
        y = 600 - value * 4
        pygame.draw.rect(screen, RED, (x, y, 50, value * 4))
        text_surface = text_font.render(str(value), True, BLACK)
        screen.blit(text_surface, (x, 610))

    # Draw scaled data
    if scaled_data:
        for i, value in enumerate(scaled_data):
            x = 100 + i * 150
            y = 300 - value * 4
            pygame.draw.rect(screen, BLUE, (x, y, 50, value * 4))
            text_surface = text_font.render(f"{value:.2f}", True, BLACK)
            screen.blit(text_surface, (x, 310))

    # Draw buttons
    regenerate_button.draw(screen)
    scale_button.draw(screen)

    # Draw labels
    original_label = text_font.render("Original Data", True, BLACK)
    screen.blit(original_label, (50, 550))
    scaled_label = text_font.render("Scaled Data", True, BLACK)
    screen.blit(scaled_label, (50, 250))

    pygame.display.flip()

pygame.quit()
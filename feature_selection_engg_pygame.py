import pygame
import random
import math

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1600, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Feature Selection and Engineering")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
LIGHT_GRAY = (200, 200, 200)

# Fonts
title_font = pygame.font.Font(None, 64)
subtitle_font = pygame.font.Font(None, 32)
text_font = pygame.font.Font(None, 24)

# Data points
class DataPoint:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.selected = False

data_points = [DataPoint(random.randint(50, WIDTH-50), random.randint(200, HEIGHT-50), random.choice([RED, BLUE])) for _ in range(50)]

# Feature selection rectangle
selection_rect = pygame.Rect(0, 0, 0, 0)
selecting = False

# Button class
class Button:
    def __init__(self, x, y, width, height, text, color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = tuple(max(0, c - 50) for c in color)

    def draw(self):
        color = self.hover_color if self.is_hovered() else self.color
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)  # Add border
        text_surface = text_font.render(self.text, True, BLACK)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def is_hovered(self):
        return self.rect.collidepoint(pygame.mouse.get_pos())

# Buttons
engineer_button = Button(WIDTH - 220, 50, 200, 50, "Engineer Features", YELLOW)
reset_button = Button(WIDTH - 220, 120, 200, 50, "Reset", YELLOW)

# Main game loop
running = True
clock = pygame.time.Clock()

def engineer_features():
    red_points = [p for p in data_points if p.selected and p.color == RED]
    blue_points = [p for p in data_points if p.selected and p.color == BLUE]
    
    if red_points:
        red_centroid = (sum(p.x for p in red_points) / len(red_points),
                        sum(p.y for p in red_points) / len(red_points))
        for p in red_points:
            p.x += (red_centroid[0] - p.x) * 0.1
            p.y += (red_centroid[1] - p.y) * 0.1
    
    if blue_points:
        blue_centroid = (sum(p.x for p in blue_points) / len(blue_points),
                         sum(p.y for p in blue_points) / len(blue_points))
        for p in blue_points:
            p.x += (blue_centroid[0] - p.x) * 0.1
            p.y += (blue_centroid[1] - p.y) * 0.1

def reset_points():
    global data_points
    data_points = [DataPoint(random.randint(50, WIDTH-50), random.randint(200, HEIGHT-50), random.choice([RED, BLUE])) for _ in range(50)]

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                if engineer_button.rect.collidepoint(event.pos):
                    engineer_features()
                elif reset_button.rect.collidepoint(event.pos):
                    reset_points()
                else:
                    selecting = True
                    selection_rect.topleft = event.pos
                    selection_rect.size = (0, 0)
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left mouse button
                selecting = False
                # Select points within the rectangle
                for point in data_points:
                    if selection_rect.collidepoint(point.x, point.y):
                        point.selected = not point.selected  # Toggle selection
        elif event.type == pygame.MOUSEMOTION:
            if selecting:
                selection_rect.width = event.pos[0] - selection_rect.x
                selection_rect.height = event.pos[1] - selection_rect.y

    # Clear the screen
    screen.fill(WHITE)

    # Draw title and subtitle
    title_surface = title_font.render("Feature Selection and Engineering", True, BLACK)
    subtitle_surface = subtitle_font.render("Developed by: Venugopal Adep", True, BLACK)
    screen.blit(title_surface, (20, 20))
    screen.blit(subtitle_surface, (20, 100))

    # Draw data points
    for point in data_points:
        pygame.draw.circle(screen, point.color, (int(point.x), int(point.y)), 5)
        if point.selected:
            pygame.draw.circle(screen, GREEN, (int(point.x), int(point.y)), 8, 2)

    # Draw selection rectangle
    if selecting:
        pygame.draw.rect(screen, GREEN, selection_rect, 2)

    # Draw buttons
    engineer_button.draw()
    reset_button.draw()

    # Draw instructions
    instructions = [
        "Instructions:",
        "1. Click and drag to select data points",
        "2. Click 'Engineer Features' to move selected points",
        "3. Click 'Reset' to generate new data points",
        "4. Observe how feature engineering affects data distribution"
    ]
    for i, instruction in enumerate(instructions):
        instruction_surface = text_font.render(instruction, True, BLACK)
        screen.blit(instruction_surface, (20, HEIGHT - 150 + i * 30))

    # Update the display
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
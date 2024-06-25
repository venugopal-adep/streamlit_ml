import pygame
import random
import math

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1600, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Feature Selection")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (200, 200, 200)

# Fonts
title_font = pygame.font.Font(None, 64)
subtitle_font = pygame.font.Font(None, 32)
text_font = pygame.font.Font(None, 24)

# Data point class
class DataPoint:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.visible = True

# Generate random data points
data_points = [DataPoint(random.randint(50, WIDTH-250), random.randint(200, HEIGHT-50), 
                         random.choice([RED, BLUE])) for _ in range(100)]

# Feature toggles
feature_x = True
feature_y = True

# Button class
class Button:
    def __init__(self, x, y, width, height, text, color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.active = True

    def draw(self):
        color = self.color if self.active else GRAY
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)
        text_surface = text_font.render(self.text, True, BLACK)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

# Buttons
toggle_x_button = Button(WIDTH - 220, 50, 200, 50, "Toggle X Feature", YELLOW)
toggle_y_button = Button(WIDTH - 220, 120, 200, 50, "Toggle Y Feature", YELLOW)
reset_button = Button(WIDTH - 220, 190, 200, 50, "Reset", GREEN)

# Function to calculate separability score
def calculate_separability():
    red_points = [p for p in data_points if p.color == RED and p.visible]
    blue_points = [p for p in data_points if p.color == BLUE and p.visible]
    
    if not red_points or not blue_points:
        return 0

    red_center = (sum(p.x for p in red_points) / len(red_points), 
                  sum(p.y for p in red_points) / len(red_points))
    blue_center = (sum(p.x for p in blue_points) / len(blue_points), 
                   sum(p.y for p in blue_points) / len(blue_points))

    center_distance = math.sqrt((red_center[0] - blue_center[0])**2 + 
                                (red_center[1] - blue_center[1])**2)

    red_spread = sum(math.sqrt((p.x - red_center[0])**2 + (p.y - red_center[1])**2) 
                     for p in red_points) / len(red_points)
    blue_spread = sum(math.sqrt((p.x - blue_center[0])**2 + (p.y - blue_center[1])**2) 
                      for p in blue_points) / len(blue_points)

    return center_distance / (red_spread + blue_spread)

# Main game loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if toggle_x_button.rect.collidepoint(event.pos):
                feature_x = not feature_x
                toggle_x_button.active = feature_x
                for point in data_points:
                    point.visible = (feature_x or not feature_y) and (feature_y or not feature_x)
            elif toggle_y_button.rect.collidepoint(event.pos):
                feature_y = not feature_y
                toggle_y_button.active = feature_y
                for point in data_points:
                    point.visible = (feature_x or not feature_y) and (feature_y or not feature_x)
            elif reset_button.rect.collidepoint(event.pos):
                feature_x = True
                feature_y = True
                toggle_x_button.active = True
                toggle_y_button.active = True
                for point in data_points:
                    point.visible = True

    # Clear the screen
    screen.fill(WHITE)

    # Draw title and subtitle
    title_surface = title_font.render("Feature Selection", True, BLACK)
    subtitle_surface = subtitle_font.render("Developed by: Venugopal Adep", True, BLACK)
    screen.blit(title_surface, (20, 20))
    screen.blit(subtitle_surface, (20, 100))

    # Draw data points
    for point in data_points:
        if point.visible:
            x = point.x if feature_x else WIDTH // 2
            y = point.y if feature_y else HEIGHT // 2
            pygame.draw.circle(screen, point.color, (x, y), 5)

    # Draw buttons
    toggle_x_button.draw()
    toggle_y_button.draw()
    reset_button.draw()

    # Calculate and display separability score
    score = calculate_separability()
    score_text = f"Separability Score: {score:.2f}"
    score_surface = text_font.render(score_text, True, BLACK)
    screen.blit(score_surface, (WIDTH - 250, HEIGHT - 50))

    # Draw instructions
    instructions = [
        "Instructions:",
        "1. Toggle X/Y features to see their impact",
        "2. Observe how feature selection affects class separation",
        "3. Higher separability score indicates better feature selection",
        "4. Reset to return to the original state"
    ]
    for i, instruction in enumerate(instructions):
        instruction_surface = text_font.render(instruction, True, BLACK)
        screen.blit(instruction_surface, (20, HEIGHT - 150 + i * 30))

    # Update the display
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
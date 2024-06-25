import pygame
import numpy as np
import pygame.gfxdraw

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1600, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Standard Scaling Demo")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# Fonts
title_font = pygame.font.Font(None, 48)
text_font = pygame.font.Font(None, 24)

# Data points
data = np.random.randint(1, 100, 20)

# Function to perform standard scaling
def standard_scale(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

# Function to draw a scatter plot
def draw_scatter_plot(data, x, y, width, height, color):
    for i, value in enumerate(data):
        pygame.draw.circle(screen, color, (int(x + i * width / len(data)), int(y + height - value * height / 100)), 5)

# Main game loop
running = True
scaled_data = standard_scale(data)
show_scaled = False
animate = False
animation_progress = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                show_scaled = not show_scaled
                animate = True
                animation_progress = 0

    screen.fill(WHITE)

    # Draw title
    title_text = title_font.render("Standard Scaling Demo", True, BLACK)
    screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, 20))

    # Draw developer credit
    dev_text = text_font.render("Developed by: Venugopal Adep", True, BLACK)
    screen.blit(dev_text, (WIDTH // 2 - dev_text.get_width() // 2, 70))

    # Draw original data plot
    pygame.draw.rect(screen, BLACK, (100, 150, 600, 400), 2)
    draw_scatter_plot(data, 100, 150, 600, 400, BLUE)
    orig_text = text_font.render("Original Data", True, BLACK)
    screen.blit(orig_text, (350 - orig_text.get_width() // 2, 570))

    # Draw scaled data plot
    pygame.draw.rect(screen, BLACK, (900, 150, 600, 400), 2)
    if show_scaled:
        draw_scatter_plot(scaled_data, 900, 150, 600, 400, RED)
    scaled_text = text_font.render("Scaled Data", True, BLACK)
    screen.blit(scaled_text, (1150 - scaled_text.get_width() // 2, 570))

    # Draw arrows and scaling explanation
    if animate:
        progress = min(1, animation_progress / 60)
        pygame.draw.line(screen, GREEN, (700, 350), (700 + 200 * progress, 350), 3)
        pygame.draw.polygon(screen, GREEN, [(900, 340), (900, 360), (920, 350)])
        
        explanation = [
            "Standard Scaling:",
            "1. Calculate mean (μ) and standard deviation (σ) of the data",
            "2. For each data point x, calculate: (x - μ) / σ",
            "3. Result: Mean = 0, Standard Deviation = 1"
        ]
        
        for i, line in enumerate(explanation):
            text = text_font.render(line, True, BLACK)
            screen.blit(text, (WIDTH // 2 - text.get_width() // 2, 600 + i * 30))
        
        animation_progress += 1
        if animation_progress >= 60:
            animate = False

    # Draw instructions
    instructions = text_font.render("Click anywhere to toggle between original and scaled data", True, BLACK)
    screen.blit(instructions, (WIDTH // 2 - instructions.get_width() // 2, HEIGHT - 50))

    pygame.display.flip()

pygame.quit()
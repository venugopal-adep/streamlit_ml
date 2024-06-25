import pygame
import sys
import random
import math
import numpy as np

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1600, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Bias-Variance Tradeoff")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GRAY = (200, 200, 200)

# Fonts
title_font = pygame.font.Font(None, 64)
text_font = pygame.font.Font(None, 32)
small_font = pygame.font.Font(None, 24)

# Data points
data_points = []
model_complexity = 1
max_complexity = 30
num_samples = 50

def generate_data():
    global data_points
    pattern = random.choice(['sine', 'quadratic', 'exponential', 'linear'])
    data_points = []
    for x in range(0, WIDTH, WIDTH // num_samples):
        if pattern == 'sine':
            y = 100 * math.sin(x / 100) + random.gauss(0, 20)
        elif pattern == 'quadratic':
            y = 0.001 * (x - WIDTH/2)**2 - 100 + random.gauss(0, 20)
        elif pattern == 'exponential':
            y = 200 * math.exp(-0.005 * x) - 50 + random.gauss(0, 20)
        else:  # linear
            y = 0.2 * x - 100 + random.gauss(0, 20)
        data_points.append((x, y))

def draw_text(text, font, color, x, y):
    surface = font.render(text, True, color)
    screen.blit(surface, (x, y))

def fit_model(complexity):
    x = np.array([p[0] for p in data_points])
    y = np.array([p[1] for p in data_points])
    coeffs = np.polyfit(x, y, complexity)
    return np.poly1d(coeffs)

def draw_curve(model, color):
    x = np.array(range(0, WIDTH, 5))
    y = model(x)
    points = [(int(x[i]), int(y[i] + HEIGHT // 2)) for i in range(len(x))]
    pygame.draw.lines(screen, color, False, points, 2)

def calculate_metrics(model):
    x = np.array([p[0] for p in data_points])
    y = np.array([p[1] for p in data_points])
    predictions = model(x)
    mse = np.mean((y - predictions)**2)
    bias = np.mean(np.abs(y - predictions))
    variance = np.var(predictions)
    return mse, bias, variance

# Generate initial data
generate_data()

# Main game loop
clock = pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                model_complexity = min(max_complexity, model_complexity + 1)
            elif event.key == pygame.K_DOWN:
                model_complexity = max(1, model_complexity - 1)
            elif event.key == pygame.K_r:
                generate_data()

    screen.fill(WHITE)

    # Draw title and developer info
    draw_text("Bias-Variance Tradeoff", title_font, BLACK, 20, 20)
    draw_text("Developed by: Venugopal Adep", text_font, BLACK, 20, 80)

    # Draw instructions
    draw_text("UP/DOWN: Adjust model complexity", text_font, BLACK, 20, HEIGHT - 90)
    draw_text("R: Regenerate data with new pattern", text_font, BLACK, 20, HEIGHT - 60)

    # Draw data points
    for point in data_points:
        pygame.draw.circle(screen, BLUE, (int(point[0]), int(point[1] + HEIGHT // 2)), 3)

    # Fit and draw model
    model = fit_model(model_complexity)
    draw_curve(model, RED)

    # Calculate and display metrics
    mse, bias, variance = calculate_metrics(model)
    draw_text(f"Model Complexity: {model_complexity}", text_font, BLACK, WIDTH - 400, 20)
    draw_text(f"MSE: {mse:.2f}", text_font, BLACK, WIDTH - 400, 60)
    draw_text(f"Bias: {bias:.2f}", text_font, BLACK, WIDTH - 400, 100)
    draw_text(f"Variance: {variance:.2f}", text_font, BLACK, WIDTH - 400, 140)

    # Draw legend
    draw_text("Data Points", small_font, BLUE, WIDTH - 150, HEIGHT - 90)
    draw_text("Fitted Model", small_font, RED, WIDTH - 150, HEIGHT - 60)

    pygame.display.flip()
    clock.tick(60)
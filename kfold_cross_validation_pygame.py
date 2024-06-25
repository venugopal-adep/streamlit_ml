import pygame
import random
import math

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1600, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("K-fold Cross-Validation")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
COLORS = [RED, GREEN, BLUE, YELLOW]

# Fonts
title_font = pygame.font.Font(None, 64)
subtitle_font = pygame.font.Font(None, 32)
text_font = pygame.font.Font(None, 24)

# K-fold parameters
K = 5
data_points = 100
fold_size = data_points // K

# Generate random data points
data = [(random.randint(50, WIDTH-50), random.randint(300, HEIGHT-50)) for _ in range(data_points)]

# Current fold and animation state
current_fold = 0
animating = False
animation_progress = 0
animation_speed = 0.02

def draw_title():
    title = title_font.render("K-fold Cross-Validation", True, BLACK)
    screen.blit(title, (WIDTH//2 - title.get_width()//2, 20))
    
    subtitle = subtitle_font.render("Developed by: Venugopal Adep", True, BLACK)
    screen.blit(subtitle, (WIDTH//2 - subtitle.get_width()//2, 80))

def draw_folds():
    for i, point in enumerate(data):
        fold_index = i // fold_size
        color = COLORS[fold_index % len(COLORS)]
        pygame.draw.circle(screen, color, point, 5)

def draw_validation_set():
    start = current_fold * fold_size
    end = start + fold_size
    for i in range(start, end):
        point = data[i]
        pygame.draw.circle(screen, BLACK, point, 7, 2)

def draw_train_test_split():
    validation_start = current_fold * fold_size
    validation_end = validation_start + fold_size
    
    # Draw training set
    train_text = text_font.render("Training Set", True, BLACK)
    screen.blit(train_text, (50, 150))
    for i, point in enumerate(data):
        if i < validation_start or i >= validation_end:
            x = 50 + (i % 10) * 30
            y = 180 + (i // 10) * 30
            color = COLORS[(i // fold_size) % len(COLORS)]
            pygame.draw.circle(screen, color, (x, y), 5)
    
    # Draw validation set
    test_text = text_font.render("Validation Set", True, BLACK)
    screen.blit(test_text, (WIDTH - 350, 150))
    for i in range(validation_start, validation_end):
        x = WIDTH - 350 + ((i - validation_start) % 10) * 30
        y = 180 + ((i - validation_start) // 10) * 30
        color = COLORS[(i // fold_size) % len(COLORS)]
        pygame.draw.circle(screen, color, (x, y), 5)

def draw_info():
    info_text = [
        f"K = {K}",
        f"Total data points: {data_points}",
        f"Points per fold: {fold_size}",
        f"Current fold: {current_fold + 1}",
        "",
        "Press SPACE to animate",
        "Press LEFT/RIGHT to change fold"
    ]
    
    for i, text in enumerate(info_text):
        rendered_text = text_font.render(text, True, BLACK)
        screen.blit(rendered_text, (50, HEIGHT - 180 + i * 25))

def animate_fold_change():
    global animation_progress, animating, current_fold
    
    if animation_progress < 1:
        animation_progress += animation_speed
        
        for i in range(fold_size):
            start_index = current_fold * fold_size + i
            end_index = ((current_fold + 1) % K) * fold_size + i
            
            start_point = data[start_index]
            end_point = data[end_index]
            
            x = start_point[0] + (end_point[0] - start_point[0]) * animation_progress
            y = start_point[1] + (end_point[1] - start_point[1]) * animation_progress
            
            color = COLORS[current_fold % len(COLORS)]
            pygame.draw.circle(screen, color, (int(x), int(y)), 5)
    else:
        animating = False
        animation_progress = 0
        current_fold = (current_fold + 1) % K

# Main game loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                animating = True
            elif event.key == pygame.K_LEFT:
                current_fold = (current_fold - 1) % K
            elif event.key == pygame.K_RIGHT:
                current_fold = (current_fold + 1) % K

    screen.fill(WHITE)
    
    draw_title()
    draw_folds()
    draw_validation_set()
    draw_train_test_split()
    draw_info()
    
    if animating:
        animate_fold_change()
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
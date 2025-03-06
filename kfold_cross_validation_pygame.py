import pygame
import random
import math
import colorsys

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1600, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("K-fold Cross-Validation Visualization")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)

# Generate a nice color palette
def generate_colors(n):
    colors = []
    for i in range(n):
        h = i / n
        s = 0.8
        v = 0.9
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors

# Fonts
title_font = pygame.font.Font(None, 72)
subtitle_font = pygame.font.Font(None, 36)
text_font = pygame.font.Font(None, 28)
small_font = pygame.font.Font(None, 24)

# K-fold parameters
K = 5
data_points = 100
fold_size = data_points // K

# Generate better distributed random data points
def generate_data():
    points = []
    for i in range(data_points):
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(100, 300)
        x = WIDTH // 2 + int(radius * math.cos(angle))
        y = HEIGHT // 2 + 50 + int(radius * math.sin(angle))
        points.append((x, y))
    return points

data = generate_data()
COLORS = generate_colors(K)

# Current fold and animation state
current_fold = 0
animating = False
animation_progress = 0
animation_speed = 0.02
auto_animate = False
auto_animate_timer = 0

def draw_title():
    # Draw background for title
    pygame.draw.rect(screen, GRAY, (0, 0, WIDTH, 120))
    pygame.draw.line(screen, DARK_GRAY, (0, 120), (WIDTH, 120), 2)
    
    title = title_font.render("K-fold Cross-Validation", True, BLACK)
    screen.blit(title, (WIDTH//2 - title.get_width()//2, 20))
    
    subtitle = subtitle_font.render("Developed by: Venugopal Adep", True, BLACK)
    screen.blit(subtitle, (WIDTH//2 - subtitle.get_width()//2, 80))

def draw_folds():
    # Draw a circular background
    pygame.draw.circle(screen, GRAY, (WIDTH//2, HEIGHT//2 + 50), 320, 2)
    
    for i, point in enumerate(data):
        fold_index = i // fold_size
        color = COLORS[fold_index]
        
        # Draw point with a shadow effect
        pygame.draw.circle(screen, DARK_GRAY, (point[0]+2, point[1]+2), 6)
        pygame.draw.circle(screen, color, point, 6)
        
        # Highlight current validation fold
        if fold_index == current_fold:
            pygame.draw.circle(screen, BLACK, point, 8, 2)

def draw_train_test_split():
    validation_start = current_fold * fold_size
    validation_end = validation_start + fold_size
    
    # Draw training set box
    pygame.draw.rect(screen, GRAY, (50, 150, 400, 300), 0, 10)
    pygame.draw.rect(screen, DARK_GRAY, (50, 150, 400, 300), 2, 10)
    
    train_text = subtitle_font.render("Training Set", True, BLACK)
    screen.blit(train_text, (50 + 200 - train_text.get_width()//2, 160))
    
    # Draw validation set box
    pygame.draw.rect(screen, GRAY, (WIDTH - 450, 150, 400, 300), 0, 10)
    pygame.draw.rect(screen, DARK_GRAY, (WIDTH - 450, 150, 400, 300), 2, 10)
    
    test_text = subtitle_font.render("Validation Set", True, BLACK)
    screen.blit(test_text, (WIDTH - 450 + 200 - test_text.get_width()//2, 160))
    
    # Draw training points
    train_count = 0
    for i, point in enumerate(data):
        if i < validation_start or i >= validation_end:
            x = 80 + (train_count % 10) * 35
            y = 200 + (train_count // 10) * 35
            color = COLORS[i // fold_size]
            pygame.draw.circle(screen, DARK_GRAY, (x+2, y+2), 6)
            pygame.draw.circle(screen, color, (x, y), 6)
            train_count += 1
    
    # Draw validation points
    for i in range(validation_start, validation_end):
        x = WIDTH - 420 + ((i - validation_start) % 10) * 35
        y = 200 + ((i - validation_start) // 10) * 35
        color = COLORS[i // fold_size]
        pygame.draw.circle(screen, DARK_GRAY, (x+2, y+2), 6)
        pygame.draw.circle(screen, color, (x, y), 6)

def draw_info():
    # Draw info box
    pygame.draw.rect(screen, GRAY, (50, HEIGHT - 220, 300, 180), 0, 10)
    pygame.draw.rect(screen, DARK_GRAY, (50, HEIGHT - 220, 300, 180), 2, 10)
    
    info_text = [
        f"K = {K}",
        f"Total data points: {data_points}",
        f"Points per fold: {fold_size}",
        f"Current fold: {current_fold + 1}",
        "",
        "Controls:",
        "SPACE: Toggle animation",
        "A: Toggle auto-animation",
        "LEFT/RIGHT: Change fold"
    ]
    
    for i, text in enumerate(info_text):
        rendered_text = text_font.render(text, True, BLACK)
        screen.blit(rendered_text, (70, HEIGHT - 200 + i * 25))

def draw_explanation():
    # Draw explanation box
    pygame.draw.rect(screen, GRAY, (WIDTH - 450, HEIGHT - 220, 400, 180), 0, 10)
    pygame.draw.rect(screen, DARK_GRAY, (WIDTH - 450, HEIGHT - 220, 400, 180), 2, 10)
    
    explanation_title = subtitle_font.render("K-fold Cross-Validation", True, BLACK)
    screen.blit(explanation_title, (WIDTH - 450 + 200 - explanation_title.get_width()//2, HEIGHT - 210))
    
    explanation_text = [
        "A technique to evaluate machine learning models",
        "by partitioning the data into K subsets.",
        "",
        "Each fold serves once as validation data,",
        "while the remaining K-1 folds form the training set.",
        "This ensures all data points are used for both",
        "training and validation."
    ]
    
    for i, text in enumerate(explanation_text):
        rendered_text = small_font.render(text, True, BLACK)
        screen.blit(rendered_text, (WIDTH - 430, HEIGHT - 180 + i * 22))

def animate_fold_change():
    global animation_progress, animating, current_fold
    
    if animation_progress < 1:
        animation_progress += animation_speed
    else:
        animating = False
        animation_progress = 0
        current_fold = (current_fold + 1) % K

def handle_auto_animation():
    global auto_animate_timer, animating, current_fold
    
    auto_animate_timer += 1
    if auto_animate_timer > 120:  # Change fold every 2 seconds (60 FPS * 2)
        auto_animate_timer = 0
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
                animating = not animating
                animation_progress = 0
            elif event.key == pygame.K_a:
                auto_animate = not auto_animate
                auto_animate_timer = 0
            elif event.key == pygame.K_LEFT:
                current_fold = (current_fold - 1) % K
                animating = False
                animation_progress = 0
            elif event.key == pygame.K_RIGHT:
                current_fold = (current_fold + 1) % K
                animating = False
                animation_progress = 0

    screen.fill(WHITE)
    
    draw_title()
    draw_folds()
    draw_train_test_split()
    draw_info()
    draw_explanation()
    
    if animating:
        animate_fold_change()
    
    if auto_animate and not animating:
        handle_auto_animation()
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()

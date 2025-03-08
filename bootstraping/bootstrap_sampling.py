import pygame
import random
import math
import colorsys
import numpy as np
from pygame.locals import *

# Initialize Pygame with OpenGL support
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1600, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF)
pygame.display.set_caption("3D Bootstrap Sampling Visualization")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
LIGHT_BLUE = (200, 220, 255)

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

# Bootstrap parameters
original_sample_size = 50
bootstrap_sample_size = original_sample_size
num_bootstrap_samples = 5
current_bootstrap = 0

# Generate 3D data points for original sample
def generate_3d_data():
    points_3d = []
    points_2d = []
    
    for i in range(original_sample_size):
        # Generate points on a 3D sphere
        phi = random.uniform(0, 2 * math.pi)
        theta = random.uniform(0, math.pi)
        radius = random.uniform(150, 250)
        
        x = radius * math.sin(theta) * math.cos(phi)
        y = radius * math.sin(theta) * math.sin(phi)
        z = radius * math.cos(theta)
        
        # Store 3D coordinates
        points_3d.append((x, y, z))
        
        # Project to 2D for visualization
        screen_x = WIDTH // 2 + int(x)
        screen_y = HEIGHT // 2 + int(y)
        points_2d.append((screen_x, screen_y, z))  # Include z for depth information
    
    return points_3d, points_2d

# Generate 3D data
original_data_3d, original_data_2d = generate_3d_data()
COLORS = generate_colors(num_bootstrap_samples + 1)

# Generate bootstrap samples using 3D data indices
bootstrap_samples_indices = []
for i in range(num_bootstrap_samples):
    # Sample with replacement - store indices instead of points
    bootstrap = [random.randint(0, original_sample_size-1) for _ in range(bootstrap_sample_size)]
    bootstrap_samples_indices.append(bootstrap)

# Initialize rotation angles
rotation_x = 0
rotation_y = 0
rotation_z = 0
rotation_speed = 0.005

# Animation state
animating = False
animation_progress = 0
animation_speed = 0.02
auto_animate = False
auto_animate_timer = 0

def rotate_point(point, angle_x, angle_y, angle_z):
    x, y, z = point
    
    # Rotate around X-axis
    y1 = y * math.cos(angle_x) - z * math.sin(angle_x)
    z1 = y * math.sin(angle_x) + z * math.cos(angle_x)
    x1, y, z = x, y1, z1
    
    # Rotate around Y-axis
    x2 = x1 * math.cos(angle_y) + z * math.sin(angle_y)
    z2 = -x1 * math.sin(angle_y) + z * math.cos(angle_y)
    x, y, z = x2, y, z2
    
    # Rotate around Z-axis
    x3 = x * math.cos(angle_z) - y * math.sin(angle_z)
    y3 = x * math.sin(angle_z) + y * math.cos(angle_z)
    x, y = x3, y3
    
    # Project to 2D
    screen_x = WIDTH // 2 + int(x)
    screen_y = HEIGHT // 2 + int(y)
    
    return (screen_x, screen_y, z)

def draw_title():
    # Draw background for title
    pygame.draw.rect(screen, GRAY, (0, 0, WIDTH, 120))
    pygame.draw.line(screen, DARK_GRAY, (0, 120), (WIDTH, 120), 2)
    
    title = title_font.render("3D Bootstrap Sampling Visualization", True, BLACK)
    screen.blit(title, (WIDTH//2 - title.get_width()//2, 20))
    
    subtitle = subtitle_font.render("Developed by: Venugopal Adep", True, BLACK)
    screen.blit(subtitle, (WIDTH//2 - subtitle.get_width()//2, 80))

def draw_3d_visualization():
    # Create a list of points with their depth info
    points_to_draw = []
    
    # Update the 2D projections based on current rotation
    rotated_points = []
    for point in original_data_3d:
        rotated_point = rotate_point(point, rotation_x, rotation_y, rotation_z)
        rotated_points.append(rotated_point)
    
    # Draw a sphere wireframe for reference
    radius = 250
    num_segments = 30
    for i in range(num_segments):
        angle1 = (i / num_segments) * 2 * math.pi
        angle2 = ((i + 1) / num_segments) * 2 * math.pi
        
        # Draw latitude circles
        for j in range(0, num_segments, 2):
            lat = (j / num_segments) * math.pi
            x1 = radius * math.sin(lat) * math.cos(angle1)
            y1 = radius * math.sin(lat) * math.sin(angle1)
            z1 = radius * math.cos(lat)
            
            x2 = radius * math.sin(lat) * math.cos(angle2)
            y2 = radius * math.sin(lat) * math.sin(angle2)
            z2 = radius * math.cos(lat)
            
            p1 = rotate_point((x1, y1, z1), rotation_x, rotation_y, rotation_z)
            p2 = rotate_point((x2, y2, z2), rotation_x, rotation_y, rotation_z)
            
            pygame.draw.line(screen, (220, 220, 220, 50), (p1[0], p1[1]), (p2[0], p2[1]), 1)
    
    # Get current bootstrap sample indices
    current_bs_indices = bootstrap_samples_indices[current_bootstrap]
    
    # Count occurrences of each index in the bootstrap sample
    index_counts = {}
    for idx in current_bs_indices:
        if idx in index_counts:
            index_counts[idx] += 1
        else:
            index_counts[idx] = 1
    
    # Add all points with their metadata to points_to_draw
    for i, point in enumerate(rotated_points):
        x, y, z = point
        if i in index_counts:
            # This point is in the current bootstrap sample
            size = 6 + 4 * index_counts[i]  # Size based on frequency
            color = COLORS[current_bootstrap + 1]
            is_selected = True
            count = index_counts[i]
        else:
            # Original point not in bootstrap
            size = 4
            color = DARK_GRAY
            is_selected = False
            count = 0
        
        # Add to list with z-coordinate for depth sorting
        points_to_draw.append((x, y, z, size, color, is_selected, count))
    
    # Sort points by z-coordinate (depth) - draw farthest points first
    points_to_draw.sort(key=lambda p: p[2], reverse=True)
    
    # Draw all points based on depth
    for x, y, z, size, color, is_selected, count in points_to_draw:
        # Add shading based on depth
        shade = min(255, max(100, 255 - int(abs(z) * 0.2)))
        shaded_color = (
            min(255, int(color[0] * shade / 255)),
            min(255, int(color[1] * shade / 255)),
            min(255, int(color[2] * shade / 255))
        )
        
        # Draw shadow
        pygame.draw.circle(screen, (30, 30, 30), (x+2, y+2), size)
        
        # Draw point
        pygame.draw.circle(screen, shaded_color, (x, y), size)
        
        # Draw highlight for selected points
        if is_selected:
            # Draw ring around the point
            pygame.draw.circle(screen, WHITE, (x, y), size + 2, 1)
            
            # Show count if more than 1
            if count > 1:
                count_text = small_font.render(str(count), True, WHITE)
                screen.blit(count_text, (x + size + 2, y - size - 2))

def draw_bootstrap_sample():
    # Draw bootstrap sample box
    box_width = 400
    box_height = 300
    box_x = WIDTH - box_width - 50
    box_y = 150
    
    # Draw box with gradient background
    pygame.draw.rect(screen, LIGHT_BLUE, (box_x, box_y, box_width, box_height), 0, 10)
    pygame.draw.rect(screen, DARK_GRAY, (box_x, box_y, box_width, box_height), 2, 10)
    
    bootstrap_text = subtitle_font.render(f"Bootstrap Sample #{current_bootstrap + 1}", True, BLACK)
    screen.blit(bootstrap_text, (box_x + box_width//2 - bootstrap_text.get_width()//2, box_y + 10))
    
    # Get current bootstrap sample indices
    current_bs_indices = bootstrap_samples_indices[current_bootstrap]
    
    # Count occurrences of each index
    index_counts = {}
    for idx in current_bs_indices:
        if idx in index_counts:
            index_counts[idx] += 1
        else:
            index_counts[idx] = 1
    
    # Draw sample with counts
    bs_idx = 0
    for idx, count in index_counts.items():
        x = box_x + 50 + (bs_idx % 7) * 50
        y = box_y + 60 + (bs_idx // 7) * 50
        
        # Draw the point
        pygame.draw.circle(screen, DARK_GRAY, (x+2, y+2), 15)
        pygame.draw.circle(screen, COLORS[current_bootstrap + 1], (x, y), 15)
        
        # Draw index number inside the circle
        idx_text = small_font.render(str(idx), True, WHITE)
        screen.blit(idx_text, (x - idx_text.get_width()//2, y - idx_text.get_height()//2))
        
        # Draw count if more than 1
        if count > 1:
            count_text = text_font.render(f"×{count}", True, BLACK)
            screen.blit(count_text, (x + 20, y - 10))
        
        bs_idx += 1

def draw_info():
    # Draw info box
    pygame.draw.rect(screen, GRAY, (50, HEIGHT - 220, 300, 180), 0, 10)
    pygame.draw.rect(screen, DARK_GRAY, (50, HEIGHT - 220, 300, 180), 2, 10)
    
    info_text = [
        f"Original sample size: {original_sample_size}",
        f"Bootstrap sample size: {bootstrap_sample_size}",
        f"Number of bootstrap samples: {num_bootstrap_samples}",
        f"Current bootstrap: {current_bootstrap + 1}",
        "",
        "Controls:",
        "SPACE: Toggle animation",
        "A: Toggle auto-animation",
        "LEFT/RIGHT: Change bootstrap sample",
        "ARROWS: Rotate visualization"
    ]
    
    for i, text in enumerate(info_text):
        rendered_text = text_font.render(text, True, BLACK)
        screen.blit(rendered_text, (70, HEIGHT - 200 + i * 20))

def draw_explanation():
    # Draw explanation box
    pygame.draw.rect(screen, GRAY, (WIDTH - 450, HEIGHT - 220, 400, 180), 0, 10)
    pygame.draw.rect(screen, DARK_GRAY, (WIDTH - 450, HEIGHT - 220, 400, 180), 2, 10)
    
    explanation_title = subtitle_font.render("Bootstrap Sampling", True, BLACK)
    screen.blit(explanation_title, (WIDTH - 450 + 200 - explanation_title.get_width()//2, HEIGHT - 210))
    
    explanation_text = [
        "A resampling technique that involves drawing samples",
        "with replacement from the original dataset.",
        "",
        "Each bootstrap sample is the same size as the original,",
        "but some points may appear multiple times while",
        "others may not appear at all.",
        "",
        "This helps estimate statistics and their variability."
    ]
    
    for i, text in enumerate(explanation_text):
        rendered_text = small_font.render(text, True, BLACK)
        screen.blit(rendered_text, (WIDTH - 430, HEIGHT - 180 + i * 22))

def animate_bootstrap_change():
    global animation_progress, animating, current_bootstrap
    
    if animation_progress < 1:
        animation_progress += animation_speed
    else:
        animating = False
        animation_progress = 0
        current_bootstrap = (current_bootstrap + 1) % num_bootstrap_samples

def handle_auto_animation():
    global auto_animate_timer, animating, current_bootstrap
    
    auto_animate_timer += 1
    if auto_animate_timer > 120:  # Change bootstrap every 2 seconds (60 FPS * 2)
        auto_animate_timer = 0
        current_bootstrap = (current_bootstrap + 1) % num_bootstrap_samples

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
                current_bootstrap = (current_bootstrap - 1) % num_bootstrap_samples
                animating = False
                animation_progress = 0
            elif event.key == pygame.K_RIGHT:
                current_bootstrap = (current_bootstrap + 1) % num_bootstrap_samples
                animating = False
                animation_progress = 0
    
    # Handle rotation with key presses
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        rotation_x += rotation_speed
    if keys[pygame.K_DOWN]:
        rotation_x -= rotation_speed
    if keys[pygame.K_LEFT]:
        rotation_y -= rotation_speed
    if keys[pygame.K_RIGHT]:
        rotation_y += rotation_speed
    if keys[pygame.K_q]:
        rotation_z += rotation_speed
    if keys[pygame.K_e]:
        rotation_z -= rotation_speed

    # Always rotate slightly for dynamic effect
    rotation_y += rotation_speed * 0.2

    screen.fill(WHITE)
    
    draw_title()
    draw_3d_visualization()
    draw_bootstrap_sample()
    draw_info()
    draw_explanation()
    
    if animating:
        animate_bootstrap_change()
    
    if auto_animate and not animating:
        handle_auto_animation()
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()

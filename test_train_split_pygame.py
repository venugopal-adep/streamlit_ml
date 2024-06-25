import pygame
import random
import math

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1600, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Test Train Split Demo")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
GRAY = (200, 200, 200)

# Fonts
title_font = pygame.font.Font(None, 64)
text_font = pygame.font.Font(None, 32)

# Data points
num_points = 100
data_points = [(random.randint(50, WIDTH - 50), random.randint(300, HEIGHT - 50)) for _ in range(num_points)]

# Split ratio
split_ratio = 0.7
split_line_x = int(WIDTH * split_ratio)

# Buttons
button_width, button_height = 200, 50
shuffle_button = pygame.Rect(WIDTH // 2 - 220, HEIGHT - 100, button_width, button_height)
add_point_button = pygame.Rect(WIDTH // 2 + 20, HEIGHT - 100, button_width, button_height)

# Slider
slider_width, slider_height = 300, 20
slider_x = WIDTH // 2 - slider_width // 2
slider_y = HEIGHT - 150
slider_rect = pygame.Rect(slider_x, slider_y, slider_width, slider_height)
slider_handle_radius = 15
slider_handle_x = slider_x + int(split_ratio * slider_width)

# Function to split data
def split_data():
    random.shuffle(data_points)
    train_data = data_points[:int(len(data_points) * split_ratio)]
    test_data = data_points[int(len(data_points) * split_ratio):]
    return train_data, test_data

train_data, test_data = split_data()

# Main game loop
running = True
clock = pygame.time.Clock()
dragging_slider = False
adding_point = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if shuffle_button.collidepoint(event.pos):
                train_data, test_data = split_data()
            elif add_point_button.collidepoint(event.pos):
                adding_point = True
            elif (slider_handle_x - slider_handle_radius <= event.pos[0] <= slider_handle_x + slider_handle_radius
                  and slider_y - slider_handle_radius <= event.pos[1] <= slider_y + slider_handle_radius):
                dragging_slider = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if adding_point:
                x, y = event.pos
                if 50 <= x <= WIDTH - 50 and 300 <= y <= HEIGHT - 150:
                    data_points.append((x, y))
                    train_data, test_data = split_data()
                adding_point = False
            dragging_slider = False
        elif event.type == pygame.MOUSEMOTION and dragging_slider:
            slider_handle_x = max(slider_x, min(event.pos[0], slider_x + slider_width))
            split_ratio = (slider_handle_x - slider_x) / slider_width
            split_line_x = int(WIDTH * split_ratio)
            train_data, test_data = split_data()

    # Clear the screen
    screen.fill(WHITE)

    # Draw title and developer info
    title_text = title_font.render("Test Train Split Demo", True, BLACK)
    screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, 20))
    
    dev_text = text_font.render("Developed by: Venugopal Adep", True, BLACK)
    screen.blit(dev_text, (WIDTH // 2 - dev_text.get_width() // 2, 80))

    # Draw split line
    pygame.draw.line(screen, BLACK, (split_line_x, 150), (split_line_x, HEIGHT - 150), 2)

    # Draw data points
    for point in train_data:
        pygame.draw.circle(screen, BLUE, point, 5)
    for point in test_data:
        pygame.draw.circle(screen, RED, point, 5)

    # Draw labels
    train_label = text_font.render("Training Data", True, BLUE)
    test_label = text_font.render("Test Data", True, RED)
    screen.blit(train_label, (split_line_x // 2 - train_label.get_width() // 2, 200))
    screen.blit(test_label, (split_line_x + (WIDTH - split_line_x) // 2 - test_label.get_width() // 2, 200))

    # Draw split ratio
    split_text = text_font.render(f"Split Ratio: {split_ratio:.2f}", True, BLACK)
    screen.blit(split_text, (WIDTH // 2 - split_text.get_width() // 2, HEIGHT - 180))

    # Draw buttons
    pygame.draw.rect(screen, GREEN, shuffle_button)
    pygame.draw.rect(screen, YELLOW, add_point_button)
    shuffle_text = text_font.render("Shuffle & Split", True, BLACK)
    add_point_text = text_font.render("Add Point", True, BLACK)
    screen.blit(shuffle_text, (shuffle_button.centerx - shuffle_text.get_width() // 2, shuffle_button.centery - shuffle_text.get_height() // 2))
    screen.blit(add_point_text, (add_point_button.centerx - add_point_text.get_width() // 2, add_point_button.centery - add_point_text.get_height() // 2))

    # Draw slider
    pygame.draw.rect(screen, GRAY, slider_rect)
    pygame.draw.circle(screen, BLUE, (slider_handle_x, slider_y + slider_height // 2), slider_handle_radius)

    # Draw data counts
    train_count = text_font.render(f"Training: {len(train_data)}", True, BLUE)
    test_count = text_font.render(f"Test: {len(test_data)}", True, RED)
    screen.blit(train_count, (20, HEIGHT - 50))
    screen.blit(test_count, (WIDTH - test_count.get_width() - 20, HEIGHT - 50))

    # Update the display
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
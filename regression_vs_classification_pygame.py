import pygame
import sys
import math

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1600, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Regression vs Classification ML Demo")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

# Fonts
title_font = pygame.font.Font(None, 48)
text_font = pygame.font.Font(None, 24)

# Data points
regression_points = []
classification_points = []

# Button class
class Button:
    def __init__(self, x, y, width, height, text, color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color

    def draw(self):
        pygame.draw.rect(screen, self.color, self.rect)
        text_surface = text_font.render(self.text, True, BLACK)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

# Create buttons
reset_button = Button(WIDTH // 2 - 100, HEIGHT - 60, 200, 40, "Reset", YELLOW)

# Function to add regression point
def add_regression_point(pos):
    x, y = pos
    if 50 < x < WIDTH // 2 - 50 and 100 < y < HEIGHT - 100:
        regression_points.append((x, y))

# Function to add classification point
def add_classification_point(pos, color):
    x, y = pos
    if WIDTH // 2 + 50 < x < WIDTH - 50 and 100 < y < HEIGHT - 100:
        classification_points.append((x, y, color))

# Function to calculate regression line
def calculate_regression_line():
    if len(regression_points) < 2:
        return None

    x_mean = sum(point[0] for point in regression_points) / len(regression_points)
    y_mean = sum(point[1] for point in regression_points) / len(regression_points)

    numerator = sum((point[0] - x_mean) * (point[1] - y_mean) for point in regression_points)
    denominator = sum((point[0] - x_mean) ** 2 for point in regression_points)

    if denominator == 0:
        return None

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    return slope, intercept

# Function to calculate decision boundary
def calculate_decision_boundary():
    if len(classification_points) < 2:
        return None

    red_points = [point for point in classification_points if point[2] == RED]
    blue_points = [point for point in classification_points if point[2] == BLUE]

    if not red_points or not blue_points:
        return None

    red_center = (sum(p[0] for p in red_points) / len(red_points), sum(p[1] for p in red_points) / len(red_points))
    blue_center = (sum(p[0] for p in blue_points) / len(blue_points), sum(p[1] for p in blue_points) / len(blue_points))

    midpoint = ((red_center[0] + blue_center[0]) / 2, (red_center[1] + blue_center[1]) / 2)
    perpendicular_slope = -1 / ((blue_center[1] - red_center[1]) / (blue_center[0] - red_center[0]))

    return perpendicular_slope, midpoint

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if reset_button.is_clicked(event.pos):
                regression_points.clear()
                classification_points.clear()
            else:
                if event.pos[0] < WIDTH // 2:
                    add_regression_point(event.pos)
                else:
                    if event.button == 1:  # Left mouse button
                        add_classification_point(event.pos, BLUE)
                    elif event.button == 3:  # Right mouse button
                        add_classification_point(event.pos, RED)

    # Clear the screen
    screen.fill(WHITE)

    # Draw title and author
    title_surface = title_font.render("Regression vs Classification ML Demo", True, BLACK)
    screen.blit(title_surface, (WIDTH // 2 - title_surface.get_width() // 2, 20))

    author_surface = text_font.render("Developed by: Venugopal Adep", True, BLACK)
    screen.blit(author_surface, (WIDTH // 2 - author_surface.get_width() // 2, 70))

    # Draw dividing line
    pygame.draw.line(screen, BLACK, (WIDTH // 2, 100), (WIDTH // 2, HEIGHT - 100), 2)

    # Draw regression points and line
    for point in regression_points:
        pygame.draw.circle(screen, GREEN, point, 5)

    regression_line = calculate_regression_line()
    if regression_line:
        slope, intercept = regression_line
        start_x, end_x = 50, WIDTH // 2 - 50
        start_y = int(slope * start_x + intercept)
        end_y = int(slope * end_x + intercept)
        pygame.draw.line(screen, GREEN, (start_x, start_y), (end_x, end_y), 2)

    # Draw classification points and decision boundary
    for point in classification_points:
        pygame.draw.circle(screen, point[2], (point[0], point[1]), 5)

    decision_boundary = calculate_decision_boundary()
    if decision_boundary:
        slope, midpoint = decision_boundary
        start_x, end_x = WIDTH // 2 + 50, WIDTH - 50
        start_y = int(slope * (start_x - midpoint[0]) + midpoint[1])
        end_y = int(slope * (end_x - midpoint[0]) + midpoint[1])
        pygame.draw.line(screen, GREEN, (start_x, start_y), (end_x, end_y), 2)

    # Draw labels
    regression_label = text_font.render("Regression", True, BLACK)
    screen.blit(regression_label, (WIDTH // 4 - regression_label.get_width() // 2, HEIGHT - 40))

    classification_label = text_font.render("Classification", True, BLACK)
    screen.blit(classification_label, (3 * WIDTH // 4 - classification_label.get_width() // 2, HEIGHT - 40))

    # Draw instructions
    instructions1 = text_font.render("Left: Click to add points | Right: Left-click for blue, Right-click for red", True, BLACK)
    instructions2 = text_font.render("Green line shows trend/boundary", True, BLACK)
    screen.blit(instructions1, (20, HEIGHT - 60))
    screen.blit(instructions2, (20, HEIGHT - 30))

    # Draw reset button
    reset_button.draw()

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()
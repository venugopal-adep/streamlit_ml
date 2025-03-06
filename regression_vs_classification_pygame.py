import pygame
import sys
import math
import random

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1600, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Regression vs Classification ML Demo")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 50, 50)
BLUE = (50, 100, 255)
GREEN = (50, 200, 50)
YELLOW = (255, 220, 0)
LIGHT_GRAY = (240, 240, 240)
DARK_GRAY = (80, 80, 80)
PURPLE = (150, 50, 200)

# Background colors
BG_COLOR = (245, 245, 250)
PANEL_COLOR = (235, 235, 245)

# Fonts - Try to use system fonts for better appearance
try:
    title_font = pygame.font.SysFont("Arial", 48, bold=True)
    subtitle_font = pygame.font.SysFont("Arial", 32, bold=True)
    text_font = pygame.font.SysFont("Arial", 24)
    small_font = pygame.font.SysFont("Arial", 18)
except:
    # Fallback to default font if system fonts are not available
    title_font = pygame.font.Font(None, 48)
    subtitle_font = pygame.font.Font(None, 36)
    text_font = pygame.font.Font(None, 24)
    small_font = pygame.font.Font(None, 18)

# Data points
regression_points = []
classification_points = []

# Button class with hover effect
class Button:
    def __init__(self, x, y, width, height, text, color, hover_color=None, text_color=BLACK):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color or self._lighten_color(color)
        self.text_color = text_color
        self.is_hovered = False

    def _lighten_color(self, color):
        return tuple(min(c + 30, 255) for c in color)

    def draw(self):
        color = self.hover_color if self.is_hovered else self.color
        # Draw button with rounded corners
        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        pygame.draw.rect(screen, DARK_GRAY, self.rect, 2, border_radius=8)
        
        text_surface = text_font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def update(self, mouse_pos):
        self.is_hovered = self.rect.collidepoint(mouse_pos)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

# Create buttons
reset_button = Button(WIDTH // 2 - 100, HEIGHT - 70, 200, 40, "Reset All", YELLOW)
clear_regression_button = Button(WIDTH // 4 - 100, HEIGHT - 70, 200, 40, "Clear Regression", LIGHT_GRAY)
clear_classification_button = Button(3 * WIDTH // 4 - 100, HEIGHT - 70, 200, 40, "Clear Classification", LIGHT_GRAY)

# Function to add regression point
def add_regression_point(pos):
    x, y = pos
    if 80 < x < WIDTH // 2 - 80 and 150 < y < HEIGHT - 150:
        regression_points.append((x, y))

# Function to add classification point
def add_classification_point(pos, color):
    x, y = pos
    if WIDTH // 2 + 80 < x < WIDTH - 80 and 150 < y < HEIGHT - 150:
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

    # Avoid division by zero
    if blue_center[0] - red_center[0] == 0:
        return 0, ((red_center[0] + blue_center[0]) / 2, (red_center[1] + blue_center[1]) / 2)

    midpoint = ((red_center[0] + blue_center[0]) / 2, (red_center[1] + blue_center[1]) / 2)
    
    # Calculate perpendicular slope
    original_slope = (blue_center[1] - red_center[1]) / (blue_center[0] - red_center[0])
    if original_slope == 0:
        perpendicular_slope = float('inf')  # Vertical line
    else:
        perpendicular_slope = -1 / original_slope

    return perpendicular_slope, midpoint

# Function to draw a panel with a title
def draw_panel(x, y, width, height, title):
    panel_rect = pygame.Rect(x, y, width, height)
    pygame.draw.rect(screen, PANEL_COLOR, panel_rect, border_radius=10)
    pygame.draw.rect(screen, DARK_GRAY, panel_rect, 2, border_radius=10)
    
    title_surface = subtitle_font.render(title, True, DARK_GRAY)
    screen.blit(title_surface, (x + width // 2 - title_surface.get_width() // 2, y + 15))

# Function to draw grid lines
def draw_grid(x, y, width, height, cell_size=50):
    for i in range(0, width + 1, cell_size):
        alpha = 20 if i % 100 == 0 else 10
        color = (*DARK_GRAY[:3], alpha)
        surface = pygame.Surface((1, height), pygame.SRCALPHA)
        pygame.draw.line(surface, color, (0, 0), (0, height))
        screen.blit(surface, (x + i, y))
    
    for i in range(0, height + 1, cell_size):
        alpha = 20 if i % 100 == 0 else 10
        color = (*DARK_GRAY[:3], alpha)
        surface = pygame.Surface((width, 1), pygame.SRCALPHA)
        pygame.draw.line(surface, color, (0, 0), (width, 0))
        screen.blit(surface, (x, y + i))

# Main game loop
running = True
clock = pygame.time.Clock()

# Add some initial points for demonstration
for _ in range(5):
    x = random.randint(100, WIDTH // 2 - 100)
    y = random.randint(200, HEIGHT - 200)
    regression_points.append((x, y))

for _ in range(5):
    x = random.randint(WIDTH // 2 + 100, WIDTH - 100)
    y = random.randint(200, HEIGHT - 200)
    color = RED if random.random() > 0.5 else BLUE
    classification_points.append((x, y, color))

while running:
    mouse_pos = pygame.mouse.get_pos()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if reset_button.is_clicked(event.pos):
                regression_points.clear()
                classification_points.clear()
            elif clear_regression_button.is_clicked(event.pos):
                regression_points.clear()
            elif clear_classification_button.is_clicked(event.pos):
                classification_points.clear()
            else:
                if event.pos[0] < WIDTH // 2:
                    add_regression_point(event.pos)
                else:
                    if event.button == 1:  # Left mouse button
                        add_classification_point(event.pos, BLUE)
                    elif event.button == 3:  # Right mouse button
                        add_classification_point(event.pos, RED)

    # Update button hover states
    reset_button.update(mouse_pos)
    clear_regression_button.update(mouse_pos)
    clear_classification_button.update(mouse_pos)

    # Clear the screen
    screen.fill(BG_COLOR)

    # Draw title and author
    title_surface = title_font.render("Regression vs Classification ML Demo", True, PURPLE)
    screen.blit(title_surface, (WIDTH // 2 - title_surface.get_width() // 2, 20))

    author_surface = text_font.render("Developed by: Venugopal Adep", True, DARK_GRAY)
    screen.blit(author_surface, (WIDTH // 2 - author_surface.get_width() // 2, 70))

    # Draw panels
    panel_width = WIDTH // 2 - 40
    panel_height = HEIGHT - 200
    draw_panel(20, 120, panel_width, panel_height, "Regression")
    draw_panel(WIDTH // 2 + 20, 120, panel_width, panel_height, "Classification")

    # Draw grid lines
    draw_grid(20, 120, panel_width, panel_height)
    draw_grid(WIDTH // 2 + 20, 120, panel_width, panel_height)

    # Draw regression points and line
    for point in regression_points:
        pygame.draw.circle(screen, GREEN, point, 8)
        pygame.draw.circle(screen, DARK_GRAY, point, 8, 1)

    regression_line = calculate_regression_line()
    if regression_line:
        slope, intercept = regression_line
        
        # Calculate points where the line intersects the panel boundaries
        panel_left = 20
        panel_right = 20 + panel_width
        panel_top = 120
        panel_bottom = 120 + panel_height
        
        # Calculate y values at panel left and right edges
        left_y = slope * panel_left + intercept
        right_y = slope * panel_right + intercept
        
        # Calculate x values at panel top and bottom edges
        if slope != 0:
            top_x = (panel_top - intercept) / slope
            bottom_x = (panel_bottom - intercept) / slope
        else:
            top_x = bottom_x = None
        
        # Find the intersection points with the panel boundaries
        intersections = []
        
        # Check left edge
        if panel_top <= left_y <= panel_bottom:
            intersections.append((panel_left, left_y))
            
        # Check right edge
        if panel_top <= right_y <= panel_bottom:
            intersections.append((panel_right, right_y))
            
        # Check top edge
        if top_x is not None and panel_left <= top_x <= panel_right:
            intersections.append((top_x, panel_top))
            
        # Check bottom edge
        if bottom_x is not None and panel_left <= bottom_x <= panel_right:
            intersections.append((bottom_x, panel_bottom))
        
        # If we have at least two intersection points, draw the line
        if len(intersections) >= 2:
            start_point, end_point = intersections[:2]
            
            # Draw line with shadow for better visibility
            pygame.draw.line(screen, (*GREEN[:3], 100), start_point, end_point, 8)
            pygame.draw.line(screen, GREEN, start_point, end_point, 3)
        
        # Display equation
        equation = f"y = {slope:.2f}x + {intercept:.2f}"
        eq_surface = text_font.render(equation, True, DARK_GRAY)
        screen.blit(eq_surface, (30, 150))

    # Draw classification points and decision boundary
    for point in classification_points:
        pygame.draw.circle(screen, point[2], (point[0], point[1]), 8)
        pygame.draw.circle(screen, DARK_GRAY, (point[0], point[1]), 8, 1)

    decision_boundary = calculate_decision_boundary()
    if decision_boundary:
        slope, midpoint = decision_boundary
        
        # Draw colored regions for classification
        red_region = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        blue_region = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        
        for x in range(0, panel_width, 5):
            for y in range(0, panel_height, 5):
                real_x = WIDTH // 2 + 20 + x
                real_y = 120 + y
                
                # Determine which side of the boundary the point is on
                if slope == float('inf'):  # Vertical line
                    is_red_side = real_x < midpoint[0]
                else:
                    expected_y = slope * (real_x - midpoint[0]) + midpoint[1]
                    is_red_side = real_y > expected_y
                
                if is_red_side:
                    pygame.draw.rect(red_region, (*RED[:3], 5), (x, y, 5, 5))
                else:
                    pygame.draw.rect(blue_region, (*BLUE[:3], 5), (x, y, 5, 5))
        
        screen.blit(red_region, (WIDTH // 2 + 20, 120))
        screen.blit(blue_region, (WIDTH // 2 + 20, 120))
        
        # Calculate points where the line intersects the panel boundaries
        panel_left = WIDTH // 2 + 20
        panel_right = WIDTH // 2 + 20 + panel_width
        panel_top = 120
        panel_bottom = 120 + panel_height
        
        if slope == float('inf'):  # Vertical line
            # For vertical lines, x is constant
            start_point = (midpoint[0], panel_top)
            end_point = (midpoint[0], panel_bottom)
        else:
            # Calculate y values at panel left and right edges
            left_y = slope * (panel_left - midpoint[0]) + midpoint[1]
            right_y = slope * (panel_right - midpoint[0]) + midpoint[1]
            
            # Calculate x values at panel top and bottom edges
            if slope != 0:
                top_x = midpoint[0] + (panel_top - midpoint[1]) / slope
                bottom_x = midpoint[0] + (panel_bottom - midpoint[1]) / slope
            else:
                # For horizontal lines, y is constant
                top_x = panel_left
                bottom_x = panel_right
            
            # Find the intersection points with the panel boundaries
            intersections = []
            
            # Check left edge
            if panel_top <= left_y <= panel_bottom:
                intersections.append((panel_left, left_y))
                
            # Check right edge
            if panel_top <= right_y <= panel_bottom:
                intersections.append((panel_right, right_y))
                
            # Check top edge
            if panel_left <= top_x <= panel_right:
                intersections.append((top_x, panel_top))
                
            # Check bottom edge
            if panel_left <= bottom_x <= panel_right:
                intersections.append((bottom_x, panel_bottom))
            
            # If we have at least two intersection points, use them
            if len(intersections) >= 2:
                start_point, end_point = intersections[:2]
            else:
                # Fallback if something went wrong with the calculations
                start_point = (panel_left, left_y)
                end_point = (panel_right, right_y)
        
        # Draw the decision boundary line
        pygame.draw.line(screen, (*GREEN[:3], 100), start_point, end_point, 8)
        pygame.draw.line(screen, GREEN, start_point, end_point, 3)

    # Draw instructions
    instruction_box = pygame.Rect(WIDTH // 2 - 400, HEIGHT - 140, 800, 60)
    pygame.draw.rect(screen, WHITE, instruction_box, border_radius=8)
    pygame.draw.rect(screen, DARK_GRAY, instruction_box, 2, border_radius=8)
    
    instructions1 = text_font.render("Left panel: Click to add regression points", True, DARK_GRAY)
    instructions2 = text_font.render("Right panel: Left-click for blue points, Right-click for red points", True, DARK_GRAY)
    
    screen.blit(instructions1, (WIDTH // 2 - instructions1.get_width() // 2, HEIGHT - 130))
    screen.blit(instructions2, (WIDTH // 2 - instructions2.get_width() // 2, HEIGHT - 100))

    # Draw buttons
    reset_button.draw()
    clear_regression_button.draw()
    clear_classification_button.draw()

    # Draw explanation text
    reg_explain = small_font.render("Regression finds a line that best fits the data points", True, DARK_GRAY)
    class_explain = small_font.render("Classification finds a boundary that separates different classes", True, DARK_GRAY)
    
    screen.blit(reg_explain, (30, HEIGHT - 160))
    screen.blit(class_explain, (WIDTH // 2 + 30, HEIGHT - 160))

    # Update the display
    pygame.display.flip()
    clock.tick(60)  # Limit to 60 FPS

# Quit Pygame
pygame.quit()
sys.exit()

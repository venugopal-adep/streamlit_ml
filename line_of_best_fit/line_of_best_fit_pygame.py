import pygame
import numpy as np
import random
import math
from pygame.locals import *

# Initialize Pygame
pygame.init()
pygame.font.init()

# Set up the display (increased size)
width, height = 1600, 900
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Developed by : Venugopal Adep")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 128, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
LIGHT_BLUE = (173, 216, 230)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)

# Fonts
font = pygame.font.SysFont('Arial', 18)
title_font = pygame.font.SysFont('Arial', 32, bold=True)
axis_font = pygame.font.SysFont('Arial', 16)

# Graph area (leaving space for axes and labels)
margin_left = 80
margin_right = 220
margin_top = 80
margin_bottom = 150
graph_width = width - margin_left - margin_right
graph_height = height - margin_top - margin_bottom

# Function to convert graph coordinates to screen coordinates
def graph_to_screen(x, y):
    screen_x = margin_left + x
    screen_y = margin_top + y
    return screen_x, screen_y

# Function to convert screen coordinates to graph coordinates
def screen_to_graph(screen_x, screen_y):
    x = screen_x - margin_left
    y = screen_y - margin_top
    return x, y

# Generate initial random data points with more linear pattern
def generate_linear_points(num_points, noise_level=0.3):
    # Generate x values spread across graph area
    x_values = np.linspace(50, graph_width-50, num_points)
    
    # Add some randomness to x positions
    x_values = np.array([x + random.uniform(-30, 30) for x in x_values])
    
    # Generate y values with linear relationship and some noise
    slope = random.uniform(0.3, 1.5) * (-1 if random.random() > 0.5 else 1)
    intercept = random.uniform(graph_height * 0.3, graph_height * 0.7)
    
    y_values = np.array([
        (slope * x) + intercept + random.uniform(-noise_level * graph_height, noise_level * graph_height)
        for x in x_values
    ])
    
    # Ensure points stay within graph area
    y_values = np.clip(y_values, 20, graph_height - 20)
    
    return x_values, y_values

# Generate initial data
num_points = 20
x, y = generate_linear_points(num_points)

# Function to calculate line of best fit using least squares method
def best_fit_line(x, y):
    if len(x) < 2:  # Need at least 2 points
        return 0, graph_height // 2
    
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calculate slope (m) using covariance and variance
    numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
    
    # Avoid division by zero
    if denominator == 0:
        m = 0
    else:
        m = numerator / denominator
    
    # Calculate y-intercept (b)
    b = y_mean - m * x_mean
    
    return m, b

# Function to calculate R-squared value
def calculate_r_squared(x, y, m, b):
    if len(x) < 2:
        return 0
    
    # Calculate predicted y values
    y_pred = m * x + b
    
    # Calculate total sum of squares (TSS)
    y_mean = np.mean(y)
    tss = sum((y_i - y_mean) ** 2 for y_i in y)
    
    # Calculate residual sum of squares (RSS)
    rss = sum((y[i] - y_pred[i]) ** 2 for i in range(len(y)))
    
    # Calculate R-squared
    if tss == 0:
        return 0
    return 1 - (rss / tss)

# Function to calculate mean squared error
def calculate_mse(x, y, m, b):
    if len(x) == 0:
        return 0
    y_pred = m * x + b
    return np.mean((y - y_pred) ** 2)

# Animation variables
current_m, current_b = 0, graph_height // 2
target_m, target_b = best_fit_line(x, y)
animation_speed = 0.05

# Interactive mode variables
interactive_mode = False
dragging_point = None
adding_point = False
removing_point = False
show_residuals = True  # Default show residuals
show_grid = True
show_stats = True
custom_line_mode = False
custom_m, custom_b = 0, graph_height // 2

# Button dimensions
button_width, button_height = 180, 40
button_x = width - button_width - 20
button_spacing = 50

# Define buttons (add the new "Clear Points" button)
buttons = [
    {"text": "Reset Points", "y": 20, "action": "reset"},
    {"text": "Clear Points", "y": 70, "action": "clear"},  # New clear button
    {"text": "Toggle Grid", "y": 120, "action": "grid"},
    {"text": "Toggle Residuals", "y": 170, "action": "residuals"},
    {"text": "Toggle Stats", "y": 220, "action": "stats"},
    {"text": "Custom Line Mode", "y": 270, "action": "custom_line", "toggle": True},
    {"text": "Add Points Mode", "y": 320, "action": "add_points", "toggle": True},
    {"text": "Remove Points Mode", "y": 370, "action": "remove_points", "toggle": True}
]

# Grid parameters
grid_spacing = 100

# Main game loop
running = True
clock = pygame.time.Clock()

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        # Mouse button down event
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            
            # Check if clicked on buttons
            button_clicked = False
            for button in buttons:
                button_rect = pygame.Rect(button_x, button["y"], button_width, button_height)
                if button_rect.collidepoint(mouse_pos):
                    button_clicked = True
                    action = button["action"]
                    
                    if action == "reset":
                        x, y = generate_linear_points(num_points)
                        target_m, target_b = best_fit_line(x, y)
                    elif action == "clear":
                        x, y = np.array([]), np.array([])  # Clear all points
                        target_m, target_b = 0, graph_height // 2  # Reset line
                    elif action == "grid":
                        show_grid = not show_grid
                    elif action == "residuals":
                        show_residuals = not show_residuals
                    elif action == "stats":
                        show_stats = not show_stats
                    elif action == "custom_line":
                        custom_line_mode = not custom_line_mode
                        if custom_line_mode:
                            # Disable other modes
                            adding_point = False
                            removing_point = False
                            # Set custom line to current position
                            custom_m, custom_b = current_m, current_b
                    elif action == "add_points":
                        adding_point = not adding_point
                        if adding_point:
                            # Disable other modes
                            removing_point = False
                            custom_line_mode = False
                    elif action == "remove_points":
                        removing_point = not removing_point
                        if removing_point:
                            # Disable other modes
                            adding_point = False
                            custom_line_mode = False
            
            if button_clicked:
                continue
            
            # Convert screen coordinates to graph coordinates
            graph_x, graph_y = screen_to_graph(mouse_pos[0], mouse_pos[1])
            
            # Check if click is within the graph area
            if (0 <= graph_x <= graph_width and 0 <= graph_y <= graph_height):
                # Check if clicked on existing point for dragging
                if not adding_point and not removing_point and not custom_line_mode:
                    for i in range(len(x)):
                        dx = graph_x - x[i]
                        dy = graph_y - y[i]
                        if math.sqrt(dx*dx + dy*dy) < 10:  # 10px radius for selection
                            dragging_point = i
                            break
                
                # Add new point if in add mode
                elif adding_point:
                    x = np.append(x, graph_x)
                    y = np.append(y, graph_y)
                    target_m, target_b = best_fit_line(x, y)
                
                # Remove point if in remove mode
                elif removing_point:
                    for i in range(len(x)):
                        dx = graph_x - x[i]
                        dy = graph_y - y[i]
                        if math.sqrt(dx*dx + dy*dy) < 10:  # 10px radius for selection
                            x = np.delete(x, i)
                            y = np.delete(y, i)
                            target_m, target_b = best_fit_line(x, y)
                            break
        
        # Mouse button up event
        elif event.type == pygame.MOUSEBUTTONUP:
            if dragging_point is not None:
                dragging_point = None
                # Update best fit line
                target_m, target_b = best_fit_line(x, y)
        
        # Mouse motion event
        elif event.type == pygame.MOUSEMOTION:
            mouse_pos = pygame.mouse.get_pos()
            graph_x, graph_y = screen_to_graph(mouse_pos[0], mouse_pos[1])
            
            # Only handle inside graph area
            if (0 <= graph_x <= graph_width and 0 <= graph_y <= graph_height):
                if dragging_point is not None:
                    x[dragging_point] = graph_x
                    y[dragging_point] = graph_y
                
                # Update custom line in custom line mode
                if custom_line_mode:
                    # Calculate slope based on mouse position relative to center
                    center_x, center_y = graph_width/2, graph_height/2
                    dx = graph_x - center_x
                    dy = center_y - graph_y  # Inverted y-axis
                    
                    # Avoid division by zero and limit extreme slopes
                    if abs(dx) < 5:
                        dx = 5 if dx >= 0 else -5
                    
                    custom_m = dy / dx
                    # Keep slope in reasonable range
                    custom_m = max(min(custom_m, 5), -5)
                    
                    # Adjust intercept to pass through mouse position
                    custom_b = graph_y - custom_m * graph_x

    # Clear the screen
    screen.fill(WHITE)
    
    # Draw axes
    # X-axis
    pygame.draw.line(screen, BLACK, 
                    graph_to_screen(0, graph_height), 
                    graph_to_screen(graph_width, graph_height), 3)
    # Y-axis
    pygame.draw.line(screen, BLACK, 
                    graph_to_screen(0, 0), 
                    graph_to_screen(0, graph_height), 3)
    
    # Draw axis labels
    x_label = axis_font.render("X", True, BLACK)
    y_label = axis_font.render("Y", True, BLACK)
    screen.blit(x_label, graph_to_screen(graph_width - 20, graph_height + 25))
    screen.blit(y_label, graph_to_screen(-20, 10))
    
    # Draw axis ticks and values
    for i in range(0, graph_width, grid_spacing):
        if i > 0:  # Skip origin
            # Tick mark
            pygame.draw.line(screen, BLACK, 
                           graph_to_screen(i, graph_height - 5), 
                           graph_to_screen(i, graph_height + 5), 2)
            # Value
            tick_label = axis_font.render(str(i), True, BLACK)
            screen.blit(tick_label, graph_to_screen(i - 10, graph_height + 15))
    
    for i in range(0, graph_height, grid_spacing):
        if i > 0:  # Skip origin
            # Tick mark
            pygame.draw.line(screen, BLACK, 
                           graph_to_screen(-5, graph_height - i), 
                           graph_to_screen(5, graph_height - i), 2)
            # Value
            tick_label = axis_font.render(str(i), True, BLACK)
            screen.blit(tick_label, graph_to_screen(-40, graph_height - i - 10))
    
    # Draw origin label
    origin_label = axis_font.render("0", True, BLACK)
    screen.blit(origin_label, graph_to_screen(-15, graph_height + 15))
    
    # Draw grid if enabled
    if show_grid:
        for i in range(0, graph_width, grid_spacing):
            pygame.draw.line(screen, GRAY, 
                           graph_to_screen(i, 0), 
                           graph_to_screen(i, graph_height), 1)
        for i in range(0, graph_height, grid_spacing):
            pygame.draw.line(screen, GRAY, 
                           graph_to_screen(0, i), 
                           graph_to_screen(graph_width, i), 1)
    
    # Draw data points
    for i in range(len(x)):
        screen_x, screen_y = graph_to_screen(x[i], y[i])
        pygame.draw.circle(screen, RED, (int(screen_x), int(screen_y)), 7)
        # Highlight dragged point
        if dragging_point == i:
            pygame.draw.circle(screen, ORANGE, (int(screen_x), int(screen_y)), 10, 2)
    
    # Calculate current best fit line metrics
    r_squared = calculate_r_squared(x, y, current_m, current_b)
    mse = calculate_mse(x, y, current_m, current_b)
    
    if custom_line_mode:
        # Draw custom line across graph area
        start_x, start_y = graph_to_screen(0, custom_b)
        end_x, end_y = graph_to_screen(graph_width, custom_m * graph_width + custom_b)
        pygame.draw.line(screen, GREEN, (start_x, start_y), (end_x, end_y), 3)
        
        # Calculate and draw residuals for custom line if enabled
        if show_residuals and len(x) > 0:
            for i in range(len(x)):
                y_pred = custom_m * x[i] + custom_b
                start_x, start_y = graph_to_screen(x[i], y[i])
                end_x, end_y = graph_to_screen(x[i], y_pred)
                pygame.draw.line(screen, GREEN, (start_x, start_y), (end_x, end_y), 2)
        
        # Calculate custom line metrics
        custom_r_squared = calculate_r_squared(x, y, custom_m, custom_b)
        custom_mse = calculate_mse(x, y, custom_m, custom_b)
    else:
        # Animate the line of best fit
        current_m += (target_m - current_m) * animation_speed
        current_b += (target_b - current_b) * animation_speed
        
        # Draw the current line of best fit across graph area
        start_x, start_y = graph_to_screen(0, current_b)
        end_x, end_y = graph_to_screen(graph_width, current_m * graph_width + current_b)
        pygame.draw.line(screen, BLUE, (start_x, start_y), (end_x, end_y), 3)
        
        # Draw residuals if enabled
        if show_residuals and len(x) > 0:
            for i in range(len(x)):
                y_pred = current_m * x[i] + current_b
                start_x, start_y = graph_to_screen(x[i], y[i])
                end_x, end_y = graph_to_screen(x[i], y_pred)
                pygame.draw.line(screen, LIGHT_BLUE, (start_x, start_y), (end_x, end_y), 2)
    
    # Draw buttons
    for button in buttons:
        # Draw button background
        button_rect = pygame.Rect(button_x, button["y"], button_width, button_height)
        
        # Check if this is a toggle button that's active
        is_active = False
        if "toggle" in button and button["toggle"]:
            if (button["action"] == "custom_line" and custom_line_mode) or \
               (button["action"] == "add_points" and adding_point) or \
               (button["action"] == "remove_points" and removing_point):
                is_active = True
        
        # Use different color for active toggle buttons
        if is_active:
            pygame.draw.rect(screen, LIGHT_BLUE, button_rect)
        else:
            pygame.draw.rect(screen, WHITE, button_rect)
        
        pygame.draw.rect(screen, BLACK, button_rect, 2)
        
        # Draw button text
        text_surface = font.render(button["text"], True, BLACK)
        text_rect = text_surface.get_rect(center=button_rect.center)
        screen.blit(text_surface, text_rect)
    
    # Draw title
    title = "Interactive Line of Best Fit Demonstration"
    title_surface = title_font.render(title, True, BLACK)
    screen.blit(title_surface, (20, 20))
    
    # Draw status messages and statistics
    if show_stats:
        stat_x = 20
        stat_y = height - 120
        
        # Instructions
        instructions = [
            "Drag points to move them",
            f"Number of points: {len(x)}",
            f"Current mode: {'Custom Line' if custom_line_mode else 'Add Points' if adding_point else 'Remove Points' if removing_point else 'Drag Points'}"
        ]
        
        for i, instruction in enumerate(instructions):
            text = font.render(instruction, True, BLACK)
            screen.blit(text, (stat_x, stat_y + i * 25))
        
        # Statistics
        stats_x = width - 400
        stats_y = height - 120
        
        if custom_line_mode:
            stats = [
                f"Custom Line: y = {custom_m:.3f}x + {custom_b:.2f}",
                f"R² = {custom_r_squared:.3f}",
                f"MSE = {custom_mse:.2f}"
            ]
        else:
            stats = [
                f"Best Fit Line: y = {current_m:.3f}x + {current_b:.2f}",
                f"R² = {r_squared:.3f}",
                f"MSE = {mse:.2f}"
            ]
        
        for i, stat in enumerate(stats):
            text = font.render(stat, True, BLUE if not custom_line_mode else GREEN)
            screen.blit(text, (stats_x, stats_y + i * 25))
    
    # Update the display
    pygame.display.flip()
    
    # Control the frame rate
    clock.tick(60)

# Quit Pygame
pygame.quit()

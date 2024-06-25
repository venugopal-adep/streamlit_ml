import pygame
import pygame.gfxdraw
import numpy as np
from scipy import stats

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1600, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Boxplot vs Violinplot vs Swarmplot Demo")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
PURPLE = (128, 0, 128)

# Fonts
title_font = pygame.font.Font(None, 48)
text_font = pygame.font.Font(None, 24)

# Generate initial data
data = np.random.normal(loc=400, scale=50, size=200)

# Function to draw a boxplot
def draw_boxplot(surface, data, x, y, width, height):
    q1, median, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    lower_whisker = max(np.min(data), q1 - 1.5 * iqr)
    upper_whisker = min(np.max(data), q3 + 1.5 * iqr)
    
    # Draw box
    pygame.draw.rect(surface, BLUE, (x, y + (q1 - lower_whisker) / (upper_whisker - lower_whisker) * height,
                                     width, (q3 - q1) / (upper_whisker - lower_whisker) * height), 2)
    
    # Draw median line
    pygame.draw.line(surface, RED, (x, y + (median - lower_whisker) / (upper_whisker - lower_whisker) * height),
                     (x + width, y + (median - lower_whisker) / (upper_whisker - lower_whisker) * height), 2)
    
    # Draw whiskers
    pygame.draw.line(surface, BLACK, (x + width // 2, y), (x + width // 2, y + height), 1)
    
    # Draw caps
    pygame.draw.line(surface, BLACK, (x + width // 4, y), (x + width * 3 // 4, y), 2)
    pygame.draw.line(surface, BLACK, (x + width // 4, y + height), (x + width * 3 // 4, y + height), 2)

# Function to draw a violin plot
def draw_violinplot(surface, data, x, y, width, height):
    kde = stats.gaussian_kde(data)
    x_range = np.linspace(min(data), max(data), 100)
    kde_values = kde(x_range)
    max_kde = max(kde_values)
    
    points = []
    for i, kd in enumerate(kde_values):
        points.append((x + width // 2 + kd / max_kde * width // 2, y + i * height // 100))
        points.insert(0, (x + width // 2 - kd / max_kde * width // 2, y + i * height // 100))
    
    pygame.gfxdraw.aapolygon(surface, points, GREEN)
    pygame.gfxdraw.filled_polygon(surface, points, GREEN)

# Function to generate swarmplot positions
def generate_swarmplot_positions(data, x, y, width, height):
    min_val, max_val = min(data), max(data)
    positions = []
    for point in data:
        normalized_y = y + height - (point - min_val) / (max_val - min_val) * height
        offset = np.random.uniform(-width // 3, width // 3)
        positions.append((int(x + width // 2 + offset), int(normalized_y)))
    return positions

# Function to draw a swarmplot
def draw_swarmplot(surface, positions):
    for position in positions:
        pygame.draw.circle(surface, PURPLE, position, 3)

# Generate initial swarmplot positions
swarm_positions = generate_swarmplot_positions(data, WIDTH * 3 // 4 - 100, 150, 200, 500)

# Main game loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                # Generate new data when clicked
                data = np.random.normal(loc=np.random.randint(300, 500), scale=np.random.randint(20, 80), size=200)
                # Regenerate swarmplot positions
                swarm_positions = generate_swarmplot_positions(data, WIDTH * 3 // 4 - 100, 150, 200, 500)

    screen.fill(WHITE)

    # Draw title
    title_text = title_font.render("Boxplot vs Violinplot vs Swarmplot Demo", True, BLACK)
    screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, 20))

    # Draw developer credit
    credit_text = text_font.render("Developed by: Venugopal Adep", True, BLACK)
    screen.blit(credit_text, (WIDTH // 2 - credit_text.get_width() // 2, 70))

    # Draw boxplot
    draw_boxplot(screen, data, WIDTH // 4 - 100, 150, 200, 500)
    boxplot_text = text_font.render("Boxplot", True, BLACK)
    screen.blit(boxplot_text, (WIDTH // 4 - boxplot_text.get_width() // 2, 670))

    # Draw violin plot
    draw_violinplot(screen, data, WIDTH // 2 - 100, 150, 200, 500)
    violinplot_text = text_font.render("Violin Plot", True, BLACK)
    screen.blit(violinplot_text, (WIDTH // 2 - violinplot_text.get_width() // 2, 670))

    # Draw swarmplot
    draw_swarmplot(screen, swarm_positions)
    swarmplot_text = text_font.render("Swarm Plot", True, BLACK)
    screen.blit(swarmplot_text, (WIDTH * 3 // 4 - swarmplot_text.get_width() // 2, 670))

    # Draw instructions
    instructions_text = text_font.render("Click anywhere to generate new data", True, BLACK)
    screen.blit(instructions_text, (WIDTH // 2 - instructions_text.get_width() // 2, 720))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
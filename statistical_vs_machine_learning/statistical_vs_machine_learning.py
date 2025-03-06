import pygame
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib
matplotlib.use("Agg")

# Initialize Pygame
pygame.init()
width, height = 1920, 1080
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Statistics vs Machine Learning Demo")

# Colors
BLUE = (25, 118, 210)
DARK_BLUE = (13, 71, 161)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_GRAY = (240, 240, 240)
GREEN = (76, 175, 80)
RED = (244, 67, 54)

# Fonts
title_font = pygame.font.SysFont('Arial', 50, bold=True)
header_font = pygame.font.SysFont('Arial', 36, bold=True)
text_font = pygame.font.SysFont('Arial', 24)
small_font = pygame.font.SysFont('Arial', 20)

# Current mode
current_mode = "comparison"  # Options: "comparison", "statistics", "machine_learning"
current_model = "linear"  # Fixed to linear now that we've removed the toggle
show_residuals = False
noise_level = 2  # Fixed noise level now that we've removed the toggle
animation_frame = 0

# Generate initial sample data with varied x values
def generate_initial_data():
    global x, y, data
    np.random.seed(42)
    x = np.linspace(0.1, 10, 100)  # Ensure x values are different
    y = 2 * x + 1 + np.random.normal(0, noise_level, 100)
    data = pd.DataFrame({'x': x, 'y': y})

# Initialize data
generate_initial_data()

def plot_to_surface(fig):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    
    # Fix the deprecated methods
    buffer = renderer.buffer_rgba()
    size = canvas.get_width_height()
    
    surf = pygame.image.frombuffer(buffer, size, "RGBA")
    plt.close(fig)  # Close the figure to prevent memory leak
    return surf

def draw_statistical_view():
    # Create matplotlib figure for statistics
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.scatter(data['x'], data['y'], alpha=0.6, color='blue')
    
    # Fit linear model
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(data['x'], data['y'])
    line_x = np.array([min(data['x']), max(data['x'])])
    line_y = slope * line_x + intercept
    
    ax.plot(line_x, line_y, color='red', linewidth=3)
    
    # Add confidence intervals
    if show_residuals:
        pred_y = slope * data['x'] + intercept
        residuals = data['y'] - pred_y
        ax.vlines(data['x'], data['y'], pred_y, colors='gray', alpha=0.3)
        
        # Add statistical information
        equation = f"y = {slope:.2f}x + {intercept:.2f}"
        r_squared = f"R² = {r_value**2:.2f}"
        p_val = f"p-value = {p_value:.4f}"
        std = f"Std Error = {std_err:.4f}"
        ax.text(0.05, 0.95, equation + "\n" + r_squared + "\n" + p_val + "\n" + std, 
                transform=ax.transAxes, verticalalignment='top', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8))
    
    ax.set_title("Statistical Approach: Focus on Inference", fontsize=16, fontweight='bold')
    ax.set_xlabel("X Variable", fontsize=12)
    ax.set_ylabel("Y Variable", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plot_to_surface(fig)

def draw_ml_view():
    # Create matplotlib figure for machine learning
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.scatter(data['x'], data['y'], alpha=0.6, color='green')
    
    # Fit ML model (Linear Regression since we removed the toggle)
    X = data['x'].values.reshape(-1, 1)
    y = data['y'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate predictions
    x_pred = np.linspace(min(data['x']), max(data['x']), 300).reshape(-1, 1)
    y_pred = model.predict(x_pred)
    
    ax.plot(x_pred, y_pred, color='blue', linewidth=3)
    
    # Show prediction metrics
    if show_residuals:
        from sklearn.metrics import mean_squared_error, r2_score
        y_model = model.predict(X)
        mse = mean_squared_error(y, y_model)
        r2 = r2_score(y, y_model)
        
        # Add ML metrics
        metrics = f"MSE = {mse:.2f}\nR² = {r2:.2f}"
        ax.text(0.05, 0.95, metrics, transform=ax.transAxes, verticalalignment='top', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Show residuals
        ax.vlines(data['x'], data['y'], y_model, colors='gray', alpha=0.3)
    
    ax.set_title("Machine Learning Approach: Focus on Prediction", fontsize=16, fontweight='bold')
    ax.set_xlabel("X Variable", fontsize=12)
    ax.set_ylabel("Y Variable", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plot_to_surface(fig)

def draw_comparison_table():
    table_surface = pygame.Surface((width-100, 350))
    table_surface.fill(WHITE)
    
    # Draw table headers
    pygame.draw.rect(table_surface, BLUE, (0, 0, (width-100)//2, 60))
    pygame.draw.rect(table_surface, BLUE, ((width-100)//2, 0, (width-100)//2, 60))
    
    # Draw table grid
    for i in range(5):  # 4 rows + header
        pygame.draw.line(table_surface, BLACK, (0, i*70), (width-100, i*70), 2)
    pygame.draw.line(table_surface, BLACK, ((width-100)//2, 0), ((width-100)//2, 350), 2)
    pygame.draw.rect(table_surface, BLACK, (0, 0, width-100, 350), 2)  # Border
    
    # Headers
    stats_header = header_font.render("Statistics", True, WHITE)
    ml_header = header_font.render("Machine Learning", True, WHITE)
    table_surface.blit(stats_header, (((width-100)//4) - stats_header.get_width()//2, 15))
    table_surface.blit(ml_header, (((width-100)//4)*3 - ml_header.get_width()//2, 15))
    
    # Row 1
    stats_text1 = text_font.render("Emphasis on deep theorems on complex models", True, BLACK)
    ml_text1 = text_font.render("Emphasis on the underlying algorithm", True, BLACK)
    table_surface.blit(stats_text1, (20, 75))
    table_surface.blit(ml_text1, ((width-100)//2 + 20, 75))
    
    # Row 2
    stats_text2 = text_font.render("Focus on hypothesis testing and interpretability", True, BLACK)
    ml_text2 = text_font.render("Focus on predicting accuracy of the model", True, BLACK)
    table_surface.blit(stats_text2, (20, 145))
    table_surface.blit(ml_text2, ((width-100)//2 + 20, 145))
    
    # Row 3
    stats_text3 = text_font.render("Inference on parameter estimation, errors, and predictions", True, BLACK)
    ml_text3 = text_font.render("Inference on prediction", True, BLACK)
    table_surface.blit(stats_text3, (20, 215))
    table_surface.blit(ml_text3, ((width-100)//2 + 20, 215))
    
    # Row 4
    stats_text4 = text_font.render("Deep understanding of simple models", True, BLACK)
    ml_text4 = text_font.render("Theory does not always explain success", True, BLACK)
    table_surface.blit(stats_text4, (20, 285))
    table_surface.blit(ml_text4, ((width-100)//2 + 20, 285))
    
    return table_surface

def draw_button(screen, text, x, y, width, height, color, hover=False):
    if hover:
        pygame.draw.rect(screen, DARK_BLUE, (x, y, width, height))
    else:
        pygame.draw.rect(screen, color, (x, y, width, height))
    pygame.draw.rect(screen, BLACK, (x, y, width, height), 2)  # Border
    
    button_text = text_font.render(text, True, WHITE)
    screen.blit(button_text, (x + width//2 - button_text.get_width()//2, 
                             y + height//2 - button_text.get_height()//2))

def draw_ui():
    global animation_frame
    screen.fill(LIGHT_GRAY)
    
    # Draw title with animation effect
    title = title_font.render("Statistics vs Machine Learning Interactive Demo", True, BLUE)
    offset = np.sin(animation_frame * 0.05) * 5  # Subtle floating effect
    screen.blit(title, (width//2 - title.get_width()//2, 30 + offset))
    animation_frame += 1
    
    # Draw subtitle
    subtitle = text_font.render("The difference between machine learning and statistical learning is their purpose.", True, BLACK)
    screen.blit(subtitle, (width//2 - subtitle.get_width()//2, 100))
    
    subtitle2 = text_font.render("Machine learning models are designed to make accurate predictions, whereas statistical models are designed for inference.", True, BLACK)
    screen.blit(subtitle2, (width//2 - subtitle2.get_width()//2, 130))
    
    if current_mode == "comparison":
        # Draw comparison view
        stats_surf = draw_statistical_view()
        ml_surf = draw_ml_view()
        screen.blit(stats_surf, (width//4 - stats_surf.get_width()//2, 180))
        screen.blit(ml_surf, (3*width//4 - ml_surf.get_width()//2, 180))
        
        # Draw comparison table
        table = draw_comparison_table()
        screen.blit(table, (50, 650))
        
    elif current_mode == "statistics":
        # Full statistics view
        stats_surf = draw_statistical_view()
        stats_title = header_font.render("Statistical Approach", True, BLUE)
        screen.blit(stats_title, (width//2 - stats_title.get_width()//2, 180))
        screen.blit(stats_surf, (width//2 - stats_surf.get_width()//2, 230))
        
        # Add explanation text
        explanation = [
            "Statistical models focus on understanding relationships between variables",
            "They emphasize hypothesis testing and interpretability of parameters",
            "Statistical inference provides insights about confidence intervals and significance",
            "These models are designed to explain rather than just predict"
        ]
        
        for i, line in enumerate(explanation):
            text = small_font.render(line, True, BLACK)
            screen.blit(text, (width//2 - text.get_width()//2, 650 + i*30))
        
    elif current_mode == "machine_learning":
        # Full machine learning view
        ml_surf = draw_ml_view()
        ml_title = header_font.render("Machine Learning Approach", True, GREEN)
        screen.blit(ml_title, (width//2 - ml_title.get_width()//2, 180))
        screen.blit(ml_surf, (width//2 - ml_surf.get_width()//2, 230))
        
        # Add explanation text
        explanation = [
            "Machine learning models focus on making accurate predictions",
            "They emphasize algorithm performance over parameter interpretation",
            "The success of ML models is measured by prediction accuracy metrics",
            "These models can capture complex patterns that may not be theoretically explained"
        ]
        
        for i, line in enumerate(explanation):
            text = small_font.render(line, True, BLACK)
            screen.blit(text, (width//2 - text.get_width()//2, 650 + i*30))
    
    # Get mouse position for hover effects
    mouse_x, mouse_y = pygame.mouse.get_pos()
    
    # Button positions - now with only 2 buttons
    button_width = 300
    button_height = 50
    button_y = height - 100
    button_spacing = 100
    total_buttons_width = 2 * button_width + button_spacing
    button1_x = (width - total_buttons_width) // 2
    button2_x = button1_x + button_width + button_spacing
    
    # Draw buttons with hover effect
    draw_button(screen, "Toggle View Mode", button1_x, button_y, button_width, button_height, 
                BLUE, button1_x <= mouse_x <= button1_x + button_width and button_y <= mouse_y <= button_y + button_height)
    
    draw_button(screen, "Toggle Residuals", button2_x, button_y, button_width, button_height, 
                BLUE, button2_x <= mouse_x <= button2_x + button_width and button_y <= mouse_y <= button_y + button_height)
    
    # Current settings display
    settings_text = text_font.render(
        f"Mode: {current_mode.capitalize()} | Residuals: {'On' if show_residuals else 'Off'}", 
        True, BLACK
    )
    screen.blit(settings_text, (width//2 - settings_text.get_width()//2, height-150))
    
    # Add watermark
    watermark = small_font.render("Developed by : Venugopal Adep", True, (100, 100, 100))
    screen.blit(watermark, (width - watermark.get_width() - 20, height - 30))
    
    pygame.display.flip()

# Main game loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            
            # Button positions (must match those in draw_ui)
            button_width = 300
            button_height = 50
            button_y = height - 100
            button_spacing = 100
            total_buttons_width = 2 * button_width + button_spacing
            button1_x = (width - total_buttons_width) // 2
            button2_x = button1_x + button_width + button_spacing
            
            # Toggle view mode button
            if button1_x <= x <= button1_x + button_width and button_y <= y <= button_y + button_height:
                if current_mode == "comparison":
                    current_mode = "statistics"
                elif current_mode == "statistics":
                    current_mode = "machine_learning"
                else:
                    current_mode = "comparison"
            
            # Toggle residuals button
            elif button2_x <= x <= button2_x + button_width and button_y <= y <= button_y + button_height:
                show_residuals = not show_residuals
    
    draw_ui()
    clock.tick(60)  # Limit to 60 FPS
    
pygame.quit()

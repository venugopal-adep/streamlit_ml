import pygame
import sys

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1600, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("One Hot Encoding Demo")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Fonts
title_font = pygame.font.Font(None, 64)
text_font = pygame.font.Font(None, 32)
small_font = pygame.font.Font(None, 24)

# Categories and their one-hot encodings
categories = ["Dog", "Cat", "Bird", "Fish"]
one_hot_encodings = {
    "Dog": [1, 0, 0, 0],
    "Cat": [0, 1, 0, 0],
    "Bird": [0, 0, 1, 0],
    "Fish": [0, 0, 0, 1]
}

# Button class
class Button:
    def __init__(self, x, y, width, height, text, color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.is_hovered = False

    def draw(self):
        pygame.draw.rect(screen, self.color if not self.is_hovered else GRAY, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)
        text_surface = text_font.render(self.text, True, BLACK)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

# Create buttons
buttons = [Button(200 + i * 300, 300, 200, 50, category, color) 
           for i, (category, color) in enumerate(zip(categories, [RED, GREEN, BLUE, YELLOW]))]

selected_category = None

# Main game loop
clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                for button in buttons:
                    if button.is_clicked(event.pos):
                        selected_category = button.text

    # Clear the screen
    screen.fill(WHITE)

    # Draw title
    title_surface = title_font.render("One Hot Encoding Demo", True, BLACK)
    screen.blit(title_surface, (WIDTH // 2 - title_surface.get_width() // 2, 50))

    # Draw developer credit
    credit_surface = small_font.render("Developed by: Venugopal Adep", True, BLACK)
    screen.blit(credit_surface, (WIDTH // 2 - credit_surface.get_width() // 2, 120))

    # Draw buttons
    for button in buttons:
        button.is_hovered = button.rect.collidepoint(pygame.mouse.get_pos())
        button.draw()

    # Draw explanation
    explanation = text_font.render("Click on a category to see its one-hot encoding", True, BLACK)
    screen.blit(explanation, (WIDTH // 2 - explanation.get_width() // 2, 200))

    # Draw one-hot encoding
    if selected_category:
        encoding = one_hot_encodings[selected_category]
        for i, value in enumerate(encoding):
            color = RED if value == 1 else WHITE
            pygame.draw.rect(screen, color, (200 + i * 300, 400, 200, 100))
            pygame.draw.rect(screen, BLACK, (200 + i * 300, 400, 200, 100), 2)
            text = text_font.render(str(value), True, BLACK)
            screen.blit(text, (300 + i * 300 - text.get_width() // 2, 450 - text.get_height() // 2))

        # Draw explanation of the encoding
        explanation = f"One-hot encoding for '{selected_category}':"
        explanation_surface = text_font.render(explanation, True, BLACK)
        screen.blit(explanation_surface, (WIDTH // 2 - explanation_surface.get_width() // 2, 550))

        binary_explanation = f"[{', '.join(map(str, encoding))}]"
        binary_surface = text_font.render(binary_explanation, True, BLACK)
        screen.blit(binary_surface, (WIDTH // 2 - binary_surface.get_width() // 2, 600))

    # Update the display
    pygame.display.flip()

    # Control the frame rate
    clock.tick(60)

# Quit Pygame
pygame.quit()
sys.exit()
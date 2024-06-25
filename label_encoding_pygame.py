import pygame
import random

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1600, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Label Encoding Demo")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (100, 149, 237)
GREEN = (50, 205, 50)
RED = (255, 99, 71)
YELLOW = (255, 215, 0)

# Fonts
title_font = pygame.font.Font(None, 64)
text_font = pygame.font.Font(None, 32)
button_font = pygame.font.Font(None, 36)

# Categories and their encoded values
categories = ["Dog", "Cat", "Bird", "Fish", "Rabbit"]
encoded_values = {}

# Button properties
button_width, button_height = 200, 50

# Animal properties
animal_size = 120
animals = []

def draw_text(text, font, color, x, y):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=(x, y))
    screen.blit(text_surface, text_rect)

def draw_button(text, x, y, width, height, color):
    pygame.draw.rect(screen, color, (x, y, width, height))
    draw_text(text, button_font, BLACK, x + width // 2, y + height // 2)

def create_animal():
    category = random.choice(categories)
    x = random.randint(50, WIDTH - 350 - animal_size)
    y = random.randint(150, HEIGHT - 200 - animal_size)
    return {"category": category, "x": x, "y": y, "encoded": False}

def main():
    clock = pygame.time.Clock()
    running = True
    encoding_mode = False
    current_encoding = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                
                # Add Animal button
                if WIDTH - 220 <= mouse_x <= WIDTH - 20 and 150 <= mouse_y <= 200:
                    animals.append(create_animal())
                
                # Start Encoding button
                elif WIDTH - 220 <= mouse_x <= WIDTH - 20 and 250 <= mouse_y <= 300:
                    encoding_mode = True
                    current_encoding = 0
                    encoded_values.clear()
                
                # Reset button
                elif WIDTH - 220 <= mouse_x <= WIDTH - 20 and 350 <= mouse_y <= 400:
                    animals.clear()
                    encoded_values.clear()
                    encoding_mode = False
                
                # Check if an animal is clicked during encoding mode
                elif encoding_mode:
                    for animal in animals:
                        if (animal["x"] <= mouse_x <= animal["x"] + animal_size and 
                            animal["y"] <= mouse_y <= animal["y"] + animal_size):
                            if animal["category"] not in encoded_values:
                                encoded_values[animal["category"]] = current_encoding
                                current_encoding += 1
                            animal["encoded"] = True

        screen.fill(WHITE)

        # Draw title and developer info
        draw_text("Label Encoding Demo", title_font, BLACK, WIDTH // 2, 50)
        draw_text("Developed by: Venugopal Adep", text_font, GRAY, WIDTH // 2, 100)

        # Draw animals and their labels
        for animal in animals:
            color = BLUE if animal["encoded"] else RED
            pygame.draw.rect(screen, color, (animal["x"], animal["y"], animal_size, animal_size))
            draw_text(animal["category"], text_font, BLACK, 
                      animal["x"] + animal_size // 2, animal["y"] + animal_size // 2)

            if animal["encoded"]:
                encoded_value = encoded_values[animal["category"]]
                draw_text(str(encoded_value), text_font, YELLOW, 
                          animal["x"] + animal_size // 2, animal["y"] + animal_size + 20)

        # Draw buttons
        draw_button("Add Animal", WIDTH - 220, 150, button_width, button_height, GREEN)
        draw_button("Start Encoding", WIDTH - 220, 250, button_width, button_height, YELLOW)
        draw_button("Reset", WIDTH - 220, 350, button_width, button_height, RED)

        # Draw legend
        legend_x, legend_y = 50, HEIGHT - 150
        draw_text("Legend:", text_font, BLACK, legend_x, legend_y)
        draw_text("Red: Not Encoded", text_font, RED, legend_x + 150, legend_y)
        draw_text("Blue: Encoded", text_font, BLUE, legend_x + 350, legend_y)
        draw_text("Yellow: Encoded Value", text_font, YELLOW, legend_x + 550, legend_y)

        # Draw encoding explanation
        explanation_x, explanation_y = WIDTH - 200, 450
        draw_text("Label Encoding:", text_font, BLACK, explanation_x, explanation_y)
        for idx, (category, value) in enumerate(encoded_values.items()):
            draw_text(f"{category} -> {value}", text_font, BLACK, explanation_x, explanation_y + 40 * (idx + 1))

        # Draw instructions
        if encoding_mode:
            draw_text("Click on animals to encode them", text_font, GREEN, WIDTH // 2, HEIGHT - 50)
        else:
            draw_text("Press 'Start Encoding' to begin", text_font, BLACK, WIDTH // 2, HEIGHT - 50)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
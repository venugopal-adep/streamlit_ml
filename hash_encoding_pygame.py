import pygame
import pygame.freetype
import hashlib

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1600, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hash Encoding Demo")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# Fonts
title_font = pygame.freetype.Font(None, 48)
text_font = pygame.freetype.Font(None, 24)
input_font = pygame.freetype.Font(None, 32)

# Input box
input_box = pygame.Rect(200, 200, 400, 50)
input_text = ''
input_active = False

# Hash visualization
hash_boxes = []
for i in range(8):
    hash_boxes.append(pygame.Rect(200 + i * 150, 400, 100, 100))

# Button
button = pygame.Rect(650, 200, 150, 50)
button_text = "Hash It!"

def get_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

def draw():
    screen.fill(WHITE)
    
    # Draw title and developer info
    title_font.render_to(screen, (WIDTH // 2 - 150, 50), "Hash Encoding Demo", BLACK)
    text_font.render_to(screen, (WIDTH // 2 - 100, 100), "Developed by: Venugopal Adep", BLACK)
    
    # Draw input box
    pygame.draw.rect(screen, BLACK, input_box, 2)
    text_surface, _ = input_font.render(input_text, BLACK)
    screen.blit(text_surface, (input_box.x + 5, input_box.y + 5))
    
    # Draw button
    pygame.draw.rect(screen, BLUE, button)
    text_font.render_to(screen, (button.x + 20, button.y + 15), button_text, WHITE)
    
    # Draw hash visualization
    for i, box in enumerate(hash_boxes):
        pygame.draw.rect(screen, GRAY, box)
        if input_text:
            hash_value = get_hash(input_text)
            text_font.render_to(screen, (box.x + 10, box.y + 40), hash_value[i*4:(i+1)*4], BLACK)
    
    # Draw explanation
    text_font.render_to(screen, (200, 550), "Hash encoding converts input text into a fixed-size string of characters.", BLACK)
    text_font.render_to(screen, (200, 600), "Each box represents 4 characters of the 32-character MD5 hash.", BLACK)
    text_font.render_to(screen, (200, 650), "Try different inputs to see how the hash changes!", BLACK)

    pygame.display.flip()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if input_box.collidepoint(event.pos):
                input_active = True
            else:
                input_active = False
            
            if button.collidepoint(event.pos):
                print(f"Hash of '{input_text}': {get_hash(input_text)}")
        
        if event.type == pygame.KEYDOWN:
            if input_active:
                if event.key == pygame.K_RETURN:
                    print(f"Hash of '{input_text}': {get_hash(input_text)}")
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                else:
                    input_text += event.unicode
    
    draw()

pygame.quit()
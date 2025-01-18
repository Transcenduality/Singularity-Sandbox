import pygame
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from scipy.ndimage import sobel
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

# Constants
WINDOW_SIZE = 1024
VIRTUAL_RESOLUTION = 512 # Lower the computation resolution for better performance
FPS = 60
ZOOM_SPEED = 0.05
PAN_SPEED = 10.0
WHITE = 255, 255, 255

# Initialize Pygame
pygame.init()
pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE), pygame.OPENGL | pygame.DOUBLEBUF)
pygame.display.set_caption("Singularity Sandbox")
clock = pygame.time.Clock()

# Initialize Pygame font
pygame.font.init()
font = pygame.font.SysFont('Arial', 24)

# OpenGL setup
glEnable(GL_TEXTURE_2D)
glClearColor(0.0, 0.0, 0.0, 1.0)

# Texture setup
grid_texture_id = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, grid_texture_id)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

# Simulation parameters
zoom_level = -1.0
iteration = 0.0
offset_x, offset_y = 0.0, 0.0

# Energy calculation function
def calculate_energy(x, y, iteration, zoom_level):
    scale = 2 ** -zoom_level
    r = np.sqrt(x**2 + y**2) * scale
    theta = np.arctan2(y, x)
    movement = iteration * 0.2
    energy = np.cos(movement + r) + 0.5 * np.sin(1.5 * r) + 0.25 * np.sin(2.5 * theta)
    return energy

# Grid rendering
def render_visible_grid(zoom_level, iteration):
    scale = 2 ** -zoom_level
    half_res = VIRTUAL_RESOLUTION // 2
    x_indices = (np.arange(VIRTUAL_RESOLUTION) - half_res) * scale + offset_x
    y_indices = (np.arange(VIRTUAL_RESOLUTION) - half_res) * scale + offset_y
    x_grid, y_grid = np.meshgrid(x_indices, y_indices)
    energy = calculate_energy(x_grid, y_grid, iteration, zoom_level)
    return energy.astype(np.float32)

def invert_color(rgb):
    """Invert an RGB color."""
    return 1.0 - rgb  # Invert each channel (assumes values in range [0, 1])

def grid_to_rgb_array(grid):
    global_min, global_max = np.min(grid), np.max(grid)
    normalized = (grid - global_min) / (global_max - global_min + 1e-6)

    # Compute gradient magnitude to represent changes between neighbors
    grad_x = sobel(normalized, axis=0, mode='reflect')
    grad_y = sobel(normalized, axis=1, mode='reflect')
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude = (gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min() + 1e-6)

    # Map normalized values to a gradient between red, green, and blue
    r = normalized
    g = 1 - np.abs(0.5 - normalized) * 2  # Peaks at 0.5 (green in RGB space)
    b = 1 - normalized

    # Create RGB channels and apply inversion
    rgb = np.stack([r, g, b], axis=-1)
    inverted_rgb = invert_color(rgb)  # Invert the colors

    # Convert RGB to HSV
    hsv = rgb_to_hsv(inverted_rgb)

    # Maximize saturation
    hsv[..., 1] = 1.0

    # Increase value (brightness)
    hsv[..., 2] = np.clip(hsv[..., 2] * 1.5, 0, 1)  # Increase brightness by 50%

    # Convert HSV back to RGB
    saturated_rgb = hsv_to_rgb(hsv)

    # Apply brightness scaling using gradient magnitude
    brightness = gradient_magnitude[..., np.newaxis] * 2 # Broadcast brightness to RGB channels
    rgb_array = saturated_rgb * brightness

    # Clamp values and convert to 0-255
    rgb_array = np.clip(rgb_array * 255, 0, 255).astype(np.uint8)
    return rgb_array

# Render text overlay
def draw_text_overlay(screen, text_list, font, text_color, bg_color=(0, 0, 0, 128)):
    """Render text onto a Pygame surface with a translucent background, then convert it to an OpenGL texture."""
    text_surface = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE), pygame.SRCALPHA)

    # Calculate dimensions of the background box
    y_offset = 10
    padding = 10
    max_text_width = 0
    total_height = 0
    for text in text_list:
        rendered_text = font.render(text, True, text_color)
        max_text_width = max(max_text_width, rendered_text.get_width())
        total_height += rendered_text.get_height() + 5

    # Draw the translucent background box
    bg_rect = pygame.Rect(5, 5, max_text_width + 2 * padding, total_height + padding)
    pygame.draw.rect(text_surface, bg_color, bg_rect)

    # Render each text line
    y_offset = bg_rect.top + padding
    for text in text_list:
        rendered_text = font.render(text, True, text_color)
        text_surface.blit(rendered_text, (bg_rect.left + padding, y_offset))
        y_offset += rendered_text.get_height() + 5

    # Convert Pygame surface to a string buffer for OpenGL
    text_data = pygame.image.tostring(text_surface, "RGBA", True)

    # Bind OpenGL texture
    text_texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, text_texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WINDOW_SIZE, WINDOW_SIZE, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

    # Render the texture as a fullscreen quad
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0)
    glVertex2f(-1, -1)
    glTexCoord2f(1, 0)
    glVertex2f(1, -1)
    glTexCoord2f(1, 1)
    glVertex2f(1, 1)
    glTexCoord2f(0, 1)
    glVertex2f(-1, 1)
    glEnd()

    glDeleteTextures(1, [text_texture])  # Clean up the texture

# Main loop
running = True
while running:
    glClear(GL_COLOR_BUFFER_BIT)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        offset_y += PAN_SPEED * (2 ** -zoom_level)
    if keys[pygame.K_s]:
        offset_y -= PAN_SPEED * (2 ** -zoom_level)
    if keys[pygame.K_a]:
        offset_x -= PAN_SPEED * (2 ** -zoom_level)
    if keys[pygame.K_d]:
        offset_x += PAN_SPEED * (2 ** -zoom_level)
    if keys[pygame.K_UP]:
        zoom_level += ZOOM_SPEED
    if keys[pygame.K_DOWN]:
        zoom_level -= ZOOM_SPEED

    iteration += 1
    visible_grid = render_visible_grid(zoom_level, iteration)
    rgb_array = grid_to_rgb_array(visible_grid)

    glBindTexture(GL_TEXTURE_2D, grid_texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, VIRTUAL_RESOLUTION, VIRTUAL_RESOLUTION, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb_array)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0)
    glVertex2f(-1, -1)
    glTexCoord2f(1, 0)
    glVertex2f(1, -1)
    glTexCoord2f(1, 1)
    glVertex2f(1, 1)
    glTexCoord2f(0, 1)
    glVertex2f(-1, 1)
    glEnd()

    # Draw overlay text
    text_list = [
        f"Zoom Level: {zoom_level:.2f}",
        f"Iteration: {int(iteration)}",
    ]
    draw_text_overlay(pygame.display.get_surface(), text_list, font, (255, 255, 255))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()

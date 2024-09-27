import pygame
import time
from game import *
from Projeto_Eco.vision import *
# Inicializa o Pygame
pygame.init()

# Define as dimensões da janela
sw = 800
sh = 800

# Carrega a imagem de fundo em pixel art do universo
background = pygame.image.load('C:/Users/pwsou/Documents/Projeto_prototipagem/src/pixel_universe_bg.jpg')

# Define as cores
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)

# Configura a janela
win = pygame.display.set_mode((sw, sh))
pygame.display.set_caption('Space Game')

# Variáveis do jogo
clock = pygame.time.Clock()
running = True
credits = 5
game_started = False
press_start_visible = True
blink_timer = 0

# Carrega fontes
font_title = pygame.font.Font(None, 80)  # Fonte para o título
font_menu = pygame.font.Font(None, 60)   # Fonte para o menu
font_credits = pygame.font.Font(None, 40) # Fonte para os créditos

def draw_start_screen():
    win.blit(background, (0, 0))
    
    # Desenha o título
    title_text = font_title.render("ECORAMA", True, YELLOW)
    win.blit(title_text, (sw // 2 - title_text.get_width() // 2, sh // 4))
    
    # Desenha o menu
    menu_text = font_credits.render("Reciclar é dar uma nova vida ao que já existe", True, WHITE)
    win.blit(menu_text, (sw // 2 - menu_text.get_width() // 2, sh // 2))
    
    # Desenha o texto "Press Start" piscando
    if press_start_visible:
        start_text = font_menu.render("INSIRA UM MATERIAL RECICLAVEL", True, WHITE)
        win.blit(start_text, (sw // 2 - start_text.get_width() // 2, sh // 2 + 100))
    
    # Desenha os créditos
    credits_text = font_credits.render(f"CREDITS: {credits}", True, WHITE)
    win.blit(credits_text, (sw - credits_text.get_width() - 20, 20))
    
    pygame.display.update()

# Loop principal
while running:
    clock.tick(60)
    blink_timer += 1
    
    if blink_timer % 30 == 0:
        press_start_visible = not press_start_visible

    for event in pygame.event.get():
        
        
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            cod = valida_item()
            if cod == 1:
                game_started = True    
                running = False
                startGame()
            # if event.key == pygame.K_RETURN:  # Enter key to start the game
               
    draw_start_screen()

pygame.quit()

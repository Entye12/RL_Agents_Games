import matplotlib.pyplot as plt
import numpy as np
import pygame
import random
from enum import Enum
from collections import namedtuple

pygame.init()


font = pygame.font.Font(None, 36) 

class Direction(Enum):
  RIGHT = 1
  LEFT = 2
  UP = 3
  DOWN = 4

# RGB COLORS
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (200,0,0)
BLUE1 = (0,0,255)
BLUE2 = (0,100,255)

Point = namedtuple('Point', 'x, y')
BLOCK_SIZE = 20
SPEED = 10

class SnakeGame:

  def __init__(self, weight = 640, height = 480):
    self.weight = weight
    self.height = height

    # init the display
    self.display = pygame.display.set_mode((self.weight,self.height))
    pygame.display.set_caption("Snake")
    self.clock = pygame.time.Clock()

    # init the game state 
    self.direction = Direction.RIGHT

    self.head = Point(self.weight//2, self.height//2)
    self.snake = [self.head,
                  Point(self.head.x - BLOCK_SIZE, self.head.y),
                  Point(self.head.x - 2*BLOCK_SIZE, self.head.y)
                  ]

    self.score = 0
    self.food = None
    self._place_food()
  

  def _place_food(self):
    x = random.randint(0,(self.weight - BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
    y = random.randint(0,(self.height - BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
    self.food = Point(x,y)
    if self.food in self.snake :
      self._place_food()


  
  def play_step(self):

    # 1 collect user input
    for event in pygame.event.get():
      if event.type == pygame.QUIT :
        pygame.quit()
        quit()
      
      if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_LEFT:
          self.direction = Direction.LEFT
        elif event.key == pygame.K_RIGHT:
          self.direction = Direction.RIGHT
        elif event.key == pygame.K_UP:
          self.direction = Direction.UP
        elif event.key == pygame.K_DOWN:
          self.direction = Direction.DOWN



    # 2 move
    self._move(self.direction)             # update the head
    self.snake.insert(0,self.head)

    # 3 check if game over
    game_over = False

    if self._is_collision():
      game_over = True
      return game_over, self.score

    # 4 place new food or just move the snake
    if self.head == self.food : 
      # eat the food
      self.score += 1
      self._place_food()
    else : 
      # remove the last part of our snake so it doesn't get bigger
      self.snake.pop()

    # 5 update the ui and clock
    self._update_ui()
    self.clock.tick(SPEED)

    # 6 return score if game over
    

    return game_over, self.score

  def _update_ui(self):
    self.display.fill(BLACK)
    # draw the snake in blue
    for pt in self.snake:
      pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
      pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
    
    # draw the food in red
    pygame.draw.rect(self.display , RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

    text = font.render("Score : " + str(self.score), True, WHITE)
    self.display.blit(text, [0,0])
    pygame.display.flip()

  
  def _move(self,direction):
    x, y = self.head.x, self.head.y

    if direction == Direction.RIGHT:
      x += BLOCK_SIZE
    elif direction == Direction.LEFT:
      x -= BLOCK_SIZE
    elif direction == Direction.UP:
      y -= BLOCK_SIZE
    elif direction == Direction.DOWN:
      y += BLOCK_SIZE
    
    self.head = Point(x,y)
  
  def _is_collision(self):
    # hits boundary
    if self.head.x > self.weight - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.height - BLOCK_SIZE or self.head.y < 0:
      return True
    # hits de snake
    if self.head in self.snake[1:] :
      return True
    return False




if __name__ == '__main__' : 
  game = SnakeGame()

  # game loop
  while True:
    game_over, score = game.play_step()

    # break if game over
    if game_over :
      break
  print('Final Score : ', score)
  
  pygame.quit()


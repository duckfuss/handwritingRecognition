'''INCOMPLETE - use interface.py
Idea is to have discrete steps downwards in colour based off neighbour's colours
- see changes to code in dimlyLit(Cell)/udpate()

'''
import pygame
import numpy as np
import neuralNetwork
import barGraph

print("begin")
#neuralNetwork.trainNetwork()


#PYGAME Innit
WIDTH = 400
HEIGHT= 400

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()


colours = {
    "Dark0":(237,239,240),
    "Dark1":(225,226,227),
    "Dark2":(98,99,99)
}

class Grid(): 
    def __init__(self) -> None:
        self.cell_size = 10
        self.cell_padding = 1
        self.grid_width = 28
        self.grid_height = 28
        self.grid = np.zeros((self.grid_width, self.grid_height), dtype=object)
    def fill_grid(self):
        self.grid = np.zeros((self.grid_width, self.grid_height), dtype=object)
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                    self.grid[x][y] = DimlyLit(self.cell_size, x, y, (self.cell_padding+self.cell_size))

    def fillGridInput(self, data):
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                    col = data[x][y]
                    self.grid[x][y]= Lit(self.cell_size, x, y, (self.cell_padding+self.cell_size), (col,col,col))

    def update(self, mouse_pos=None, draw_only=True):
        new_grid = np.copy(self.grid)
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                cell = self.grid[x][y]
                if not draw_only:
                    cell.update(self.grid)
                if mouse_pos != None: #mouse click code
                    if self.mouseOnCell(mouse_pos,x,y):
                        new_grid[x][y] = Lit(self.cell_size, x, y, (self.cell_padding+self.cell_size))
                        if checkInBounds(x,y+1, self.grid):
                            new_grid[x][y+1] = Lit(self.cell_size, x, y+1, (self.cell_padding+self.cell_size))
                        self.grid = np.copy(new_grid)
                cell.draw()
    
    def numericOutput(self):
        data = np.zeros((self.grid_width, self.grid_height))
        for row in range(len(self.grid)):
            for col in range(len(self.grid[row])):
                data[row][col] = self.grid[row][col].colour[0]
        return data.T
        

    def mouseOnCell(self, mouse_pos, x,y):
        if mouse_pos[0]/(self.cell_padding+self.cell_size) >= x and mouse_pos[1]/(self.cell_padding+self.cell_size) >= y and mouse_pos[0]/(self.cell_padding+self.cell_size) <= x+(self.cell_size/(self.cell_padding+self.cell_size)) and mouse_pos[1]/(self.cell_padding+self.cell_size) <= y+self.cell_size/((self.cell_padding+self.cell_size)):
            return True
        else:
            return False

class Cell():
    def __init__(self, w, x, y, padding, colour=(0,0,0)) -> None:
        self.colour = colour
        self.width = w
        self.grid_coords = (x,y)
        self.rect = pygame.Rect(self.grid_coords[0]*padding, self.grid_coords[1]*padding, self.width, self.width)
        self.neighbours = {(1,0):"", (0,1):"", (1,1):"", (-1,0):"", (0,-1):"", (-1,-1):"", (-1,1):"", (1,-1):""}
    def find_neighbours(self, grid):
        cumColour = 0
        for neighbour in self.neighbours:
            if checkInBounds(self.grid_coords[0]+neighbour[0],self.grid_coords[1]+neighbour[1], grid):
                neighbourCol = list(grid[self.grid_coords[0]+neighbour[0]][self.grid_coords[1]+neighbour[1]].colour)
                cumColour += neighbourCol[0]
        return cumColour
    def setColour(self,colour):
        self.colour = colour
    def draw(self):
        pygame.draw.rect(screen, self.colour, self.rect)
    
def checkInBounds(x, y, grid):
    if len(grid) > x and len(grid[0]) > y:
        if x > 0 and y > 0:
            return True
    return False

class DimlyLit(Cell):
    def __init__(self, w, x, y, padding) -> None:
        super().__init__(w, x, y, padding)
        self.p = padding
    def update(self, grid):
        nColour = self.find_neighbours(grid)
        if nColour > 240*3:
            self.setColour((200,200,200))
        elif nColour > 360:
            self.setColour((180,180,180))
        elif nColour > 240:
            self.setColour((80,80,80))

        


class Lit(Cell):
    def __init__(self, w, x, y, padding, colour = (240,240,240)) -> None:
        super().__init__(w, x, y, padding)
        self.p = padding
        self.setColour(colour)
    def update(self, grid):
        #self.setColour((255,255,255))
        pass


grid = Grid()
grid.fill_grid()

tick = pygame.USEREVENT + 0
pygame.time.set_timer(tick, 100)


# TEST:
tData = neuralNetwork.Data()
tData.loadMNIST()
tdIndex= 0

# MAIN LOOP
while True:
    screen.fill((100,100,100))
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            pygame.quit()
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_SPACE:
                grid.fill_grid()
                data = grid.numericOutput()
                output = neuralNetwork.compute(data)
                barGraph.update(drawOnly=False, data=output)
            if ev.key == pygame.K_RETURN:
                data = grid.numericOutput()
                output = neuralNetwork.compute(data)#, verbose=True)
                print(output)
                barGraph.update(drawOnly=False, data=output)
            if ev.key == pygame.K_t:
                print("t")
                grid.fillGridInput(tData.testIm[tdIndex].T)
                print("answer:", tData.testLab[tdIndex])
            if ev.key == pygame.K_UP:
                tdIndex += 1
            if ev.key == pygame.K_DOWN and tdIndex > 0:
                tdIndex -=1

        if ev.type == tick:
            grid.update(draw_only=False)
    if pygame.mouse.get_pressed()[0] == True:
        grid.update(mouse_pos=pygame.mouse.get_pos())
        data = grid.numericOutput()
        output = neuralNetwork.compute(data)
        barGraph.update(drawOnly=False, data=output)
    barGraph.update(drawOnly=True)
    grid.update()
    pygame.display.update()
    clock.tick(240)


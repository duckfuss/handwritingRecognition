import pygame
import numpy as np
import neuralNetwork
import barGraph

print("begin")
neuralNetwork.trainNetwork() # If the network hasn't learnt anything, check that this line is uncommented


#PYGAME Innit
WIDTH = 308
HEIGHT= 308

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('neural network input')
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
        # don't bother trying to read this, it works
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
        cumColour = [0,0,0]
        for neighbour in self.neighbours:
            if checkInBounds(self.grid_coords[0]+neighbour[0],self.grid_coords[1]+neighbour[1], grid):
                neighbourCol = list(grid[self.grid_coords[0]+neighbour[0]][self.grid_coords[1]+neighbour[1]].colour)
                cumColour = [sum(x) for x in zip(cumColour, neighbourCol)]
        return [x//10 for x in cumColour]
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
        avgColour = self.find_neighbours(grid)
        if avgColour[0] > 50:
            self.setColour(avgColour)

class Lit(Cell):
    def __init__(self, w, x, y, padding, colour = (240,240,240)) -> None:
        super().__init__(w, x, y, padding)
        self.p = padding
        self.setColour(colour)
    def update(self, grid):
        pass


grid = Grid()
grid.fill_grid()

tick = pygame.USEREVENT + 0
pygame.time.set_timer(tick, 100)


# TEST - this doesn't load into the neural net, only the test data:
tData = neuralNetwork.Data()
#tData.loadFashionMNIST()
tData.loadMNIST()
tdIndex= 0

def reCalc(verbose=False):
    data = grid.numericOutput()
    output = neuralNetwork.compute(data, verbose=verbose)#, verbose=True)
    barGraph.update(drawOnly=False, data=output)


# MAIN LOOP
while True:
    screen.fill((100,100,100))
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            pygame.quit()
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_SPACE:
                grid.fill_grid()
                reCalc()
            if ev.key == pygame.K_RETURN:
                reCalc()
            if ev.key == pygame.K_UP:
                tdIndex += 1
                grid.fillGridInput(tData.testIm[tdIndex].T)
                print("answer:", tData.testLab[tdIndex])
                reCalc()
            if ev.key == pygame.K_DOWN and tdIndex > 0:
                tdIndex -=1
                grid.fillGridInput(tData.testIm[tdIndex].T)
                print("answer:", tData.testLab[tdIndex])
                reCalc()
        if ev.type == tick:
            grid.update(draw_only=False)
    if pygame.mouse.get_pressed()[0] == True:
        grid.update(mouse_pos=pygame.mouse.get_pos())
        reCalc()
    barGraph.update(drawOnly=True)
    grid.update()
    pygame.display.update()
    clock.tick(240)


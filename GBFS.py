import os, copy
#from os import path
import matplotlib.pyplot as plt

def findSE(maze): # Return the pos of start node and end node.
    for i in range(len(maze)):
      for j in range(len(maze[0])):
          if maze[i][j]=='S': # Find start node
              start=(i,j)
          elif maze[i][j]==' ': # Find end node
              if (i==0) or (i==len(maze)-1) or (j==0) or (j==len(maze[0])-1):
                  end=(i,j)
          else:
              pass
    return start, end

def read_file(file_name: str):
    f=open(file_name,'r')
    n_bonus_points = int(next(f)[:-1])
    bonus_points = []
    for i in range(n_bonus_points):
        x, y, reward = map(int, next(f)[:-1].split(' '))
        bonus_points.append((x, y, reward))

    text=f.read()
    matrix=[list(i) for i in text.splitlines()]
    f.close()

    return bonus_points, matrix

#Greedy Best First Search
def matrixToIndex(state): # Takes matrix and returns index of actor
    temp = state.copy()
    idx = [(index, rows.index(agent)) for index, rows in enumerate(temp) if agent in rows]      # Getting the index of the Agent (A)
    x = idx[0][0] # Will store the x-coordinate of agent
    y = idx[0][1] # Will store the y-coordinate of agent
    return idx,x,y
class Node:
    def __init__(self, state, parent, operator, moves): # Default Constructor
        self.state = state
        self.parent = parent # Moves
        self.operator = operator
        self.moves = moves
        self.m_dist = 0 # Initially Manhatten distance is 0
        
        manhattan_distance(self)

    def __eq__(self, other): # Comparing two nodes
        return (type(self) == type(other)) and (self.state == other.state)

    def __lt__(self, other): # Sorting nodes
        return self.m_dist < other.m_dist
def create_node(state, parent, operator, cost): # This function creates a node of current state
    return Node(state, parent, operator, cost)
def manhattan_distance(node):
    idx,ax,ay = matrixToIndex(node.state)
    endx,endy = goal_index[0][0],goal_index[0][1]
    distance = abs(ax - endx) + abs(ay -endy)
    node.m_dist = distance
def expand_node(node, n): # This function performs all possible operations
    expanded_nodes = []

    temp_state1 = move_up(node.state,n)
    if (temp_state1 is not None):
        # The state is expanded with upward operation
        temp_node1 = create_node(temp_state1,node,"up",node.moves+1) 
        expanded_nodes.append(temp_node1) # Appending the expanded nodes in the list
    temp_state2 = move_left(node.state,n)
    
    if (temp_state2 is not None):
        temp_node2 = create_node(temp_state2,node,"left",node.moves+1) 
        expanded_nodes.append(temp_node2)
    temp_state3 = move_right(node.state,n)
    
    if (temp_state3 is not None):
        temp_node3 = create_node(temp_state3,node,"right",node.moves+1)   
        expanded_nodes.append(temp_node3)
    temp_state = move_down(node.state,n)
    
    if (temp_state is not None):
        temp_node = create_node(temp_state,node,"down",node.moves+1) 
        expanded_nodes.append(temp_node)
    
    return expanded_nodes

def move_left(state, n):
    swap = copy.deepcopy(state)
    idx,x,y = matrixToIndex(swap) # Returning index of actor

    if (swap[x][y-1] == "x" or y <= 0): # Checks for unallowed moves 
        return None
    else:
        # Moving the agent one cell left
        swap[x][y-1], swap[x][y] = swap[x][y] , swap[x][y-1]
        return swap
def move_right(state, n):
    swap = copy.deepcopy(state)
    idx,x,y = matrixToIndex(swap) # Returning index of actor
    
    if (y >= n-1 or swap[x][y+1] == "x"): # Checks for unallowed moves
        return None
    else:
        # Moving the agent one cell left
        swap[x][y+1], swap[x][y] = swap[x][y] , swap[x][y+1]
        return swap
def move_up(state, n):
    swap = copy.deepcopy(state)
    idx,x,y = matrixToIndex(swap) # Returning index of actor
    
    if (swap[x-1][y] == "x" or x <= 0 ): # Checks for unallowed moves
        return None
    else:
        # Moving the agent one cell above
        swap[x-1][y], swap[x][y] = swap[x][y] , swap[x-1][y]
        return swap
def move_down(state, n):
    swap = copy.deepcopy(state)
    idx,x,y = matrixToIndex(swap) # Returning index of actor
    
    if (swap[x+1][y] == "x" or x >= n-1): # Checks for unallowed moves
        return None
    else:
        # Moving the agent one cell left
        swap[x+1][y], swap[x][y] = swap[x][y] , swap[x+1][y]
        return swap

def gbfs(start, n, path, bonus):
    i, bpoint = 1, 0
    # Getting agent's current position
    temp_idx,x1,y1 = matrixToIndex(start)

    to_be_expanded = [] # Array of all nodes in one level/depth
    visited_nodes = [] # Tuple that will contain the nodes already visited
    start_node = create_node(start,None,None,0) # Starting node is stored
    to_be_expanded.append(start_node) # Adding first node to the expanding array
        
    while to_be_expanded:
        to_be_expanded.sort() # Sorting the nodes wrt to cost (ascending)            
        current_node = to_be_expanded.pop(0) # Getting the node with the smallest cost
        visited_nodes.append(current_node) # Adding the index to visited array
        new_idx,x2,y2 = matrixToIndex(current_node.state) # Getting agent's new position
            
        #print('Times:', i, ': Popping', (x2,y2))
        
        for _, point in enumerate(bonus): # Check if node is bonus-point?
            if (x2,y2) == ((point[0],point[1])):
                bpoint += point[2] # Update the bpoint
        path.append(tuple((x2,y2)))

        if (new_idx == goal_index): # If goal state is found, return
            print('Cost: ', i + bpoint)
            return current_node
        else:
            node_array = expand_node(current_node,n) # Expanding the neigbours
                
            for node in node_array: # Checking conditions of A*
                if (node not in to_be_expanded):
                    if (node not in visited_nodes):
                        to_be_expanded.append(node)
        i += 1
    return path



#Visualization
def visualize_maze(matrix, bonus, start, end, route=None):
    """
    Args:
      1. matrix: The matrix read from the input file,
      2. bonus: The array of bonus points,
      3. start, end: The starting and ending points,
      4. route: The route from the starting point to the ending one, defined by an array of (x, y), e.g. route = [(1, 2), (1, 3), (1, 4)]
    """
    #1. Define walls and array of direction based on the route
    walls=[(i,j) for i in range(len(matrix)) for j in range(len(matrix[0])) if matrix[i][j]=='x']
    
    
    route = []

    # GBFS
    gbfs(maze,len(maze[0]),route, bonus)

    # Path where node is popped
    print(route)

    if route:
        direction=[]
        for i in range(1,len(route)):
            if route[i][0]-route[i-1][0]>0:
                direction.append('v') #^
            elif route[i][0]-route[i-1][0]<0:
                direction.append('^') #v        
            elif route[i][1]-route[i-1][1]>0:
                direction.append('>')
            else:
                direction.append('<')
        #print(direction)
        direction.pop(0)
        

    #2. Drawing the map
    ax=plt.figure(dpi=100).add_subplot(111)

    for i in ['top','bottom','right','left']:
        ax.spines[i].set_visible(False)


    #print([i[1] for i in walls])
    #print([-i[0] for i in walls])

    plt.scatter([i[1] for i in walls],[-i[0] for i in walls],marker='X',s=100,color='black')
    
    plt.scatter([i[1] for i in bonus],[-i[0] for i in bonus],marker='P',s=100,color='green')

    plt.scatter(start[1],-start[0],marker='*', s=100,color='gold')

    if route:
        for i in range(len(route)-2):
            plt.scatter(route[i+1][1],-route[i+1][0],marker=direction[i],color='silver')

    plt.text(end[1],-end[0],'EXIT',color='red', horizontalalignment='center', verticalalignment='center')
    plt.xticks([])
    plt.yticks([])
    plt.show() #Show the map

    print(f'Starting point (x, y) = {start[0], start[1]}')
    print(f'Ending point (x, y) = {end[0], end[1]}')
    
    for _, point in enumerate(bonus):
      print(f'Bonus point at position (x, y) = {point[0], point[1]} with point {point[2]}')

if __name__=='__main__':
    # Change file name to change maze (maze:1-5, maze1:1-3)
    bonus_points, maze = read_file('maze1.txt')
    start, end = findSE(maze)
    
    agent = maze[start[0]][start[1]]    #Size of maze
    goal_index = []
    goal_index.append(end)
    agent_index = [(index, rows.index(agent)) for index, rows in enumerate(maze) if agent in rows]  # Storing the current index of agent
    x = agent_index[0][0] # Will store the x-coordinate of agent
    y = agent_index[0][1]
    
    visualize_maze(maze,bonus_points,start,end)
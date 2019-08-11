import util
from sudoku import SudokuSearchProblem
from maps import MapSearchProblem
import copy

################ Node structure to use for the search algorithm ################
class Node:
    def __init__(self, state, action, path_cost, parent_node, depth):
        self.state = state
        self.action = action
        self.path_cost = path_cost
        self.parent_node = parent_node
        self.depth = depth

########################## DFS for Sudoku ########################
## Choose some node to expand from the frontier with Stack like implementation
def sudokuDepthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    Return the final values dictionary, i.e. the values dictionary which is the goal state  
    """

    def convertStateToHash(values):
        """ 
        values as a dictionary is not hashable and hence cannot be used directly in the explored/visited set.
        This function changes values dict into a unique hashable string which can be used in the explored set.
        You may or may not use this
        """
        l = list(sorted(values.items()))
        modl = [a+b for (a, b) in l]
        return ''.join(modl)

    ## YOUR CODE HERE
    # print(convertStateToHash(problem.getStartState()))
    stack1 = util.Stack();
    stack1.push(problem.getStartState())
    visit = []
    visit.append(convertStateToHash(problem.getStartState()))
    while not stack1.isEmpty():
        top = stack1.pop()
        if(problem.isGoalState(top)):
            return top

        successors = problem.getSuccessors(top)
        # print(successors)
        for succ in successors:
            str1 = convertStateToHash(succ[0])
            if str1 not in visit:
                visit.append(str1)
                stack1.push(succ[0])

    # util.raiseNotDefined()

######################## A-Star and DFS for Map Problem ########################
## Choose some node to expand from the frontier with priority_queue like implementation

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """

    return 0

def heuristic(state, problem):
    # It would take a while for Flat Earther's to get accustomed to this paradigm
    # but hang in there.

    """
        Takes the state and the problem as input and returns the heuristic for the state
        Returns a real number(Float)
    """
    return util.points2distance(((problem.G.node[state]['x'],0,0), (problem.G.node[state]['y'],0,0)),
                                ((problem.G.node[problem.end_node]['x'],0,0), (problem.G.node[problem.end_node]['y'],0,0)))

    # util.raiseNotDefined()

def AStar_search(problem, heuristic=nullHeuristic):

    """
        Search the node that has the lowest combined cost and heuristic first.
        Return the route as a list of nodes(Int) iterated through starting from the first to the final.
    """
    queue = util.PriorityQueue()
    st = problem.getStartState()
    queue.push([st,[st],0],heuristic(st,problem))
    visit = []

    # while not queue.isEmpty():
    #     top = queue.pop()
    #     # print(top[1])
    #     if(problem.isGoalState(top[0])):
    #         return top[1]

    #     successors = problem.getSuccessors(top[0])
    #     # print(successors)

    #     for succ in successors:
    #         # print(succ)
    #         if succ[0] not in visit:
    #             # print("h",top)
    #             visit.append(succ[0])
    #             x1 = top[1][:]
    #             x1.append(succ[0])
    #             # print(top[1])
    #             # print([succ[0],top1])
    #             queue.push([succ[0],x1,top[2]+succ[2]],top[2]+succ[2])


    while not queue.isEmpty():
        top = queue.pop()
        # print(top[1])
        if(problem.isGoalState(top[0])):
            return top[1]

        # print(successors)
        if top[0] not in visit:
            visit.append(top[0])
            successors = problem.getSuccessors(top[0])
            for succ in successors:
                # print("h",top)
                x1 = top[1][:]
                x1.append(succ[0])
                # print(top[1])
                # print([succ[0],top1])
                queue.push([succ[0],x1,top[2]+succ[2]],top[2]+succ[2]+heuristic(succ[0],problem))


    # util.raiseNotDefined()
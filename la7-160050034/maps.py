import networkx as nx
import osmnx as ox
import util
import itertools

class MapSearchProblem:
    """
    This class outlines the structure of a search problem
    """

    def  __init__(self, G, start_node, end_node):
        self.G = G
        self.start_node = start_node
        self.end_node = end_node
        self.nodes_expanded = 0

    def getStartState(self):
        """
        Returns the start state for the search problem which will be of type Int
        """
        return self.start_node
        # util.raiseNotDefined()

    def isGoalState(self, node):
        """
        node: Search state of type Int
        Returns True if node is the goal state otherwise False
        """
        if(node == self.end_node):
            return True
        return False
        # util.raiseNotDefined()

    def getSuccessors(self, node):
        """
        node: Search state of type Int

        For a given state, this should return a list of triples, (successor(Int),
        action(Edge ID of type Int), stepCost(Float)), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        ## Maintain for bookkeeping purposes    
        self.nodes_expanded += 1 
        ## Dont overuse this function since we calculate the nodes expanded using this

        successors = []
        ## YOUR CODE HERE
        # print(self.G.nodes())
        x1 = self.G.nodes()
        # print(x1[0])
        # print(type(x1))
        # print(self.G.node[self.start_node])
        # print(self.G[self.start_node][self.end_node])

        for succ in self.G.neighbors(node):
            successors.append((succ,0,self.G[node][succ][0]['length']))

        return successors

if __name__ == "__main__":

    pass
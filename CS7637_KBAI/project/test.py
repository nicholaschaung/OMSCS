#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 13:10:21 2019

@author: nicholas
"""

# DO NOT MODIFY THIS FILE.
#
# Any modifications to this file will not be used when grading your project.
# If you have any questions, please email the TAs.
#
# The main driver file for the project. You may edit this file to change which
# problems your Agent addresses while debugging and designing, but you should
# not depend on changes to this file for final execution of your project. Your
# project will be graded using our own version of this file.

import os
import sys
import csv

from Agent import Agent
from ProblemSet import ProblemSet
from RavensGrader import grade

def getNextLine(r):
    return r.readline().rstrip()

# The project's main solve method. This will generate your agent's answers
# to all the current problems.
#
# You do not need to use this method.
def solve():
    sets=[] # The variable 'sets' stores multiple problem sets.
            # Each problem set comes from a different folder in /Problems/
            # Additional sets of problems will be used when grading projects.
            # You may also write your own problems.

    r = open(os.path.join("Problems","ProblemSetList.txt"))    # ProblemSetList.txt lists the sets to solve.
    line = getNextLine(r)                                   # Sets will be solved in the order they appear in the file.
    while not line=="":                                     # You may modify ProblemSetList.txt for design and debugging.
        sets.append(ProblemSet(line))                       # We will use a fresh copy of all problem sets when grading.
        line=getNextLine(r)                                 # We will also use some problem sets not given in advance.

    #print(sets)
    #print()
    for s in sets:
        if s.name == "Basic Problems B":
            #print(s.name)
            #print(s.problems)
            #print()
            for p in s.problems:
                if p.name == "Basic Problem B-11":
                    print(p.name, p.problemType, p.problemSetName, p.hasVisual, p.hasVerbal)
                    print(p.figures)
                    print()
                    for f in p.figures:
                        if f == "A":
                            print(f, p.figures[f].name, p.figures[f].visualFilename)
                            print(p.figures[f].objects)
                            print()
                            for o in p.figures[f].objects:
                                print(p.figures[f].objects[o].name, p.figures[f].objects[o].attributes)
                            print()

    # Initializing problem-solving agent from Agent.java
    agent=Agent()   # Your agent will be initialized with its default constructor.
                    # You may modify the default constructor in Agent.java

    # Running agent against each problem set
    for set in sets:
        if set.name == "Basic Problems B":
            for problem in set.problems:   # Your agent will solve one problem at a time.
                if problem.name == "Basic Problem B-11":
                    answer = agent.Solve(problem)
                    print('Final answer:', str(answer))
    
    
    #with open("AgentAnswers.csv","w") as results:     # Results will be written to ProblemResults.csv.
                                                        # Note that each run of the program will overwrite the previous results.
                                                        # Do not write anything else to ProblemResults.txt during execution of the program.
        #results.write("ProblemSet,RavensProblem,Agent's Answer\n")
        #for set in sets:
            #for problem in set.problems:   # Your agent will solve one problem at a time.
                #try:
                #answer = agent.Solve(problem)  # The problem will be passed to your agent as a RavensProblem object as a parameter to the Solve method
                                                # Your agent should return its answer at the conclusion of the execution of Solve.

                #results.write("%s,%s,%d\n" % (set.name, problem.name, answer))
    r.close()

# The main execution will have your agent generate answers for all the problems,
# then generate the grades for them.
def main():
    solve()
    #grade()

if __name__ == "__main__":
    main()

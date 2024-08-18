# Q-Learning-Implementation
---
An intrepid adventurer must make his way through a maze. He must collect the key, located somewhere in the maze, before leaving through the door.
Without the key, the door will not open. Additionally, the maze contains dangerous fire hazards, which the adventurer wishes to avoid.
The environment is modelled as a Markov Decision Process: the states correspond to the adventurers (x,y) position in the maze and whether he currently possesses the key.
Rewards are 0 on every time step the agent has not reached the end of the key, 1 for successfully completing the maze, and -100 if the agent touches the fire.
The agent would like to finish the maze as quickly as possible, so the environment's discount rate is γ = 0.95.
---
Here I have implemented Q-Learning. The agent uses an epsilon-greedy behavior policy to ensure adequate exploration of the maze. 
By running main.py, an automated graphic will display a final episode after training has been completed, and a learning curve, showing the length of episodes over time will be generated in "learning_curve.png".

This code was from a project I completed for my CMPUT 261 course at the University of Alberta.

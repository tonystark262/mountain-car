# OPENAI GYM MOUNTAIN CAR IMPLEMENTATION
This is my second game playing AI [Open AI mountain car](https://gym.openai.com/envs/MountainCar-v0/). In this I have tried out Deep-Q-Learning(DQN) along with more sophisticated techniques such as Double DQN and DDQN with prioritized experience replay.

I apologise that the repository is not well maintained but I will try to organize it in future. Till the it can be used as reference for anyone trying out above techniques though you have to open and look inside the code to know it. There are certain difference in the files with same name and I apologise again for not keeping track of those. 

### Common Errors in Implementation
I had spent a lot of time debugging the errors and was frustated when the model could not converge after 15 days of attempt. So, I provide my errors which took long time to debug:
- First was the shape of the output matrix for a batch to compare it with the true output. Carefully look that numerical computation considers series( shape [** ]) and 1-D matrix (shape [1, **]) differently.
- Correctly look at the type conversion. Look where there are chances of a value to become float rather than int types. this mostly happens in the label matrix.

### Note
I would like to point out that the each model is not able to converge perfectly though it should have been a childs play for DDQN+PER to make it converge. Someohw it converges to approx -100 reward during 1800-2000 iterations but then quickly falls back to -200.0 avg. reward. I haven't been able to sort out the problem for months. If someone has suggestions please push your repository and if possible with documentation(:P) for everyones benefit.   

### References 
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf)
- https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
- [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf)

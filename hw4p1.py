
import numpy as np
import matplotlib.pyplot as plt

'''
==============================================================================================================================================================================
Plotting Optimal Policy using value iteration
==============================================================================================================================================================================
'''


def value_iteration_for_gamblers(p_h, theta=0.0001, discount_factor=1.0):
    """
    Args:
        p_h: Probability of the coin coming up heads
    """
    
    # The reward is zero on all transitions except those on which the gambler reaches his goal ($100),
    # when it is +1.
    rewards = np.zeros(101)
    rewards[100] = 1 
    
    # We introduce two dummy states corresponding to termination with capital of 0 and 100
    V = np.zeros(101)
    
    def one_step_lookahead(s, V, rewards):
        """
        Helper function to calculate the value for all action in a given state.
        
        Args:
            s: The gamblers capital. Integer.
            V: The vector that contains values at each state. 
            rewards: The reward vector.
                        
        Returns:
            A vector containing the expected value of each action. 
            Its length equals to the number of actions.
        """
        
        A = np.zeros(101)
        stakes = range(1, min(s, 100-s)+1) # Your minimum bet is 1, maximum bet is min(s, 100-s).
        for a in stakes:
            # rewards[s+a], rewards[s-a] are immediate rewards.
            # V[s+a], V[s-a] are values of the next states.
            # This is the core of the Bellman equation: The expected value of your action is 
            # the sum of immediate rewards and the value of the next state.
            A[a] = p_h * (rewards[s+a] + discount_factor * V[s+a]) + (1-p_h) * (rewards[s-a] + discount_factor * V[s-a])
        
        return A
    
    while True:
        # Stopping condition
        delta = 0
        # Update each state...
        for s in range(1, 100):
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, V, rewards)
            # print(s,A,V) Use this to debug 
            best_action_value = np.max(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function
            V[s] = best_action_value        
        # Check if we can stop 
        if delta < theta:
            break
    
    # Create a deterministic policy using the optimal value function
    policy = np.zeros(100)
    for s in range(1, 100):
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(s, V, rewards)
        best_action = np.argmax(A)
        # Always take the best action
        policy[s] = best_action
    
    return policy, V

def plot_policy(policy, p_h):
    plt.figure(figsize=(10, 5))
    plt.stem(policy, use_line_collection=True)
    plt.xlabel('Current Capital')
    plt.ylabel('Stakes')
    plt.title('Optimal Policy with ph = {}'.format(p_h))
    plt.grid()
    plt.show()

# Compute and plot the policy for ph=0.25
policy, _ = value_iteration_for_gamblers(0.25)
plot_policy(policy, 0.25)

# Compute and plot the policy for ph=0.55
policy, _ = value_iteration_for_gamblers(0.55)
plot_policy(policy, 0.55)



'''
==============================================================================================================================================================================
Every-Visit Monte Carlo prediction algorithm to estimate the value function of the optimal policy
==============================================================================================================================================================================
'''

import numpy as np

# Define parameters
num_states = 101
num_episodes = 10000
gamma = 1.0  # discount factor

# Get optimal policy for ph=0.55
policy, _ = value_iteration_for_gamblers(0.55)

# Initialize value function and count of visits for each state
V = np.zeros(num_states)
returns_sum = np.zeros(num_states)
returns_count = np.zeros(num_states)

# Monte Carlo loop
for i_episode in range(1, num_episodes+1):
    # Print out the episode number every 1000 episodes
    if i_episode % 1000 == 0:
        print(f"Episode: {i_episode}/{num_episodes}")
        break

    # Generate an episode
    state = np.random.randint(1, num_states-1)  # start at a random state
    episode = []  # initialize episode list
    done = False  # game is not done

    while not done:
        action = policy[state]  # get action from policy
        next_state = state + action if np.random.uniform() < 0.55 else state - action  # simulate game
        next_state = int(next_state)
        i = 0 # Variable to void infinite loop

        # Get reward, check if game is done
        if next_state == 100:
            reward = 1
            done = True
        elif next_state == 0:
            reward = 0
            done = True
        else:
            reward = 0

        episode.append((state, reward))  # store state and reward in episode
        state = next_state


    # Update value function
    for i in range(len(episode)):
        state, reward = episode[i]
        # Calculate discounted return

        G = sum([x[1]*(gamma**i) for i, x in enumerate(episode[i:])])

        # Accumulate returns
        returns_sum[state] += G
        returns_count[state] += 1.0

        # Calculate average return
        V[state] = returns_sum[state] / returns_count[state]
  


print("Value function:")
print(V)

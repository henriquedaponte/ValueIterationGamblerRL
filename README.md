# ValueIterationGamblerRL
Dynamic programming value iteration for gambler's problem (problem statement in README file)


Problem statement: 
a gambler repeatedly places bets on the outcomes of a sequence of coin
flips. If the flip is heads, then she wins as many dollars as she has bet on that flip; if it is tails, she
loses all the bet. The game ends when the gambler reaches her goal of making $100, or when she runs
out of money. At each time, the gambler chooses how much of her money to bet on the next flip.
This situation can be formulated as an undiscounted, episodic, finite MDP. The state is the
gambler’s capital s ∈ {0, 1, 2, . . . , 99, 100}, and her actions are stakes (i.e., how much to bet) a ∈
{0, 1, . . . , min{s, 100 − s}}. The rewards are zero for all state transitions, except when the transition
leads to the gambler reaching her $100 goal, in which case the reward is +1. Let ph be the probability
that the coin flips heads.

Work:
Wrote a code that uses Value Iteration to find the gambler’s optimal policy (i.e, a mapping from levels
of capital to the stakes). In particular, plotted the optimal policy for ph = 0.25 and ph = 0.55. These
plots should have the current capital on the x-axis, and the stakes chosen on the y-axis

Implemented the Every-Visit Monte Carlo prediction algorithm to estimate the value function of the
optimal policy you found in part (a) for ph = 0.55. 

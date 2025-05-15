# Plan
- use Atropos agent.
    - each turn ScoredDataGroup is system prompt + observation (inc memories) + previous actions (summarised thoughts) + current FULL thoughts & memories
    - Each turn has window of system prompt + previous trajectory, generates group_size alternatives 
- use Atropos RM
    - for all group_size alternatives
        - Provide same input as agent
        - thinking block + Q score estimate

best-of-n selection (only 1 canonical trajectory, greedy selection, same as blackjack)
    - for each alternative in G alternatives, get Q scores (includes plan implicitly)
    - Mean of all Q scores used for alternative
    - select best alternative action for actual play

Per-turn
    - get any rewards. Use as scores for policy, compute advantage for previous steps, with gamma factor for discounts
    - Use actual rewards to score RM Q-score estimate accuracy
    - ScoredDataGroups for Policy & RM created

End of episode:
    - Get all final episode rewards (win/loss)
    - Use to recalculate scores on ALL previous ScoredDataGroups (both Policy and RM)
    - Return

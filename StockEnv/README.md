# Environment
- StockEnvLS-v0
    - action sapce : Continuous space ranging from -1 to 1.
    - observation space : [balance, shares, close, macd, rsi, cci, adx]
    - reward funciton : (end asset - begin asset) * 1e-4
    - This environment allows the agent to take both short and long positions.

- StockEnvSharp-v0
    - action sapce : Continuous space ranging from -1 to 1.
    - observation space : [balance, shares, close, macd, rsi, cci, adx]
    - reward funciton : (end asset - begin asset) * 1e-4 + Sharp_t
    - This environment allows the agent to take both short and long positions.

- StockLongOnly-v1
    - action space : Continuous space ranging from -1 to 1.
    - observation space : $[balance, shares] + [close, macd, rsi, cci, adx] * WINDOW SIZE$
    - reward funciton : [end asset - begin asset] * 1e-4
    - This environment allows the agent to take only long positions. 

- StockMarketLongDiscreteMask-v1
    - action space : Discrete space ranging from -100 to 100.
    - observation space : $[balance, shares]+ [close, macd, rsi, cci, adx] * WINDOW SIZE$  
    - reward funciton : (end asset - begin asset) * 1e-4
    - This environment uses a discrete action space, enables an action mask to filter out invalid actions, and allows only long positions.

- StockEnvLongDiscreteMask-v2
    - action space : Discrete space ranging from -100 to 100.
    - observation space : $[balance, shares]+ [close, macd, rsi, cci, adx] * WINDOW SIZE$  
    - reward funciton : The reward is the points earned from a single trade. If the points earned are positive, the reward is calculated as $earned * 0.1$; if negative, it is $earned * 1.5 * 0.1$.
    - This environment uses a discrete action space, enables an action mask to filter out invalid actions, and allows only long positions.

- StockEnvOnlyOneShare-v1
    - action space : Discrete space ranging from -1 to 1.
    - observation space : $[balance, shares]+ [close, macd, rsi, cci, adx] * WINDOW SIZE$  
    - reward funciton : The reward is the points earned from a single trade. If the points earned are positive, the reward is calculated as $earned * 0.1$; if negative, it is $earned * 1.5 * 0.1$.
    - This environment uses a discrete action space restricted to -1 to 1, enables an action mask to filter out invalid actions, and allows only long positions. The observation of the balance is fixed at the initial balance.

- StockEnvOnlyOneShare-v2
    - action space : Discrete space ranging from -1 to 1.
    - observation space : $[balance, shares]+ [close, macd, rsi, cci, adx] * WINDOW SIZE$  
    - reward funciton : The reward is the points earned from a single trade. If the points earned are positive, the reward is calculated as $earned * 0.1$; if negative, it is $earned * 1.5 * 0.1$.
    - This environment uses a discrete action space restricted to -1 to 1, incorporates an action mask to filter out invalid actions, and allows only long positions. The contract price in this environment changes dynamically, making it more realistic. The observation of the balance is fixed at the initial balance.


- StockEnvOnlyOneShare-v3
    - action space : Discrete space ranging from -1 to 1.
    - observation space : $[balance, shares]+ [close, macd, rsi, cci, adx] * WINDOW SIZE$  
    - reward funciton : The reward is the points earned from a single trade. If the points earned are positive, the reward is calculated as $earned * 0.1$; if negative, it is $earned * 1.5 * 0.1$.
    - This environment uses a discrete action space restricted to -1 to 1, incorporates an action mask to filter out invalid actions, and allows short and long positions. The contract price in this environment changes dynamically, making it more realistic. It now allows both long and short positions. The observation of the balance is fixed at the initial balance.

- StockEnvOnlyOneShare-v4
    - action space : Discrete space ranging from -1 to 1.
    - observation space : $[balance, shares]+ [close, macd, rsi, cci, adx] * WINDOW SIZE$  
    - reward funciton : The reward is the points earned from a single trade. If the points earned are positive, the reward is calculated as $earned * 0.1$; if negative, it is $earned * 1.5 * 0.1$.
    - This environment uses a discrete action space restricted to -1 to 1, incorporates an action mask to filter out invalid actions. The contract price in this environment changes dynamically, making it more realistic. It now allows only short positions. The observation of the balance is fixed at the initial balance.

- StockEnvOnlyOneShare-v5
    - action space : Discrete space ranging from -1 to 1.
    - observation space : $[balance, shares]+ [close, macd, rsi, cci, adx] * WINDOW SIZE$  
    - reward funciton : The reward is the points earned from a single trade. If the points earned are positive, the reward is calculated as $earned * 0.1$; if negative, it is $earned * 1.5 * 0.1$.
    - This environment uses a discrete action space restricted to -1 to 1, incorporates an action mask to filter out invalid actions. The contract price in this environment changes dynamically, making it more realistic. It now allows only long positions. The observation of the balance is fixed at the initial balance. In this environment, synthetic data is used. I added synthetic data by reversing the price trend of each month in the original dataset to balance it (the dataset includes both the original and synthetic data).

- StockEnvOnlyOneShare-v6
    - action space : Discrete space ranging from -1 to 1.
    - observation space : $[balance, shares]+ [close, macd, rsi, cci, adx] * WINDOW SIZE$  
    - reward funciton : The reward is the points earned from a single trade. If the points earned are positive, the reward is calculated as $earned * 0.1$; if negative, it is $earned * 1.5 * 0.1$.
    - This environment uses a discrete action space restricted to -1 to 1, incorporates an action mask to filter out invalid actions. The contract price in this environment changes dynamically, making it more realistic. It now allows only short positions. The observation of the balance is fixed at the initial balance. In this environment, synthetic data is used. I added synthetic data by reversing the price trend of each month in the original dataset to balance it (the dataset includes both the original and synthetic data).

- StockEnvNoLimit-v1
    - action space : Discrete space ranging from -1 to 1.
    - observation space : $[balance, shares]+ [close, macd, rsi, cci, adx] * WINDOW SIZE$  
    - reward funciton : The reward is the points earned from a single trade. If the points earned are positive, the reward is calculated as $earned * 0.1$; if negative, it is $earned * 1.5 * 0.1$.
    - This environment uses a discrete action space(-100 ~ 100), enables an action mask to filter out invalid actions, and allows only long positions.


- StockEnvNoLimit-v2
    - action space : Discrete space ranging from -1 to 1.
    - observation space : $[balance, shares]+ [close, macd, rsi, cci, adx] * WINDOW SIZE$  
    - reward funciton : The reward is the points earned from a single trade. If the points earned are positive, the reward is calculated as $earned * 0.1$; if negative, it is $earned * 1.5 * 0.1$.
    - This environment uses a discrete action space restricted to -100 to 100, incorporates an action mask to filter out invalid actions, and allows only long positions. The contract price in this environment changes dynamically, making it more realistic.

- StockEnvMargin-v1
    - action space : Discrete space ranging from -1 to 1.
    - observation space : $[balance, shares]+ [close, macd, rsi, cci, adx] * WINDOW SIZE$  
    - reward funciton : The reward is the points earned from a single trade. If the points earned are positive, the reward is calculated as $earned * 0.1$; if negative, it is $earned * 1.5 * 0.1$.
    - This environment uses a discrete action space (-100 to 100), enables an action mask to filter out invalid actions, and allows only long positions. A margin system has been added to the environment.
    
- StockEnvMargin-v2
    - action space : Discrete space ranging from -1 to 1.
    - observation space : $[balance, shares]+ [close, macd, rsi, cci, adx] * WINDOW SIZE$  
    - reward funciton : The reward is the points earned from a single trade. If the points earned are positive, the reward is calculated as $earned * 0.1$; if negative, it is $earned * 1.5 * 0.1$.
    - This environment uses a discrete action space (-100 to 100), enables an action mask to filter out invalid actions, and allows only long positions. A margin system has been added to the environment. The contract price in this environment changes dynamically, making it more realistic.